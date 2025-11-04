#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChannelAlign — Tkinter GUI (no PySimpleGUI) for multi‑channel TIFF registration (translation-only)
and export of aligned multi‑channel TIFFs, with an alignment CSV log.

License: MIT

Dependencies (pip):
    numpy tifffile scikit-image scipy

GUI stack:
    tkinter / ttk (bundled with Python; on Linux you may need to install `python3-tk`)

Summary of features:
    • Batch mode (“Per‑channel folders”): put channel A files in one folder, channel B in another, etc.
      Files are paired by sorted order across folders (make sure lists line up!).
    • Multi‑channel mode (“Multi‑channel TIFF(s)”): load TIFFs that already contain multiple channels.
    • Sub‑pixel translation-only registration via phase cross‑correlation.
    • Optional cropping to the common overlap to avoid padded edges.
    • Saves a (C,Y,X) TIFF per image with simple channel metadata.
    • Writes a CSV log with per‑channel shifts (dy, dx) and magnitudes.
    • Responsive UI: progress bars, logs, cancel/stop.

Limitations:
    • Translation-only (no rotation/scale).
    • All channels in a set are center-cropped to a common (min H, min W) before alignment.
    • Batch pairing is by sorted filename order across folders.

Packaging tip:
    pyinstaller --noconsole --onefile ChannelAlign_Tk.py
"""

from __future__ import annotations

import os
import sys
import glob
import csv
import math
import queue
import threading
import argparse
import datetime as _dt
from typing import List, Tuple, Optional, Callable

import numpy as np
from tifffile import imread, imwrite
from skimage.registration import phase_cross_correlation
from scipy import ndimage

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None
    _HAVE_MPL = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


SUPPORTED_EXTS = ('.tif', '.tiff')


# ------------------------------- Utility functions -------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def natural_sort_key(s: str):
    """Sort helper so that 'img2.tif' comes before 'img10.tif'."""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def list_tiff_files(folder: str, pattern: str = '*.tif') -> List[str]:
    """Return sorted list of TIFF file paths matching pattern within folder."""
    if not folder:
        return []
    # Expand both *.tif and *.tiff if user uses *.tif
    patterns = []
    base = (pattern or '*.tif').strip()
    if base.lower().endswith('.tif'):
        patterns = [base, base + 'f']  # '*.tif' and '*.tiff'
    elif base.lower().endswith('.tiff'):
        patterns = [base]
    else:
        patterns = [base]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files = [f for f in files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    files = sorted(set(files), key=natural_sort_key)
    return files


def read_2d_tiff(path: str) -> np.ndarray:
    """Read a TIFF and return a 2D numpy array (Y, X)."""
    arr = imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale TIFF at '{path}', got shape {arr.shape}")
    return arr


def center_crop_to(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Center-crop array to (H, W); center-pad with zeros if target bigger."""
    Ht, Wt = target_hw
    H, W = arr.shape[-2], arr.shape[-1]
    # Crop if larger
    y0 = max((H - Ht) // 2, 0)
    x0 = max((W - Wt) // 2, 0)
    y1 = y0 + min(Ht, H)
    x1 = x0 + min(Wt, W)
    cropped = arr[y0:y1, x0:x1]
    # Pad if smaller
    pad_top = max((Ht - cropped.shape[0]) // 2, 0)
    pad_bottom = max(Ht - cropped.shape[0] - pad_top, 0)
    pad_left = max((Wt - cropped.shape[1]) // 2, 0)
    pad_right = max(Wt - cropped.shape[1] - pad_left, 0)
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)
    return cropped


def compute_overlap_bounds(h: int, w: int, shifts: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """Largest (y0:y1, x0:x1) rectangle valid across images after applying shifts."""
    y_starts, y_stops, x_starts, x_stops = [], [], [], []
    for dy, dx in shifts:
        y_start_i = max(0, int(math.ceil(dy)))
        y_stop_i  = min(h, int(math.floor(h + dy)))
        x_start_i = max(0, int(math.ceil(dx)))
        x_stop_i  = min(w, int(math.floor(w + dx)))
        y_starts.append(y_start_i); y_stops.append(y_stop_i)
        x_starts.append(x_start_i); x_stops.append(x_stop_i)
    y0 = max([0] + y_starts); y1 = min([h] + y_stops)
    x0 = max([0] + x_starts); x1 = min([w] + x_stops)
    if y1 <= y0 or x1 <= x0:
        y0, y1, x0, x1 = 0, h, 0, w
    return y0, y1, x0, x1


def promote_dtype(dtypes: List[np.dtype]) -> np.dtype:
    """Choose a common dtype for stacking; prefer highest integer type among inputs; float->float32."""
    if any(np.issubdtype(dt, np.floating) for dt in dtypes):
        return np.dtype('float32')
    ranks = [np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32')]
    max_rank = max((ranks.index(np.dtype(dt)) if np.dtype(dt) in ranks else -1) for dt in dtypes)
    if max_rank >= 0:
        return ranks[max_rank]
    return np.result_type(*dtypes)


def cast_back(arr: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    if arr.dtype == target_dtype:
        return arr
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        arr = np.clip(arr, info.min, info.max)
        return arr.round().astype(target_dtype, copy=False)
    return arr.astype(target_dtype, copy=False)


def stack_and_save_tiff(
    out_path: str,
    aligned_images: List[np.ndarray],
    channel_names: Optional[List[str]] = None,
    dtype: Optional[np.dtype] = None
) -> None:
    """Stack images as (C, Y, X) and save TIFF with axes metadata."""
    if dtype is None:
        dtype = promote_dtype([im.dtype for im in aligned_images])
    stacked = np.stack([cast_back(im, dtype) for im in aligned_images], axis=0)
    metadata = {'axes': 'CYX'}
    if channel_names:
        metadata['Channel'] = list(channel_names)
    imwrite(out_path, stacked, metadata=metadata)


def sanitize_filename_component(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "sample"


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.percentile(arr, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _make_dual_overlay(ref_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    ref_norm = _normalize_for_display(ref_image)
    target_norm = _normalize_for_display(target_image)
    zeros = np.zeros_like(ref_norm)
    return np.stack([target_norm, zeros, ref_norm], axis=-1)


def generate_pair_overlay(
    qc_path: str,
    ref_fixed: np.ndarray,
    target_before: np.ndarray,
    target_after: np.ndarray
) -> None:
    if not _HAVE_MPL or plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    try:
        axes[0].imshow(_make_dual_overlay(ref_fixed, target_before))
        axes[0].set_title("Before alignment")
        axes[1].imshow(_make_dual_overlay(ref_fixed, target_after))
        axes[1].set_title("After alignment")
        for ax in axes:
            ax.axis("off")
        fig.savefig(qc_path, dpi=150)
    finally:
        plt.close(fig)


def generate_dual_channel_qc(
    qc_path: str,
    fixed_ch0: np.ndarray,
    moving_ch0: np.ndarray,
    aligned_ch0: np.ndarray,
    fixed_ch1: np.ndarray,
    moving_ch1: np.ndarray,
    aligned_ch1: np.ndarray,
    dy: float,
    dx: float,
    error: float,
    fixed_label: str = "Fixed",
    moving_label: str = "Moving"
) -> None:
    """
    Generate a comprehensive QC overlay showing both channel 0 and channel 1 alignments.
    
    CONSISTENT LABELING:
        Channel 0 (Ch0) = Nuclei (DAPI) - shown in Row 0
        Channel 1 (Ch1) = Protein marker - shown in Row 1
    """
    if not _HAVE_MPL or plt is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    try:
        # Row 0: Channel 0 (Nuclei)
        axes[0, 0].imshow(_normalize_for_display(fixed_ch0), cmap="gray")
        axes[0, 0].set_title(f"{fixed_label}\nChannel 0 (Nuclei)", fontsize=10)
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(_normalize_for_display(aligned_ch0), cmap="gray")
        axes[0, 1].set_title(f"{moving_label} (aligned)\nChannel 0 (Nuclei)", fontsize=10)
        axes[0, 1].axis("off")
        
        overlay_ch0 = _make_dual_overlay(fixed_ch0, aligned_ch0)
        axes[0, 2].imshow(overlay_ch0)
        axes[0, 2].set_title(f"Ch0 Overlay (Nuclei)\nRed=Aligned, Blue=Fixed", fontsize=10)
        axes[0, 2].axis("off")
        
        # Row 1: Channel 1 (Protein)
        axes[1, 0].imshow(_normalize_for_display(fixed_ch1), cmap="gray")
        axes[1, 0].set_title(f"{fixed_label}\nChannel 1 (Protein)", fontsize=10)
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(_normalize_for_display(aligned_ch1), cmap="gray")
        axes[1, 1].set_title(f"{moving_label} (aligned)\nChannel 1 (Protein)", fontsize=10)
        axes[1, 1].axis("off")
        
        overlay_ch1 = _make_dual_overlay(fixed_ch1, aligned_ch1)
        axes[1, 2].imshow(overlay_ch1)
        axes[1, 2].set_title(f"Ch1 Overlay (Protein)\nRed=Aligned, Blue=Fixed", fontsize=10)
        axes[1, 2].axis("off")
        
        fig.suptitle(f"Alignment QC: shift dy={dy:.3f}, dx={dx:.3f} (error={error:.4f})", fontsize=12, fontweight='bold')
        fig.savefig(qc_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)


def generate_raw_comparison_qc(
    qc_path: str,
    fixed_ch0: np.ndarray,
    moving_ch0: np.ndarray,
    fixed_ch1: np.ndarray,
    moving_ch1: np.ndarray,
    fixed_label: str = "Fixed",
    moving_label: str = "Moving"
) -> None:
    """
    Generate a 2x3 grid showing raw (unaligned) images for both channels.
    Useful for initial diagnostics before alignment.
    
    CONSISTENT LABELING:
        Row 0: Channel 0 (Nuclei) - Fixed | Moving | Overlay (no alignment)
        Row 1: Channel 1 (Protein) - Fixed | Moving | Overlay (no alignment)
    """
    if not _HAVE_MPL or plt is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    try:
        # Row 0: Channel 0 (Nuclei)
        axes[0, 0].imshow(_normalize_for_display(fixed_ch0), cmap='gray')
        axes[0, 0].set_title(f'{fixed_label}\nChannel 0 (Nuclei)', fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(_normalize_for_display(moving_ch0), cmap='gray')
        axes[0, 1].set_title(f'{moving_label}\nChannel 0 (Nuclei)', fontsize=10)
        axes[0, 1].axis('off')
        
        overlay_ch0 = _make_dual_overlay(fixed_ch0, moving_ch0)
        axes[0, 2].imshow(overlay_ch0)
        axes[0, 2].set_title('Ch0 Overlay (Nuclei - No alignment)\nRed=Moving, Blue=Fixed', fontsize=10)
        axes[0, 2].axis('off')
        
        # Row 1: Channel 1 (Protein)
        axes[1, 0].imshow(_normalize_for_display(fixed_ch1), cmap='gray')
        axes[1, 0].set_title(f'{fixed_label}\nChannel 1 (Protein)', fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(_normalize_for_display(moving_ch1), cmap='gray')
        axes[1, 1].set_title(f'{moving_label}\nChannel 1 (Protein)', fontsize=10)
        axes[1, 1].axis('off')
        
        overlay_ch1 = _make_dual_overlay(fixed_ch1, moving_ch1)
        axes[1, 2].imshow(overlay_ch1)
        axes[1, 2].set_title('Ch1 Overlay (Protein - No alignment)\nRed=Moving, Blue=Fixed', fontsize=10)
        axes[1, 2].axis('off')
        
        fig.suptitle('Raw Images - Before Alignment', fontsize=14, fontweight='bold')
        fig.savefig(qc_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def make_log_writer(csv_path: str) -> Tuple[csv.writer, object]:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["timestamp", "mode", "sample_id", "channel", "reference_channel",
                     "dy_px", "dx_px", "distance_px", "H_used", "W_used",
                     "upsample_factor", "interp_order", "reg_error"])
    f.flush()
    return writer, f


# ------------------------------- Alignment core -------------------------------

def align_channel_set(
    files_per_channel: List[str],
    channel_names: List[str],
    ref_index: int = 0,
    upsample_factor: int = 10,
    interpolation_order: int = 1,
    crop_to_overlap: bool = True,
    log_writer: Optional[csv.writer] = None,
    log_flush_file: Optional[object] = None,
    log_print = print
) -> Tuple[List[np.ndarray], List[Tuple[float,float]], Tuple[int,int,int,int], Tuple[int,int]]:
    """Align a single set of files (one file per channel)."""
    assert len(files_per_channel) == len(channel_names) >= 2
    C = len(files_per_channel)

    imgs = []
    for fp in files_per_channel:
        arr = read_2d_tiff(fp)
        imgs.append(arr)

    # Pre-crop all to the minimum (H, W)
    H = min(im.shape[0] for im in imgs); W = min(im.shape[1] for im in imgs)
    if any((im.shape[0] != H or im.shape[1] != W) for im in imgs):
        log_print(f"  ⤷ Cropping all channels to common size {H}×{W} (center crop).")
    imgs_c = [center_crop_to(im, (H, W)) for im in imgs]

    ref = imgs_c[ref_index]
    shifts: List[Tuple[float, float]] = []
    aligned: List[np.ndarray] = []

    for i, (name, img) in enumerate(zip(channel_names, imgs_c)):
        if i == ref_index:
            shifts.append((0.0, 0.0))
            aligned.append(img.copy())
            continue
        shift, error, diffphase = phase_cross_correlation(ref, img, upsample_factor=upsample_factor)
        dy, dx = float(shift[0]), float(shift[1])
        moved = ndimage.shift(img, shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)
        shifts.append((dy, dx))
        aligned.append(moved)
        if log_writer is not None:
            dist = math.hypot(dy, dx)
            log_writer.writerow([_timestamp(), "single-set", "", name, channel_names[ref_index],
                                 dy, dx, dist, H, W, upsample_factor, interpolation_order, error])
            if log_flush_file is not None:
                log_flush_file.flush()
        log_print(f"  • {name} ← {channel_names[ref_index]}  shift dy={dy:.3f}, dx={dx:.3f} (|Δ|={math.hypot(dy,dx):.3f} px)")

    # Crop to overlap
    y0, y1, x0, x1 = (0, H, 0, W)
    if crop_to_overlap:
        y0, y1, x0, x1 = compute_overlap_bounds(H, W, shifts)
        aligned = [im[y0:y1, x0:x1] for im in aligned]

    return aligned, shifts, (y0, y1, x0, x1), (H, W)


def detect_channel_axis(arr: np.ndarray) -> int:
    """Heuristic: prefer axis with small size (2..8) as channel axis; else last axis if 3/4; else 0."""
    if arr.ndim < 3:
        raise ValueError(f"Need a 3D array to find channel axis; got shape {arr.shape}")
    candidates = [ax for ax, size in enumerate(arr.shape) if 2 <= size <= 8]
    if candidates:
        return candidates[0]
    if arr.shape[-1] in (3, 4):
        return arr.ndim - 1
    return 0


def align_multichannel_array(
    arr: np.ndarray,
    channel_axis: int,
    channel_names: Optional[List[str]],
    ref_index: int,
    upsample_factor: int,
    interpolation_order: int,
    crop_to_overlap: bool,
    log_writer: Optional[csv.writer],
    log_flush_file: Optional[object],
    log_print = print
) -> Tuple[np.ndarray, List[Tuple[float,float]], Tuple[int,int,int,int]]:
    """Align channels within a single multi-channel array and return (C, Y, X)."""
    arr = np.moveaxis(arr, channel_axis, 0)
    C, H, W = arr.shape
    if C < 2:
        raise ValueError(f"Expected at least 2 channels; found {C}")
    if not channel_names or len(channel_names) != C:
        channel_names = [f"ch{i}" for i in range(C)]
    ref = arr[ref_index]
    shifts = []
    aligned = []
    for i in range(C):
        if i == ref_index:
            shifts.append((0.0, 0.0))
            aligned.append(ref.copy())
            continue
        shift, error, diffphase = phase_cross_correlation(ref, arr[i], upsample_factor=upsample_factor)
        dy, dx = float(shift[0]), float(shift[1])
        moved = ndimage.shift(arr[i], shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)
        shifts.append((dy, dx))
        aligned.append(moved)
        if log_writer is not None:
            dist = math.hypot(dy, dx)
            log_writer.writerow([_timestamp(), "single-file", "", channel_names[i], channel_names[ref_index],
                                 dy, dx, dist, H, W, upsample_factor, interpolation_order, error])
            if log_flush_file is not None:
                log_flush_file.flush()
        log_print(f"  • {channel_names[i]} ← {channel_names[ref_index]}  shift dy={dy:.3f}, dx={dx:.3f} (|Δ|={math.hypot(dy,dx):.3f} px)")

    y0, y1, x0, x1 = (0, H, 0, W)
    if crop_to_overlap:
        y0, y1, x0, x1 = compute_overlap_bounds(H, W, shifts)
        aligned = [im[y0:y1, x0:x1] for im in aligned]
    out = np.stack(aligned, axis=0)  # (C, Hc, Wc)
    return out, shifts, (y0, y1, x0, x1)


def align_channel1_between_dirs(
    fixed_dir: str,
    moving_dir: str,
    out_dir: str,
    pattern: str = "*.tif",
    ref_index: int = 0,
    target_index: int = 1,
    upsample_factor: int = 10,
    interpolation_order: int = 1,
    crop_to_overlap: bool = True,
    log_print = print,
    progress_cb: Optional[Callable[[int, int], None]] = None
) -> None:
    """Align channel `target_index` images between two folders using channel `ref_index` as reference."""
    if not os.path.isdir(fixed_dir):
        raise FileNotFoundError(f"Fixed directory not found: {fixed_dir}")
    if not os.path.isdir(moving_dir):
        raise FileNotFoundError(f"Moving directory not found: {moving_dir}")

    fixed_files = list_tiff_files(fixed_dir, pattern)
    moving_files = list_tiff_files(moving_dir, pattern)
    if not fixed_files:
        raise FileNotFoundError(f"No TIFF files matching '{pattern}' found in {fixed_dir}")
    if not moving_files:
        raise FileNotFoundError(f"No TIFF files matching '{pattern}' found in {moving_dir}")

    total = min(len(fixed_files), len(moving_files))
    if total == 0:
        raise RuntimeError("No paired files available for alignment.")
    if len(fixed_files) != len(moving_files):
        log_print(f"⚠️  File counts differ ({len(fixed_files)} vs {len(moving_files)}); aligning first {total} pairs.")

    os.makedirs(out_dir, exist_ok=True)
    qc_dir = os.path.join(out_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "alignment_channel1_log.csv")
    with open(log_path, 'w', newline='') as f_log:
        writer = csv.writer(f_log)
        writer.writerow([
            "timestamp", "pair_index", "sample_id",
            "fixed_file", "moving_file",
            "ref_index", "target_index",
            "dy_px", "dx_px", "distance_px",
            "upsample_factor", "interp_order",
            "roi_height", "roi_width", "reg_error"
        ])

        for idx in range(total):
            fixed_file = fixed_files[idx]
            moving_file = moving_files[idx]
            log_print(f"[{idx+1}/{total}] Aligning channel {target_index} using reference channel {ref_index}:")
            log_print(f"    Fixed : {fixed_file}")
            log_print(f"    Moving: {moving_file}")

            arr_fixed = imread(fixed_file)
            arr_moving = imread(moving_file)

            ax_fixed = detect_channel_axis(arr_fixed)
            ax_moving = detect_channel_axis(arr_moving)
            arr_fixed = np.moveaxis(arr_fixed, ax_fixed, 0)
            arr_moving = np.moveaxis(arr_moving, ax_moving, 0)

            C_fixed, Hf, Wf = arr_fixed.shape
            C_moving, Hm, Wm = arr_moving.shape
            if ref_index >= C_fixed or ref_index >= C_moving:
                raise ValueError(f"Reference channel index {ref_index} missing (sizes {C_fixed}, {C_moving}).")
            if target_index >= C_fixed or target_index >= C_moving:
                raise ValueError(f"Target channel index {target_index} missing (sizes {C_fixed}, {C_moving}).")

            H = min(Hf, Hm)
            W = min(Wf, Wm)
            fixed_center = [center_crop_to(arr_fixed[c], (H, W)) for c in range(C_fixed)]
            moving_center = [center_crop_to(arr_moving[c], (H, W)) for c in range(C_moving)]

            ref_fixed = fixed_center[ref_index]
            ref_moving = moving_center[ref_index]
            shift, error, diffphase = phase_cross_correlation(ref_fixed, ref_moving, upsample_factor=upsample_factor)
            dy, dx = float(shift[0]), float(shift[1])
            dist = math.hypot(dy, dx)
            log_print(f"    ↳ shift dy={dy:.4f}, dx={dx:.4f} (|Δ|={dist:.4f} px)")

            moved_ref = ndimage.shift(ref_moving, shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)
            moved_target = ndimage.shift(moving_center[target_index], shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)

            y0, y1, x0, x1 = (0, H, 0, W)
            if crop_to_overlap:
                y0, y1, x0, x1 = compute_overlap_bounds(H, W, [(0.0, 0.0), (dy, dx)])

            ref_fixed_c = ref_fixed[y0:y1, x0:x1]
            target_fixed_c = fixed_center[target_index][y0:y1, x0:x1]
            before_target_c = moving_center[target_index][y0:y1, x0:x1]
            moved_target_c = moved_target[y0:y1, x0:x1]
            moved_ref_c = moved_ref[y0:y1, x0:x1]

            sample_id_raw = os.path.basename(os.path.commonprefix([stem(fixed_file), stem(moving_file)])).strip("_-. ")
            sample_id = sanitize_filename_component(sample_id_raw or stem(fixed_file) or stem(moving_file) or f"pair_{idx+1}")
            pair_stack = [target_fixed_c, moved_target_c]
            channel_names = [
                f"{stem(fixed_file)}_ch{target_index}_fixed",
                f"{stem(moving_file)}_ch{target_index}_aligned"
            ]
            out_path = os.path.join(out_dir, f"{sample_id}_ch{target_index}_pair.tif")
            stack_and_save_tiff(out_path, pair_stack, channel_names=channel_names)
            log_print(f"    ✅ Saved aligned pair: {out_path}")

            # Generate raw comparison QC (before alignment)
            raw_qc_path = os.path.join(qc_dir, f"{sample_id}_raw_comparison.png")
            generate_raw_comparison_qc(
                raw_qc_path,
                fixed_ch0=ref_fixed_c,
                moving_ch0=moving_center[ref_index][y0:y1, x0:x1],
                fixed_ch1=target_fixed_c,
                moving_ch1=before_target_c,
                fixed_label=stem(fixed_file),
                moving_label=stem(moving_file)
            )

            # Generate comprehensive QC for both channels (after alignment)
            qc_path = os.path.join(qc_dir, f"{sample_id}_qc.png")
            generate_dual_channel_qc(
                qc_path,
                fixed_ch0=ref_fixed_c,
                moving_ch0=moving_center[ref_index][y0:y1, x0:x1],
                aligned_ch0=moved_ref_c,
                fixed_ch1=target_fixed_c,
                moving_ch1=before_target_c,
                aligned_ch1=moved_target_c,
                dy=dy,
                dx=dx,
                error=error,
                fixed_label=stem(fixed_file),
                moving_label=stem(moving_file)
            )

            writer.writerow([
                _timestamp(), idx + 1, sample_id,
                os.path.basename(fixed_file), os.path.basename(moving_file),
                ref_index, target_index,
                dy, dx, dist,
                upsample_factor, interpolation_order,
                y1 - y0, x1 - x0, error
            ])
            f_log.flush()
            if progress_cb is not None:
                progress_cb(idx + 1, total)

    log_print(f"📄 Channel 1 alignment log written to: {log_path}")
    log_print(f"🖼️ QC overlays in: {qc_dir}")


# ------------------------------- Tkinter GUI -------------------------------

class ScrollableFrame(ttk.Frame):
    """Simple scrollable frame (vertical) using a Canvas."""
    def __init__(self, parent, height=200, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Mousewheel scrolling
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel)
        self.inner.bind_all("<Button-4>", self._on_mousewheel)  # Linux
        self.inner.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")


class ChannelAlignApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChannelAlign — Multi‑Channel TIFF Registration (translation‑only) — Tkinter")
        self.geometry("1050x780")
        try:
            self.tk.call("source", "sun-valley.tcl")  # If you happen to have a ttk theme; silently ignore.
            self.tk.call("set_theme", "light")
        except Exception:
            pass

        # Ensure button text is visible across different Tk themes/platforms.
        # Some macOS / third-party themes render ttk button text with the same
        # color as the button background (appearing invisible). Configure a
        # sensible default foreground and provide a conservative macOS fallback.
        try:
            style = ttk.Style(self)
            # Try to set a clear, high-contrast foreground for enabled buttons.
            style.configure("TButton", foreground="#000000")
            style.map("TButton",
                      foreground=[("disabled", "#6b6b6b"), ("!disabled", "#000000")])
            # On some macOS Tk builds the native theme ignores foreground colors;
            # fallback to a theme that respects styling which also looks reasonable.
            if sys.platform == "darwin":
                try:
                    style.theme_use("clam")
                    style.configure("TButton", foreground="#000000", background="#f0f0f0")
                except Exception:
                    # If switching theme fails, continue without crashing.
                    pass
        except Exception:
            # Never let style issues prevent the GUI from launching.
            pass

        self.notebook = ttk.Notebook(self)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_mc = ttk.Frame(self.notebook)
        self.tab_pair = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_batch, text="Per‑channel folders (Batch)")
        self.notebook.add(self.tab_mc, text="Multi‑channel TIFF(s)")
        self.notebook.add(self.tab_pair, text="Dataset pair (Channel 0→1)")
        self.notebook.pack(fill="both", expand=True)

        # Threading helpers
        self._batch_thread: Optional[threading.Thread] = None
        self._mc_thread: Optional[threading.Thread] = None
        self._pair_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Build tabs
        self._build_batch_tab()
        self._build_mc_tab()
        self._build_pair_tab()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --------- Tab: Batch ---------
    def _build_batch_tab(self):
        f = self.tab_batch

        pad = {"padx": 8, "pady": 6}

        ttk.Label(f, text="Batch mode: provide one folder per channel. Files across folders are paired by sorted order.").grid(row=0, column=0, columnspan=6, sticky="w", **pad)

        # Channels list (scrollable)
        ttk.Label(f, text="Channels").grid(row=1, column=0, sticky="w", **pad)
        self.batch_sframe = ScrollableFrame(f, height=220)
        self.batch_sframe.grid(row=2, column=0, columnspan=6, sticky="nsew", padx=8)
        f.grid_rowconfigure(2, weight=1)
        f.grid_columnconfigure(1, weight=1)

        header = ("#", "Name", "Folder", "Browse", "Pattern", "Remove")
        for c, h in enumerate(header):
            ttk.Label(self.batch_sframe.inner, text=h, style="Heading.TLabel").grid(row=0, column=c, padx=4, pady=2, sticky="w")

        self._batch_rows = {}  # row_id -> dict
        self._next_row_id = 0
        self._add_batch_row()
        self._add_batch_row()

        ttk.Button(f, text="Add Channel", command=self._add_batch_row).grid(row=3, column=0, sticky="w", **pad)

        # Options
        ttk.Label(f, text="Reference channel name:").grid(row=4, column=0, sticky="w", **pad)
        self.batch_refname = tk.StringVar(value="ch0")
        ttk.Entry(f, textvariable=self.batch_refname, width=12).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(f, text="Output folder:").grid(row=5, column=0, sticky="w", **pad)
        self.batch_outdir = tk.StringVar()
        ttk.Entry(f, textvariable=self.batch_outdir, width=70).grid(row=5, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.batch_outdir)).grid(row=5, column=4, sticky="w", **pad)

        ttk.Label(f, text="Output suffix:").grid(row=5, column=5, sticky="w", **pad)
        self.batch_suffix = tk.StringVar(value="_aligned")
        ttk.Entry(f, textvariable=self.batch_suffix, width=12).grid(row=5, column=6, sticky="w", **pad)

        self.batch_crop = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Crop to common overlap", variable=self.batch_crop).grid(row=6, column=0, sticky="w", **pad)

        ttk.Label(f, text="Subpixel upsample factor:").grid(row=6, column=1, sticky="e", **pad)
        self.batch_upsample = tk.StringVar(value="10")
        tk.Spinbox(f, from_=1, to=50, increment=1, textvariable=self.batch_upsample, width=6).grid(row=6, column=2, sticky="w", **pad)

        ttk.Label(f, text="Interpolation order (0..5):").grid(row=6, column=3, sticky="e", **pad)
        self.batch_order = tk.StringVar(value="1")
        tk.Spinbox(f, from_=0, to=5, increment=1, textvariable=self.batch_order, width=6).grid(row=6, column=4, sticky="w", **pad)

        # Progress + buttons
        self.batch_prog = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.batch_prog.grid(row=7, column=0, columnspan=7, sticky="we", padx=8, pady=(6, 4))

        btn_row = ttk.Frame(f)
        btn_row.grid(row=8, column=0, columnspan=7, sticky="w", padx=8, pady=4)
        self.btn_runbatch = ttk.Button(btn_row, text="Run Batch", command=self._on_run_batch)
        self.btn_stop = ttk.Button(btn_row, text="Stop", command=self._on_stop, state="disabled")
        self.btn_help_b = ttk.Button(btn_row, text="Help", command=self._on_help_batch)
        self.btn_quit_b = ttk.Button(btn_row, text="Quit", command=self._on_close)
        self.btn_runbatch.pack(side="left", padx=(0, 6))
        self.btn_stop.pack(side="left", padx=(0, 6))
        self.btn_help_b.pack(side="left", padx=(0, 6))
        self.btn_quit_b.pack(side="left")

        # Log
        ttk.Label(f, text="Log").grid(row=9, column=0, sticky="w", **pad)
        self.batch_log = ScrolledText(f, width=120, height=14, state="disabled")
        self.batch_log.grid(row=10, column=0, columnspan=7, sticky="nsew", padx=8, pady=(0, 8))
        f.grid_rowconfigure(10, weight=1)

    def _add_batch_row(self):
        r = self._next_row_id
        self._next_row_id += 1
        row = {}

        idx_lbl = ttk.Label(self.batch_sframe.inner, text=str(r+1))
        name_var = tk.StringVar(value=f"ch{r}")
        name_ent = ttk.Entry(self.batch_sframe.inner, textvariable=name_var, width=12)

        folder_var = tk.StringVar()
        folder_ent = ttk.Entry(self.batch_sframe.inner, textvariable=folder_var, width=55)
        browse_btn = ttk.Button(self.batch_sframe.inner, text="Browse…",
                                command=lambda v=folder_var: self._choose_dir(v))

        patt_var = tk.StringVar(value="*.tif")
        patt_ent = ttk.Entry(self.batch_sframe.inner, textvariable=patt_var, width=12)

        rm_btn = ttk.Button(self.batch_sframe.inner, text="Remove", command=lambda rid=r: self._remove_batch_row(rid))

        widgets = [idx_lbl, name_ent, folder_ent, browse_btn, patt_ent, rm_btn]
        for c, w in enumerate(widgets):
            w.grid(row=r+1, column=c, padx=4, pady=2, sticky="w")

        row.update(dict(idx_lbl=idx_lbl, name_var=name_var, folder_var=folder_var,
                        patt_var=patt_var, widgets=widgets, active=True))
        self._batch_rows[r] = row

    def _remove_batch_row(self, rid: int):
        row = self._batch_rows.get(rid)
        if not row or not row.get("active", False):
            return
        for w in row["widgets"]:
            try:
                w.grid_remove()
            except Exception:
                pass
        row["active"] = False

    def _choose_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _choose_files(self, var: tk.StringVar):
        files = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")])
        if files:
            var.set(";".join(files))

    def _gather_batch_channels(self) -> Tuple[List[str], List[List[str]]]:
        names, file_lists = [], []
        active_rows = [rid for rid, r in self._batch_rows.items() if r.get("active")]
        if not active_rows:
            return [], []
        for rid in sorted(active_rows):
            row = self._batch_rows[rid]
            name = (row["name_var"].get() or f"ch{rid}").strip()
            folder = (row["folder_var"].get() or "").strip()
            patt = (row["patt_var"].get() or "*.tif").strip()
            if not folder:
                continue
            files = list_tiff_files(folder, patt)
            if not files:
                continue
            names.append(name)
            file_lists.append(files)
        return names, file_lists

    def _append_log(self, widget: ScrolledText, msg: str):
        widget.configure(state="normal")
        widget.insert("end", msg + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _post_log_batch(self, msg: str):
        self.after(0, lambda: self._append_log(self.batch_log, msg))

    def _post_log_mc(self, msg: str):
        self.after(0, lambda: self._append_log(self.mc_log, msg))

    def _post_log_pair(self, msg: str):
        self.after(0, lambda: self._append_log(self.pair_log, msg))

    def _on_stop(self):
        self._stop_event.set()
        self._post_log_batch("⛔ Stop requested; will finish current item then stop.")
        self._post_log_mc("⛔ Stop requested; will finish current item then stop.")
        self._post_log_pair("⛔ Stop requested; will finish current item then stop.")

    def _toggle_run_buttons(self, busy: bool, which: str):
        if which == "batch":
            self.btn_runbatch.configure(state="disabled" if busy else "normal")
            self.btn_stop.configure(state="normal" if busy else "disabled")
        elif which == "mc":
            self.btn_runmc.configure(state="disabled" if busy else "normal")
            self.btn_stop_mc.configure(state="normal" if busy else "disabled")
        elif which == "pair":
            self.btn_runpair.configure(state="disabled" if busy else "normal")
            self.btn_stop_pair.configure(state="normal" if busy else "disabled")

    def _on_run_batch(self):
        if self._batch_thread and self._batch_thread.is_alive():
            return
        self._stop_event.clear()
        self.batch_prog.configure(value=0, maximum=100)
        self._toggle_run_buttons(True, "batch")
        self._append_log(self.batch_log, "\n— Starting Batch —")
        self._batch_thread = threading.Thread(target=self._run_batch_thread, daemon=True)
        self._batch_thread.start()

    def _run_batch_thread(self):
        try:
            names, file_lists = self._gather_batch_channels()
            if len(names) < 2:
                self._post_log_batch("⚠️  Need at least two valid channels (name + folder with TIFFs). Nothing to do.")
                return
            ref_name = (self.batch_refname.get() or names[0]).strip()
            if ref_name not in names:
                self._post_log_batch(f"⚠️  Reference channel '{ref_name}' not found among {names}. Using '{names[0]}' instead.")
                ref_name = names[0]
            ref_idx = names.index(ref_name)

            outdir = (self.batch_outdir.get() or "").strip()
            if not outdir:
                self._post_log_batch("⚠️  Please choose an Output folder.")
                return
            os.makedirs(outdir, exist_ok=True)

            suffix = (self.batch_suffix.get() or "_aligned").strip()
            try:
                upsample = int(self.batch_upsample.get() or 10)
                order = int(self.batch_order.get() or 1)
            except Exception:
                self._post_log_batch("⚠️  Invalid numeric options for upsample/order.")
                return
            crop_overlap = bool(self.batch_crop.get())

            min_count = min(len(lst) for lst in file_lists)
            if any(len(lst) != min_count for lst in file_lists):
                self._post_log_batch(f"⚠️  Channel file counts differ {[len(lst) for lst in file_lists]}; processing only the first {min_count} sets.")
            total = min_count
            self.after(0, lambda: self.batch_prog.configure(value=0, maximum=total))

            # Prepare CSV log
            csv_path = os.path.join(outdir, "alignment_log.csv")
            writer, log_file = make_log_writer(csv_path)
            self._post_log_batch(f"📝 Writing log to: {csv_path}")

            for idx in range(total):
                if self._stop_event.is_set():
                    self._post_log_batch("🛑 Stopping before next image set as requested.")
                    break
                set_files = [lst[idx] for lst in file_lists]
                # Sample ID from common stem prefix
                try:
                    stems = [stem(p) for p in set_files]
                    prefix = os.path.basename(os.path.commonprefix(stems)).strip("_-. ")
                    sample_id = prefix or f"set_{idx+1}"
                except Exception:
                    sample_id = f"set_{idx+1}"
                self._post_log_batch(f"\n[{idx+1}/{total}] Aligning set: {sample_id}")
                try:
                    aligned, shifts, roi, base_hw = align_channel_set(
                        files_per_channel=set_files,
                        channel_names=names,
                        ref_index=ref_idx,
                        upsample_factor=upsample,
                        interpolation_order=order,
                        crop_to_overlap=crop_overlap,
                        log_writer=writer,
                        log_flush_file=log_file,
                        log_print=self._post_log_batch
                    )
                    # Save TIFF
                    ref_file = set_files[ref_idx]
                    out_name = f"{stem(ref_file)}{suffix}.tif"
                    out_path = os.path.join(outdir, out_name)
                    stack_and_save_tiff(out_path, aligned, channel_names=names)
                    Hc, Wc = aligned[0].shape
                    self._post_log_batch(f"  ✅ Saved: {out_path}  (shape: C={len(aligned)}, H={Hc}, W={Wc})")
                    # Per-set summary row
                    writer.writerow([_timestamp(), "batch-summary", sample_id, "", names[ref_idx],
                                     "", "", "", base_hw[0], base_hw[1], upsample, order, ""])
                    log_file.flush()
                except Exception as e:
                    self._post_log_batch(f"  ❌ ERROR aligning set {idx+1}: {e}")
                self.after(0, lambda v=idx+1: self.batch_prog.configure(value=v))

            try:
                log_file.close()
            except Exception:
                pass
            self._post_log_batch("🎉 Done.")
        finally:
            self.after(0, lambda: self._toggle_run_buttons(False, "batch"))

    def _on_help_batch(self):
        messagebox.showinfo(
            "Help — Batch mode",
            "How to use (Batch / Per‑channel folders):\n"
            "1) Click 'Add Channel' and create one row for each channel (e.g., DAPI, FITC, TRITC).\n"
            "2) For each row, set a 'name', choose the folder with its TIFFs, and an optional pattern (default '*.tif').\n"
            "3) Set 'Reference channel name' to one of the listed names.\n"
            "4) Choose an Output folder and (optional) suffix.\n"
            "5) (Optional) 'Crop to common overlap' removes padded borders; adjust upsample/order as needed.\n"
            "6) Click 'Run Batch'. Files across folders are paired by sorted filename order.\n\n"
            "Notes:\n"
            "• Registration is translation‑only (no rotation/scale).\n"
            "• All images in a set are center‑cropped to the minimum H×W before alignment.\n"
            "• If folder file counts differ, only the first min-count sets are processed."
        )

    # --------- Tab: Multi‑channel ---------
    def _build_mc_tab(self):
        f = self.tab_mc
        pad = {"padx": 8, "pady": 6}

        ttk.Label(f, text="Multi‑channel TIFF(s): load files that already contain multiple channels (3D arrays).").grid(row=0, column=0, columnspan=6, sticky="w", **pad)

        ttk.Label(f, text="Select file(s):").grid(row=1, column=0, sticky="w", **pad)
        self.mc_files = tk.StringVar()
        ttk.Entry(f, textvariable=self.mc_files, width=80).grid(row=1, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_files(self.mc_files)).grid(row=1, column=4, sticky="w", **pad)

        ttk.Label(f, text="Channel names (comma‑separated, optional):").grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        self.mc_names = tk.StringVar()
        ttk.Entry(f, textvariable=self.mc_names, width=50).grid(row=2, column=2, columnspan=3, sticky="we", **pad)

        ttk.Label(f, text="Reference channel index (0‑based):").grid(row=3, column=0, sticky="w", **pad)
        self.mc_ref = tk.StringVar(value="0")
        tk.Spinbox(f, from_=0, to=63, increment=1, textvariable=self.mc_ref, width=6).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(f, text="Output folder:").grid(row=4, column=0, sticky="w", **pad)
        self.mc_outdir = tk.StringVar()
        ttk.Entry(f, textvariable=self.mc_outdir, width=70).grid(row=4, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.mc_outdir)).grid(row=4, column=4, sticky="w", **pad)

        ttk.Label(f, text="Output suffix:").grid(row=4, column=5, sticky="w", **pad)
        self.mc_suffix = tk.StringVar(value="_aligned")
        ttk.Entry(f, textvariable=self.mc_suffix, width=12).grid(row=4, column=6, sticky="w", **pad)

        self.mc_crop = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Crop to common overlap", variable=self.mc_crop).grid(row=5, column=0, sticky="w", **pad)

        ttk.Label(f, text="Subpixel upsample factor:").grid(row=5, column=1, sticky="e", **pad)
        self.mc_upsample = tk.StringVar(value="10")
        tk.Spinbox(f, from_=1, to=50, increment=1, textvariable=self.mc_upsample, width=6).grid(row=5, column=2, sticky="w", **pad)

        ttk.Label(f, text="Interpolation order (0..5):").grid(row=5, column=3, sticky="e", **pad)
        self.mc_order = tk.StringVar(value="1")
        tk.Spinbox(f, from_=0, to=5, increment=1, textvariable=self.mc_order, width=6).grid(row=5, column=4, sticky="w", **pad)

        # Progress + buttons
        self.mc_prog = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.mc_prog.grid(row=6, column=0, columnspan=7, sticky="we", padx=8, pady=(6, 4))

        btn_row = ttk.Frame(f)
        btn_row.grid(row=7, column=0, columnspan=7, sticky="w", padx=8, pady=4)
        self.btn_runmc = ttk.Button(btn_row, text="Run", command=self._on_run_mc)
        self.btn_stop_mc = ttk.Button(btn_row, text="Stop", command=self._on_stop, state="disabled")
        self.btn_help_mc = ttk.Button(btn_row, text="Help", command=self._on_help_mc)
        self.btn_quit_mc = ttk.Button(btn_row, text="Quit", command=self._on_close)
        self.btn_runmc.pack(side="left", padx=(0, 6))
        self.btn_stop_mc.pack(side="left", padx=(0, 6))
        self.btn_help_mc.pack(side="left", padx=(0, 6))
        self.btn_quit_mc.pack(side="left")

        # Log
        ttk.Label(f, text="Log").grid(row=8, column=0, sticky="w", **pad)
        self.mc_log = ScrolledText(f, width=120, height=14, state="disabled")
        self.mc_log.grid(row=9, column=0, columnspan=7, sticky="nsew", padx=8, pady=(0, 8))
        f.grid_rowconfigure(9, weight=1)
        f.grid_columnconfigure(2, weight=1)

    def _build_pair_tab(self):
        f = self.tab_pair
        pad = {"padx": 8, "pady": 6}

        ttk.Label(f, text="Align channel 1 in Folder B to Folder A by registering channel 0 between matching TIFFs.").grid(row=0, column=0, columnspan=6, sticky="w", **pad)

        ttk.Label(f, text="Dataset A (fixed, channel 0 reference):").grid(row=1, column=0, sticky="w", **pad)
        self.pair_dir_a = tk.StringVar(value=os.path.join(os.getcwd(), "4"))
        ttk.Entry(f, textvariable=self.pair_dir_a, width=70).grid(row=1, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.pair_dir_a)).grid(row=1, column=4, sticky="w", **pad)

        ttk.Label(f, text="Dataset B (moving, channel 1 corrected):").grid(row=2, column=0, sticky="w", **pad)
        self.pair_dir_b = tk.StringVar(value=os.path.join(os.getcwd(), "4-2"))
        ttk.Entry(f, textvariable=self.pair_dir_b, width=70).grid(row=2, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.pair_dir_b)).grid(row=2, column=4, sticky="w", **pad)

        ttk.Label(f, text="Filename pattern:").grid(row=3, column=0, sticky="w", **pad)
        self.pair_pattern = tk.StringVar(value="*.tif")
        ttk.Entry(f, textvariable=self.pair_pattern, width=16).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(f, text="Output folder:").grid(row=4, column=0, sticky="w", **pad)
        self.pair_outdir = tk.StringVar(value=os.path.join(os.getcwd(), "aligned_channel1"))
        ttk.Entry(f, textvariable=self.pair_outdir, width=70).grid(row=4, column=1, columnspan=3, sticky="we", **pad)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.pair_outdir)).grid(row=4, column=4, sticky="w", **pad)

        ttk.Label(f, text="Reference channel index:").grid(row=5, column=0, sticky="w", **pad)
        self.pair_ref = tk.StringVar(value="0")
        tk.Spinbox(f, from_=0, to=15, increment=1, textvariable=self.pair_ref, width=6).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(f, text="Target channel index (aligned):").grid(row=5, column=2, sticky="w", **pad)
        self.pair_target = tk.StringVar(value="1")
        tk.Spinbox(f, from_=0, to=15, increment=1, textvariable=self.pair_target, width=6).grid(row=5, column=3, sticky="w", **pad)

        self.pair_crop = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Crop to common overlap", variable=self.pair_crop).grid(row=6, column=0, sticky="w", **pad)

        ttk.Label(f, text="Upsample factor:").grid(row=6, column=1, sticky="e", **pad)
        self.pair_upsample = tk.StringVar(value="10")
        tk.Spinbox(f, from_=1, to=50, increment=1, textvariable=self.pair_upsample, width=6).grid(row=6, column=2, sticky="w", **pad)

        ttk.Label(f, text="Interpolation order:").grid(row=6, column=3, sticky="e", **pad)
        self.pair_interp = tk.StringVar(value="1")
        tk.Spinbox(f, from_=0, to=5, increment=1, textvariable=self.pair_interp, width=6).grid(row=6, column=4, sticky="w", **pad)

        self.pair_prog = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.pair_prog.grid(row=7, column=0, columnspan=6, sticky="we", padx=8, pady=(6, 4))

        btn_row = ttk.Frame(f)
        btn_row.grid(row=8, column=0, columnspan=6, sticky="w", padx=8, pady=4)
        self.btn_runpair = ttk.Button(btn_row, text="Run pair alignment", command=self._on_run_pair)
        self.btn_stop_pair = ttk.Button(btn_row, text="Stop", command=self._on_stop, state="disabled")
        self.btn_help_pair = ttk.Button(btn_row, text="Help", command=self._on_help_pair)
        self.btn_quit_pair = ttk.Button(btn_row, text="Quit", command=self._on_close)
        self.btn_runpair.pack(side="left", padx=(0, 6))
        self.btn_stop_pair.pack(side="left", padx=(0, 6))
        self.btn_help_pair.pack(side="left", padx=(0, 6))
        self.btn_quit_pair.pack(side="left")

        ttk.Label(f, text="Log").grid(row=9, column=0, sticky="w", **pad)
        self.pair_log = ScrolledText(f, width=120, height=14, state="disabled")
        self.pair_log.grid(row=10, column=0, columnspan=6, sticky="nsew", padx=8, pady=(0, 8))
        f.grid_rowconfigure(10, weight=1)
        f.grid_columnconfigure(3, weight=1)

    def _on_run_mc(self):
        if self._mc_thread and self._mc_thread.is_alive():
            return
        self._stop_event.clear()
        self.mc_prog.configure(value=0, maximum=100)
        self._toggle_run_buttons(True, "mc")
        self._append_log(self.mc_log, "\n— Starting Multi‑channel —")
        self._mc_thread = threading.Thread(target=self._run_mc_thread, daemon=True)
        self._mc_thread.start()

    def _run_mc_thread(self):
        try:
            files_raw = (self.mc_files.get() or "").strip()
            if not files_raw:
                self._post_log_mc("⚠️  Please select at least one multi‑channel TIFF.")
                return
            files = [f for f in files_raw.split(';') if f]
            outdir = (self.mc_outdir.get() or "").strip()
            if not outdir:
                self._post_log_mc("⚠️  Please choose an Output folder.")
                return
            os.makedirs(outdir, exist_ok=True)

            suffix = (self.mc_suffix.get() or "_aligned").strip()
            try:
                upsample = int(self.mc_upsample.get() or 10)
                order = int(self.mc_order.get() or 1)
                ref_index = int(self.mc_ref.get() or 0)
            except Exception:
                self._post_log_mc("Invalid numeric options for upsample/order/ref index.")
                return
            crop_overlap = bool(self.mc_crop.get())
            name_list = [s.strip() for s in (self.mc_names.get() or "").split(',') if s.strip()]

            total = len(files)
            self.after(0, lambda: self.mc_prog.configure(value=0, maximum=total))

            csv_path = os.path.join(outdir, "alignment_log.csv")
            writer, log_file = make_log_writer(csv_path)
            self._post_log_mc(f"📝 Writing log to: {csv_path}")

            for i, fp in enumerate(sorted(files, key=natural_sort_key)):
                if self._stop_event.is_set():
                    self._post_log_mc("Stopping before next file as requested.")
                    break
                try:
                    self._post_log_mc(f"\n[{i+1}/{total}] Aligning file: {fp}")
                    arr = imread(fp)
                    if arr.ndim < 3:
                        raise ValueError(f"File '{fp}' does not appear to have multiple channels (shape={arr.shape}).")
                    ch_axis = detect_channel_axis(arr)
                    if name_list and len(name_list) != arr.shape[ch_axis]:
                        self._post_log_mc(f"Provided {len(name_list)} channel names but file has {arr.shape[ch_axis]} channels; ignoring provided names.")
                        ch_names = None
                    else:
                        ch_names = name_list if name_list else None

                    out_arr, shifts, roi = align_multichannel_array(
                        arr=arr,
                        channel_axis=ch_axis,
                        channel_names=ch_names,
                        ref_index=ref_index,
                        upsample_factor=upsample,
                        interpolation_order=order,
                        crop_to_overlap=crop_overlap,
                        log_writer=writer,
                        log_flush_file=log_file,
                        log_print=self._post_log_mc
                    )
                    base = stem(fp)
                    out_path = os.path.join(outdir, f"{base}{suffix}.tif")
                    if ch_names is None:
                        ch_names_save = [f"ch{i}" for i in range(out_arr.shape[0])]
                    else:
                        ch_names_save = ch_names
                    stack_and_save_tiff(out_path, [out_arr[i] for i in range(out_arr.shape[0])], channel_names=ch_names_save)
                    self._post_log_mc(f"  ✅ Saved: {out_path}  (shape: {tuple(out_arr.shape)})")
                except Exception as e:
                    self._post_log_mc(f"  ❌ ERROR: {e}")
                self.after(0, lambda v=i+1: self.mc_prog.configure(value=v))

            try:
                log_file.close()
            except Exception:
                pass
            self._post_log_mc("🎉 Done.")
        finally:
            self.after(0, lambda: self._toggle_run_buttons(False, "mc"))

    def _on_help_mc(self):
        messagebox.showinfo(
            "Help — Multi‑channel mode",
            "How to use (Multi‑channel TIFFs):\n"
            "1) Click 'Select file(s)' and pick one or more TIFFs that already contain multiple channels (shape C×Y×X or Y×X×C).\n"
            "2) (Optional) Enter channel names separated by commas (e.g., DAPI,FITC,TRITC). If omitted -> ch0,ch1,...\n"
            "3) Choose the reference channel index (0‑based).\n"
            "4) Choose an Output folder and suffix.\n"
            "5) Click 'Run'.\n\n"
            "Notes:\n"
            "• Program will try to detect which axis is the channel axis (prefers small axis 2..8).\n"
            "• Registration is translation‑only; images can be cropped to the common overlap."
        )

    def _on_run_pair(self):
        if self._pair_thread and self._pair_thread.is_alive():
            return
        self._stop_event.clear()
        self.pair_prog.configure(value=0, maximum=100)
        self._toggle_run_buttons(True, "pair")
        self._append_log(self.pair_log, "\n— Starting Dataset pair alignment —")
        self._pair_thread = threading.Thread(target=self._run_pair_thread, daemon=True)
        self._pair_thread.start()

    def _run_pair_thread(self):
        try:
            dir_a = (self.pair_dir_a.get() or "").strip()
            dir_b = (self.pair_dir_b.get() or "").strip()
            pattern = (self.pair_pattern.get() or "*.tif").strip()
            outdir = (self.pair_outdir.get() or "").strip()
            if not dir_a or not dir_b:
                self._post_log_pair("⚠️  Please choose both Dataset A and Dataset B folders.")
                return
            if not outdir:
                self._post_log_pair("⚠️  Please choose an output folder.")
                return
            if not os.path.isdir(dir_a):
                self._post_log_pair(f"⚠️  Dataset A folder does not exist: {dir_a}")
                return
            if not os.path.isdir(dir_b):
                self._post_log_pair(f"⚠️  Dataset B folder does not exist: {dir_b}")
                return
            try:
                ref_index = int(self.pair_ref.get() or 0)
                target_index = int(self.pair_target.get() or 1)
                upsample = int(self.pair_upsample.get() or 10)
                interp = int(self.pair_interp.get() or 1)
            except Exception:
                self._post_log_pair("⚠️  Invalid numeric options for reference/target/upsample/interpolation.")
                return
            crop = bool(self.pair_crop.get())

            files_a = list_tiff_files(dir_a, pattern)
            files_b = list_tiff_files(dir_b, pattern)
            if not files_a:
                self._post_log_pair(f"⚠️  No files matching '{pattern}' found in {dir_a}.")
                return
            if not files_b:
                self._post_log_pair(f"⚠️  No files matching '{pattern}' found in {dir_b}.")
                return
            total = min(len(files_a), len(files_b))
            if total == 0:
                self._post_log_pair("⚠️  No overlapping file pairs to process.")
                return
            if len(files_a) != len(files_b):
                self._post_log_pair(f"⚠️  File counts differ ({len(files_a)} vs {len(files_b)}); aligning first {total} pairs.")
            self.after(0, lambda: self.pair_prog.configure(value=0, maximum=total))

            def progress_cb(done: int, total_pairs: int):
                self.after(0, lambda: self.pair_prog.configure(value=done, maximum=max(total_pairs, 1)))

            align_channel1_between_dirs(
                fixed_dir=dir_a,
                moving_dir=dir_b,
                out_dir=outdir,
                pattern=pattern,
                ref_index=ref_index,
                target_index=target_index,
                upsample_factor=upsample,
                interpolation_order=interp,
                crop_to_overlap=crop,
                log_print=self._post_log_pair,
                progress_cb=progress_cb
            )
            self._post_log_pair("🎉 Dataset pair alignment complete.")
        except Exception as exc:
            self._post_log_pair(f"❌ ERROR: {exc}")
        finally:
            self.after(0, lambda: self._toggle_run_buttons(False, "pair"))

    def _on_help_pair(self):
        messagebox.showinfo(
            "Help — Dataset pair",
            "Align channel 1 across two folders that share a channel 0 reference:\n"
            "1) Point Dataset A to the fixed/reference folder (channel 0).\n"
            "2) Point Dataset B to the moving folder that requires correction.\n"
            "3) Ensure files match by sorted order; adjust the filename pattern if needed.\n"
            "4) Set channel indices (defaults: reference 0, target 1), output folder, and options.\n"
            "5) Click 'Run pair alignment'.\n\n"
            "Each pair produces aligned channel-1 TIFFs, QC overlays, and a CSV log."
        )

    def _on_close(self):
        if ((self._batch_thread and self._batch_thread.is_alive()) or
                (self._mc_thread and self._mc_thread.is_alive()) or
                (self._pair_thread and self._pair_thread.is_alive())):
            if not messagebox.askyesno("Quit", "A task is still running. Stop and quit?"):
                return
            self._stop_event.set()
        self.destroy()


def parse_cli_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="ChannelAlign Tk GUI and command-line helpers",
        add_help=True
    )
    parser.add_argument(
        "--align-channel1",
        action="store_true",
        help="Align channel `target-index` between two folders using channel `ref-index` as the reference."
    )
    parser.add_argument("--fixed-dir", default="4", help="Fixed/reference folder (default: ./4).")
    parser.add_argument("--moving-dir", default="4-2", help="Moving folder to align (default: ./4-2).")
    parser.add_argument("--output-dir", default="aligned_channel1", help="Output directory for aligned channel 1 results.")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern to match TIFF files (default: *.tif).")
    parser.add_argument("--ref-index", type=int, default=0, help="Reference channel index (default: 0).")
    parser.add_argument("--target-index", type=int, default=1, help="Target channel index to align (default: 1).")
    parser.add_argument("--upsample", type=int, default=10, help="Upsample factor for phase cross-correlation (default: 10).")
    parser.add_argument("--interp-order", type=int, default=1, help="Interpolation order for ndimage.shift (default: 1).")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping to the common overlap after alignment.")
    parser.add_argument("--with-gui", action="store_true", help="Launch the GUI after completing command-line tasks.")
    parser.add_argument("--no-gui", action="store_true", help="Skip launching the GUI entirely.")
    return parser.parse_known_args(argv)


def main(argv: Optional[List[str]] = None):
    args, unknown = parse_cli_args(argv)
    ran_cli_task = False

    if args.align_channel1:
        align_channel1_between_dirs(
            fixed_dir=args.fixed_dir,
            moving_dir=args.moving_dir,
            out_dir=args.output_dir,
            pattern=args.pattern,
            ref_index=args.ref_index,
            target_index=args.target_index,
            upsample_factor=args.upsample,
            interpolation_order=args.interp_order,
            crop_to_overlap=not args.no_crop,
            log_print=print
        )
        ran_cli_task = True

    launch_gui = not args.no_gui
    if ran_cli_task and not args.with_gui:
        launch_gui = False

    if not launch_gui:
        return

    sys.argv = [sys.argv[0]] + unknown
    app = ChannelAlignApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
