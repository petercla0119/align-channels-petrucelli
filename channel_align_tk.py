#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChannelAlign — Tkinter GUI (no PySimpleGUI) for multi‑channel TIFF registration (translation-only)
and export of aligned multi‑channel TIFFs, with an alignment CSV log.

Author: ChatGPT (GPT-5 Pro)
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
import datetime as _dt
from typing import List, Tuple, Optional

import numpy as np
from tifffile import imread, imwrite
from skimage.registration import phase_cross_correlation
from scipy import ndimage

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

        self.notebook = ttk.Notebook(self)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_mc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_batch, text="Per‑channel folders (Batch)")
        self.notebook.add(self.tab_mc, text="Multi‑channel TIFF(s)")
        self.notebook.pack(fill="both", expand=True)

        # Threading helpers
        self._batch_thread: Optional[threading.Thread] = None
        self._mc_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Build tabs
        self._build_batch_tab()
        self._build_mc_tab()

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

    def _on_stop(self):
        self._stop_event.set()
        self._post_log_batch("⛔ Stop requested; will finish current item then stop.")
        self._post_log_mc("⛔ Stop requested; will finish current item then stop.")

    def _toggle_run_buttons(self, busy: bool, which: str):
        if which == "batch":
            self.btn_runbatch.configure(state="disabled" if busy else "normal")
            self.btn_stop.configure(state="normal" if busy else "disabled")
        elif which == "mc":
            self.btn_runmc.configure(state="disabled" if busy else "normal")
            self.btn_stop_mc.configure(state="normal" if busy else "disabled")

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
                self._post_log_mc("⚠️  Invalid numeric options for upsample/order/ref index.")
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
                    self._post_log_mc("🛑 Stopping before next file as requested.")
                    break
                try:
                    self._post_log_mc(f"\n[{i+1}/{total}] Aligning file: {fp}")
                    arr = imread(fp)
                    if arr.ndim < 3:
                        raise ValueError(f"File '{fp}' does not appear to have multiple channels (shape={arr.shape}).")
                    ch_axis = detect_channel_axis(arr)
                    if name_list and len(name_list) != arr.shape[ch_axis]:
                        self._post_log_mc(f"  ⚠️ Provided {len(name_list)} channel names but file has {arr.shape[ch_axis]} channels; ignoring provided names.")
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

    def _on_close(self):
        if (self._batch_thread and self._batch_thread.is_alive()) or (self._mc_thread and self._mc_thread.is_alive()):
            if not messagebox.askyesno("Quit", "A task is still running. Stop and quit?"):
                return
            self._stop_event.set()
        self.destroy()


def main():
    app = ChannelAlignApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
