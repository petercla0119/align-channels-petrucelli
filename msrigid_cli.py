#!/usr/bin/env python3
"""CLI for msrigid — TurboReg/MultiStackReg-compatible rigid alignment in Python."""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import os
import tempfile

import numpy as np
import tifffile

try:  # Preferred TIFF IO
    import tifffile as tiff  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    tiff = None  # type: ignore

plt = None  # lazily imported matplotlib.pyplot

try:  # fallback for simple image reads (estimate mode)
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

from msrigid import (
    RigidTransformTurbo,
    anchor_points,
    apply_transform,
    estimate_rigid_from_images,
    intensity_principal_landmarks,
    invert_transform,
    parse_msreg_file,
    rms_error,
    save_transform_json,
    transform_from_landmarks,
)

Point = Tuple[float, float]


###############################################################################
# Utilities
###############################################################################


def _natural_key(name: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _require_tifffile() -> None:
    if tiff is None:
        raise RuntimeError("tifffile is required for TIFF IO. Install via 'pip install tifffile'.")


def _read_stack(path: Path) -> np.ndarray:
    _require_tifffile()
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Identify channel axis (prefer axis with <=4 entries)
        channel_axis = 0
        for ax, size in enumerate(arr.shape):
            if ax in (-2, -1):
                continue
            if size <= 4:
                channel_axis = ax
                break
        if channel_axis != 0:
            arr = np.moveaxis(arr, channel_axis, 0)
        elif arr.shape[0] > 16:
            raise ValueError(f"Ambiguous TIFF layout for {path}: shape {arr.shape}")
    else:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}; expected 2D or 3D (C,Y,X)")
    return arr


def _write_tiff(path: Path, data: np.ndarray, **kwargs) -> None:
    _require_tifffile()
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), data, **kwargs)


def _read_gray_image(path: Path) -> np.ndarray:
    if tiff is not None and path.suffix.lower() in {".tif", ".tiff"}:
        arr = tiff.imread(str(path))
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # prefer channel-last for RGB, else take first slice
            if arr.shape[0] <= arr.shape[-1]:
                return arr[0]
            return arr[..., 0]
    if Image is not None:
        with Image.open(path) as im:
            if im.mode not in ("L", "I;16"):
                im = im.convert("L")
            return np.array(im)
    raise RuntimeError("Unable to read image; install tifffile or Pillow.")


def _normalize_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out


def _cast_array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if arr.dtype == dtype:
        return arr
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(arr, info.min, info.max)
        return np.rint(clipped).astype(dtype, copy=False)
    return arr.astype(dtype, copy=False)


def _apply_points(rt: RigidTransformTurbo, pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    R = np.array([[rt.m00, rt.m01], [rt.m10, rt.m11]], dtype=np.float64)
    t = np.array([rt.m02, rt.m12], dtype=np.float64)
    arr = np.asarray(pts, dtype=np.float64)
    res = (arr @ R.T) + t
    return [tuple(map(float, row)) for row in res]


def _synthetic_landmarks(rt: RigidTransformTurbo, width: int, height: int, coord_base: float) -> Dict[str, Any]:
    anchors = anchor_points("rigid", width, height, coord_base=coord_base)
    target_pts = [(float(x), float(y)) for (x, y, _) in anchors]
    moving_pts = _apply_points(invert_transform(rt), target_pts)
    return {
        "mode": "synthetic_anchor",
        "coord_base": coord_base,
        "moving": moving_pts,
        "fixed": target_pts,
    }


def _rgb_composite(dapi: np.ndarray, ch_fixed: np.ndarray, ch_moving: np.ndarray) -> np.ndarray:
    b = _normalize_display(dapi)
    g = _normalize_display(ch_fixed)
    r = _normalize_display(ch_moving)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def _ensure_matplotlib() -> None:
    global plt
    if plt is not None:
        return
    try:
        import matplotlib  # type: ignore
        if "MPLCONFIGDIR" not in os.environ:
            tmp = Path(tempfile.gettempdir()) / "msrigid_mpl"
            tmp.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(tmp)
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt  # type: ignore
        plt = _plt
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("matplotlib is required for QC output") from exc


def _write_qc_grid(path: Path, *,
                   fixed_dapi: np.ndarray,
                   moving_dapi: np.ndarray,
                   aligned_dapi: np.ndarray,
                   fixed_signal: np.ndarray,
                   moving_signal: np.ndarray,
                   aligned_signal: np.ndarray) -> None:
    _ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 0: DAPI channels (blue colormap for clarity)
    axes[0, 0].set_title("Fixed DAPI")
    axes[0, 0].imshow(_normalize_display(fixed_dapi), cmap="Blues", interpolation="nearest")
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Moving DAPI (raw)")
    axes[0, 1].imshow(_normalize_display(moving_dapi), cmap="Blues", interpolation="nearest")
    axes[0, 1].axis("off")

    overlay_dapi = np.zeros(fixed_dapi.shape + (3,), dtype=np.float32)
    overlay_dapi[..., 2] = _normalize_display(fixed_dapi)  # blue
    overlay_dapi[..., 1] = _normalize_display(moving_dapi)  # green
    axes[0, 2].set_title("DAPI overlay")
    axes[0, 2].imshow(overlay_dapi)
    axes[0, 2].axis("off")

    # Row 1: signal channels (magenta colormap)
    axes[1, 0].set_title("Fixed signal")
    axes[1, 0].imshow(_normalize_display(fixed_signal), cmap="magma", interpolation="nearest")
    axes[1, 0].axis("off")

    axes[1, 1].set_title("Moving signal (raw)")
    axes[1, 1].imshow(_normalize_display(moving_signal), cmap="magma", interpolation="nearest")
    axes[1, 1].axis("off")

    overlay_signal = np.zeros(fixed_signal.shape + (3,), dtype=np.float32)
    overlay_signal[..., 0] = _normalize_display(aligned_signal)  # red
    overlay_signal[..., 1] = _normalize_display(fixed_signal)   # green
    axes[1, 2].set_title("Aligned overlay")
    axes[1, 2].imshow(overlay_signal)
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _inspect_channels(input_path: Path, output_path: Path) -> None:
    arr = _read_stack(input_path)
    H, W = arr.shape[-2:]
    pages = []
    axes_text = []
    for idx, plane in enumerate(arr):
        label = f"Channel {idx}"
        pages.append(plane)
        axes_text.append(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        np.asarray(pages),
        metadata={"axes": "CYX", "Channel": axes_text},
    )


def _parse_refine_steps(text: Optional[str]) -> Optional[List[float]]:
    if not text:
        return None
    vals = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals or None


def _seed_points(width: int, height: int, coord_base: float = 0.0) -> List[Tuple[float, float]]:
    cx = (width - 1) * 0.5 + coord_base
    cy = (height - 1) * 0.5 + coord_base
    return [
        (cx, cy),
        (cx, (height - 1) * 0.25 + coord_base),
        (cx, (height - 1) * 0.75 + coord_base),
        ((width - 1) * 0.25 + coord_base, cy),
        ((width - 1) * 0.75 + coord_base, cy),
    ]


def _refine_points_with_blobs(
    image: np.ndarray,
    seeds: Sequence[Point],
    *,
    window: int = 96,
) -> List[Point]:
    arr = np.asarray(image, dtype=np.float32)
    H, W = arr.shape
    refined: List[Point] = []
    half = max(4, window // 2)
    sigma = max(4.0, window * 0.25)
    for (x_seed, y_seed) in seeds:
        x_c = float(np.clip(x_seed, 0, W - 1))
        y_c = float(np.clip(y_seed, 0, H - 1))
        x0 = max(0, int(round(x_c)) - half)
        x1 = min(W, int(round(x_c)) + half)
        y0 = max(0, int(round(y_c)) - half)
        y1 = min(H, int(round(y_c)) + half)
        patch = arr[y0:y1, x0:x1]
        if patch.size == 0:
            refined.append((x_c, y_c))
            continue
        patch_norm = patch - float(patch.min())
        max_val = float(patch_norm.max())
        if max_val <= 0:
            refined.append((x_c, y_c))
            continue
        patch_norm /= max_val
        yy, xx = np.indices(patch_norm.shape)
        sigma_local = max(4.0, 0.25 * min(patch_norm.shape[0], patch_norm.shape[1], window))
        cx_local = float(np.clip(x_c - x0, 0, patch_norm.shape[1] - 1))
        cy_local = float(np.clip(y_c - y0, 0, patch_norm.shape[0] - 1))
        gauss = np.exp(-(((xx - cx_local) ** 2) + ((yy - cy_local) ** 2)) / (2.0 * (sigma_local ** 2)))
        weights = patch_norm * gauss
        wsum = float(weights.sum())
        if wsum <= 0:
            refined.append((x_c, y_c))
            continue
        cx = float((weights * xx).sum() / wsum)
        cy = float((weights * yy).sum() / wsum)
        refined.append((float(x0 + cx), float(y0 + cy)))
    return refined


def _auto_landmarks_from_images(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    *,
    coord_base: float,
    method: str,
) -> Tuple[RigidTransformTurbo, Dict[str, Any], float]:
    moving_pts = intensity_principal_landmarks(moving_image, coord_base=coord_base)
    fixed_pts = intensity_principal_landmarks(fixed_image, coord_base=coord_base)
    rt = transform_from_landmarks(moving_pts, fixed_pts, method=method)
    rms = rms_error(rt, moving_pts, fixed_pts)
    landmarks = {
        "mode": "auto_intensity",
        "coord_base": coord_base,
        "moving": moving_pts,
        "fixed": fixed_pts,
    }
    return rt, landmarks, rms


###############################################################################
# Transform helpers
###############################################################################


def _select_block(blocks: List[Any], block_index: int) -> Any:
    if not blocks:
        raise RuntimeError("No RIGID_BODY blocks found in MultiStackReg file")
    if block_index < 1 or block_index > len(blocks):
        raise IndexError(f"block index {block_index} out of range (1..{len(blocks)})")
    return blocks[block_index - 1]


def _transform_from_block(block, method: str) -> Tuple[RigidTransformTurbo, Dict[str, Any], float]:
    rt = transform_from_landmarks(block.source_points, block.target_points, method=method)
    err = rms_error(rt, block.source_points, block.target_points)
    landmarks = {
        "mode": "msreg_file",
        "moving": [(float(x), float(y)) for (x, y) in block.source_points],
        "fixed": [(float(x), float(y)) for (x, y) in block.target_points],
        "source_index": int(block.source_index),
        "target_index": int(block.target_index),
    }
    return rt, landmarks, err


def _estimate_transform_from_images(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    *,
    angle_min: float,
    angle_max: float,
    coarse_step: float,
    refine_steps: Optional[Sequence[float]],
    downsample_to: int,
    upsample: int,
    order: int,
) -> Tuple[RigidTransformTurbo, Dict[str, Any]]:
    rt, info = estimate_rigid_from_images(
        fixed_img,
        moving_img,
        angle_min=angle_min,
        angle_max=angle_max,
        coarse_step=coarse_step,
        refine_steps=refine_steps,
        downsample_to=downsample_to,
        upsample_factor=upsample,
        order=order,
    )
    return rt, info


###############################################################################
# Command implementations
###############################################################################


def cmd_estimate(args: argparse.Namespace) -> int:
    fixed = _read_gray_image(Path(args.fixed))
    moving = _read_gray_image(Path(args.moving))
    rt, info = _estimate_transform_from_images(
        fixed,
        moving,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        coarse_step=args.coarse_step,
        refine_steps=_parse_refine_steps(args.refine_steps),
        downsample_to=args.downsample_to,
        upsample=args.upsample,
        order=args.order,
    )
    print(f"[estimate] θ={rt.theta_deg:.6f}°, tx={rt.m02:.3f}, ty={rt.m12:.3f}, error={info.get('error', float('nan')):.6f}")
    if args.out:
        aligned = apply_transform(moving[np.newaxis, ...], rt, order=args.order)[0]
        aligned = _cast_array(aligned, moving.dtype)
        _write_tiff(Path(args.out), aligned)
        print(f"[estimate] wrote aligned image: {args.out}")
    if args.json:
        meta = {"mode": "estimate", "info": info, "paths": {"fixed": args.fixed, "moving": args.moving}}
        save_transform_json(Path(args.json), rt, meta=meta)
        print(f"[estimate] wrote transform JSON: {args.json}")
    return 0


def _prepare_channels(arr: np.ndarray, *, ref_idx: int, signal_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if ref_idx < 0 or ref_idx >= arr.shape[0]:
        raise IndexError(f"Reference channel index {ref_idx} out of range for array with {arr.shape[0]} channels")
    if signal_idx < 0 or signal_idx >= arr.shape[0]:
        raise IndexError(f"Signal channel index {signal_idx} out of range for array with {arr.shape[0]} channels")
    return arr[ref_idx], arr[signal_idx]


def _run_alignment(
    *,
    fixed_path: Path,
    moving_path: Path,
    outdir: Path,
    stem: str,
    args: argparse.Namespace,
    block: Optional[Any],
) -> Dict[str, Any]:
    fixed_stack = _read_stack(fixed_path)
    moving_stack = _read_stack(moving_path)
    if fixed_stack.shape[-2:] != moving_stack.shape[-2:]:
        raise ValueError("Fixed and moving images must have matching spatial dimensions")

    fixed_dapi, fixed_signal = _prepare_channels(fixed_stack, ref_idx=args.fixed_ref_channel, signal_idx=args.fixed_signal_channel)
    moving_dapi, moving_signal = _prepare_channels(moving_stack, ref_idx=args.moving_ref_channel, signal_idx=args.moving_signal_channel)

    strategy = getattr(args, "landmark_strategy", "anchors")

    if block is not None and not args.force_estimate:
        rt, landmarks, rms = _transform_from_block(block, args.method)
        estimation_info = None
    else:
        if strategy == "auto":
            rt, landmarks, rms = _auto_landmarks_from_images(
                moving_dapi,
                fixed_dapi,
                coord_base=args.coord_base,
                method=args.method,
            )
            estimation_info = None
        elif strategy == "blend":
            rt_coarse, estimation_info = _estimate_transform_from_images(
                fixed_dapi,
                moving_dapi,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
                coarse_step=args.coarse_step,
                refine_steps=_parse_refine_steps(args.refine_steps),
                downsample_to=args.downsample_to,
                upsample=args.upsample,
                order=args.order,
            )
            seeds_fixed = _seed_points(fixed_dapi.shape[1], fixed_dapi.shape[0], coord_base=args.coord_base)
            inv = invert_transform(rt_coarse)
            seeds_moving_guess = _apply_points(inv, seeds_fixed)
            refined_fixed = _refine_points_with_blobs(fixed_dapi, seeds_fixed)
            refined_moving = _refine_points_with_blobs(moving_dapi, seeds_moving_guess)
            if len(refined_fixed) >= 3 and len(refined_moving) >= 3:
                rt = transform_from_landmarks(refined_moving, refined_fixed, method="ls")
                rms = rms_error(rt, refined_moving, refined_fixed)
                landmarks = {
                    "mode": "blend",
                    "coarse": {
                        "theta_deg": rt_coarse.theta_deg,
                        "tx": rt_coarse.m02,
                        "ty": rt_coarse.m12,
                    },
                    "seeds_fixed": seeds_fixed,
                    "seeds_moving_guess": seeds_moving_guess,
                    "refined_fixed": refined_fixed,
                    "refined_moving": refined_moving,
                }
            else:
                rt = rt_coarse
                rms = None
                landmarks = _synthetic_landmarks(rt, width=fixed_dapi.shape[1], height=fixed_dapi.shape[0], coord_base=args.coord_base)
        else:
            rt, estimation_info = _estimate_transform_from_images(
                fixed_dapi,
                moving_dapi,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
                coarse_step=args.coarse_step,
                refine_steps=_parse_refine_steps(args.refine_steps),
                downsample_to=args.downsample_to,
                upsample=args.upsample,
                order=args.order,
            )
            rms = None
            landmarks = _synthetic_landmarks(rt, width=fixed_dapi.shape[1], height=fixed_dapi.shape[0], coord_base=args.coord_base)

    aligned_stack = apply_transform(moving_stack, rt, order=args.order)
    aligned_dapi, aligned_signal = _prepare_channels(aligned_stack, ref_idx=args.moving_ref_channel, signal_idx=args.moving_signal_channel)

    dtype_out = np.result_type(fixed_stack.dtype, moving_stack.dtype)
    aligned_cyx = np.stack([fixed_dapi, fixed_signal, aligned_signal], axis=0)
    aligned_cyx = _cast_array(aligned_cyx, dtype_out)
    stack_path = outdir / f"{stem}_aligned.tif"
    _write_tiff(stack_path, aligned_cyx, metadata={"axes": "CYX", "Channel": ["DAPI", "fixed", "moving_aligned"]})

    rgb_path = outdir / f"{stem}_aligned_rgb.tif"
    rgb = _rgb_composite(fixed_dapi, fixed_signal, aligned_signal)
    _write_tiff(rgb_path, rgb)

    qc_path = outdir / f"{stem}_qc.png"
    if not args.no_qc:
        _write_qc_grid(
            qc_path,
            fixed_dapi=fixed_dapi,
            moving_dapi=moving_dapi,
            aligned_dapi=aligned_dapi,
            fixed_signal=fixed_signal,
            moving_signal=moving_signal,
            aligned_signal=aligned_signal,
        )

    json_path = outdir / f"{stem}_aligned.json"
    meta = {
        "paths": {"fixed": str(fixed_path), "moving": str(moving_path)},
        "channels": {
            "fixed_ref": args.fixed_ref_channel,
            "fixed_signal": args.fixed_signal_channel,
            "moving_ref": args.moving_ref_channel,
            "moving_signal": args.moving_signal_channel,
        },
        "outputs": {
            "stack": str(stack_path),
            "rgb": str(rgb_path),
            "qc": str(qc_path) if not args.no_qc else None,
        },
        "method": args.method,
        "strategy": strategy,
        "coord_base": args.coord_base,
        "landmarks": landmarks,
        "estimation": estimation_info,
        "rms_error": rms,
        "msreg_file": str(args.msreg_file) if getattr(args, "msreg_file", None) else None,
    }
    save_transform_json(json_path, rt, meta=meta)

    if rms is not None and args.rms_max is not None and len(landmarks.get("moving", [])) == 3:
        if rms > args.rms_max:
            raise RuntimeError(f"Landmark RMS {rms:.3e} exceeded threshold {args.rms_max:.3e}")

    print(f"[aligned] {moving_path.name} → {stack_path.name}  θ={rt.theta_deg:.3f}°, tx={rt.m02:.2f}, ty={rt.m12:.2f}")
    if rms is not None:
        print(f"          landmark RMS = {rms:.3e} px")
    return {
        "stack": stack_path,
        "rgb": rgb_path,
        "qc": qc_path,
        "json": json_path,
        "transform": rt,
        "rms": rms,
    }


def cmd_single(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir or Path(args.moving).parent / "aligned")
    outdir.mkdir(parents=True, exist_ok=True)
    block = None
    if args.msreg_file:
        blocks = parse_msreg_file(args.msreg_file)
        block = _select_block(blocks, args.block)
    stem = args.stem or Path(args.moving).stem
    _run_alignment(
        fixed_path=Path(args.fixed),
        moving_path=Path(args.moving),
        outdir=outdir,
        stem=stem,
        args=args,
        block=block,
    )
    return 0


def _iter_pairs(dir_path: Path, pattern: str) -> List[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file()]
    if pattern:
        files = [p for p in files if fnmatch.fnmatch(p.name, pattern)]
    files.sort(key=lambda p: _natural_key(p.name))
    return files


def cmd_batch(args: argparse.Namespace) -> int:
    fixed_dir = Path(args.fixed_dir)
    moving_dir = Path(args.moving_dir)
    fixed_files = _iter_pairs(fixed_dir, args.pattern)
    moving_files = _iter_pairs(moving_dir, args.pattern)
    if not fixed_files:
        raise RuntimeError(f"No files matched in {fixed_dir} with pattern {args.pattern}")
    if not moving_files:
        raise RuntimeError(f"No files matched in {moving_dir} with pattern {args.pattern}")
    total = min(len(fixed_files), len(moving_files))

    blocks_map: Dict[int, Any] = {}
    if args.msreg_file:
        for blk in parse_msreg_file(args.msreg_file):
            blocks_map[int(blk.source_index)] = blk

    outdir = Path(args.outdir or (moving_dir / "aligned"))
    outdir.mkdir(parents=True, exist_ok=True)
    start = max(1, args.start)
    stop = min(args.stop or total, total)
    for idx in range(start, stop + 1):
        fixed_path = fixed_files[idx - 1]
        moving_path = moving_files[idx - 1]
        blk = blocks_map.get(idx)
        stem = f"{idx:04d}_{Path(moving_path).stem}"
        _run_alignment(
            fixed_path=fixed_path,
            moving_path=moving_path,
            outdir=outdir,
            stem=stem,
            args=args,
            block=blk,
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.out)
    _inspect_channels(input_path, output_path)
    print(f"[inspect] wrote {output_path}")
    return 0


###############################################################################
# CLI plumbing
###############################################################################


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="msrigid CLI — TurboReg-equivalent rigid alignment")
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict(
        method=dict(choices=["turbo", "ls"], default="turbo"),
        coord_base=dict(type=float, default=0.0, help="Anchor coordinate base offset (0.0 for ImageJ, 0.5 for pixel centers)."),
        order=dict(type=int, default=1, help="Interpolation order (0=nearest, 1=bilinear, 3=cubic)."),
        angle_min=dict(type=float, default=-45.0),
        angle_max=dict(type=float, default=45.0),
        coarse_step=dict(type=float, default=1.0),
        refine_steps=dict(type=str, default="0.25,0.05"),
        downsample_to=dict(type=int, default=1024),
        upsample=dict(type=int, default=10),
    )

    # estimate
    p_est = sub.add_parser("estimate", help="Estimate rigid transform between two grayscale images")
    p_est.add_argument("--fixed", required=True, help="Fixed/reference 2D image")
    p_est.add_argument("--moving", required=True, help="Moving 2D image")
    p_est.add_argument("--out", help="Optional path for aligned moving image")
    p_est.add_argument("--json", help="Optional JSON output path for transform metadata")
    p_est.add_argument("--angle-min", **common["angle_min"])
    p_est.add_argument("--angle-max", **common["angle_max"])
    p_est.add_argument("--coarse-step", **common["coarse_step"])
    p_est.add_argument("--refine-steps", **common["refine_steps"])
    p_est.add_argument("--downsample-to", **common["downsample_to"])
    p_est.add_argument("--upsample", **common["upsample"])
    p_est.add_argument("--order", **common["order"])
    p_est.set_defaults(func=cmd_estimate)

    # single
    p_single = sub.add_parser("single", help="Align one fixed/moving TIFF pair and emit outputs")
    p_single.add_argument("--fixed", required=True, help="Fixed/reference multi-channel TIFF")
    p_single.add_argument("--moving", required=True, help="Moving multi-channel TIFF")
    p_single.add_argument("--msreg-file", help="Optional MultiStackReg transformation file (.txt)")
    p_single.add_argument("--block", type=int, default=1, help="1-based block index inside msreg file")
    p_single.add_argument("--method", **common["method"])
    p_single.add_argument("--coord-base", **common["coord_base"])
    p_single.add_argument("--order", **common["order"])
    p_single.add_argument("--angle-min", **common["angle_min"])
    p_single.add_argument("--angle-max", **common["angle_max"])
    p_single.add_argument("--coarse-step", **common["coarse_step"])
    p_single.add_argument("--refine-steps", **common["refine_steps"])
    p_single.add_argument("--downsample-to", **common["downsample_to"])
    p_single.add_argument("--upsample", **common["upsample"])
    p_single.add_argument("--fixed-ref-channel", type=int, default=0)
    p_single.add_argument("--fixed-signal-channel", type=int, default=1)
    p_single.add_argument("--moving-ref-channel", type=int, default=0)
    p_single.add_argument("--moving-signal-channel", type=int, default=1)
    p_single.add_argument("--outdir", help="Output directory (default: moving parent/aligned)")
    p_single.add_argument("--stem", help="Output file stem (default: moving filename stem)")
    p_single.add_argument("--no-qc", action="store_true", help="Skip QC PNG generation")
    p_single.add_argument("--force-estimate", action="store_true", help="Ignore msreg file and re-estimate from DAPI channels")
    p_single.add_argument("--rms-max", type=float, default=1e-6, help="Max RMS for 3-point blocks (turbo mode)")
    p_single.add_argument("--landmark-strategy", choices=["anchors", "auto", "blend"], default="anchors",
                         help="anchors = estimator + fixed anchors, auto = global intensity landmarks, blend = coarse estimator then local blob refinement.")
    p_single.set_defaults(func=cmd_single)

    # batch
    p_batch = sub.add_parser("batch", help="Align many TIFF pairs using natural order")
    p_batch.add_argument("--fixed-dir", required=True, help="Directory with fixed/reference TIFFs")
    p_batch.add_argument("--moving-dir", required=True, help="Directory with moving TIFFs")
    p_batch.add_argument("--pattern", default="*.tif", help="Glob pattern for TIFF selection")
    p_batch.add_argument("--msreg-file", help="Optional MultiStackReg transformation file (.txt)")
    p_batch.add_argument("--method", **common["method"])
    p_batch.add_argument("--coord-base", **common["coord_base"])
    p_batch.add_argument("--order", **common["order"])
    p_batch.add_argument("--angle-min", **common["angle_min"])
    p_batch.add_argument("--angle-max", **common["angle_max"])
    p_batch.add_argument("--coarse-step", **common["coarse_step"])
    p_batch.add_argument("--refine-steps", **common["refine_steps"])
    p_batch.add_argument("--downsample-to", **common["downsample_to"])
    p_batch.add_argument("--upsample", **common["upsample"])
    p_batch.add_argument("--fixed-ref-channel", type=int, default=0)
    p_batch.add_argument("--fixed-signal-channel", type=int, default=1)
    p_batch.add_argument("--moving-ref-channel", type=int, default=0)
    p_batch.add_argument("--moving-signal-channel", type=int, default=1)
    p_batch.add_argument("--outdir", help="Output directory (default: moving_dir/aligned)")
    p_batch.add_argument("--no-qc", action="store_true")
    p_batch.add_argument("--force-estimate", action="store_true")
    p_batch.add_argument("--rms-max", type=float, default=1e-6)
    p_batch.add_argument("--landmark-strategy", choices=["anchors", "auto", "blend"], default="anchors")
    p_batch.add_argument("--start", type=int, default=1, help="1-based start index")
    p_batch.add_argument("--stop", type=int, help="1-based inclusive stop index")
    p_batch.set_defaults(func=cmd_batch)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Export a labeled multi-page TIFF with one page per channel")
    p_inspect.add_argument("--input", required=True, help="Path to a multi-channel TIFF")
    p_inspect.add_argument("--out", required=True, help="Output TIFF path")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
