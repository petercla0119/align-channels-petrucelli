"""msrigid — TurboReg/MultiStackReg-compatible rigid-body alignment utilities.

This module re-implements the rigid-body math from TurboReg and MultiStackReg so we
can estimate, compose, and apply rotation+translation transforms entirely in Python.
It intentionally mirrors ImageJ's coordinate conventions (top-left origin, x→right,
y→down) and supports both TurboReg-exact (`method="turbo"`) and least-squares
(`method="ls"`) solvers. Optional helpers cover anchor generation, MultiStackReg
transformation parsing, Fourier–polar angle estimation, and array warping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional
import json
import math
import re

import numpy as np

try:  # SciPy is preferred for affine warping & resizing
    import scipy.ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ndi = None  # type: ignore

try:  # scikit-image is used for fallbacks + Fourier-polar angle estimation
    from skimage import transform as sktf  # type: ignore
    from skimage.registration import phase_cross_correlation  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sktf = None  # type: ignore
    phase_cross_correlation = None  # type: ignore

Point = Tuple[float, float]


class AlignmentError(RuntimeError):
    """Raised when parsing or math steps encounter invalid inputs."""


@dataclass
class RigidTransformTurbo:
    """TurboReg-style rigid transform mapping moving→fixed coordinates."""

    m00: float
    m01: float
    m02: float
    m10: float
    m11: float
    m12: float

    def as_matrix3x3(self) -> np.ndarray:
        """Return 3×3 homogeneous matrix with bottom row [0, 0, 1]."""
        return np.array(
            [
                [self.m00, self.m01, self.m02],
                [self.m10, self.m11, self.m12],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def theta_rad(self) -> float:
        return math.atan2(self.m10, self.m00)

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta_rad)

    def macro_fields(self) -> Dict[str, float]:
        return {
            "m00": float(self.m00),
            "m01": float(self.m01),
            "m02": float(self.m02),
            "m10": float(self.m10),
            "m11": float(self.m11),
            "m12": float(self.m12),
            "theta_deg": float(self.theta_deg),
        }

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "matrix": [
                [float(self.m00), float(self.m01), float(self.m02)],
                [float(self.m10), float(self.m11), float(self.m12)],
            ],
            "theta_deg": float(self.theta_deg),
            "convention": "fixed = R * moving + t",
            "macro": self.macro_fields(),
        }

    @classmethod
    def from_matrix(cls, mat: Sequence[Sequence[float]]) -> "RigidTransformTurbo":
        if len(mat) != 3 or len(mat[0]) != 3:
            raise ValueError("matrix must be 3x3 for homogeneous coords")
        return cls(mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2])


@dataclass
class LandmarkBlock:
    source_index: int  # 1-based index in MultiStackReg file
    target_index: int
    source_points: List[Point]
    target_points: List[Point]


_ANCHOR_MODES = {"translation", "rigid", "scaled_rotation", "affine"}


def anchor_points(mode: str, width: int, height: int, coord_base: float = 0.0) -> np.ndarray:
    """Return anchor points (N×3 array) following MultiStackReg presets."""

    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive integers")
    mode = (mode or "").strip().lower()
    if mode not in _ANCHOR_MODES:
        raise ValueError(f"Unknown anchor mode: {mode}")

    def _pt(x: int, y: int) -> Tuple[float, float, float]:
        return (float(x) + coord_base, float(y) + coord_base, 1.0)

    w2 = width // 2
    h2 = height // 2
    h4 = height // 4
    w4 = width // 4

    if mode == "translation":
        pts = [_pt(w2, h2)]
    elif mode == "rigid":
        pts = [_pt(w2, h2), _pt(w2, h4), _pt(w2, (3 * height) // 4)]
    elif mode == "scaled_rotation":
        pts = [_pt(w4, h2), _pt((3 * width) // 4, h2)]
    else:  # affine
        pts = [_pt(w2, h4), _pt(w4, (3 * height) // 4), _pt((3 * width) // 4, (3 * height) // 4)]

    return np.asarray(pts, dtype=np.float64)


def intensity_principal_landmarks(
    image: np.ndarray,
    *,
    coord_base: float = 0.0,
    scale: float = 0.25,
    min_weight: float = 1e-6,
) -> List[Point]:
    """Derive three deterministic landmarks from intensity moments (center + orthogonal axes).

    Falls back to geometric rigid anchors if the image has no dynamic range.
    """

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("intensity_principal_landmarks expects a 2D image")
    h, w = arr.shape
    if h == 0 or w == 0:
        raise ValueError("empty image provided for landmark detection")
    data = arr - float(arr.min())
    data[data < 0.0] = 0.0
    total = float(data.sum())
    if not np.isfinite(total) or total < min_weight:
        anchors = anchor_points("rigid", w, h, coord_base=coord_base)
        return [(float(x), float(y)) for (x, y, _) in anchors]

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    cx = float((data * xx).sum() / total)
    cy = float((data * yy).sum() / total)
    dx = xx - cx
    dy = yy - cy
    cov_xx = float((data * dx * dx).sum() / total)
    cov_yy = float((data * dy * dy).sum() / total)
    cov_xy = float((data * dx * dy).sum() / total)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    cov += 1e-9 * np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    axis_major = evecs[:, int(np.argmax(evals))]
    norm = float(np.linalg.norm(axis_major))
    if not np.isfinite(norm) or norm < 1e-9:
        axis_major = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis_major /= norm
    axis_minor = np.array([-axis_major[1], axis_major[0]], dtype=np.float64)
    axis_scale = max(1.0, scale * float(min(h, w)))

    pts = [
        (cx + coord_base, cy + coord_base),
        (cx + axis_major[0] * axis_scale + coord_base,
         cy + axis_major[1] * axis_scale + coord_base),
        (cx + axis_minor[0] * axis_scale + coord_base,
         cy + axis_minor[1] * axis_scale + coord_base),
    ]
    return [(float(x), float(y)) for (x, y) in pts]


_MSREG_COORD_RE = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def parse_msreg_file(path: str | Path) -> List[LandmarkBlock]:
    """Parse a MultiStackReg "Transformation File" (.txt) and collect rigid blocks."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: List[LandmarkBlock] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "RIGID_BODY":
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            header = lines[i]
            i += 1
            m = re.search(r"Source\s*img:\s*(\d+).+Target\s*img:\s*(\d+)", header)
            if not m:
                raise AlignmentError(f"Malformed RIGID_BODY header near line {i}: {header!r}")
            src_idx = int(m.group(1))
            tgt_idx = int(m.group(2))

            src_pts: List[Point] = []
            while i < len(lines) and lines[i].strip():
                cm = _MSREG_COORD_RE.match(lines[i])
                if not cm:
                    break
                src_pts.append((float(cm.group(1)), float(cm.group(2))))
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            tgt_pts: List[Point] = []
            while i < len(lines) and lines[i].strip():
                cm = _MSREG_COORD_RE.match(lines[i])
                if not cm:
                    break
                tgt_pts.append((float(cm.group(1)), float(cm.group(2))))
                i += 1
            if len(src_pts) < 3 or len(tgt_pts) < 3:
                raise AlignmentError("Rigid blocks require at least 3 landmarks")
            blocks.append(LandmarkBlock(src_idx, tgt_idx, src_pts, tgt_pts))
        else:
            i += 1
    return blocks


def _angle_turbo(moving_pts: Sequence[Point], fixed_pts: Sequence[Point]) -> float:
    if len(moving_pts) < 3 or len(fixed_pts) < 3:
        raise AlignmentError("Need ≥3 landmarks for rigid TurboReg solve")
    dx_m = moving_pts[2][0] - moving_pts[1][0]
    dy_m = moving_pts[2][1] - moving_pts[1][1]
    dx_f = fixed_pts[2][0] - fixed_pts[1][0]
    dy_f = fixed_pts[2][1] - fixed_pts[1][1]
    return math.atan2(dx_m, dy_m) - math.atan2(dx_f, dy_f)


def rigid_from_points_turbo(moving_pts: Sequence[Point], fixed_pts: Sequence[Point]) -> RigidTransformTurbo:
    """TurboReg-exact rigid transform; maps moving coords → fixed coords."""

    angle = _angle_turbo(moving_pts, fixed_pts)
    c = math.cos(angle)
    s = math.sin(angle)
    mx0, my0 = moving_pts[0]
    fx0, fy0 = fixed_pts[0]
    tx = fx0 - c * mx0 + s * my0
    ty = fy0 - s * mx0 - c * my0
    return RigidTransformTurbo(c, -s, tx, s, c, ty)


def rigid_from_points_ls(moving_pts: Sequence[Point], fixed_pts: Sequence[Point]) -> RigidTransformTurbo:
    """Least-squares (Kabsch) rigid transform with no scaling."""

    A = np.asarray(moving_pts, dtype=np.float64)
    B = np.asarray(fixed_pts, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        n = min(A.shape[0], B.shape[0])
        A, B = A[:n], B[:n]
    if A.shape[0] < 3:
        raise AlignmentError("Least-squares rigid requires ≥3 shared landmarks")
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    Ac = A - ca
    Bc = B - cb
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cb - (R @ ca)
    return RigidTransformTurbo(float(R[0, 0]), float(R[0, 1]), float(t[0]), float(R[1, 0]), float(R[1, 1]), float(t[1]))


def transform_from_landmarks(moving_pts: Sequence[Point], fixed_pts: Sequence[Point], method: str = "turbo") -> RigidTransformTurbo:
    method = (method or "turbo").lower()
    if method in {"turbo", "turboreg", "msreg"}:
        return rigid_from_points_turbo(moving_pts, fixed_pts)
    if method in {"ls", "least_squares", "kabsch"}:
        return rigid_from_points_ls(moving_pts, fixed_pts)
    raise ValueError(f"Unknown method: {method}")


def compose_transforms(first: RigidTransformTurbo, second: RigidTransformTurbo) -> RigidTransformTurbo:
    """Return first ∘ second (matrix multiply, left-to-right)."""

    A = first.as_matrix3x3()
    B = second.as_matrix3x3()
    C = A @ B
    return RigidTransformTurbo.from_matrix(C)


def invert_transform(rt: RigidTransformTurbo) -> RigidTransformTurbo:
    """Inverse rigid transform (fixed→moving)."""

    inv = np.linalg.inv(rt.as_matrix3x3())
    return RigidTransformTurbo.from_matrix(inv)


def rms_error(rt: RigidTransformTurbo, moving_pts: Sequence[Point], fixed_pts: Sequence[Point]) -> float:
    M = np.asarray(moving_pts, dtype=np.float64)
    F = np.asarray(fixed_pts, dtype=np.float64)
    R = np.array([[rt.m00, rt.m01], [rt.m10, rt.m11]], dtype=np.float64)
    t = np.array([rt.m02, rt.m12], dtype=np.float64)
    pred = (M @ R.T) + t
    resid = F - pred
    return float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))


def _ndimage_matrix_and_offset(rt: RigidTransformTurbo) -> Tuple[np.ndarray, np.ndarray]:
    """Convert x/y-forward matrix into ndimage's (y,x) inverse mapping."""

    forward = rt.as_matrix3x3()
    inv = np.linalg.inv(forward)
    a = inv[0, 0]
    b = inv[0, 1]
    c = inv[1, 0]
    d = inv[1, 1]
    tx = rt.m02
    ty = rt.m12
    matrix = np.array([[d, c], [b, a]], dtype=np.float64)
    offset = np.array([-(c * tx + d * ty), -(a * tx + b * ty)], dtype=np.float64)
    return matrix, offset


def apply_transform(
    image: np.ndarray,
    rt: RigidTransformTurbo,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Apply rigid transform to the last two axes (Y,X) of `image`."""

    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError("image must have at least 2 dimensions (Y,X)")
    leading = arr.shape[:-2]
    H, W = arr.shape[-2:]
    reshaped = arr.reshape((-1, H, W))

    def _warp2d(img2d: np.ndarray) -> np.ndarray:
        if ndi is not None:
            matrix, offset = _ndimage_matrix_and_offset(rt)
            return ndi.affine_transform(
                img2d,
                matrix=matrix,
                offset=offset,
                order=order,
                mode=mode,
                cval=cval,
                output_shape=(H, W),
                prefilter=bool(order > 1),
            )
        if sktf is not None:
            forward = rt.as_matrix3x3()
            inv = np.linalg.inv(forward)
            tf = sktf.AffineTransform(matrix=inv)
            warped = sktf.warp(
                img2d,
                inverse_map=tf,
                order=order,
                preserve_range=True,
                mode=mode,
                cval=cval,
                output_shape=(H, W),
            )
            return warped.astype(img2d.dtype, copy=False)
        raise RuntimeError("SciPy or scikit-image required for affine warping")

    out_slices = [_warp2d(slice2d) for slice2d in reshaped]
    out = np.stack(out_slices, axis=0)
    return out.reshape(leading + (H, W))


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    return (wy * wx).astype(np.float32)


def _downsample(arr: np.ndarray, max_side: int) -> np.ndarray:
    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    scale = 1.0
    if max_side is None or max_side <= 0:
        return arr.astype(np.float32, copy=False)
    if max(h, w) > max_side > 0:
        scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return arr.astype(np.float32, copy=False)
    if ndi is not None:
        return ndi.zoom(arr, (scale, scale), order=1, mode="nearest").astype(np.float32, copy=False)
    step = max(1, int(round(1 / scale)))
    return arr[::step, ::step].astype(np.float32, copy=False)


def _normalize_percentile(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr


def estimate_rigid_from_images(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    angle_min: float = -45.0,
    angle_max: float = 45.0,
    coarse_step: float = 2.0,
    refine_steps: Optional[Sequence[float]] = (0.5, 0.1),
    downsample_to: int = 512,
    upsample_factor: int = 10,
    order: int = 1,
) -> Tuple[RigidTransformTurbo, Dict[str, Any]]:
    """Estimate rigid transform using Fourier–polar angle + PCC translation."""

    if fixed.ndim != 2 or moving.ndim != 2:
        raise ValueError("estimate_rigid_from_images expects 2D inputs")
    if sktf is None or phase_cross_correlation is None:
        raise RuntimeError("scikit-image (transform+registration) is required for estimation")

    fixed_ds = _normalize_percentile(_downsample(fixed, downsample_to))
    moving_ds = _normalize_percentile(_downsample(moving, downsample_to))
    H, W = fixed_ds.shape
    win = _hann2d(H, W)
    Fw = np.abs(np.fft.fft2(fixed_ds * win))
    Mw = np.abs(np.fft.fft2(moving_ds * win))
    Flog = np.log1p(Fw)
    Mlog = np.log1p(Mw)
    angles = 720  # 0.5°
    radius = min(H, W) // 2
    P1 = sktf.warp_polar(Flog, radius=radius, output_shape=(radius, angles), scaling="linear")
    P2 = sktf.warp_polar(Mlog, radius=radius, output_shape=(radius, angles), scaling="linear")
    shift, err_polar, _ = phase_cross_correlation(P1, P2, upsample_factor=upsample_factor)
    theta_guessed = shift[1] * (360.0 / angles)

    def _eval(theta: float) -> Tuple[float, float, float]:
        if ndi is None:
            rotated = moving_ds
        else:
            rotated = ndi.rotate(moving_ds, angle=theta, reshape=False, order=order, mode="constant", cval=0.0)
        delta, pcc_err, _ = phase_cross_correlation(fixed_ds, rotated, upsample_factor=upsample_factor)
        dy, dx = float(delta[0]), float(delta[1])
        shifted = ndi.shift(rotated, shift=(dy, dx), order=1, mode="constant", cval=0.0) if ndi is not None else rotated
        mse = float(np.mean((fixed_ds - shifted) ** 2))
        return mse, dy, dx

    best = {"theta": theta_guessed, "dy": 0.0, "dx": 0.0, "error": float("inf")}
    lo = max(angle_min, theta_guessed - max(10.0, 3 * coarse_step))
    hi = min(angle_max, theta_guessed + max(10.0, 3 * coarse_step))
    angles_to_try = np.arange(lo, hi + 1e-9, coarse_step, dtype=float)
    for th in angles_to_try:
        err, dy, dx = _eval(th)
        if err < best["error"]:
            best.update({"theta": th, "dy": dy, "dx": dx, "error": err})
    if refine_steps:
        for step in refine_steps:
            th0 = best["theta"]
            for th in np.arange(th0 - 3 * step, th0 + 3 * step + 1e-9, step, dtype=float):
                err, dy, dx = _eval(th)
                if err < best["error"]:
                    best.update({"theta": th, "dy": dy, "dx": dx, "error": err})

    theta_math = math.radians(best["theta"])
    theta = -theta_math  # convert to TurboReg (top-left origin) orientation
    c = math.cos(theta)
    s = math.sin(theta)
    # convert shifts (dy,dx) into TurboReg translation at full resolution
    Hy, Wx = fixed.shape
    cx = (Wx - 1) * 0.5
    cy = (Hy - 1) * 0.5
    tx = cx - c * cx + s * cy + best["dx"]
    ty = cy - s * cx - c * cy + best["dy"]
    rt = RigidTransformTurbo(c, -s, tx, s, c, ty)

    info = {
        "theta_deg": math.degrees(theta),
        "theta_math_deg": float(best["theta"]),
        "dy_px": float(best["dy"]),
        "dx_px": float(best["dx"]),
        "ds_shape": [int(H), int(W)],
        "downsample_to": int(downsample_to),
        "coarse_step": float(coarse_step),
        "refine_steps": [float(step) for step in (refine_steps or [])],
        "angle_bounds": [float(angle_min), float(angle_max)],
        "polar_error": float(err_polar),
    }
    return rt, info


def save_transform_json(path: str | Path, rt: RigidTransformTurbo, meta: Optional[Dict[str, Any]] = None) -> None:
    out = {"transform": rt.to_jsonable()}
    if meta:
        out["meta"] = meta
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = [
    "AlignmentError",
    "RigidTransformTurbo",
    "LandmarkBlock",
    "anchor_points",
    "intensity_principal_landmarks",
    "parse_msreg_file",
    "rigid_from_points_turbo",
    "rigid_from_points_ls",
    "transform_from_landmarks",
    "compose_transforms",
    "invert_transform",
    "rms_error",
    "apply_transform",
    "estimate_rigid_from_images",
    "save_transform_json",
]
