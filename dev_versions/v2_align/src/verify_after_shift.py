"""
Correlation sanity-check before/after applying a detected translation.

Usage:
    python -m src.verify_after_shift --fixed ... --moving ...
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.registration import phase_cross_correlation

from .io_utils import load_multichannel_tiff, extract_channel


def compute_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def roll_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    rolled = np.roll(img, int(round(dy)), axis=0)
    rolled = np.roll(rolled, int(round(dx)), axis=1)
    return rolled


def main(fixed_path: str, moving_path: str, overlay_prefix: str = "correlation_before_after") -> None:
    fixed = load_multichannel_tiff(fixed_path, 2)
    moving = load_multichannel_tiff(moving_path, 2)

    fixed_dapi = extract_channel(fixed, 1)
    moving_dapi = extract_channel(moving, 1)

    print("=" * 70)
    print("CORRELATION TEST: Before and After Shift")
    print("=" * 70)

    corr_before = compute_corr(fixed_dapi, moving_dapi)
    print(f"\nCorrelation BEFORE alignment: {corr_before:.4f} ← Low!")

    shift, error, diffphase = phase_cross_correlation(fixed_dapi, moving_dapi, upsample_factor=10)
    print(f"\nDetected shift: dy={shift[0]:.2f}, dx={shift[1]:.2f}")

    moving_shifted = roll_image(moving_dapi, shift[0], shift[1])
    corr_after = compute_corr(fixed_dapi, moving_shifted)
    print(f"Correlation AFTER alignment:  {corr_after:.4f}")

    print("\n" + "=" * 70)
    if corr_after > 0.7:
        print("✓ HIGH CORRELATION! Images match well after alignment.")
    elif corr_after > 0.3:
        print("⚠ MODERATE CORRELATION. Images are similar but not identical.")
    else:
        print("✗ LOW CORRELATION even after alignment. Images may differ fundamentally.")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(fixed_dapi, cmap='gray')
    axes[0].set_title('Fixed DAPI')
    axes[1].imshow(moving_dapi, cmap='gray')
    axes[1].set_title(f'Moving DAPI (before)\nCorr: {corr_before:.4f}')
    axes[2].imshow(moving_shifted, cmap='gray')
    axes[2].set_title(f'Moving DAPI (after shift)\nCorr: {corr_after:.4f}')
    plt.tight_layout()
    plot_path = f"{overlay_prefix}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")

    fixed_norm = cv2.normalize(fixed_dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    moving_norm = cv2.normalize(moving_dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    shifted_norm = cv2.normalize(moving_shifted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    overlay_before = np.zeros((*fixed_dapi.shape, 3), dtype=np.uint8)
    overlay_after = np.zeros_like(overlay_before)
    overlay_before[:, :, 1] = fixed_norm
    overlay_before[:, :, 0] = moving_norm
    overlay_after[:, :, 1] = fixed_norm
    overlay_after[:, :, 0] = shifted_norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(overlay_before)
    axes[0].set_title(f'Before shift (Corr={corr_before:.4f})')
    axes[1].imshow(overlay_after)
    axes[1].set_title(f'After shift (Corr={corr_after:.4f})')
    plt.tight_layout()
    overlay_path = f"{overlay_prefix}_overlay.png"
    plt.savefig(overlay_path, dpi=150)
    print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation before/after phase shift.")
    parser.add_argument("--fixed", required=True, help="Path to fixed/reference TIFF")
    parser.add_argument("--moving", required=True, help="Path to moving TIFF")
    parser.add_argument(
        "--prefix",
        default="correlation_before_after",
        help="Output prefix for figures"
    )
    args = parser.parse_args()
    main(args.fixed, args.moving, args.prefix)

