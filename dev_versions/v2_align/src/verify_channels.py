"""
Quick channel sanity-check utility.

Usage:
    python -m src.verify_channels --fixed <path> --moving <path> [--output channel_verification.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .io_utils import load_multichannel_tiff, extract_channel


def compute_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def main(fixed_path: str, moving_path: str, output_path: str) -> None:
    fixed = load_multichannel_tiff(fixed_path, expected_channels=2)
    moving = load_multichannel_tiff(moving_path, expected_channels=2)

    fixed_ch0 = extract_channel(fixed, 0)
    fixed_ch1 = extract_channel(fixed, 1)
    moving_ch0 = extract_channel(moving, 0)
    moving_ch1 = extract_channel(moving, 1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(fixed_ch0, cmap='gray')
    axes[0, 0].set_title('Fixed - Channel 0', fontsize=10)
    axes[0, 1].imshow(fixed_ch1, cmap='gray')
    axes[0, 1].set_title('Fixed - Channel 1', fontsize=10)
    axes[0, 2].imshow(moving_ch0, cmap='gray')
    axes[0, 2].set_title('Moving - Channel 0', fontsize=10)
    axes[0, 3].imshow(moving_ch1, cmap='gray')
    axes[0, 3].set_title('Moving - Channel 1', fontsize=10)

    axes[1, 0].hist(fixed_ch0.flatten(), bins=100, alpha=0.7, color='steelblue')
    axes[1, 0].set_title('Fixed Ch0 histogram', fontsize=10)
    axes[1, 1].hist(fixed_ch1.flatten(), bins=100, alpha=0.7, color='seagreen')
    axes[1, 1].set_title('Fixed Ch1 histogram', fontsize=10)
    axes[1, 2].hist(moving_ch0.flatten(), bins=100, alpha=0.7, color='darkorange')
    axes[1, 2].set_title('Moving Ch0 histogram', fontsize=10)
    axes[1, 3].hist(moving_ch1.flatten(), bins=100, alpha=0.7, color='orchid')
    axes[1, 3].set_title('Moving Ch1 histogram', fontsize=10)

    for ax in axes.flat:
        ax.axis('off') if ax in axes[0] else ax.set_xlabel('Intensity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

    corr_ch0_ch0 = compute_corr(fixed_ch0, moving_ch0)
    corr_ch0_ch1 = compute_corr(fixed_ch0, moving_ch1)
    corr_ch1_ch0 = compute_corr(fixed_ch1, moving_ch0)
    corr_ch1_ch1 = compute_corr(fixed_ch1, moving_ch1)

    print("\n" + "=" * 70)
    print("CHANNEL CORRELATION MATRIX")
    print("=" * 70)
    print(f"Fixed Ch0 vs Moving Ch0: {corr_ch0_ch0:.4f}")
    print(f"Fixed Ch0 vs Moving Ch1: {corr_ch0_ch1:.4f}")
    print(f"Fixed Ch1 vs Moving Ch0: {corr_ch1_ch0:.4f}")
    print(f"Fixed Ch1 vs Moving Ch1: {corr_ch1_ch1:.4f} ← Expected highest for DAPI")
    print("=" * 70)

    stats = {
        "Fixed Ch0": fixed_ch0,
        "Fixed Ch1": fixed_ch1,
        "Moving Ch0": moving_ch0,
        "Moving Ch1": moving_ch1,
    }
    print("\nChannel Statistics:")
    for name, arr in stats.items():
        print(f"{name:>11}: min={arr.min():8.1f}, max={arr.max():8.1f}, mean={arr.mean():8.1f}")
    print("=" * 70)

    correlations = {
        'Ch0-Ch0': corr_ch0_ch0,
        'Ch0-Ch1': corr_ch0_ch1,
        'Ch1-Ch0': corr_ch1_ch0,
        'Ch1-Ch1': corr_ch1_ch1
    }
    best = max(correlations, key=correlations.get)
    print(f"\nBest channel pairing: {best} with correlation {correlations[best]:.4f}")
    if best != 'Ch1-Ch1':
        print("⚠️  WARNING: Channel 1 vs Channel 1 is NOT the best match!")
        print("   Your channel assignments might be swapped or DAPI is not consistently stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Channel verification utility")
    parser.add_argument("--fixed", required=True, help="Path to fixed/reference 2-channel TIFF")
    parser.add_argument("--moving", required=True, help="Path to moving 2-channel TIFF")
    parser.add_argument(
        "--output",
        default="channel_verification.png",
        help="Output path for the montage/plots (PNG)."
    )
    args = parser.parse_args()
    main(args.fixed, args.moving, args.output)

