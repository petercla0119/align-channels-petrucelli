#!/usr/bin/env python3
"""
Comprehensive nuclei-based channel alignment script.

This script:
1. Uses phase cross-correlation on nuclei (Channel 0) to compute alignment shift
2. Applies that shift to both Channel 0 and Channel 1
3. Generates comprehensive QC images (before and after)
4. Saves aligned images and alignment log

Usage:
    python align_using_nuclei.py fixed.tiff moving.tiff output_dir

Or use as a module:
    from align_using_nuclei import align_pair_using_nuclei
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import csv

import numpy as np
from tifffile import imread, imwrite
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def detect_channel_axis(arr: np.ndarray) -> int:
    """Detect which axis is the channel axis (prefer size 2-8)."""
    if arr.ndim < 3:
        raise ValueError(f"Expected at least 3D array, got shape {arr.shape}")
    for axis, size in enumerate(arr.shape):
        if 2 <= size <= 8:
            return axis
    if arr.shape[-1] in (3, 4):
        return arr.ndim - 1
    return 0


def center_crop(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Center crop image to target size."""
    y0 = (img.shape[0] - H) // 2
    x0 = (img.shape[1] - W) // 2
    return img[y0:y0+H, x0:x0+W]


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize image using 1-99 percentile for display."""
    arr = np.asarray(img, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=float)
    p1, p99 = np.percentile(arr, (1, 99))
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return np.zeros_like(arr, dtype=float)
    norm = (arr - p1) / (p99 - p1)
    return np.clip(norm, 0.0, 1.0)


def make_overlay(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Create red-blue overlay (image1=red, image2=blue)."""
    norm1 = normalize_for_display(image1)
    norm2 = normalize_for_display(image2)
    zeros = np.zeros_like(norm1)
    return np.stack([norm1, zeros, norm2], axis=-1)


def compute_nuclei_alignment(
    fixed_nuclei: np.ndarray,
    moving_nuclei: np.ndarray,
    upsample_factor: int = 100,
    try_center_region: bool = True,
    verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Compute alignment shift using phase cross-correlation on nuclei channel.
    
    Returns:
        (dy, dx, error): Shift in y, shift in x, and registration error
    """
    # Convert to float for numerical stability
    fixed_float = fixed_nuclei.astype(np.float64)
    moving_float = moving_nuclei.astype(np.float64)
    
    if verbose:
        print(f"  Computing phase cross-correlation (upsample={upsample_factor})...")
    
    # Try full image first
    shift, error, _ = phase_cross_correlation(
        fixed_float,
        moving_float,
        upsample_factor=upsample_factor
    )
    
    dy, dx = float(shift[0]), float(shift[1])
    
    if verbose:
        print(f"    Full image: dy={dy:.3f}, dx={dx:.3f}, error={error:.6f}")
    
    # If error is high, try center region
    if error > 0.5 and try_center_region:
        center_size = min(2048, fixed_float.shape[0] // 2, fixed_float.shape[1] // 2)
        if center_size >= 512:
            if verbose:
                print(f"  ⚠️  High error, trying center {center_size}x{center_size} region...")
            
            y_start = (fixed_float.shape[0] - center_size) // 2
            x_start = (fixed_float.shape[1] - center_size) // 2
            
            fixed_center = fixed_float[y_start:y_start+center_size, x_start:x_start+center_size]
            moving_center = moving_float[y_start:y_start+center_size, x_start:x_start+center_size]
            
            shift2, error2, _ = phase_cross_correlation(
                fixed_center,
                moving_center,
                upsample_factor=upsample_factor
            )
            
            if verbose:
                print(f"    Center region: dy={shift2[0]:.3f}, dx={shift2[1]:.3f}, error={error2:.6f}")
            
            if error2 < error:
                if verbose:
                    print("    ✓ Using center region result (better error)")
                dy, dx = float(shift2[0]), float(shift2[1])
                error = error2
    
    return dy, dx, error


def align_pair_using_nuclei(
    fixed_path: Path,
    moving_path: Path,
    output_dir: Path,
    upsample_factor: int = 100,
    interpolation_order: int = 3,
    crop_to_overlap: bool = True,
    generate_qc: bool = True,
    verbose: bool = True
) -> dict:
    """
    Align a pair of multi-channel images using nuclei (Channel 0) as reference.
    
    Args:
        fixed_path: Path to fixed (reference) image
        moving_path: Path to moving image to align
        output_dir: Directory for outputs
        upsample_factor: Subpixel precision (10=0.1px, 100=0.01px)
        interpolation_order: 0=nearest, 1=linear, 3=cubic
        crop_to_overlap: Whether to crop to common overlap after alignment
        generate_qc: Whether to generate QC images
        verbose: Print progress
    
    Returns:
        dict with alignment statistics and output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("Nuclei-Based Multi-Channel Alignment")
        print("=" * 70)
        print(f"Fixed:  {fixed_path.name}")
        print(f"Moving: {moving_path.name}")
        print(f"Output: {output_dir}")
        print()
    
    # Load images
    arr_fixed = imread(fixed_path)
    arr_moving = imread(moving_path)
    
    if verbose:
        print(f"Loaded images:")
        print(f"  Fixed:  {arr_fixed.shape}, {arr_fixed.dtype}")
        print(f"  Moving: {arr_moving.shape}, {arr_moving.dtype}")
    
    # Detect and move channel axis to front
    ch_axis_fixed = detect_channel_axis(arr_fixed)
    ch_axis_moving = detect_channel_axis(arr_moving)
    
    arr_fixed = np.moveaxis(arr_fixed, ch_axis_fixed, 0)
    arr_moving = np.moveaxis(arr_moving, ch_axis_moving, 0)
    
    if arr_fixed.shape[0] < 2 or arr_moving.shape[0] < 2:
        raise ValueError(f"Expected at least 2 channels, got {arr_fixed.shape[0]} and {arr_moving.shape[0]}")
    
    # Extract channels
    fixed_ch0 = arr_fixed[0]  # Nuclei
    fixed_ch1 = arr_fixed[1]  # Protein marker
    moving_ch0 = arr_moving[0]  # Nuclei
    moving_ch1 = arr_moving[1]  # Protein marker
    
    # Crop to common size
    min_h = min(fixed_ch0.shape[0], moving_ch0.shape[0])
    min_w = min(fixed_ch0.shape[1], moving_ch0.shape[1])
    
    fixed_ch0_c = center_crop(fixed_ch0, min_h, min_w)
    moving_ch0_c = center_crop(moving_ch0, min_h, min_w)
    fixed_ch1_c = center_crop(fixed_ch1, min_h, min_w)
    moving_ch1_c = center_crop(moving_ch1, min_h, min_w)
    
    if verbose:
        print(f"\nCropped to common size: {fixed_ch0_c.shape}")
    
    # Compute alignment using NUCLEI (Channel 0)
    if verbose:
        print("\nComputing alignment from nuclei (Channel 0):")
    
    dy, dx, error = compute_nuclei_alignment(
        fixed_ch0_c,
        moving_ch0_c,
        upsample_factor=upsample_factor,
        verbose=verbose
    )
    
    distance = np.hypot(dy, dx)
    
    if verbose:
        print(f"\nFinal shift: dy={dy:.3f}, dx={dx:.3f}")
        print(f"  Distance: {distance:.3f} pixels")
        print(f"  Error: {error:.6f}")
        
        if error > 0.5:
            print("\n  ⚠️  WARNING: High error suggests poor match!")
            print("     Images may not correspond well or alignment failed.")
        elif distance < 1.0:
            print("\n  ✓ Very small shift - images may already be well-aligned!")
        else:
            print(f"\n  ✓ Shift computed successfully")
    
    # Apply shift to BOTH channels
    if verbose:
        print(f"\nApplying shift to both channels (order={interpolation_order})...")
    
    aligned_ch0 = nd_shift(moving_ch0_c, shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)
    aligned_ch1 = nd_shift(moving_ch1_c, shift=(dy, dx), order=interpolation_order, mode='constant', cval=0.0)
    
    # Save uncropped versions for QC
    fixed_ch0_qc = fixed_ch0_c.copy()
    fixed_ch1_qc = fixed_ch1_c.copy()
    moving_ch0_qc = moving_ch0_c.copy()
    moving_ch1_qc = moving_ch1_c.copy()
    aligned_ch0_qc = aligned_ch0.copy()
    aligned_ch1_qc = aligned_ch1.copy()
    
    # Optionally crop to overlap
    if crop_to_overlap and distance > 0.5:
        # Simple crop - remove edges affected by shift
        pad_y = int(np.ceil(abs(dy)))
        pad_x = int(np.ceil(abs(dx)))
        
        if pad_y > 0 or pad_x > 0:
            y0, y1 = pad_y, fixed_ch0_c.shape[0] - pad_y
            x0, x1 = pad_x, fixed_ch0_c.shape[1] - pad_x
            
            fixed_ch0_c = fixed_ch0_c[y0:y1, x0:x1]
            fixed_ch1_c = fixed_ch1_c[y0:y1, x0:x1]
            aligned_ch0 = aligned_ch0[y0:y1, x0:x1]
            aligned_ch1 = aligned_ch1[y0:y1, x0:x1]
            
            if verbose:
                print(f"  Cropped to overlap: {fixed_ch0_c.shape}")
    
    # Save aligned images
    sample_name = fixed_path.stem
    
    aligned_stack = np.stack([aligned_ch0, aligned_ch1], axis=0)
    stack_path = output_dir / f"{sample_name}_aligned.tif"
    imwrite(stack_path, aligned_stack, metadata={'axes': 'CYX'})
    
    # Save individual channels
    imwrite(output_dir / f"{sample_name}_fixed_ch0.tif", fixed_ch0_c)
    imwrite(output_dir / f"{sample_name}_fixed_ch1.tif", fixed_ch1_c)
    imwrite(output_dir / f"{sample_name}_aligned_ch0.tif", aligned_ch0)
    imwrite(output_dir / f"{sample_name}_aligned_ch1.tif", aligned_ch1)
    
    if verbose:
        print(f"\n✅ Saved aligned images:")
        print(f"   {stack_path.name}")
    
    # Generate QC images
    qc_paths = {}
    if generate_qc:
        if verbose:
            print("\nGenerating QC images...")
        
        # Raw comparison (before alignment) - use uncropped versions
        raw_qc_path = output_dir / f"{sample_name}_raw_comparison.png"
        _generate_raw_qc(
            fixed_ch0_qc, moving_ch0_qc, fixed_ch1_qc, moving_ch1_qc,
            fixed_path.stem, moving_path.stem, raw_qc_path
        )
        qc_paths['raw'] = raw_qc_path
        
        # Aligned comparison (after alignment) - use uncropped versions
        aligned_qc_path = output_dir / f"{sample_name}_aligned_qc.png"
        _generate_aligned_qc(
            fixed_ch0_qc, aligned_ch0_qc, fixed_ch1_qc, aligned_ch1_qc,
            dy, dx, distance, error,
            fixed_path.stem, moving_path.stem, aligned_qc_path
        )
        qc_paths['aligned'] = aligned_qc_path
        
        if verbose:
            print(f"   {raw_qc_path.name}")
            print(f"   {aligned_qc_path.name}")
    
    # Save alignment log
    log_path = output_dir / f"{sample_name}_alignment_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fixed_file', 'moving_file', 'dy_px', 'dx_px', 'distance_px', 
                        'error', 'upsample_factor', 'interp_order', 'final_shape'])
        writer.writerow([fixed_path.name, moving_path.name, dy, dx, distance, 
                        error, upsample_factor, interpolation_order, f"{fixed_ch0_c.shape}"])
    
    if verbose:
        print(f"   {log_path.name}")
        print("\n" + "=" * 70)
        print("✅ Complete!")
        print("=" * 70)
    
    return {
        'dy': dy,
        'dx': dx,
        'distance': distance,
        'error': error,
        'aligned_stack': stack_path,
        'qc_paths': qc_paths,
        'log_path': log_path
    }


def _generate_raw_qc(fixed_ch0, moving_ch0, fixed_ch1, moving_ch1, 
                     fixed_label, moving_label, output_path):
    """Generate before-alignment QC image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 0: Channel 0 (Nuclei)
    axes[0, 0].imshow(normalize_for_display(fixed_ch0), cmap='gray')
    axes[0, 0].set_title(f'{fixed_label}\nChannel 0 (Nuclei)', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_for_display(moving_ch0), cmap='gray')
    axes[0, 1].set_title(f'{moving_label}\nChannel 0 (Nuclei)', fontsize=10)
    axes[0, 1].axis('off')
    
    overlay_ch0 = make_overlay(moving_ch0, fixed_ch0)
    axes[0, 2].imshow(overlay_ch0)
    axes[0, 2].set_title('Ch0 Overlay (No alignment)\nRed=Moving, Blue=Fixed', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 1: Channel 1 (Protein)
    axes[1, 0].imshow(normalize_for_display(fixed_ch1), cmap='gray')
    axes[1, 0].set_title(f'{fixed_label}\nChannel 1 (Protein)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalize_for_display(moving_ch1), cmap='gray')
    axes[1, 1].set_title(f'{moving_label}\nChannel 1 (Protein)', fontsize=10)
    axes[1, 1].axis('off')
    
    overlay_ch1 = make_overlay(moving_ch1, fixed_ch1)
    axes[1, 2].imshow(overlay_ch1)
    axes[1, 2].set_title('Ch1 Overlay (No alignment)\nRed=Moving, Blue=Fixed', fontsize=10)
    axes[1, 2].axis('off')
    
    fig.suptitle('Raw Images - Before Alignment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_aligned_qc(fixed_ch0, aligned_ch0, fixed_ch1, aligned_ch1,
                         dy, dx, distance, error, fixed_label, moving_label, output_path):
    """Generate after-alignment QC image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 0: Channel 0 (Nuclei)
    axes[0, 0].imshow(normalize_for_display(fixed_ch0), cmap='gray')
    axes[0, 0].set_title(f'{fixed_label}\nChannel 0 (Nuclei)', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_for_display(aligned_ch0), cmap='gray')
    axes[0, 1].set_title(f'{moving_label} (Aligned)\nChannel 0 (Nuclei)', fontsize=10)
    axes[0, 1].axis('off')
    
    overlay_ch0 = make_overlay(aligned_ch0, fixed_ch0)
    axes[0, 2].imshow(overlay_ch0)
    axes[0, 2].set_title('Ch0 Overlay (Aligned)\nRed=Aligned, Blue=Fixed', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 1: Channel 1 (Protein)
    axes[1, 0].imshow(normalize_for_display(fixed_ch1), cmap='gray')
    axes[1, 0].set_title(f'{fixed_label}\nChannel 1 (Protein)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalize_for_display(aligned_ch1), cmap='gray')
    axes[1, 1].set_title(f'{moving_label} (Aligned)\nChannel 1 (Protein)', fontsize=10)
    axes[1, 1].axis('off')
    
    overlay_ch1 = make_overlay(aligned_ch1, fixed_ch1)
    axes[1, 2].imshow(overlay_ch1)
    axes[1, 2].set_title('Ch1 Overlay (Aligned)\nRed=Aligned, Blue=Fixed', fontsize=10)
    axes[1, 2].axis('off')
    
    fig.suptitle(f'After Alignment using Nuclei (Ch0)\nShift: dy={dy:.2f}, dx={dx:.2f} | Distance: {distance:.2f} px | Error: {error:.4f}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Command-line interface."""
    if len(sys.argv) < 4:
        print("Usage: python align_using_nuclei.py <fixed.tiff> <moving.tiff> <output_dir>")
        print("\nExample:")
        print("  python align_using_nuclei.py 23/23.1_Stack01-Ctx-MIP.tiff 23-2/23.1_Stack01-TMEM-Ctx-MIP.tiff aligned_output")
        sys.exit(1)
    
    fixed_path = Path(sys.argv[1])
    moving_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    if not fixed_path.exists():
        print(f"Error: Fixed file not found: {fixed_path}")
        sys.exit(1)
    
    if not moving_path.exists():
        print(f"Error: Moving file not found: {moving_path}")
        sys.exit(1)
    
    result = align_pair_using_nuclei(
        fixed_path=fixed_path,
        moving_path=moving_path,
        output_dir=output_dir,
        upsample_factor=100,
        interpolation_order=3,
        crop_to_overlap=True,
        generate_qc=True,
        verbose=True
    )
    
    print(f"\nAlignment summary:")
    print(f"  Shift: dy={result['dy']:.3f}, dx={result['dx']:.3f}")
    print(f"  Distance: {result['distance']:.3f} pixels")
    print(f"  Error: {result['error']:.6f}")


if __name__ == "__main__":
    main()

