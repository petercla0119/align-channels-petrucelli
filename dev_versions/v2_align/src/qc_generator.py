"""
QC visualization generators for alignment quality assessment.

Creates overlay images and RGB composites for visual inspection.
"""

import numpy as np
from typing import Tuple, Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def normalize_to_8bit(
    image: np.ndarray,
    percentile_clip: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize image to 8-bit range [0, 255].
    
    Args:
        image: Input image (any dtype)
        percentile_clip: If provided, clip to (low, high) percentiles before normalizing
        
    Returns:
        uint8 image normalized to [0, 255]
    """
    img = image.astype(float)
    
    # Optional percentile clipping for better contrast
    if percentile_clip is not None:
        low, high = percentile_clip
        vmin, vmax = np.percentile(img, [low, high])
        img = np.clip(img, vmin, vmax)
    else:
        vmin, vmax = img.min(), img.max()
    
    # Normalize
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin) * 255.0
    else:
        img = np.zeros_like(img)
    
    return img.astype(np.uint8)


def create_overlay(
    img1: np.ndarray,
    img2: np.ndarray,
    color1: str = 'cyan',
    color2: str = 'magenta',
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    mask_zero_regions: bool = True
) -> np.ndarray:
    """
    Create a two-color overlay for comparing two grayscale images.
    
    Args:
        img1: First image (H, W), will be colored with color1
        img2: Second image (H, W), will be colored with color2
        color1: Color for img1 ('cyan', 'green', 'red', 'blue')
        color2: Color for img2 ('magenta', 'red', 'green', 'blue')
        percentile_clip: Percentile range for contrast adjustment
        mask_zero_regions: If True, regions where img2 is zero (cropped) will be black, not showing img1
        
    Returns:
        RGB overlay image, shape (H, W, 3), dtype uint8
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
    
    h, w = img1.shape
    
    # Identify zero/cropped regions in img2 (moving image after alignment)
    # Use a small threshold to account for numerical errors
    zero_mask = img2 < 1  # Pixels that are essentially zero
    
    # Normalize both images
    img1_norm = normalize_to_8bit(img1, percentile_clip)
    img2_norm = normalize_to_8bit(img2, percentile_clip)
    
    # Color mappings
    colors = {
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'cyan': (0, 1, 1),
        'magenta': (1, 0, 1),
        'yellow': (1, 1, 0),
        'white': (1, 1, 1)
    }
    
    if color1 not in colors:
        raise ValueError(f"Unknown color: {color1}")
    if color2 not in colors:
        raise ValueError(f"Unknown color: {color2}")
    
    # Create RGB channels
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors
    r1, g1, b1 = colors[color1]
    r2, g2, b2 = colors[color2]
    
    # Blend: add both images with their colors
    rgb[:, :, 0] = np.clip(img1_norm * r1 + img2_norm * r2, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(img1_norm * g1 + img2_norm * g2, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(img1_norm * b1 + img2_norm * b2, 0, 255).astype(np.uint8)
    
    # Mask out cropped regions (where moving image is zero after transformation)
    if mask_zero_regions:
        rgb[zero_mask] = 0  # Set to black where img2 is zero (cropped edges)
    
    logger.info(f"Created overlay: {color1} + {color2}")
    return rgb


def create_rgb_composite(
    dapi: np.ndarray,
    protein_fixed: np.ndarray,
    protein_moving: np.ndarray,
    dapi_color: str = 'blue',
    fixed_color: str = 'green',
    moving_color: str = 'magenta',
    percentile_clip: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Create RGB composite with three channels.
    
    Args:
        dapi: DAPI channel (H, W)
        protein_fixed: Fixed protein channel (H, W)
        protein_moving: Moving (aligned) protein channel (H, W)
        dapi_color: Color for DAPI (default 'blue')
        fixed_color: Color for fixed protein (default 'green')
        moving_color: Color for moving protein (default 'magenta')
        percentile_clip: Percentile range for contrast adjustment
        
    Returns:
        RGB composite image, shape (H, W, 3), dtype uint8
    """
    if not (dapi.shape == protein_fixed.shape == protein_moving.shape):
        raise ValueError("All images must have same shape")
    
    h, w = dapi.shape
    
    # Normalize all channels
    dapi_norm = normalize_to_8bit(dapi, percentile_clip)
    fixed_norm = normalize_to_8bit(protein_fixed, percentile_clip)
    moving_norm = normalize_to_8bit(protein_moving, percentile_clip)
    
    # Color mappings
    colors = {
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'cyan': (0, 1, 1),
        'magenta': (1, 0, 1),
        'yellow': (1, 1, 0)
    }
    
    # Create RGB
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add each channel with its color
    for img_norm, color_name in [
        (dapi_norm, dapi_color),
        (fixed_norm, fixed_color),
        (moving_norm, moving_color)
    ]:
        if color_name in colors:
            r, g, b = colors[color_name]
            rgb[:, :, 0] = np.clip(rgb[:, :, 0] + img_norm * r, 0, 255).astype(np.uint8)
            rgb[:, :, 1] = np.clip(rgb[:, :, 1] + img_norm * g, 0, 255).astype(np.uint8)
            rgb[:, :, 2] = np.clip(rgb[:, :, 2] + img_norm * b, 0, 255).astype(np.uint8)
    
    logger.info(f"Created RGB composite: DAPI={dapi_color}, fixed={fixed_color}, moving={moving_color}")
    return rgb


def save_rgb_png(
    image: np.ndarray,
    path: str,
    downsample_factor: int = 2,
    quality: int = 85
) -> None:
    """
    Save RGB image as PNG with optional downsampling to reduce file size.
    
    Args:
        image: RGB image, shape (H, W, 3), dtype uint8
        path: Output path
        downsample_factor: Factor to downsample by (2 = half resolution, 1 = no downsampling)
        quality: JPEG quality if saving as JPEG (not used for PNG, but kept for compatibility)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape: {image.shape}")
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Create PIL image
    pil_img = Image.fromarray(image, mode='RGB')
    
    # Downsample if requested
    if downsample_factor > 1:
        h, w = image.shape[:2]
        new_size = (w // downsample_factor, h // downsample_factor)
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Downsampled from {(w, h)} to {new_size} ({downsample_factor}x)")
    
    # Save with compression
    pil_img.save(path, optimize=True)
    
    logger.info(f"Saved RGB PNG: {path}")


def create_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    labels: Optional[Tuple[str, str]] = None,
    border_width: int = 2
) -> np.ndarray:
    """
    Create side-by-side comparison of two images.
    
    Args:
        img1: First image (H, W) or (H, W, 3)
        img2: Second image (H, W) or (H, W, 3)
        labels: Optional (label1, label2) text labels
        border_width: Width of border between images
        
    Returns:
        Side-by-side image
    """
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    
    # Ensure same height
    h = max(img1.shape[0], img2.shape[0])
    if img1.shape[0] < h:
        pad = h - img1.shape[0]
        img1 = np.pad(img1, ((0, pad), (0, 0), (0, 0)), mode='constant')
    if img2.shape[0] < h:
        pad = h - img2.shape[0]
        img2 = np.pad(img2, ((0, pad), (0, 0), (0, 0)), mode='constant')
    
    # Create border
    border = np.ones((h, border_width, 3), dtype=img1.dtype) * 255
    
    # Concatenate
    combined = np.concatenate([img1, border, img2], axis=1)
    
    logger.info(f"Created side-by-side comparison: {combined.shape}")
    return combined


def create_checkerboard(
    img1: np.ndarray,
    img2: np.ndarray,
    square_size: int = 64
) -> np.ndarray:
    """
    Create checkerboard overlay for alignment inspection.
    
    Args:
        img1: First image (H, W)
        img2: Second image (H, W)
        square_size: Size of checkerboard squares in pixels
        
    Returns:
        Checkerboard composite image
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have same shape")
    
    h, w = img1.shape
    
    # Normalize both
    img1_norm = normalize_to_8bit(img1)
    img2_norm = normalize_to_8bit(img2)
    
    # Create checkerboard mask
    mask = np.zeros((h, w), dtype=bool)
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            # Alternate pattern
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                mask[i:i+square_size, j:j+square_size] = True
    
    # Apply mask
    result = np.where(mask, img1_norm, img2_norm)
    
    logger.info(f"Created checkerboard: square_size={square_size}")
    return result.astype(np.uint8)


def create_difference_map(
    img1: np.ndarray,
    img2: np.ndarray,
    colormap: str = 'coolwarm'
) -> np.ndarray:
    """
    Create difference map between two images.
    
    Args:
        img1: First image (H, W)
        img2: Second image (H, W)
        colormap: Colormap name (currently only 'coolwarm' supported)
        
    Returns:
        RGB difference map, shape (H, W, 3)
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have same shape")
    
    # Normalize both to [0, 1]
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-10)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-10)
    
    # Compute difference
    diff = img1_norm - img2_norm  # Range: [-1, 1]
    
    # Map to RGB (simple coolwarm: blue=negative, white=0, red=positive)
    h, w = diff.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Positive differences (img1 > img2): red
    pos_mask = diff > 0
    rgb[pos_mask, 0] = (diff[pos_mask] * 255).astype(np.uint8)
    
    # Negative differences (img1 < img2): blue
    neg_mask = diff < 0
    rgb[neg_mask, 2] = (-diff[neg_mask] * 255).astype(np.uint8)
    
    # Near zero: white
    zero_mask = np.abs(diff) < 0.05
    rgb[zero_mask] = 255
    
    logger.info("Created difference map")
    return rgb


def create_qc_grid(
    fixed_ch0: np.ndarray,
    fixed_ch1: np.ndarray,
    moving_ch0_before: np.ndarray,
    moving_ch1_before: np.ndarray,
    moving_ch0_after: np.ndarray,
    moving_ch1_after: np.ndarray,
    percentile_clip: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Create 2×3 QC grid showing alignment quality for both channels.
    
    Layout:
        Row 0 (Channel 1 - DAPI):
            [Fixed Ch1] [Moving Ch1 Before] [Overlay Before: Fixed(cyan) + Moving(magenta)]
        Row 1 (Channel 0 - Protein):
            [Fixed Ch0] [Moving Ch0 Before] [Overlay Before: Fixed(green) + Moving(red)]
    
    Args:
        fixed_ch0: Fixed image channel 0 (protein)
        fixed_ch1: Fixed image channel 1 (DAPI)
        moving_ch0_before: Moving image channel 0 before alignment
        moving_ch1_before: Moving image channel 1 before alignment
        moving_ch0_after: Moving image channel 0 after alignment
        moving_ch1_after: Moving image channel 1 after alignment
        percentile_clip: Percentile range for contrast adjustment
        
    Returns:
        RGB grid image with 2 rows × 3 columns
    """
    # Normalize all images to 8-bit
    fixed_ch1_norm = normalize_to_8bit(fixed_ch1, percentile_clip)
    moving_ch1_before_norm = normalize_to_8bit(moving_ch1_before, percentile_clip)
    moving_ch1_after_norm = normalize_to_8bit(moving_ch1_after, percentile_clip)
    
    fixed_ch0_norm = normalize_to_8bit(fixed_ch0, percentile_clip)
    moving_ch0_before_norm = normalize_to_8bit(moving_ch0_before, percentile_clip)
    moving_ch0_after_norm = normalize_to_8bit(moving_ch0_after, percentile_clip)
    
    # Create overlays
    # Row 0, Col 2: DAPI overlay before (cyan + magenta)
    overlay_ch1_before = create_overlay(fixed_ch1, moving_ch1_before, 'cyan', 'magenta', percentile_clip)
    overlay_ch1_after = create_overlay(fixed_ch1, moving_ch1_after, 'cyan', 'magenta', percentile_clip)
    
    # Row 1, Col 2: Protein overlay before (green + red)
    overlay_ch0_before = create_overlay(fixed_ch0, moving_ch0_before, 'green', 'red', percentile_clip)
    overlay_ch0_after = create_overlay(fixed_ch0, moving_ch0_after, 'green', 'red', percentile_clip)
    
    # Convert grayscale to RGB for consistent grid
    def to_rgb(img):
        return np.stack([img, img, img], axis=-1)
    
    # Build grid
    # Row 0: Channel 1 (DAPI)
    row0_col0 = to_rgb(fixed_ch1_norm)
    row0_col1 = to_rgb(moving_ch1_before_norm)
    row0_col2 = overlay_ch1_before
    
    # Row 1: Channel 0 (Protein)
    row1_col0 = to_rgb(fixed_ch0_norm)
    row1_col1 = to_rgb(moving_ch0_before_norm)
    row1_col2 = overlay_ch0_before
    
    # Concatenate horizontally for each row
    row0 = np.concatenate([row0_col0, row0_col1, row0_col2], axis=1)
    row1 = np.concatenate([row1_col0, row1_col1, row1_col2], axis=1)
    
    # Concatenate vertically
    grid_before = np.concatenate([row0, row1], axis=0)
    
    # Build "after" grid
    row0_after = np.concatenate([row0_col0, to_rgb(moving_ch1_after_norm), overlay_ch1_after], axis=1)
    row1_after = np.concatenate([row1_col0, to_rgb(moving_ch0_after_norm), overlay_ch0_after], axis=1)
    grid_after = np.concatenate([row0_after, row1_after], axis=0)
    
    logger.info("Created 2×3 QC grids (before and after)")
    return grid_before, grid_after

