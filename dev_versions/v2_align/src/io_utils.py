"""
TIFF I/O utilities for multichannel images.

Handles loading and saving of multichannel TIFF files with proper metadata
for ImageJ/FIJI compatibility.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging


logger = logging.getLogger(__name__)


class IOError(Exception):
    """Raised when I/O operations fail."""
    pass


def load_multichannel_tiff(
    path: str,
    expected_channels: Optional[int] = None
) -> np.ndarray:
    """
    Load a multichannel TIFF file.
    
    Args:
        path: Path to TIFF file
        expected_channels: If provided, validate channel count
        
    Returns:
        Image array with shape (H, W, C) where C is number of channels
        
    Raises:
        IOError: If file cannot be read or has wrong format
    """
    path = Path(path)
    if not path.exists():
        raise IOError(f"File not found: {path}")
    
    logger.info(f"Loading TIFF: {path}")
    
    try:
        # Load with tifffile (handles ImageJ format)
        img = tifffile.imread(str(path))
        
        # Handle different array shapes
        if img.ndim == 2:
            # Single channel (H, W) -> (H, W, 1)
            img = img[:, :, np.newaxis]
            logger.info(f"Loaded single-channel image: {img.shape}")
            
        elif img.ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            # Heuristic: if first dimension is small (< 10), assume it's channels
            if img.shape[0] < 10 and img.shape[0] < img.shape[1]:
                # Likely (C, H, W) -> convert to (H, W, C)
                img = np.moveaxis(img, 0, -1)
                logger.info(f"Converted (C,H,W) to (H,W,C): {img.shape}")
            else:
                logger.info(f"Loaded (H,W,C) image: {img.shape}")
                
        elif img.ndim == 4:
            # Multi-page TIFF (T, H, W, C) or (T, C, H, W)
            # Take first page
            img = img[0]
            if img.shape[0] < 10:
                img = np.moveaxis(img, 0, -1)
            logger.warning(f"Multi-page TIFF detected, using first page: {img.shape}")
            
        else:
            raise IOError(f"Unexpected array dimensions: {img.ndim}")
        
        # Validate expected channels
        if expected_channels is not None:
            actual_channels = img.shape[2] if img.ndim == 3 else 1
            if actual_channels != expected_channels:
                raise IOError(
                    f"Expected {expected_channels} channels, found {actual_channels}"
                )
        
        logger.info(f"Successfully loaded image: shape={img.shape}, dtype={img.dtype}")
        return img
        
    except Exception as e:
        if isinstance(e, IOError):
            raise
        raise IOError(f"Failed to load TIFF: {e}")


def save_multichannel_tiff(
    path: str,
    image: np.ndarray,
    photometric: str = 'minisblack',
    axes: str = 'YXC',
    metadata: Optional[Dict] = None,
    compression: Optional[str] = None
) -> None:
    """
    Save a multichannel TIFF file with ImageJ-compatible metadata.
    
    Args:
        path: Output path
        image: Image array, shape (H, W) or (H, W, C)
        photometric: Color interpretation ('minisblack' for grayscale)
        axes: Dimension order (default 'YXC' for ImageJ)
        metadata: Additional metadata dict
        compression: Compression type ('lzw', 'deflate', None)
        
    Raises:
        IOError: If save fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving TIFF: {path}, shape={image.shape}, dtype={image.dtype}")
    
    try:
        # For multichannel images (H, W, C), convert to (C, H, W) for proper saving
        if image.ndim == 3 and axes == 'YXC':
            # tifffile expects (C, H, W) for multichannel, so transpose
            image_to_save = np.moveaxis(image, -1, 0)  # (H, W, C) -> (C, H, W)
            axes_to_save = 'CYX'
            logger.debug(f"Transposed to {image_to_save.shape} with axes={axes_to_save}")
        else:
            image_to_save = image
            axes_to_save = axes
        
        # Prepare metadata for ImageJ
        meta = {}
        if image.ndim == 3:
            # ImageJ metadata for multichannel
            meta['axes'] = axes_to_save
            meta['imagej'] = True
        
        if metadata:
            meta.update(metadata)
        
        # Save with tifffile
        tifffile.imwrite(
            str(path),
            image_to_save,
            photometric=photometric,
            metadata=meta if meta else None,
            compression=compression,
            imagej=True if image.ndim == 3 else False
        )
        
        logger.info(f"Successfully saved: {path}")
        
    except Exception as e:
        raise IOError(f"Failed to save TIFF to {path}: {e}")


def extract_channel(image: np.ndarray, channel_idx: int) -> np.ndarray:
    """
    Extract a single channel from multichannel image.
    
    Args:
        image: Multichannel image, shape (H, W, C)
        channel_idx: Channel index to extract (0-based)
        
    Returns:
        Single channel image, shape (H, W)
        
    Raises:
        IOError: If channel index is invalid
    """
    if image.ndim == 2:
        if channel_idx != 0:
            raise IOError(f"Image is single-channel, cannot extract channel {channel_idx}")
        return image
    
    if image.ndim != 3:
        raise IOError(f"Expected 2D or 3D image, got shape: {image.shape}")
    
    n_channels = image.shape[2]
    if channel_idx < 0 or channel_idx >= n_channels:
        raise IOError(
            f"Invalid channel index {channel_idx} for image with {n_channels} channels"
        )
    
    return image[:, :, channel_idx]


def stack_channels(*channels: np.ndarray) -> np.ndarray:
    """
    Stack multiple single-channel images into multichannel image.
    
    Args:
        *channels: Variable number of 2D arrays (H, W)
        
    Returns:
        Multichannel image, shape (H, W, C)
        
    Raises:
        IOError: If channels have different shapes
    """
    if not channels:
        raise IOError("No channels provided")
    
    # Validate all channels have same shape
    shape = channels[0].shape
    for i, ch in enumerate(channels[1:], start=1):
        if ch.shape != shape:
            raise IOError(
                f"Channel {i} shape {ch.shape} doesn't match channel 0 shape {shape}"
            )
    
    # Stack along last axis
    stacked = np.stack(channels, axis=-1)
    return stacked


def validate_input_image(
    image: np.ndarray,
    expected_channels: int = 2,
    expected_dtype: Optional[type] = None,
    min_size: int = 32
) -> None:
    """
    Validate input image meets requirements.
    
    Args:
        image: Image array to validate
        expected_channels: Expected number of channels
        expected_dtype: Expected data type (e.g., np.uint16)
        min_size: Minimum image dimension
        
    Raises:
        IOError: If validation fails
    """
    # Check dimensions
    if image.ndim not in [2, 3]:
        raise IOError(f"Image must be 2D or 3D, got {image.ndim}D")
    
    # Check size
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        raise IOError(f"Image too small: {h}x{w}, minimum is {min_size}x{min_size}")
    
    # Check channels
    if image.ndim == 3:
        actual_channels = image.shape[2]
        if actual_channels != expected_channels:
            raise IOError(
                f"Expected {expected_channels} channels, found {actual_channels}"
            )
    elif expected_channels != 1:
        raise IOError(f"Expected {expected_channels} channels, found 1 (single-channel image)")
    
    # Check dtype
    if expected_dtype is not None and image.dtype != expected_dtype:
        logger.warning(
            f"Image dtype {image.dtype} doesn't match expected {expected_dtype}"
        )
    
    logger.info(f"Image validation passed: shape={image.shape}, dtype={image.dtype}")


def validate_image_pair(
    fixed: np.ndarray,
    moving: np.ndarray
) -> None:
    """
    Validate that two images can be registered together.
    
    Args:
        fixed: Fixed/reference image
        moving: Moving image to align
        
    Raises:
        IOError: If images are incompatible
    """
    # Check shapes match
    if fixed.shape != moving.shape:
        raise IOError(
            f"Image shapes don't match: fixed {fixed.shape} vs moving {moving.shape}"
        )
    
    # Check dtypes
    if fixed.dtype != moving.dtype:
        logger.warning(
            f"Image dtypes differ: fixed {fixed.dtype} vs moving {moving.dtype}"
        )
    
    logger.info("Image pair validation passed")


def get_channel_names(image: np.ndarray, default_prefix: str = "Channel") -> list:
    """
    Generate channel names for an image.
    
    Args:
        image: Multichannel image
        default_prefix: Prefix for channel names
        
    Returns:
        List of channel names
    """
    n_channels = image.shape[2] if image.ndim == 3 else 1
    
    # Special case for 2-channel DAPI+Protein images
    if n_channels == 2:
        return ["Protein", "DAPI"]
    
    return [f"{default_prefix}_{i}" for i in range(n_channels)]


def normalize_to_dtype(
    image: np.ndarray,
    target_dtype: type
) -> np.ndarray:
    """
    Normalize image to target dtype range.
    
    Args:
        image: Input image
        target_dtype: Target dtype (e.g., np.uint8, np.uint16)
        
    Returns:
        Normalized image with target dtype
    """
    # Get dtype ranges
    if target_dtype == np.uint8:
        target_max = 255
    elif target_dtype == np.uint16:
        target_max = 65535
    elif target_dtype == np.float32 or target_dtype == np.float64:
        target_max = 1.0
    else:
        raise IOError(f"Unsupported target dtype: {target_dtype}")
    
    # Normalize to [0, target_max]
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        # Constant image
        normalized = np.zeros_like(image, dtype=target_dtype)
    else:
        normalized = (image - img_min) / (img_max - img_min) * target_max
        normalized = normalized.astype(target_dtype)
    
    return normalized

