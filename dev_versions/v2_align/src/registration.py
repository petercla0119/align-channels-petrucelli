"""
Pure Python image registration for DAPI channel alignment.

Implements automatic landmark detection using phase correlation and 
feature-based methods. No FIJI/ImageJ required.
"""

import numpy as np
from typing import Tuple, Optional, List
from skimage.registration import phase_cross_correlation
from skimage.feature import match_descriptors, ORB, corner_peaks, corner_harris
from skimage.transform import EuclideanTransform, SimilarityTransform, AffineTransform
from skimage.measure import ransac
import logging
import cv2

from .rigid_transform import get_transformation_matrix, TransformError


logger = logging.getLogger(__name__)


class RegistrationError(Exception):
    """Raised when image registration fails."""
    pass


def register_images_phase_correlation(
    fixed: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Register two images using phase correlation (translation only).
    
    Fast and robust for pure translation, but doesn't handle rotation.
    
    Args:
        fixed: Fixed/reference image (H, W)
        moving: Moving image to align (H, W)
        upsample_factor: Precision factor for sub-pixel alignment
        
    Returns:
        Tuple of (shift_vector, error)
        - shift_vector: [shift_y, shift_x] in pixels
        - error: Registration error metric (0.0 if not available)
    """
    if fixed.shape != moving.shape:
        raise RegistrationError(f"Image shapes don't match: {fixed.shape} vs {moving.shape}")
    
    logger.info("Computing phase correlation...")
    
    # scikit-image API changed - return_error parameter was removed
    # Now returns (shift, error, phasediff) by default
    result = phase_cross_correlation(fixed, moving, upsample_factor=upsample_factor)
    
    if len(result) == 3:
        shift, error, diffphase = result
    else:
        # Older API or different return format
        shift = result
        error = 0.0
    
    logger.info(f"Detected shift: y={shift[0]:.2f}, x={shift[1]:.2f}")
    
    return shift, error


def detect_keypoints_orb(
    image: np.ndarray,
    n_keypoints: int = 500,
    fast_threshold: float = 0.08
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect keypoints using ORB (Oriented FAST and Rotated BRIEF).
    
    Args:
        image: Grayscale image (H, W)
        n_keypoints: Maximum number of keypoints to detect
        fast_threshold: Detection threshold (lower = more keypoints)
        
    Returns:
        Tuple of (keypoints, descriptors)
        - keypoints: Array of shape (N, 2) with (y, x) coordinates
        - descriptors: Array of shape (N, descriptor_size) binary descriptors
    """
    logger.info(f"Detecting up to {n_keypoints} ORB keypoints...")
    
    descriptor_extractor = ORB(
        n_keypoints=n_keypoints,
        fast_threshold=fast_threshold
    )
    
    descriptor_extractor.detect_and_extract(image)
    
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    
    logger.info(f"Found {len(keypoints)} keypoints")
    
    return keypoints, descriptors


def detect_keypoints_harris(
    image: np.ndarray,
    min_distance: int = 10,
    threshold_rel: float = 0.01,
    num_peaks: int = 500
) -> np.ndarray:
    """
    Detect corner keypoints using Harris corner detector.
    
    Args:
        image: Grayscale image (H, W)
        min_distance: Minimum distance between corners
        threshold_rel: Relative threshold (fraction of max corner strength)
        num_peaks: Maximum number of corners to detect
        
    Returns:
        Array of shape (N, 2) with (y, x) coordinates
    """
    logger.info(f"Detecting Harris corners...")
    
    # Compute corner strength
    corner_response = corner_harris(image)
    
    # Find peaks
    corners = corner_peaks(
        corner_response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=num_peaks
    )
    
    logger.info(f"Found {len(corners)} corners")
    
    return corners


def match_keypoints(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    cross_check: bool = True,
    max_ratio: float = 0.8
) -> np.ndarray:
    """
    Match keypoint descriptors between two images.
    
    Args:
        descriptors1: Descriptors from first image
        descriptors2: Descriptors from second image
        cross_check: If True, only keep mutually best matches
        max_ratio: Maximum ratio for Lowe's ratio test
        
    Returns:
        Array of shape (N, 2) where matches[i] = [idx1, idx2]
    """
    logger.info("Matching descriptors...")
    
    matches = match_descriptors(
        descriptors1,
        descriptors2,
        cross_check=cross_check,
        max_ratio=max_ratio
    )
    
    logger.info(f"Found {len(matches)} matches")
    
    return matches


def register_images_feature_based(
    fixed: np.ndarray,
    moving: np.ndarray,
    transform_type: str = 'RIGID_BODY',
    min_matches: int = 10,
    ransac_residual_threshold: float = 2.0,
    ransac_max_trials: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Register two images using feature-based matching.
    
    Detects keypoints, matches them, and computes transformation using RANSAC.
    Handles rotation and translation.
    
    Args:
        fixed: Fixed/reference image (H, W)
        moving: Moving image to align (H, W)
        transform_type: 'TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', or 'AFFINE'
        min_matches: Minimum number of matches required
        ransac_residual_threshold: RANSAC inlier threshold in pixels
        ransac_max_trials: Maximum RANSAC iterations
        
    Returns:
        Tuple of (src_landmarks, dst_landmarks, inlier_mask)
        - src_landmarks: Matched keypoints in moving image, shape (N, 2)
        - dst_landmarks: Matched keypoints in fixed image, shape (N, 2)
        - inlier_mask: Boolean array indicating RANSAC inliers
        
    Raises:
        RegistrationError: If registration fails
    """
    if fixed.shape != moving.shape:
        raise RegistrationError(f"Image shapes don't match: {fixed.shape} vs {moving.shape}")
    
    # Normalize images to [0, 1] for feature detection
    fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-10)
    moving_norm = (moving - moving.min()) / (moving.max() - moving.min() + 1e-10)
    
    # Detect keypoints and extract descriptors
    kp_fixed, desc_fixed = detect_keypoints_orb(fixed_norm)
    kp_moving, desc_moving = detect_keypoints_orb(moving_norm)
    
    if len(kp_fixed) == 0 or len(kp_moving) == 0:
        raise RegistrationError("No keypoints detected in one or both images")
    
    # Match descriptors
    matches = match_keypoints(desc_fixed, desc_moving)
    
    if len(matches) < min_matches:
        raise RegistrationError(
            f"Insufficient matches: {len(matches)} < {min_matches}. "
            "Images may be too different or feature-poor."
        )
    
    # Extract matched point coordinates
    src_pts = kp_moving[matches[:, 1]]  # Points in moving image
    dst_pts = kp_fixed[matches[:, 0]]   # Corresponding points in fixed image
    
    # Convert from (row, col) to (x, y)
    src_pts = src_pts[:, ::-1]  # Flip to (x, y)
    dst_pts = dst_pts[:, ::-1]
    
    logger.info(f"Running RANSAC with {len(matches)} matches...")
    
    # Determine model class based on transform type
    if transform_type == 'TRANSLATION':
        model_class = 'euclidean'  # Actually just translation if we set rotation=0
        # For pure translation, we'll compute it manually
        # Compute median shift (robust to outliers)
        shifts = dst_pts - src_pts
        median_shift = np.median(shifts, axis=0)
        
        # Consider inliers as points within threshold of median
        residuals = np.linalg.norm(shifts - median_shift, axis=1)
        inlier_mask = residuals < ransac_residual_threshold
        
        if inlier_mask.sum() < min_matches:
            raise RegistrationError(
                f"Insufficient inliers after RANSAC: {inlier_mask.sum()} < {min_matches}"
            )
        
        # Use only inliers
        src_landmarks = src_pts[inlier_mask]
        dst_landmarks = dst_pts[inlier_mask]
        
        logger.info(f"RANSAC found {inlier_mask.sum()} inliers out of {len(matches)} matches")
        
        return src_landmarks, dst_landmarks, inlier_mask
        
    elif transform_type in ['RIGID_BODY', 'SCALED_ROTATION']:
        # Use Euclidean (rigid) or similarity (scaled rotation) transform
        if transform_type == 'SCALED_ROTATION':
            model_class = SimilarityTransform
        else:
            model_class = EuclideanTransform
        
    elif transform_type == 'AFFINE':
        model_class = AffineTransform
        
    else:
        raise RegistrationError(f"Unknown transform type: {transform_type}")
    
    # Run RANSAC to find inliers and estimate transform
    try:
        model, inliers = ransac(
            (src_pts, dst_pts),
            model_class,
            min_samples=3 if transform_type in ['RIGID_BODY', 'AFFINE'] else 2,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials
        )
        
        if inliers.sum() < min_matches:
            raise RegistrationError(
                f"Insufficient inliers after RANSAC: {inliers.sum()} < {min_matches}"
            )
        
        # Use only inliers
        src_landmarks = src_pts[inliers]
        dst_landmarks = dst_pts[inliers]
        
        logger.info(f"RANSAC found {inliers.sum()} inliers out of {len(matches)} matches")
        
        return src_landmarks, dst_landmarks, inliers
        
    except Exception as e:
        raise RegistrationError(f"RANSAC failed: {e}")


def select_well_distributed_landmarks(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_landmarks: int,
    image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of well-distributed landmark pairs.
    
    For rigid-body transform, we need exactly 3 landmarks that form
    a good triangle (not collinear). This function selects the best subset.
    
    Args:
        src_pts: Source landmarks, shape (N, 2)
        dst_pts: Destination landmarks, shape (N, 2)
        num_landmarks: Number of landmarks to select (typically 3)
        image_shape: (height, width) for spatial distribution
        
    Returns:
        Tuple of (selected_src, selected_dst), each shape (num_landmarks, 2)
    """
    if len(src_pts) <= num_landmarks:
        return src_pts[:num_landmarks], dst_pts[:num_landmarks]
    
    # Divide image into grid
    h, w = image_shape
    grid_size = int(np.sqrt(num_landmarks)) + 1
    
    # Try to select points from different grid cells
    selected_indices = []
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            if len(selected_indices) >= num_landmarks:
                break
            
            # Find points in this grid cell
            y_min, y_max = gy * h // grid_size, (gy + 1) * h // grid_size
            x_min, x_max = gx * w // grid_size, (gx + 1) * w // grid_size
            
            in_cell = (
                (src_pts[:, 0] >= x_min) & (src_pts[:, 0] < x_max) &
                (src_pts[:, 1] >= y_min) & (src_pts[:, 1] < y_max)
            )
            
            if in_cell.any():
                # Select first point in this cell
                idx = np.where(in_cell)[0][0]
                selected_indices.append(idx)
        
        if len(selected_indices) >= num_landmarks:
            break
    
    # If not enough, just take the first N
    if len(selected_indices) < num_landmarks:
        selected_indices = list(range(num_landmarks))
    
    selected_indices = selected_indices[:num_landmarks]
    
    return src_pts[selected_indices], dst_pts[selected_indices]


def register_images_ecc(
    fixed: np.ndarray,
    moving: np.ndarray,
    transform_type: str = 'RIGID_BODY',
    num_iterations: int = 5000,
    termination_eps: float = 1e-6,
    gauss_filt_size: int = 5,
    use_pyramid: bool = True,
    pyramid_levels: int = 5,  # Increased from 3 to 5 for better convergence
    use_phase_init: bool = True  # Initialize with phase correlation
) -> Tuple[np.ndarray, float]:
    """
    Register two images using ECC (Enhanced Correlation Coefficient).
    
    This is an intensity-based method that works well on images with uniform
    regions (like fluorescence microscopy). More robust than feature-based
    methods when features are sparse.
    
    Uses multi-scale pyramid approach for better convergence:
    - Starts at coarse resolution (downsampled)
    - Progressively refines at finer resolutions
    - Avoids local minima and improves convergence
    
    Improvements (Nov 10, 2025):
    - Phase correlation initialization for better starting point
    - 5-level pyramid (vs 3) for more robust coarse-to-fine optimization
    - Adaptive epsilon per pyramid level (looser at coarse, strict at fine)
    
    Args:
        fixed: Fixed/reference image (H, W)
        moving: Moving image to align (H, W)
        transform_type: Transform model ('TRANSLATION', 'RIGID_BODY', 'AFFINE')
        num_iterations: Maximum iterations per pyramid level
        termination_eps: Convergence threshold (for finest level)
        gauss_filt_size: Gaussian filter size for smoothing (1=no filter)
        use_pyramid: If True, use multi-scale pyramid (recommended)
        pyramid_levels: Number of pyramid levels (5 = 16x, 8x, 4x, 2x, 1x downsampling)
        use_phase_init: If True, initialize translation with phase correlation
        
    Returns:
        Tuple of (warp_matrix, correlation_coefficient)
        - warp_matrix: 2x3 transformation matrix
        - correlation_coefficient: Final correlation (higher is better)
    
    Raises:
        RegistrationError: If ECC fails to converge
    """
    if fixed.shape != moving.shape:
        raise RegistrationError(f"Image shapes don't match: {fixed.shape} vs {moving.shape}")
    
    logger.info(f"Running ECC registration ({transform_type}, pyramid={use_pyramid}, levels={pyramid_levels}, phase_init={use_phase_init})...")
    
    # Normalize to 16-bit for better precision (microscopy data is often 16-bit)
    # OpenCV ECC supports float32, so we normalize to [0, 65535] then convert to float32
    fixed_norm = cv2.normalize(fixed.astype(np.float32), None, 0, 65535, cv2.NORM_MINMAX).astype(np.float32)
    moving_norm = cv2.normalize(moving.astype(np.float32), None, 0, 65535, cv2.NORM_MINMAX).astype(np.float32)
    
    # Map transform type to OpenCV motion type
    motion_map = {
        'TRANSLATION': cv2.MOTION_TRANSLATION,
        'RIGID_BODY': cv2.MOTION_EUCLIDEAN,
        'AFFINE': cv2.MOTION_AFFINE
    }
    
    if transform_type not in motion_map:
        raise RegistrationError(f"Unsupported transform type for ECC: {transform_type}")
    
    warp_mode = motion_map[transform_type]
    
    # Initialize warp matrix (identity) and optional phase-correlation shift
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    phase_shift = None
    
    if use_phase_init:
        try:
            # Use phase correlation to estimate translation at full resolution
            shift, error = register_images_phase_correlation(fixed, moving, upsample_factor=10)
            phase_shift = (shift[0], shift[1])  # (dy, dx)
            logger.info(f"  Phase correlation init (will apply at finest level): tx={shift[1]:.2f}, ty={shift[0]:.2f}, error={error:.4f}")
        except Exception as e:
            phase_shift = None
            logger.warning(f"  Phase correlation init failed ({e}), proceeding without it")
    
    # Define adaptive termination criteria per pyramid level
    # Coarse levels: looser epsilon for faster exploration
    # Fine levels: strict epsilon for sub-pixel precision
    def get_criteria_for_level(level: int, total_levels: int, base_eps: float, base_iters: int):
        """Get adaptive criteria for pyramid level."""
        if level == 0:
            # Coarsest: very loose, more iterations for exploration
            eps = base_eps * 100  # 1e-4 if base is 1e-6
            iters = base_iters
        elif level == 1:
            # Second coarsest: loose
            eps = base_eps * 10   # 1e-5 if base is 1e-6
            iters = base_iters
        else:
            # Fine levels: strict for precision
            eps = base_eps
            iters = base_iters
        
        return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    
    if not use_pyramid:
        # Single-scale ECC (original implementation)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, termination_eps)
            logger.debug(f"  Single-scale: Iterations={num_iterations}, eps={termination_eps}")
            (cc, warp_matrix) = cv2.findTransformECC(
                fixed_norm,
                moving_norm,
                warp_matrix,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=gauss_filt_size
            )
            logger.info(f"  ECC converged with correlation: {cc:.6f}")
            return warp_matrix, cc
        except cv2.error as e:
            raise RegistrationError(f"ECC optimization failed: {e}")
    
    # Multi-scale pyramid ECC (recommended)
    try:
        # Calculate pyramid scales (e.g., [16, 8, 4, 2, 1] for 5 levels)
        scales = [2 ** i for i in range(pyramid_levels - 1, -1, -1)]
        logger.info(f"  Multi-scale pyramid: levels={pyramid_levels}, scales={scales}")
        
        for level, scale in enumerate(scales):
            # Get adaptive criteria for this level
            level_criteria = get_criteria_for_level(level, pyramid_levels, termination_eps, num_iterations)
            _, _, level_eps = level_criteria
            
            # Downsample images for this level
            if scale > 1:
                new_size = (int(fixed_norm.shape[1] / scale), int(fixed_norm.shape[0] / scale))
                fixed_scaled = cv2.resize(fixed_norm, new_size, interpolation=cv2.INTER_LINEAR)
                moving_scaled = cv2.resize(moving_norm, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                fixed_scaled = fixed_norm
                moving_scaled = moving_norm
            
            logger.debug(f"  Level {level+1}/{pyramid_levels}: scale=1/{scale}, size={fixed_scaled.shape}, eps={level_eps:.1e}")
            
            # Apply phase correlation initialization ONLY at the finest level
            if level == len(scales) - 1 and phase_shift is not None:
                warp_matrix[0, 2] += phase_shift[1]  # dx
                warp_matrix[1, 2] += phase_shift[0]  # dy
                logger.debug(f"    Applied phase init at finest level: tx+={phase_shift[1]:.2f}, ty+={phase_shift[0]:.2f}")
            
            # Run ECC at this scale with adaptive criteria
            (cc, warp_matrix) = cv2.findTransformECC(
                fixed_scaled,
                moving_scaled,
                warp_matrix,
                warp_mode,
                level_criteria,  # Use adaptive criteria per level
                inputMask=None,
                gaussFiltSize=gauss_filt_size
            )
            
            logger.debug(f"    Correlation: {cc:.6f}")
            
            # Scale up transformation for next level
            if scale > 1 and level < len(scales) - 1:
                # Translation components need to be scaled up
                warp_matrix[0, 2] *= 2.0  # x translation
                warp_matrix[1, 2] *= 2.0  # y translation
                # Rotation components (matrix[0:2, 0:2]) stay the same
        
        logger.info(f"  Multi-scale ECC converged with final correlation: {cc:.6f}")
        return warp_matrix, cc
        
    except cv2.error as e:
        raise RegistrationError(f"Multi-scale ECC optimization failed: {e}")


def warp_matrix_to_landmarks(
    warp_matrix: np.ndarray,
    image_shape: Tuple[int, int],
    transform_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ECC warp matrix to landmark pairs for compatibility.
    
    Creates anchor points and transforms them using the warp matrix
    to generate source/destination landmark pairs.
    
    Args:
        warp_matrix: 2x3 transformation matrix from ECC
        image_shape: (height, width) of the images
        transform_type: Transform model type
        
    Returns:
        Tuple of (src_landmarks, dst_landmarks) as Nx2 arrays
    """
    h, w = image_shape
    
    # Define anchor points based on transform type (matching FIJI MultiStackReg)
    if transform_type == 'RIGID_BODY':
        # 3 points along vertical centerline
        src_pts = np.array([
            [w / 2, h / 2],        # Center
            [w / 2, h / 4],        # Top center
            [w / 2, 3 * h / 4]     # Bottom center
        ], dtype=np.float32)
    elif transform_type == 'AFFINE':
        # 3 points at corners
        src_pts = np.array([
            [w / 2, h / 4],
            [w / 4, 3 * h / 4],
            [3 * w / 4, 3 * h / 4]
        ], dtype=np.float32)
    elif transform_type == 'TRANSLATION':
        # Single center point
        src_pts = np.array([[w / 2, h / 2]], dtype=np.float32)
    else:
        # Default: 3 well-distributed points
        src_pts = np.array([
            [w / 2, h / 2],
            [w / 2, h / 4],
            [w / 2, 3 * h / 4]
        ], dtype=np.float32)
    
    # Transform source points to get destination points
    # warp_matrix is 2x3, need to add homogeneous coordinate
    src_pts_h = np.column_stack([src_pts, np.ones(len(src_pts))])  # Nx3
    dst_pts = (warp_matrix @ src_pts_h.T).T  # (2x3) @ (3xN)^T -> (2xN)^T -> Nx2
    
    logger.debug(f"  Generated {len(src_pts)} landmark pairs from warp matrix")
    
    return src_pts, dst_pts


def register_dapi_channels(
    fixed_dapi: np.ndarray,
    moving_dapi: np.ndarray,
    method: str = 'feature',
    transform_type: str = 'RIGID_BODY'  # DEFAULT: 3 landmarks, rotation + translation, NO scaling
) -> dict:
    """
    Register two DAPI channels and return transformation parameters.
    
    Main entry point for DAPI-based alignment.
    
    Args:
        fixed_dapi: Fixed/reference DAPI channel (H, W)
        moving_dapi: Moving DAPI channel to align (H, W)
        method: 'ecc' (default, robust), 'phase' (translation only), or 'hybrid' (ECC with phase fallback)
        transform_type: 'TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', or 'AFFINE'
        
    Returns:
        Dictionary with registration results:
            - 'src_landmarks': Source keypoints, shape (N, 2)
            - 'dst_landmarks': Target keypoints, shape (N, 2)
            - 'transform_type': Type of transformation
            - 'method': Registration method used
            - 'num_matches': Number of matched keypoints
            
    Raises:
        RegistrationError: If registration fails
    """
    logger.info(f"Registering DAPI channels using {method} method, {transform_type} transform")
    
    if method == 'phase':
        # Phase correlation (translation only)
        shift, error = register_images_phase_correlation(fixed_dapi, moving_dapi)
        
        # Detect and fix wrapping artifacts
        # If shift is > 20% of image dimensions, it's likely wrapped
        # Most microscopy shifts should be < 100 pixels for well-aligned images
        h, w = fixed_dapi.shape
        shift_threshold_percent = 0.2  # 20% of image size
        shift_threshold_pixels = 200  # Absolute pixel threshold
        
        if (abs(shift[0]) > h * shift_threshold_percent or abs(shift[1]) > w * shift_threshold_percent or 
            abs(shift[0]) > shift_threshold_pixels or abs(shift[1]) > shift_threshold_pixels):
            logger.warning(f"  Phase correlation detected large shift (y={shift[0]:.1f}, x={shift[1]:.1f}), checking for wrapping...")
            # Try complement shift (subtract from image dimensions)
            alt_shift_y = shift[0] - h if shift[0] > 0 else shift[0] + h
            alt_shift_x = shift[1] - w if shift[1] > 0 else shift[1] + w
            
            # Use the smaller absolute shift
            if abs(alt_shift_y) < abs(shift[0]):
                shift[0] = alt_shift_y
                logger.warning(f"  Corrected Y shift to: {shift[0]:.1f} (unwrapped from periodic boundary)")
            if abs(alt_shift_x) < abs(shift[1]):
                shift[1] = alt_shift_x
                logger.warning(f"  Corrected X shift to: {shift[1]:.1f} (unwrapped from periodic boundary)")
        
        # Create single landmark pair representing the shift
        center = np.array([[w / 2, h / 2]])
        src_landmarks = center.copy()
        dst_landmarks = center + shift[::-1]  # Flip from (y, x) to (x, y)
        
        return {
            'src_landmarks': src_landmarks,
            'dst_landmarks': dst_landmarks,
            'transform_type': 'TRANSLATION',
            'method': 'phase_correlation',
            'num_matches': 1,
            'shift': shift,
            'error': error
        }
        
    elif method == 'hybrid':
        # Hybrid: Try ECC first, fall back to phase correlation if it fails
        try:
            logger.info("  Attempting ECC registration...")
            warp_matrix, correlation = register_images_ecc(
                fixed_dapi,
                moving_dapi,
                transform_type=transform_type
            )
            
            # Convert warp matrix to landmarks for compatibility
            src_landmarks, dst_landmarks = warp_matrix_to_landmarks(
                warp_matrix,
                fixed_dapi.shape,
                transform_type
            )
            
            return {
                'src_landmarks': src_landmarks,
                'dst_landmarks': dst_landmarks,
                'transform_type': transform_type,
                'method': 'ecc',
                'num_matches': len(src_landmarks),
                'warp_matrix': warp_matrix,
                'correlation': correlation
            }
        except RegistrationError as e:
            logger.warning(f"  ECC failed ({e}), falling back to phase correlation...")
            
            # Fall back to phase correlation with wrapping correction
            shift, error = register_images_phase_correlation(fixed_dapi, moving_dapi)
            
            # Detect and fix wrapping artifacts AND suspicious shifts
            h, w = fixed_dapi.shape
            shift_threshold_percent = 0.2  # 20% of image size
            shift_threshold_pixels = 200  # Absolute pixel threshold for wrapping
            suspicious_shift_threshold = 50  # Shifts > 50 pixels are suspicious when ECC failed
            
            # First check for wrapping
            if (abs(shift[0]) > h * shift_threshold_percent or abs(shift[1]) > w * shift_threshold_percent or 
                abs(shift[0]) > shift_threshold_pixels or abs(shift[1]) > shift_threshold_pixels):
                logger.warning(f"  Phase correlation detected large shift (y={shift[0]:.1f}, x={shift[1]:.1f}), checking for wrapping...")
                alt_shift_y = shift[0] - h if shift[0] > 0 else shift[0] + h
                alt_shift_x = shift[1] - w if shift[1] > 0 else shift[1] + w
                
                if abs(alt_shift_y) < abs(shift[0]):
                    shift[0] = alt_shift_y
                    logger.warning(f"  Corrected Y shift to: {shift[0]:.1f} (unwrapped from periodic boundary)")
                if abs(alt_shift_x) < abs(shift[1]):
                    shift[1] = alt_shift_x
                    logger.warning(f"  Corrected X shift to: {shift[1]:.1f} (unwrapped from periodic boundary)")
            
            # Check if shift is still suspicious (likely noise/artifact on near-identical images)
            shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
            if shift_magnitude > suspicious_shift_threshold:
                # Check if maybe the images DO need a small shift that phase correlation found
                # Calculate image correlation to assess similarity
                from scipy.stats import pearsonr
                fixed_flat = fixed_dapi.flatten()
                moving_flat = moving_dapi.flatten()
                correlation, _ = pearsonr(fixed_flat, moving_flat)
                
                logger.warning(f"  Phase correlation shift magnitude {shift_magnitude:.1f} pixels seems unreliable")
                logger.warning(f"  Image correlation: {correlation:.4f}")
                
                if correlation > 0.95:
                    # Images are very similar - likely already aligned or need sub-pixel shift
                    logger.warning(f"  High correlation ({correlation:.4f}) suggests images are nearly identical")
                    logger.warning(f"  Applying identity transform (no shift) - images may already be aligned")
                    shift = np.array([0.0, 0.0])
                else:
                    # Images are different but alignment failed - keep the phase correlation result as best guess
                    logger.warning(f"  Moderate correlation ({correlation:.4f}) suggests images differ significantly")
                    logger.warning(f"  Keeping phase correlation shift as best available estimate")
                    logger.warning(f"  Manual inspection STRONGLY recommended for this pair")
            
            center = np.array([[w / 2, h / 2]])
            src_landmarks = center.copy()
            dst_landmarks = center + shift[::-1]
            
            return {
                'src_landmarks': src_landmarks,
                'dst_landmarks': dst_landmarks,
                'transform_type': 'TRANSLATION',
                'method': 'phase_correlation_fallback',
                'num_matches': 1,
                'shift': shift,
                'error': error
            }
        
    elif method == 'ecc':
        # ECC (Enhanced Correlation Coefficient) - intensity-based registration
        warp_matrix, correlation = register_images_ecc(
            fixed_dapi,
            moving_dapi,
            transform_type=transform_type
        )
        
        # Convert warp matrix to landmarks for compatibility
        src_landmarks, dst_landmarks = warp_matrix_to_landmarks(
            warp_matrix,
            fixed_dapi.shape,
            transform_type
        )
        
        return {
            'src_landmarks': src_landmarks,
            'dst_landmarks': dst_landmarks,
            'transform_type': transform_type,
            'method': 'ecc',
            'num_matches': len(src_landmarks),
            'warp_matrix': warp_matrix,
            'correlation': correlation
        }
        
    else:
        raise RegistrationError(f"Unknown registration method: {method}. Choose from: phase, ecc, hybrid")

