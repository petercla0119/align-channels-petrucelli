"""
Main pipeline orchestration for two-channel DAPI+Protein alignment.

Coordinates all steps from loading to final output generation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict
import logging
import time

from .io_utils import (
    load_multichannel_tiff, save_multichannel_tiff,
    extract_channel, stack_channels,
    validate_input_image, validate_image_pair
)
from .registration import register_dapi_channels
from .rigid_transform import (
    get_transformation_matrix,
    apply_rigid_transform,
    rigid_from_landmarks
)
from .qc_generator import (
    create_overlay, create_rgb_composite,
    save_rgb_png, create_qc_grid
)
from .transform_parser import parse_multistackreg_file, validate_transform_data


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


def create_output_structure(output_dir: Path) -> Dict[str, Path]:
    """
    Create output directory structure.
    
    Returns dict with paths to each subdirectory.
    """
    output_dir = Path(output_dir)
    
    paths = {
        'root': output_dir,
        'aligned': output_dir / 'aligned',
        'transforms': output_dir / 'transforms',
        'composite': output_dir / 'composite',
        'channels': output_dir / 'composite' / 'channels',
        'qc': output_dir / 'qc'
    }
    
    # Create all directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output structure at: {output_dir}")
    return paths


def extract_prefix_from_filename(path: Path) -> str:
    """
    Return substring before first '-' in filename (without directories).
    If '-' not present, use stem.
    """
    name = path.name
    if '-' in name:
        return name.split('-', 1)[0]
    return path.stem


def align_two_channel_images(
    fixed_path: str,
    moving_path: str,
    output_dir: str,
    transform_type: str = 'RIGID_BODY',  # DEFAULT: 3 landmarks, rotation + translation, NO scaling
    registration_method: str = 'ecc',
    transform_file: Optional[str] = None,
    invert_transform: bool = False
) -> Dict:
    """
    Main pipeline: align two 2-channel images and generate all outputs.
    
    Args:
        fixed_path: Path to fixed/reference image (H, W, 2)
        moving_path: Path to moving image to align (H, W, 2)
        output_dir: Output directory for all results
        transform_type: Transform model (default RIGID_BODY), ignored if transform_file provided
        registration_method: 'feature' (ORB+RANSAC) or 'phase' (translation only), ignored if transform_file provided
        transform_file: Optional path to MultiStackReg transformation file (skips registration)
        invert_transform: If True, swap source and destination landmarks (invert transformation)
        
    Returns:
        Dictionary with pipeline results and metrics
        
    Raises:
        PipelineError: If any step fails
    """
    start_time = time.time()
    logger.info("="*60)
    logger.info("Starting two-channel alignment pipeline")
    logger.info(f"Fixed: {fixed_path}")
    logger.info(f"Moving: {moving_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Transform: {transform_type}")
    logger.info(f"Method: {registration_method}")
    logger.info("="*60)
    
    try:
        # 1. Create output structure
        paths = create_output_structure(Path(output_dir))
        
        fixed_path_obj = Path(fixed_path)
        moving_path_obj = Path(moving_path)
        prefix_fixed = extract_prefix_from_filename(fixed_path_obj)
        prefix_moving = extract_prefix_from_filename(moving_path_obj)
        if prefix_fixed != prefix_moving:
            logger.warning(
                "Prefix mismatch between fixed (%s) and moving (%s); using fixed prefix.",
                prefix_fixed,
                prefix_moving
            )
        prefix = prefix_fixed
        logger.debug(f"Using output filename prefix: {prefix}")
        
        # 2. Load images
        logger.info("Step 1/7: Loading images...")
        fixed = load_multichannel_tiff(fixed_path, expected_channels=2)
        moving = load_multichannel_tiff(moving_path, expected_channels=2)
        
        # 3. Validate
        logger.info("Step 2/7: Validating images...")
        validate_input_image(fixed, expected_channels=2)
        validate_input_image(moving, expected_channels=2)
        validate_image_pair(fixed, moving)
        
        # 4. Extract channels (C0=protein, C1=DAPI)
        logger.info("Step 3/7: Extracting channels...")
        fixed_protein = extract_channel(fixed, 0)
        fixed_dapi = extract_channel(fixed, 1)
        moving_protein = extract_channel(moving, 0)
        moving_dapi = extract_channel(moving, 1)
        
        logger.info(f"  Fixed protein: {fixed_protein.shape}, dtype={fixed_protein.dtype}")
        logger.info(f"  Fixed DAPI: {fixed_dapi.shape}, dtype={fixed_dapi.dtype}")
        logger.info(f"  Moving protein: {moving_protein.shape}, dtype={moving_protein.dtype}")
        logger.info(f"  Moving DAPI: {moving_dapi.shape}, dtype={moving_dapi.dtype}")
        
        # 5. Load or compute transform
        if transform_file:
            # Load transform from FIJI MultiStackReg file
            logger.info("Step 4/7: Loading transform from file...")
            logger.info(f"  Transform file: {transform_file}")
            
            transform_data = parse_multistackreg_file(transform_file)
            validate_transform_data(transform_data)
            
            src_landmarks = transform_data['src_pts']
            dst_landmarks = transform_data['dst_pts']
            loaded_transform_type = transform_data['transform_type']
            
            logger.info(f"  Loaded {loaded_transform_type} transform with {len(src_landmarks)} landmarks")
            
            # Override registration result
            reg_result = {
                'src_landmarks': src_landmarks,
                'dst_landmarks': dst_landmarks,
                'transform_type': loaded_transform_type,
                'num_matches': len(src_landmarks),
                'method': 'loaded_from_file'
            }
            registration_method = 'loaded_from_file'
            transform_type = loaded_transform_type
        else:
            # Register DAPI channels (automatic landmark detection)
            logger.info("Step 4/7: Registering DAPI channels...")
            reg_result = register_dapi_channels(
                fixed_dapi,
                moving_dapi,
                method=registration_method,
                transform_type=transform_type
            )
            
            logger.info(f"  Registration successful: {reg_result['num_matches']} landmarks")
        
        # 6. Compute transformation matrix
        logger.info("Step 5/7: Computing transformation matrix...")
        
        # Invert transformation if requested (swap source and destination)
        if invert_transform:
            logger.warning("  Inverting transformation (swapping source and destination landmarks)")
            src_landmarks = reg_result['dst_landmarks']
            dst_landmarks = reg_result['src_landmarks']
        else:
            src_landmarks = reg_result['src_landmarks']
            dst_landmarks = reg_result['dst_landmarks']
        
        transform_matrix = get_transformation_matrix(
            src_landmarks,
            dst_landmarks,
            reg_result['transform_type']
        )
        
        transform_params = rigid_from_landmarks(
            src_landmarks,
            dst_landmarks
        )
        
        def compute_landmark_residuals(src_pts: np.ndarray, dst_pts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
            """Return per-landmark Euclidean error between desired and predicted dst points."""
            if src_pts.size == 0:
                return np.array([])
            src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
            predicted = (matrix[:2, :] @ src_h.T).T
            residuals = np.linalg.norm(predicted - dst_pts, axis=1)
            return residuals
        
        landmark_residuals = compute_landmark_residuals(
            src_landmarks,
            dst_landmarks,
            transform_matrix
        )
        if landmark_residuals.size:
            max_resid = float(np.max(landmark_residuals))
            mean_resid = float(np.mean(landmark_residuals))
            logger.info(
                "  Landmark residuals (pixels): mean=%.4f max=%.4f",
                mean_resid,
                max_resid
            )
            if max_resid > 5.0:
                logger.warning(
                    "  Landmark residuals exceed 5px (max=%.3f). "
                    "Check registration quality for this pair.",
                    max_resid
                )
        else:
            logger.info("  Landmark residuals unavailable (no landmarks reported).")
        
        logger.info(f"  Transform type: {transform_params['transform_type']}")
        logger.info(f"  Rotation angle: {transform_params['angle']:.3f} degrees")
        logger.info(f"  Matrix:\n{transform_matrix}")
        
        # 7. Apply transformation to BOTH channels of moving image
        logger.info("Step 6/7: Applying transformation to moving image...")
        aligned_protein = apply_rigid_transform(
            moving_protein,
            transform_params,
            order=1  # Bilinear interpolation
        )
        aligned_dapi = apply_rigid_transform(
            moving_dapi,
            transform_params,
            order=1
        )
        
        logger.info("  Transformation applied to both channels")
        
        # 8. Generate outputs
        logger.info("Step 7/7: Generating outputs...")
        
        # 8a. Save aligned images
        logger.info("  Saving aligned TIFFs...")
        save_multichannel_tiff(
            str(paths['aligned'] / f"{prefix}_fixed.tif"),
            fixed,
            axes='YXC'
        )
        
        aligned_stack = stack_channels(aligned_protein, aligned_dapi)
        save_multichannel_tiff(
            str(paths['aligned'] / f"{prefix}_moving_aligned.tif"),
            aligned_stack,
            axes='YXC'
        )
        
        # 8b. Save individual channels
        logger.info("  Saving individual channels...")
        for name, data in [
            (f"{prefix}_fixed_protein.tif", fixed_protein),
            (f"{prefix}_fixed_DAPI.tif", fixed_dapi),
            (f"{prefix}_moving_aligned_protein.tif", aligned_protein),
            (f"{prefix}_moving_aligned_DAPI.tif", aligned_dapi)
        ]:
            save_multichannel_tiff(
                str(paths['channels'] / name),
                data,
                axes='YX'
            )
        
        # 8c. Save transform
        logger.info("  Saving transformation data...")
        transform_data = {
            'transform_type': reg_result['transform_type'],
            'registration_method': reg_result['method'],
            'num_landmarks': int(reg_result['num_matches']),
            'source_landmarks': reg_result['src_landmarks'].tolist(),
            'target_landmarks': reg_result['dst_landmarks'].tolist(),
            'matrix_2x3': transform_params['matrix_2x3'].tolist(),
            'matrix_3x3': transform_params['matrix_3x3'].tolist(),
            'angle_degrees': float(transform_params['angle']),
            'landmark_residuals': landmark_residuals.tolist() if landmark_residuals.size else [],
            'parameters': {
                'm00': float(transform_params['m00']),
                'm01': float(transform_params['m01']),
                'm02': float(transform_params['m02']),
                'm10': float(transform_params['m10']),
                'm11': float(transform_params['m11']),
                'm12': float(transform_params['m12'])
            }
        }
        
        transform_filename = f"{prefix}_rigid_transform.json"
        with open(paths['transforms'] / transform_filename, 'w') as f:
            json.dump(transform_data, f, indent=2)
        
        # 8d. Create QC overlays (2×3 grid for both channels)
        logger.info("  Creating QC overlays (2×3 grid)...")
        grid_before, grid_after = create_qc_grid(
            fixed_protein,      # Ch0 fixed
            fixed_dapi,         # Ch1 fixed  
            moving_protein,     # Ch0 moving before
            moving_dapi,        # Ch1 moving before
            aligned_protein,    # Ch0 moving after
            aligned_dapi        # Ch1 moving after
        )
        # Save with 2x downsampling to reduce file size (4x reduction in pixels)
        save_rgb_png(grid_before, str(paths['qc'] / f"{prefix}_overlay_before.png"), downsample_factor=2)
        save_rgb_png(grid_after, str(paths['qc'] / f"{prefix}_overlay_after.png"), downsample_factor=2)
        
        # 8e. Create RGB composite (DAPI=blue, fixed_protein=green, moving_protein=magenta)
        logger.info("  Creating RGB composite...")
        composite = create_rgb_composite(
            fixed_dapi,  # Use fixed DAPI (both should be aligned now)
            fixed_protein,
            aligned_protein,
            dapi_color='blue',
            fixed_color='green',
            moving_color='magenta'
        )
        # Save with 2x downsampling to reduce file size
        save_rgb_png(composite, str(paths['composite'] / f"{prefix}_composite_RGB.png"), downsample_factor=2)
        
        # 9. Compute metrics
        elapsed_time = time.time() - start_time
        
        results = {
            'status': 'success',
            'elapsed_time_seconds': elapsed_time,
            'transform_type': reg_result['transform_type'],
            'registration_method': reg_result['method'],
            'num_landmarks': reg_result['num_matches'],
            'rotation_angle_degrees': transform_params['angle'],
            'landmark_residuals': landmark_residuals.tolist() if landmark_residuals.size else [],
            'output_directory': str(paths['root']),
            'files_created': {
                'aligned': [f"{prefix}_fixed.tif", f"{prefix}_moving_aligned.tif"],
                'channels': [
                    f"{prefix}_fixed_protein.tif", f"{prefix}_fixed_DAPI.tif",
                    f"{prefix}_moving_aligned_protein.tif", f"{prefix}_moving_aligned_DAPI.tif"
                ],
                'transforms': [transform_filename],
                'qc': [f"{prefix}_overlay_before.png", f"{prefix}_overlay_after.png"],
                'composite': [f"{prefix}_composite_RGB.png"]
            }
        }
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Rotation: {transform_params['angle']:.3f} degrees")
        logger.info(f"Landmarks: {reg_result['num_matches']}")
        logger.info(f"Output: {paths['root']}")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise PipelineError(f"Pipeline execution failed: {e}")

