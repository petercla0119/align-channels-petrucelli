#!/usr/bin/env python3
"""
Two-Channel DAPI+Protein Alignment - CLI Entry Point

Pure Python implementation of rigid-body image registration.
NO FIJI/ImageJ required.
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from src.pipeline import align_two_channel_images, PipelineError
def collect_image_files(directory: Path) -> list[Path]:
    """Return list of TIFF files under directory (recursive)."""
    patterns = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
    files = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def pairing_key(path: Path) -> str:
    """Extract pairing key from filename (substring before first '-')."""
    name = path.name
    if '-' in name:
        return name.split('-', 1)[0]
    return Path(name).stem


def run_batch_alignments(
    fixed_dir: Path,
    moving_dir: Path,
    output_root: Path,
    transform_type: str,
    registration_method: str,
    transform_file: str | None,
    overwrite: bool
) -> dict:
    logger = logging.getLogger(__name__)
    
    if transform_file:
        raise PipelineError("Batch mode does not support --transform-file (transform differs per pair).")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    fixed_files = collect_image_files(fixed_dir)
    moving_files = collect_image_files(moving_dir)
    
    fixed_map = {}
    duplicates_fixed = defaultdict(list)
    for f in fixed_files:
        key = pairing_key(f)
        if key in fixed_map:
            duplicates_fixed[key].append(f)
        else:
            fixed_map[key] = f
    
    moving_map = {}
    duplicates_moving = defaultdict(list)
    for f in moving_files:
        key = pairing_key(f)
        if key in moving_map:
            duplicates_moving[key].append(f)
        else:
            moving_map[key] = f
    
    summary = {
        'pairs_total': 0,
        'success_count': 0,
        'failure_count': 0,
        'skipped_existing': [],
        'missing_from_fixed': sorted(set(moving_map.keys()) - set(fixed_map.keys())),
        'missing_from_moving': sorted(set(fixed_map.keys()) - set(moving_map.keys())),
        'failures': {},
    }
    
    common_keys = sorted(set(fixed_map.keys()) & set(moving_map.keys()))
    summary['pairs_total'] = len(common_keys)
    
    for key in sorted(duplicates_fixed):
        logger.warning(f"Multiple fixed images for key '{key}'; using {fixed_map[key]}, skipping {duplicates_fixed[key]}")
    for key in sorted(duplicates_moving):
        logger.warning(f"Multiple moving images for key '{key}'; using {moving_map[key]}, skipping {duplicates_moving[key]}")
    
    logger.info(f"Found {len(common_keys)} matching image pairs")
    
    for key in common_keys:
        fixed_path = fixed_map[key]
        moving_path = moving_map[key]
        pair_output = output_root / key
        
        if pair_output.exists():
            if not overwrite:
                logger.warning(f"Output directory exists for '{key}', skipping (use --batch-overwrite to force).")
                summary['skipped_existing'].append(str(pair_output))
                continue
            logger.info(f"Overwriting existing output directory for '{key}'")
            shutil.rmtree(pair_output)
        
        logger.info(f"[{key}] Aligning\n  Fixed : {fixed_path}\n  Moving: {moving_path}")
        
        try:
            align_two_channel_images(
                fixed_path=str(fixed_path),
                moving_path=str(moving_path),
                output_dir=str(pair_output),
                transform_type=transform_type,
                registration_method=registration_method,
                transform_file=None
            )
            summary['success_count'] += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{key}] Alignment failed: {exc}")
            summary['failure_count'] += 1
            summary['failures'][key] = str(exc)
    
    return summary


def summarize_batch_results(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("BATCH ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"Total pairs considered : {summary['pairs_total']}")
    print(f"Successful alignments   : {summary['success_count']}")
    print(f"Failures                : {summary['failure_count']}")
    print(f"Skipped (existing out.) : {len(summary['skipped_existing'])}")
    
    if summary['missing_from_fixed']:
        print("\nMoving images missing matching fixed images for keys:")
        for key in summary['missing_from_fixed']:
            print(f"  - {key}")
    
    if summary['missing_from_moving']:
        print("\nFixed images missing matching moving images for keys:")
        for key in summary['missing_from_moving']:
            print(f"  - {key}")
    
    if summary['skipped_existing']:
        print("\nSkipped pairs (output already exists):")
        for path in summary['skipped_existing']:
            print(f"  - {path}")
    
    if summary['failures']:
        print("\nFailures:")
        for key, reason in summary['failures'].items():
            print(f"  - {key}: {reason}")
    
    print("=" * 70 + "\n")



def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    elif verbose:
        level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    else:
        level = logging.WARNING
        format_str = '%(levelname)s: %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Align two 2-channel TIFF images (DAPI + protein marker)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage (ECC registration - default)
  python align_channels.py --fixed fixed.tif --moving moving.tif --output results/

  # With verbose output
  python align_channels.py --fixed fixed.tif --moving moving.tif --output results/ --verbose

  # Use phase correlation (translation only, no rotation)
  python align_channels.py --fixed fixed.tif --moving moving.tif --output results/ \\
      --method phase

  # Load transform from FIJI file (for validation)
  python align_channels.py --fixed fixed.tif --moving moving.tif --output results/ \\
      --transform-file fiji_transform.txt

Expected input format:
  - Two TIFF files, each with 2 channels
  - Channel 0: Protein of interest (marker)
  - Channel 1: DAPI (used for alignment)
  - Same dimensions (H, W)

Outputs:
  out/
    ├── aligned/
    │   ├── fixed.tif                  # Copy of fixed image
    │   └── moving_aligned.tif         # Aligned moving image
    ├── transforms/
    │   └── rigid_transform.json       # Transform parameters
    ├── composite/
    │   ├── composite_RGB.png          # Blue=DAPI, Green=fixed, Magenta=moving
    │   └── channels/                  # Individual channels
    └── qc/
        ├── overlay_before.png         # Pre-alignment DAPI overlay
        └── overlay_after.png          # Post-alignment DAPI overlay

DEFAULT: RIGID_BODY transform (3 landmarks, rotation + translation, NO scaling)
        '''
    )
    
    # Required arguments (single-run mode)
    parser.add_argument(
        '--fixed',
        type=str,
        help='Path to fixed/reference image (2-channel TIFF)'
    )
    
    parser.add_argument(
        '--moving',
        type=str,
        help='Path to moving image to align (2-channel TIFF)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--transform-file',
        type=str,
        default=None,
        help='Path to MultiStackReg transformation file (if provided, skips registration)'
    )
    
    parser.add_argument(
        '--transform',
        type=str,
        default='RIGID_BODY',
        choices=['TRANSLATION', 'RIGID_BODY'],
        help='Transform type (default: RIGID_BODY = rotation + translation). Ignored if --transform-file is provided.'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='ecc',
        choices=['ecc', 'phase'],
        help='Registration method: ecc=Enhanced Correlation Coefficient (default, robust), phase=translation only. Ignored if --transform-file is provided.'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output (INFO level)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output (DEBUG level)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Two-Channel Alignment v0.1.0 (Pure Python, NO FIJI)'
    )
    
    # Batch-processing arguments
    parser.add_argument(
        '--batch-fixed-dir',
        type=str,
        help='Batch mode: directory containing fixed images (searched recursively)'
    )
    
    parser.add_argument(
        '--batch-moving-dir',
        type=str,
        help='Batch mode: directory containing moving images (searched recursively)'
    )
    
    parser.add_argument(
        '--batch-overwrite',
        action='store_true',
        help='Overwrite existing output directories in batch mode'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, debug=args.debug)
    
    logger = logging.getLogger(__name__)
    
    # Determine mode (single vs batch)
    batch_mode = args.batch_fixed_dir and args.batch_moving_dir
    if batch_mode:
        fixed_dir = Path(args.batch_fixed_dir)
        moving_dir = Path(args.batch_moving_dir)
        
        if not fixed_dir.exists():
            logger.error(f"Batch fixed directory not found: {fixed_dir}")
            sys.exit(1)
        if not moving_dir.exists():
            logger.error(f"Batch moving directory not found: {moving_dir}")
            sys.exit(1)
        
        logger.info("Batch mode enabled")
        logger.info(f"  Fixed dir : {fixed_dir}")
        logger.info(f"  Moving dir: {moving_dir}")
    else:
        if not args.fixed or not args.moving:
            logger.error("Single-run mode requires --fixed and --moving arguments (or use --batch-fixed-dir/--batch-moving-dir)")
            sys.exit(1)
        if args.batch_fixed_dir or args.batch_moving_dir:
            logger.error("Both --batch-fixed-dir and --batch-moving-dir are required for batch mode")
            sys.exit(1)
        
        fixed_path = Path(args.fixed)
        moving_path = Path(args.moving)
        
        if not fixed_path.exists():
            logger.error(f"Fixed image not found: {fixed_path}")
            sys.exit(1)
        
        if not moving_path.exists():
            logger.error(f"Moving image not found: {moving_path}")
            sys.exit(1)
    
    # Validate transform file if provided
    transform_file_path = None
    if args.transform_file:
        transform_file_path = Path(args.transform_file)
        if not transform_file_path.exists():
            logger.error(f"Transform file not found: {transform_file_path}")
            sys.exit(1)
        logger.info(f"Using transform file: {transform_file_path}")
    
    # Run pipeline
    try:
        if batch_mode:
            results = run_batch_alignments(
                fixed_dir=fixed_dir,
                moving_dir=moving_dir,
                output_root=Path(args.output),
                transform_type=args.transform,
                registration_method=args.method,
                transform_file=str(transform_file_path) if transform_file_path else None,
                overwrite=args.batch_overwrite
            )
            summarize_batch_results(results)
            sys.exit(0 if results['success_count'] else 1)
        
        logger.info("Starting alignment pipeline...")
        
        results = align_two_channel_images(
            fixed_path=str(fixed_path),
            moving_path=str(moving_path),
            output_dir=args.output,
            transform_type=args.transform,
            registration_method=args.method,
            transform_file=str(transform_file_path) if transform_file_path else None
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ALIGNMENT SUCCESSFUL!")
        print("="*60)
        print(f"Transform type:    {results['transform_type']}")
        print(f"Registration:      {results['registration_method']}")
        print(f"Landmarks found:   {results['num_landmarks']}")
        print(f"Rotation angle:    {results['rotation_angle_degrees']:.3f}°")
        print(f"Processing time:   {results['elapsed_time_seconds']:.2f}s")
        print(f"\nOutput directory:  {results['output_directory']}")
        print("\nGenerated files:")
        for category, files in results['files_created'].items():
            print(f"  {category}/")
            for f in files:
                print(f"    - {f}")
        print("="*60)
        
        sys.exit(0)
        
    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

