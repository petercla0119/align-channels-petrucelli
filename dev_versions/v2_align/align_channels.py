#!/usr/bin/env python3
"""
Two-Channel DAPI+Protein Alignment - CLI Entry Point

Pure Python implementation of rigid-body image registration.
NO FIJI/ImageJ required.
"""

import argparse
import logging
import re
import shutil
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from src.pipeline import align_two_channel_images, PipelineError


# Shared filename for combined STDOUT + log capture
RUN_LOG_FILENAME = "run.log"
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
    name = path.stem
    if '-' in name:
        return name.split('-', 1)[0]
    return name


PAIRING_STRIP_TOKENS = (
    'fixed',
    'moving',
    'aligned',
    'align',
    'dapi',
    'protein',
    'marker',
    'nuclei',
    'before',
    'after',
    'channel'
)


def normalize_pairing_key(raw_key: str) -> str:
    """Normalize key by stripping channel labels and punctuation for loose matching."""
    normalized = raw_key.lower()
    for token in PAIRING_STRIP_TOKENS:
        normalized = normalized.replace(token, '')
    normalized = re.sub(r'ch[0-9]+', '', normalized)
    normalized = re.sub(r'[^a-z0-9]+', '', normalized)
    return normalized or raw_key.lower()


def print_and_log_block(lines: list[str]) -> None:
    """Emit identical block to stdout and the active logger."""
    block = "\n".join(lines)
    print(block)
    logging.getLogger(__name__).info(block)


def build_batch_records(files: list[Path]) -> tuple[list[dict], dict[str, list[Path]]]:
    """Create metadata records for batch matching (handles duplicate primary keys)."""
    records: list[dict] = []
    primary_groups: defaultdict[str, list[dict]] = defaultdict(list)
    duplicates: dict[str, list[Path]] = {}
    
    for file_path in files:
        primary_key = pairing_key(file_path)
        record = {
            'path': file_path,
            'primary_key': primary_key,
            'full_key': file_path.stem
        }
        primary_groups[primary_key].append(record)
        records.append(record)
    
    for primary_key, group in primary_groups.items():
        if len(group) == 1:
            rec = group[0]
            rec['key'] = rec['primary_key']
            rec['normalized'] = normalize_pairing_key(rec['key'])
            rec['key_strategy'] = 'primary'
        else:
            duplicates[primary_key] = [rec['path'] for rec in group]
            for rec in group:
                rec['key'] = rec['full_key']
                rec['normalized'] = normalize_pairing_key(rec['key'])
                rec['key_strategy'] = 'full_stem'
    
    return records, duplicates


def match_batch_file_pairs(
    fixed_records: list[dict],
    moving_records: list[dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Attempt to pair fixed/moving records using increasingly loose rules.
    
    Returns:
        matched_pairs: list of dicts with keys (key, fixed, moving, match_type, score)
        unmatched_fixed: list of unused fixed records
        unmatched_moving: list of unused moving records
    """
    matched_pairs: list[dict] = []
    used_fixed_keys: set[str] = set()
    used_moving_keys: set[str] = set()
    
    fixed_by_key = {rec['key']: rec for rec in fixed_records}
    moving_by_key = {rec['key']: rec for rec in moving_records}
    
    # Stage 1: exact key match
    for key in sorted(set(fixed_by_key.keys()) & set(moving_by_key.keys())):
        fixed_rec = fixed_by_key[key]
        moving_rec = moving_by_key[key]
        matched_pairs.append({
            'key': key,
            'fixed': fixed_rec['path'],
            'moving': moving_rec['path'],
            'match_type': 'exact',
            'score': 1.0
        })
        used_fixed_keys.add(key)
        used_moving_keys.add(key)
    
    # Stage 2: normalized match (ignore channel tokens)
    fixed_norm_map = defaultdict(list)
    for rec in fixed_records:
        if rec['key'] in used_fixed_keys:
            continue
        fixed_norm_map[rec['normalized']].append(rec)
    
    for moving_rec in moving_records:
        if moving_rec['key'] in used_moving_keys:
            continue
        norm = moving_rec['normalized']
        candidates = fixed_norm_map.get(norm)
        while candidates and candidates[0]['key'] in used_fixed_keys:
            candidates.pop(0)
        if candidates:
            fixed_rec = candidates.pop(0)
            matched_pairs.append({
                'key': fixed_rec['key'],
                'fixed': fixed_rec['path'],
                'moving': moving_rec['path'],
                'match_type': 'normalized',
                'score': 1.0
            })
            used_fixed_keys.add(fixed_rec['key'])
            used_moving_keys.add(moving_rec['key'])
            if not candidates:
                fixed_norm_map.pop(norm, None)
    
    # Stage 3: fuzzy ratio on normalized strings
    remaining_fixed = [rec for rec in fixed_records if rec['key'] not in used_fixed_keys]
    remaining_moving = [rec for rec in moving_records if rec['key'] not in used_moving_keys]
    
    for moving_rec in remaining_moving:
        best_rec = None
        best_ratio = 0.0
        for fixed_rec in remaining_fixed:
            ratio = SequenceMatcher(
                None,
                moving_rec['normalized'],
                fixed_rec['normalized']
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_rec = fixed_rec
        if best_rec and best_ratio >= 0.6:
            matched_pairs.append({
                'key': best_rec['key'],
                'fixed': best_rec['path'],
                'moving': moving_rec['path'],
                'match_type': 'fuzzy',
                'score': best_ratio
            })
            used_fixed_keys.add(best_rec['key'])
            used_moving_keys.add(moving_rec['key'])
            remaining_fixed.remove(best_rec)
    
    unmatched_fixed = [rec for rec in fixed_records if rec['key'] not in used_fixed_keys]
    unmatched_moving = [rec for rec in moving_records if rec['key'] not in used_moving_keys]
    
    return matched_pairs, unmatched_fixed, unmatched_moving


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
    
    fixed_records, duplicates_fixed_primary = build_batch_records(fixed_files)
    moving_records, duplicates_moving_primary = build_batch_records(moving_files)
    
    fixed_by_key = {}
    duplicate_keys_fixed = defaultdict(list)
    for record in fixed_records:
        key = record['key']
        if key in fixed_by_key:
            duplicate_keys_fixed[key].append(record['path'])
        else:
            fixed_by_key[key] = record
    
    moving_by_key = {}
    duplicate_keys_moving = defaultdict(list)
    for record in moving_records:
        key = record['key']
        if key in moving_by_key:
            duplicate_keys_moving[key].append(record['path'])
        else:
            moving_by_key[key] = record
    
    summary = {
        'pairs_total': 0,
        'success_count': 0,
        'failure_count': 0,
        'skipped_existing': [],
        'missing_from_fixed': [],
        'missing_from_moving': [],
        'failures': {},
    }
    
    matched_pairs, unmatched_fixed, unmatched_moving = match_batch_file_pairs(
        fixed_records,
        moving_records
    )
    summary['pairs_total'] = len(matched_pairs)
    summary['missing_from_fixed'] = sorted({rec['path'].name for rec in unmatched_moving})
    summary['missing_from_moving'] = sorted({rec['path'].name for rec in unmatched_fixed})
    
    for primary_key in sorted(duplicates_fixed_primary):
        logger.warning(
            "Fixed directory has multiple files with primary key '%s'; switched to full filename stems: %s",
            primary_key,
            [path.name for path in duplicates_fixed_primary[primary_key]]
        )
    for primary_key in sorted(duplicates_moving_primary):
        logger.warning(
            "Moving directory has multiple files with primary key '%s'; switched to full filename stems: %s",
            primary_key,
            [path.name for path in duplicates_moving_primary[primary_key]]
        )
    
    for key in sorted(duplicate_keys_fixed):
        logger.error(
            "Unexpected duplicate fixed key even after fallback '%s': %s",
            key,
            duplicate_keys_fixed[key]
        )
    for key in sorted(duplicate_keys_moving):
        logger.error(
            "Unexpected duplicate moving key even after fallback '%s': %s",
            key,
            duplicate_keys_moving[key]
        )
    
    logger.info(f"Found {len(matched_pairs)} matching image pairs")
    
    for pair in matched_pairs:
        key = pair['key']
        fixed_path = pair['fixed']
        moving_path = pair['moving']
        if pair['match_type'] == 'normalized':
            logger.info(
                "[%s] Matched by normalized filename: %s ↔ %s",
                key,
                fixed_path.name,
                moving_path.name
            )
        elif pair['match_type'] == 'fuzzy':
            logger.warning(
                "[%s] Using fuzzy filename match: %s ↔ %s (score %.2f)",
                key,
                fixed_path.name,
                moving_path.name,
                pair['score']
            )
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
    lines = [
        "",
        "=" * 70,
        "BATCH ALIGNMENT SUMMARY",
        "=" * 70,
        f"Total pairs considered : {summary['pairs_total']}",
        f"Successful alignments   : {summary['success_count']}",
        f"Failures                : {summary['failure_count']}",
        f"Skipped (existing out.) : {len(summary['skipped_existing'])}"
    ]
    
    if summary['missing_from_fixed']:
        lines.append("")
        lines.append("Moving images missing matching fixed images for keys:")
        for key in summary['missing_from_fixed']:
            lines.append(f"  - {key}")
    
    if summary['missing_from_moving']:
        lines.append("")
        lines.append("Fixed images missing matching moving images for keys:")
        for key in summary['missing_from_moving']:
            lines.append(f"  - {key}")
    
    if summary['skipped_existing']:
        lines.append("")
        lines.append("Skipped pairs (output already exists):")
        for path in summary['skipped_existing']:
            lines.append(f"  - {path}")
    
    if summary['failures']:
        lines.append("")
        lines.append("Failures:")
        for key, reason in summary['failures'].items():
            lines.append(f"  - {key}: {reason}")
    
    lines.append("=" * 70)
    lines.append("")
    
    print_and_log_block(lines)



def setup_logging(verbose: bool = False, debug: bool = False) -> tuple[int, str, str]:
    """Configure console logging; return level and format for optional file handler."""
    if debug:
        level = logging.DEBUG
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    elif verbose:
        level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    else:
        level = logging.WARNING
        format_str = '%(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return level, format_str, date_format


def attach_file_logger(
    log_path: Path,
    level: int,
    format_str: str,
    date_format: str
) -> Path | None:
    """Attach file handler to root logger."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format_str, datefmt=date_format))
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info("Writing detailed logs to %s", log_path)
        return log_path
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not create log file %s: %s", log_path, exc)
        return None


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
    log_level, log_format, log_datefmt = setup_logging(verbose=args.verbose, debug=args.debug)
    
    logger = logging.getLogger(__name__)
    file_log_path: Path | None = None
    output_root = Path(args.output)
    
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
        
        file_log_path = attach_file_logger(
            output_root / RUN_LOG_FILENAME,
            log_level,
            log_format,
            log_datefmt
        )
        logger.info("Batch mode enabled")
        logger.info(f"  Fixed dir : {fixed_dir}")
        logger.info(f"  Moving dir: {moving_dir}")
        if file_log_path:
            logger.info("Log file saved to: %s", file_log_path)
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
        
        file_log_path = attach_file_logger(
            output_root / RUN_LOG_FILENAME,
            log_level,
            log_format,
            log_datefmt
        )
        logger.info("Single-run mode enabled")
        if file_log_path:
            logger.info("Log file saved to: %s", file_log_path)
        logger.info("Fixed input path : %s (filename: %s)", fixed_path, fixed_path.name)
        logger.info("Moving input path: %s (filename: %s)", moving_path, moving_path.name)
    
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
            logger.info("Batch outputs written under: %s", Path(args.output))
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
        
        single_lines = [
            "",
            "="*60,
            "ALIGNMENT SUCCESSFUL!",
            "="*60,
            f"Transform type:    {results['transform_type']}",
            f"Registration:      {results['registration_method']}",
            f"Landmarks found:   {results['num_landmarks']}",
            f"Rotation angle:    {results['rotation_angle_degrees']:.3f}°",
            f"Processing time:   {results['elapsed_time_seconds']:.2f}s",
            "",
            f"Output directory:  {results['output_directory']}",
            "",
            "Generated files:"
        ]
        for category, files in results['files_created'].items():
            single_lines.append(f"  {category}/")
            for f in files:
                single_lines.append(f"    - {f}")
        single_lines.append("="*60)
        print_and_log_block(single_lines)
        if not batch_mode:
            logger.info("Alignment outputs saved to: %s", results['output_directory'])
            logger.info("Fixed input confirmed: %s", fixed_path)
            logger.info("Moving input confirmed: %s", moving_path)
        
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
