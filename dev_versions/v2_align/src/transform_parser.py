"""
MultiStackReg transform file parser.

Parses the landmark-based transformation files saved by FIJI/MultiStackReg.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class TransformParseError(Exception):
    """Raised when transform file parsing fails."""
    pass


def parse_multistackreg_file(filepath: str) -> Dict:
    """
    Parse a MultiStackReg transformation file containing landmark coordinates.
    
    Expected format:
        MultiStackReg Transformation File
        File Version 1.0
        1                              # 1=two-stack align, 0=single stack
        RIGID_BODY
        Source img: 1 Target img: 1
        <x1> <y1>                      # Source point 1
        <x2> <y2>                      # Source point 2  
        <x3> <y3>                      # Source point 3
                                       # Blank line
        <x1> <y1>                      # Target point 1
        <x2> <y2>                      # Target point 2
        <x3> <y3>                      # Target point 3
    
    Args:
        filepath: Path to the MultiStackReg transform file
        
    Returns:
        Dictionary with keys:
            - 'transform_type': str (e.g., 'RIGID_BODY', 'TRANSLATION', 'AFFINE')
            - 'src_pts': numpy array of shape (N, 2) with source landmarks
            - 'dst_pts': numpy array of shape (N, 2) with target landmarks
            - 'source_img': int
            - 'target_img': int
            - 'two_stack_align': bool
            
    Raises:
        TransformParseError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Transform file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Validate header
        if not lines[0].startswith("MultiStackReg"):
            raise TransformParseError("Invalid header: expected 'MultiStackReg Transformation File'")
        
        if not lines[1].startswith("File Version"):
            raise TransformParseError("Missing version line")
        
        # Parse two-stack alignment flag
        two_stack_align = int(lines[2]) == 1
        
        # Parse transform type
        transform_type = lines[3].strip()
        valid_types = ['TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE']
        if transform_type not in valid_types:
            raise TransformParseError(f"Unknown transform type: {transform_type}. Expected one of {valid_types}")
        
        # Determine number of landmarks based on transform type
        num_landmarks = {
            'TRANSLATION': 1,
            'RIGID_BODY': 3,
            'SCALED_ROTATION': 2,
            'AFFINE': 3
        }[transform_type]
        
        # Parse source/target image indices
        img_line = lines[4]
        match = re.search(r'Source img:\s*(\d+)\s*Target img:\s*(\d+)', img_line)
        if not match:
            raise TransformParseError(f"Invalid image index line: {img_line}")
        source_img = int(match.group(1))
        target_img = int(match.group(2))
        
        # Parse source landmarks
        src_pts = []
        line_idx = 5
        for i in range(num_landmarks):
            if line_idx >= len(lines):
                raise TransformParseError(f"Missing source landmark {i+1}")
            parts = lines[line_idx].split()
            if len(parts) < 2:
                raise TransformParseError(f"Invalid source landmark at line {line_idx}: {lines[line_idx]}")
            try:
                x, y = float(parts[0]), float(parts[1])
                src_pts.append([x, y])
            except ValueError:
                raise TransformParseError(f"Non-numeric coordinates at line {line_idx}: {lines[line_idx]}")
            line_idx += 1
        
        # Skip blank line(s)
        while line_idx < len(lines) and not lines[line_idx]:
            line_idx += 1
        
        # Parse target landmarks
        dst_pts = []
        for i in range(num_landmarks):
            if line_idx >= len(lines):
                raise TransformParseError(f"Missing target landmark {i+1}")
            parts = lines[line_idx].split()
            if len(parts) < 2:
                raise TransformParseError(f"Invalid target landmark at line {line_idx}: {lines[line_idx]}")
            try:
                x, y = float(parts[0]), float(parts[1])
                dst_pts.append([x, y])
            except ValueError:
                raise TransformParseError(f"Non-numeric coordinates at line {line_idx}: {lines[line_idx]}")
            line_idx += 1
        
        return {
            'transform_type': transform_type,
            'src_pts': np.array(src_pts, dtype=float),
            'dst_pts': np.array(dst_pts, dtype=float),
            'source_img': source_img,
            'target_img': target_img,
            'two_stack_align': two_stack_align
        }
        
    except IndexError as e:
        raise TransformParseError(f"File ended unexpectedly: {e}")
    except Exception as e:
        if isinstance(e, (TransformParseError, FileNotFoundError)):
            raise
        raise TransformParseError(f"Unexpected error parsing file: {e}")


def validate_transform_data(data: Dict) -> None:
    """
    Validate parsed transform data.
    
    Args:
        data: Dictionary returned by parse_multistackreg_file
        
    Raises:
        TransformParseError: If validation fails
    """
    required_keys = ['transform_type', 'src_pts', 'dst_pts', 'source_img', 'target_img']
    for key in required_keys:
        if key not in data:
            raise TransformParseError(f"Missing required key: {key}")
    
    src_pts = data['src_pts']
    dst_pts = data['dst_pts']
    
    if src_pts.shape != dst_pts.shape:
        raise TransformParseError(f"Source and target landmark shapes don't match: {src_pts.shape} vs {dst_pts.shape}")
    
    if src_pts.shape[1] != 2:
        raise TransformParseError(f"Landmarks must be 2D, got shape: {src_pts.shape}")
    
    if len(src_pts) < 1:
        raise TransformParseError("Must have at least 1 landmark pair")

