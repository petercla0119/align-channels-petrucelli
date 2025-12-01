"""
Rigid-body transformation computation and application.

Ported from MultiStackReg Java implementation (EPFL/BIG).
Implements transformation matrix computation from landmark coordinates.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.ndimage import affine_transform


class TransformError(Exception):
    """Raised when transformation computation or application fails."""
    pass


def get_transformation_matrix(
    from_coord: np.ndarray,
    to_coord: np.ndarray,
    transformation_type: str
) -> np.ndarray:
    """
    Compute transformation matrix from landmark coordinates.
    
    Ported from MultiStackReg_.java getTransformationMatrix() (lines 1328-1432).
    
    Args:
        from_coord: Source landmarks, shape (N, 2) where N depends on transform type
        to_coord: Target landmarks, shape (N, 2)
        transformation_type: One of 'TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE'
        
    Returns:
        3x3 transformation matrix in homogeneous coordinates
        
    Raises:
        TransformError: If landmarks are invalid or computation fails
    """
    if from_coord.shape != to_coord.shape:
        raise TransformError(f"Landmark shapes don't match: {from_coord.shape} vs {to_coord.shape}")
    
    if from_coord.shape[1] != 2:
        raise TransformError(f"Landmarks must be 2D, got shape: {from_coord.shape}")
    
    matrix = np.eye(3, dtype=float)
    
    if transformation_type == 'TRANSLATION':
        # Case 0: Pure translation (1 landmark)
        if len(from_coord) < 1:
            raise TransformError("TRANSLATION requires at least 1 landmark")
        matrix[0, 2] = to_coord[0, 0] - from_coord[0, 0]
        matrix[1, 2] = to_coord[0, 1] - from_coord[0, 1]
        
    elif transformation_type == 'RIGID_BODY':
        # Case 1: Rigid body (rotation + translation, 3 landmarks)
        if len(from_coord) < 3:
            raise TransformError("RIGID_BODY requires at least 3 landmarks")
        
        # Compute rotation angle from landmarks 1 and 2
        angle = (
            np.arctan2(from_coord[2, 0] - from_coord[1, 0], 
                      from_coord[2, 1] - from_coord[1, 1]) -
            np.arctan2(to_coord[2, 0] - to_coord[1, 0],
                      to_coord[2, 1] - to_coord[1, 1])
        )
        
        c = np.cos(angle)
        s = np.sin(angle)
        
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[0, 2] = to_coord[0, 0] - c * from_coord[0, 0] + s * from_coord[0, 1]
        matrix[1, 0] = s
        matrix[1, 1] = c
        matrix[1, 2] = to_coord[0, 1] - s * from_coord[0, 0] - c * from_coord[0, 1]
        
    elif transformation_type == 'SCALED_ROTATION':
        # Case 2: Scaled rotation (2 landmarks)
        if len(from_coord) < 2:
            raise TransformError("SCALED_ROTATION requires at least 2 landmarks")
        
        # Build coefficient matrix
        a = np.array([
            [from_coord[0, 0], from_coord[0, 1], 1.0],
            [from_coord[1, 0], from_coord[1, 1], 1.0],
            [from_coord[0, 1] - from_coord[1, 1] + from_coord[1, 0],
             from_coord[1, 0] + from_coord[1, 1] - from_coord[0, 0], 1.0]
        ])
        
        # Invert using Gaussian elimination (in-place)
        a_inv = invert_gauss(a.copy())
        
        # Solve for first row of transformation matrix
        v = np.array([
            to_coord[0, 0],
            to_coord[1, 0],
            to_coord[0, 1] - to_coord[1, 1] + to_coord[1, 0]
        ])
        matrix[0, :] = a_inv @ v
        
        # Solve for second row
        v = np.array([
            to_coord[0, 1],
            to_coord[1, 1],
            to_coord[1, 0] + to_coord[1, 1] - to_coord[0, 0]
        ])
        matrix[1, :] = a_inv @ v
        
    elif transformation_type == 'AFFINE':
        # Case 3: Full affine (3 landmarks)
        if len(from_coord) < 3:
            raise TransformError("AFFINE requires at least 3 landmarks")
        
        # Build coefficient matrix
        a = np.array([
            [from_coord[0, 0], from_coord[0, 1], 1.0],
            [from_coord[1, 0], from_coord[1, 1], 1.0],
            [from_coord[2, 0], from_coord[2, 1], 1.0]
        ])
        
        # Invert using Gaussian elimination
        a_inv = invert_gauss(a.copy())
        
        # Solve for first row
        v = np.array([to_coord[0, 0], to_coord[1, 0], to_coord[2, 0]])
        matrix[0, :] = a_inv @ v
        
        # Solve for second row
        v = np.array([to_coord[0, 1], to_coord[1, 1], to_coord[2, 1]])
        matrix[1, :] = a_inv @ v
        
    else:
        raise TransformError(f"Unknown transformation type: {transformation_type}")
    
    return matrix


def invert_gauss(matrix: np.ndarray) -> np.ndarray:
    """
    Invert a matrix using Gaussian elimination with partial pivoting.
    
    Ported from MultiStackReg_.java invertGauss() (lines 1435-1507).
    This modifies the input matrix in-place and returns the inverse.
    
    Args:
        matrix: Square matrix to invert (will be modified)
        
    Returns:
        Inverse matrix
        
    Raises:
        TransformError: If matrix is singular
    """
    n = matrix.shape[0]
    if matrix.shape[1] != n:
        raise TransformError(f"Matrix must be square, got shape: {matrix.shape}")
    
    inverse = np.zeros((n, n), dtype=float)
    
    # Normalize rows
    for i in range(n):
        max_val = matrix[i, 0]
        abs_max = abs(max_val)
        
        for j in range(n):
            inverse[i, j] = 0.0
            if abs_max < abs(matrix[i, j]):
                max_val = matrix[i, j]
                abs_max = abs(max_val)
        
        if abs_max < 1e-10:
            raise TransformError(f"Matrix is singular (row {i} is zero)")
        
        inverse[i, i] = 1.0 / max_val
        matrix[i, :] /= max_val
    
    # Forward elimination with partial pivoting
    for j in range(n):
        # Find pivot
        max_val = matrix[j, j]
        abs_max = abs(max_val)
        k = j
        
        for i in range(j + 1, n):
            if abs_max < abs(matrix[i, j]):
                max_val = matrix[i, j]
                abs_max = abs(max_val)
                k = i
        
        # Swap rows if needed
        if k != j:
            matrix[[j, k], j:] = matrix[[k, j], j:]
            inverse[[j, k], :] = inverse[[k, j], :]
        
        if abs_max < 1e-10:
            raise TransformError(f"Matrix is singular (column {j})")
        
        # Normalize pivot row
        for col in range(j + 1):
            inverse[j, col] /= max_val
        for col in range(j + 1, n):
            matrix[j, col] /= max_val
            inverse[j, col] /= max_val
        
        # Eliminate below pivot
        for i in range(j + 1, n):
            factor = matrix[i, j]
            for col in range(j + 1):
                inverse[i, col] -= factor * inverse[j, col]
            for col in range(j + 1, n):
                matrix[i, col] -= factor * matrix[j, col]
                inverse[i, col] -= factor * inverse[j, col]
    
    # Back substitution
    for j in range(n - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            factor = matrix[i, j]
            for col in range(j + 1):
                inverse[i, col] -= factor * inverse[j, col]
            for col in range(j + 1, n):
                matrix[i, col] -= factor * matrix[j, col]
                inverse[i, col] -= factor * inverse[j, col]
    
    return inverse


def rigid_from_landmarks(src_pts: np.ndarray, dst_pts: np.ndarray) -> Dict:
    """
    Compute rigid-body transform parameters from landmark pairs.
    
    This is a convenience wrapper around get_transformation_matrix that
    returns the transform in a more accessible format.
    
    Args:
        src_pts: Source landmarks, shape (N, 2)
        dst_pts: Target landmarks, shape (N, 2)
        
    Returns:
        Dictionary with keys:
            - 'm00', 'm01', 'm02': First row of 2x3 affine matrix
            - 'm10', 'm11', 'm12': Second row of 2x3 affine matrix
            - 'angle': Rotation angle in degrees
            - 'matrix_2x3': 2x3 numpy array
            - 'matrix_3x3': 3x3 numpy array (homogeneous)
    """
    # Determine transform type based on number of points
    n_pts = len(src_pts)
    if n_pts == 1:
        transform_type = 'TRANSLATION'
    elif n_pts == 2:
        transform_type = 'SCALED_ROTATION'
    elif n_pts >= 3:
        transform_type = 'RIGID_BODY'
    else:
        raise TransformError("Need at least 1 landmark pair")
    
    matrix_3x3 = get_transformation_matrix(src_pts, dst_pts, transform_type)
    
    # Extract 2x3 affine matrix
    matrix_2x3 = matrix_3x3[:2, :]
    
    m00, m01, m02 = matrix_2x3[0, :]
    m10, m11, m12 = matrix_2x3[1, :]
    
    # Compute rotation angle
    angle_rad = np.arctan2(m10, m00)
    angle_deg = np.degrees(angle_rad)
    
    return {
        'm00': m00, 'm01': m01, 'm02': m02,
        'm10': m10, 'm11': m11, 'm12': m12,
        'angle': angle_deg,
        'matrix_2x3': matrix_2x3,
        'matrix_3x3': matrix_3x3,
        'transform_type': transform_type
    }


def apply_rigid_transform(
    image: np.ndarray,
    transform_params: Dict,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0,
    fiji_compatible: bool = True
) -> np.ndarray:
    """
    Apply rigid-body transformation to a 2D image.
    
    Matches FIJI/ImageJ behavior: rotate around center, then translate.
    (AGENT.md lines 130-132)
    
    Args:
        image: 2D numpy array (H, W)
        transform_params: Dict with 'matrix_2x3', 'm00'-'m12', or 'angle' keys
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)
        mode: How to handle borders ('constant', 'reflect', 'wrap', etc.)
        cval: Fill value for constant mode
        fiji_compatible: If True, use FIJI's rotate-then-translate approach (default)
        
    Returns:
        Transformed image, same shape as input
        
    Raises:
        TransformError: If image or parameters are invalid
    """
    from scipy.ndimage import rotate, shift
    
    if image.ndim != 2:
        raise TransformError(f"Image must be 2D, got shape: {image.shape}")
    
    # Extract parameters - always compute all values
    if 'matrix_2x3' in transform_params:
        matrix = np.array(transform_params['matrix_2x3'], dtype=float)
        m00, m01, tx = matrix[0]
        m10, m11, ty = matrix[1]
        angle_deg = np.degrees(np.arctan2(m10, m00))
    elif 'warp_matrix_2x3' in transform_params:
        matrix = np.array(transform_params['warp_matrix_2x3'], dtype=float)
        m00, m01, tx = matrix[0]
        m10, m11, ty = matrix[1]
        angle_deg = np.degrees(np.arctan2(m10, m00))
    elif 'm00' in transform_params:
        m00 = transform_params['m00']
        m01 = transform_params['m01']
        tx = transform_params['m02']
        m10 = transform_params['m10']
        m11 = transform_params['m11']
        ty = transform_params['m12']
        angle_deg = np.degrees(np.arctan2(m10, m00))
    elif 'angle' in transform_params:
        angle_deg = transform_params['angle']
        tx = transform_params.get('m02', 0.0)
        ty = transform_params.get('m12', 0.0)
        # Build rotation matrix from angle
        angle_rad = np.radians(angle_deg)
        m00 = np.cos(angle_rad)
        m01 = -np.sin(angle_rad)
        m10 = np.sin(angle_rad)
        m11 = np.cos(angle_rad)
    else:
        raise TransformError("Transform params must contain 'matrix_2x3', 'm00'-'m12', or 'angle' keys")
    
    if fiji_compatible:
        # Use OpenCV's warpAffine with inverse mapping (WARP_INVERSE_MAP)
        # This matches FIJI's behavior correctly
        import cv2
        
        h, w = image.shape
        
        # Build 2x3 affine matrix
        matrix = np.array([
            [m00, m01, tx],
            [m10, m11, ty]
        ], dtype=np.float32)
        
        # Convert image to float32 for OpenCV
        image_f32 = image.astype(np.float32)
        
        # Apply transform with INVERSE mapping
        # This is the key: WARP_INVERSE_MAP tells OpenCV to use the inverse transform
        # Remove WARP_INVERSE_MAP - it's inverting an already-correct transform
        transformed = cv2.warpAffine(
            image_f32,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,  # ← Just use LINEAR interpolation
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(cval)
        )
        
        # Convert back to original dtype
        transformed = transformed.astype(image.dtype)
        
    else:
        # Single affine transform (original method, kept for compatibility)
        matrix = np.array([
            [m00, m01, tx],
            [m10, m11, ty]
        ])
        
        R = matrix[:, :2]
        t = matrix[:, 2]
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise TransformError("Transformation matrix is singular")
        
        t_inv = -R_inv @ t
        
        transformed = affine_transform(
            image,
            R_inv,
            offset=t_inv,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(order > 1)
        )
    
    return transformed


def create_affine_matrix_2x3(params: Dict) -> np.ndarray:
    """
    Create 2x3 affine matrix from parameters.
    
    Args:
        params: Dict with 'm00', 'm01', 'm02', 'm10', 'm11', 'm12' keys
        
    Returns:
        2x3 numpy array
    """
    return np.array([
        [params['m00'], params['m01'], params['m02']],
        [params['m10'], params['m11'], params['m12']]
    ])


def create_affine_matrix_3x3(params: Dict) -> np.ndarray:
    """
    Create 3x3 homogeneous affine matrix from parameters.
    
    Args:
        params: Dict with 'm00', 'm01', 'm02', 'm10', 'm11', 'm12' keys
        
    Returns:
        3x3 numpy array with bottom row [0, 0, 1]
    """
    return np.array([
        [params['m00'], params['m01'], params['m02']],
        [params['m10'], params['m11'], params['m12']],
        [0.0, 0.0, 1.0]
    ])

