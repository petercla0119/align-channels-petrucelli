# test_rotation_center.py
import numpy as np
import cv2
from src.io_utils import load_multichannel_tiff, extract_channel
from src.registration import register_images_ecc

fixed_path = "path/to/fixed.tif"
moving_path = "path/to/moving.tif"

fixed = load_multichannel_tiff(fixed_path, 2)
moving = load_multichannel_tiff(moving_path, 2)

fixed_dapi = extract_channel(fixed, 1)
moving_dapi = extract_channel(moving, 1)

# Get ECC result
warp_matrix, corr = register_images_ecc(fixed_dapi, moving_dapi, 
                                         transform_type='RIGID_BODY')

print(f"Original correlation: {corr:.6f}")
print(f"Original matrix:\n{warp_matrix}")

# Extract rotation angle and translation
angle = np.degrees(np.arctan2(warp_matrix[1,0], warp_matrix[0,0]))
print(f"\nRotation: {angle:.4f}°")
print(f"Translation: ({warp_matrix[0,2]:.2f}, {warp_matrix[1,2]:.2f})")

# Apply and measure correlation
h, w = fixed_dapi.shape
result = cv2.warpAffine(
    moving_dapi.astype(np.float32),
    warp_matrix.astype(np.float32),
    (w, h),
    flags=cv2.INTER_LINEAR
)

img_corr = np.corrcoef(fixed_dapi.flatten(), result.flatten())[0,1]
print(f"\nImage correlation after alignment: {img_corr:.6f}")

# Try manual rotation adjustments
print("\nTrying small rotation adjustments:")
for delta_angle in [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]:
    adjusted_angle = angle + delta_angle
    angle_rad = np.radians(adjusted_angle)
    
    # Rebuild warp matrix with adjusted angle
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Keep same translation
    adjusted_matrix = np.array([
        [cos_a, -sin_a, warp_matrix[0,2]],
        [sin_a,  cos_a, warp_matrix[1,2]]
    ], dtype=np.float32)
    
    result_adj = cv2.warpAffine(
        moving_dapi.astype(np.float32),
        adjusted_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR
    )
    
    corr_adj = np.corrcoef(fixed_dapi.flatten(), result_adj.flatten())[0,1]
    
    marker = " ←" if delta_angle == 0 else (" ← BETTER!" if corr_adj > img_corr else "")
    print(f"  Angle {adjusted_angle:7.3f}°: correlation {corr_adj:.6f}{marker}")