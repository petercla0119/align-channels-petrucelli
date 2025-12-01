import argparse
import cv2
import numpy as np

from .io_utils import load_multichannel_tiff, extract_channel
from .registration import register_images_ecc


def compute_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.flatten().astype(np.float64)
    b = img_b.flatten().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_test(fixed_path: str, moving_path: str, angle_adjustments: list[float]) -> None:
    fixed = load_multichannel_tiff(fixed_path, expected_channels=2)
    moving = load_multichannel_tiff(moving_path, expected_channels=2)

    fixed_dapi = extract_channel(fixed, 1)
    moving_dapi = extract_channel(moving, 1)

    warp_matrix, corr = register_images_ecc(
        fixed_dapi,
        moving_dapi,
        transform_type='RIGID_BODY',
        use_phase_init=True
    )

    print(f"\nOriginal ECC correlation: {corr:.6f}")
    print(f"Original warp matrix:\n{warp_matrix}\n")

    angle = np.degrees(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]))
    tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
    print(f"Rotation angle: {angle:.4f}°")
    print(f"Translation: ({tx:.2f}, {ty:.2f}) pixels\n")

    h, w = fixed_dapi.shape
    aligned = cv2.warpAffine(
        moving_dapi.astype(np.float32),
        warp_matrix.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR
    )
    base_corr = compute_correlation(fixed_dapi, aligned)
    print(f"Correlation after applying ECC warp: {base_corr:.6f}\n")

    print("Testing small rotation adjustments:")
    for delta_angle in angle_adjustments:
        adjusted_angle = angle + delta_angle
        angle_rad = np.radians(adjusted_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        adjusted_matrix = np.array([
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty]
        ], dtype=np.float32)
        adjusted = cv2.warpAffine(
            moving_dapi.astype(np.float32),
            adjusted_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        corr_adj = compute_correlation(fixed_dapi, adjusted)
        marker = ""
        if abs(delta_angle) < 1e-6:
            marker = " <-- baseline"
        elif corr_adj > base_corr:
            marker = " <-- improved"
        print(f"  Angle {adjusted_angle:7.3f}°: correlation {corr_adj:.6f}{marker}")

    print("\nGenerating difference map 'alignment_difference.png' ...")
    fixed_norm = cv2.normalize(fixed_dapi, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    aligned_norm = cv2.normalize(aligned, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    diff = np.abs(fixed_norm - aligned_norm)
    heatmap = (diff * 255).astype(np.uint8)
    cv2.imwrite('alignment_difference.png', heatmap)
    print("Saved difference heatmap to alignment_difference.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation center diagnostics for ECC alignment.")
    parser.add_argument("--fixed", required=True, help="Path to fixed/reference TIFF")
    parser.add_argument("--moving", required=True, help="Path to moving TIFF")
    parser.add_argument(
        "--angles",
        nargs="*",
        type=float,
        default=[-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5],
        help="List of angle adjustments (degrees) to test"
    )
    args = parser.parse_args()
    run_test(args.fixed, args.moving, args.angles)

