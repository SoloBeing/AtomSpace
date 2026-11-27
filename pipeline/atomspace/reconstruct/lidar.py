"""Turns ARKit's LiDAR depth maps into coloured world points.

The depth map is 256x192 while the colour frame is 1920x1440, and `odometry.csv` reports
intrinsics for the colour frame, so they must be rescaled before anything is unprojected.
Getting that wrong produces a point cloud that still looks like a room and is simply the
wrong size, which then silently poisons the navmesh — hence `scale_intrinsics` and the
tests around it.
"""

import numpy as np

# Apple quotes about 5 m for the sensor; past that the returned depth is unreliable even
# when ARKit reports high confidence.
MAX_RANGE_M = 5.0
# ARKit's levels are 0 (low), 1 (medium) and 2 (high). Low is noise.
MIN_CONFIDENCE = 1


def scale_intrinsics(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """`K` rescaled from one image resolution to another. Sizes are (width, height).

    Uses the pixel-centre convention — a pixel is sampled at its middle, so the principal
    point maps as (c + 0.5) * s - 0.5 — matching mast3r_sfm.to_original_pixels.
    """
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    out = np.eye(3)
    out[0, 0], out[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    out[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return out


def unproject(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    K_depth: np.ndarray,
    cam_to_world: np.ndarray,
    min_confidence: int = MIN_CONFIDENCE,
    max_range_m: float = MAX_RANGE_M,
) -> np.ndarray:
    """(M, 3) world points from one depth frame, keeping only pixels worth believing.

    `K_depth` must already be at the depth map's resolution. `cam_to_world` is OpenCV, so
    +Z is the viewing direction.
    """
    if confidence.shape != depth_m.shape:
        raise ValueError(
            f"Confidence shape {confidence.shape} does not match the depth shape {depth_m.shape}"
        )
    keep = (confidence >= min_confidence) & (depth_m > 0) & (depth_m <= max_range_m)
    if not keep.any():
        raise RuntimeError(
            f"No depth in range: nothing within {max_range_m} m at confidence {min_confidence} "
            "or better. The LiDAR only reaches about 5 m, so a large hall needs passes closer "
            "to its surfaces."
        )

    rows, cols = np.nonzero(keep)
    z = depth_m[rows, cols].astype(np.float64)
    camera = np.c_[
        (cols - K_depth[0, 2]) * z / K_depth[0, 0],
        (rows - K_depth[1, 2]) * z / K_depth[1, 1],
        z,
    ]
    return camera @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def sample_colors(
    points: np.ndarray, image: np.ndarray, K_color: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
    """(M, 3) uint8 RGB for `points`, read from the keyframe they were measured in.

    `image` is BGR as OpenCV loads it; the result is RGB, which is what write_ply stores.
    """
    world_to_cam = np.linalg.inv(cam_to_world)
    camera = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    depth = np.maximum(camera[:, 2], 1e-9)
    cols = np.rint(camera[:, 0] * K_color[0, 0] / depth + K_color[0, 2]).astype(int)
    rows = np.rint(camera[:, 1] * K_color[1, 1] / depth + K_color[1, 2]).astype(int)
    height, width = image.shape[:2]
    bgr = image[np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)]
    return bgr[:, ::-1].astype(np.uint8)
