"""The sensor (fast) track: ARKit poses and LiDAR depth from a Stray Scanner recording.

ARKit's poses are metric and gravity-aligned, so this track needs no scale reference. It
cannot simply hand those poses to MASt3R — `init['cam2w']` reaches an unimplemented assert
in mast3r.cloud_opt.sparse_ga — so MASt3R solves poses on its own and its reconstruction is
mapped onto ARKit's frame afterwards, by fitting a similarity transform between the two
camera trajectories over the same keyframes.

The LiDAR only reaches about 5 m, so its cloud covers the near field and MASt3R's fills in
whatever lies beyond it.
"""

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..ingest.stray import find_recording, read_confidence, read_depth_m, read_odometry
from .align import robust_similarity
from .export import Reconstruction
from .lidar import sample_colors, scale_intrinsics, unproject
from .mast3r_sfm import voxel_downsample

# ARKit's world frame is gravity-aligned, so up is +Y by construction.
WORLD_UP = np.array([0.0, 1.0, 0.0])
# Above this trajectory residual the two reconstructions disagree enough that the scene is
# not treated as metric: a navmesh carved from it would be confidently wrong.
RESIDUAL_LIMIT_M = 0.15
# The merged cloud is metric, so the voxel is an absolute size rather than a fraction of an
# arbitrary scene scale.
VOXEL_M = 0.01
MAX_DENSE_POINTS = 2_000_000


def reconstruct(
    raw_dir: Path,
    frames: list[Path],
    source_indices: list[int],
    mast3r: Callable[[], Reconstruction] | None = None,
) -> tuple[Reconstruction, dict]:
    """Builds a metric reconstruction from the recording in `raw_dir`.

    `source_indices` are the recording's own frame numbers for `frames`, which ingest keeps
    because blur and motion filtering discard most frames. `mast3r`, when given, is called
    to produce the up-to-scale vision reconstruction that fills in beyond the sensor's range.
    """
    recording = find_recording(raw_dir)
    if recording is None:
        raise RuntimeError(f"{raw_dir} is not a Stray Scanner recording (no odometry.csv or rgb.mp4).")
    if recording.depth is None or recording.confidence is None:
        raise RuntimeError(
            f"{raw_dir} has no depth/ or confidence/ folder, so there is no LiDAR geometry to "
            "ingest. Record with a LiDAR device, or use the plain-video track."
        )

    odometry = read_odometry(recording.odometry)
    rows = _rows_for(odometry, source_indices)
    cam_to_world = odometry.cam_to_world[rows]
    intrinsics = odometry.intrinsics[rows]

    points, colors = _lidar_cloud(recording, frames, source_indices, cam_to_world, intrinsics)
    lidar_count = len(points)
    report = {
        "track": "fast",
        "keyframes": len(frames),
        "lidar_points": lidar_count,
        "mast3r_points": 0,
        "residual_m": None,
        "inliers": None,
        "scale_source": "sensor",
        "world_up": WORLD_UP.tolist(),
        "warning": None,
    }

    if mast3r is not None:
        vision = mast3r()
        fit = robust_similarity(vision.cam_to_world[:, :3, 3], cam_to_world[:, :3, 3])
        report["residual_m"] = round(fit.residual_m, 6)
        report["inliers"] = int(fit.inliers.sum())
        report["mast3r_points"] = len(vision.points)
        if fit.residual_m > RESIDUAL_LIMIT_M:
            report["scale_source"] = "unscaled"
            report["warning"] = (
                f"The vision and sensor camera paths disagree by {fit.residual_m:.3f} m "
                f"(limit {RESIDUAL_LIMIT_M} m), so this residual is too large to call the scene "
                "metric. Treating it as unscaled; set a scale reference by hand, or recapture."
            )
        points = np.concatenate([points, fit.apply(vision.points)])
        colors = np.concatenate([colors, vision.colors])

    points, colors = voxel_downsample(
        points.astype(np.float32), colors.astype(np.uint8), VOXEL_M, MAX_DENSE_POINTS
    )
    recon = Reconstruction(
        image_names=[f.name for f in frames],
        intrinsics=intrinsics,
        cam_to_world=cam_to_world,
        points=points,
        colors=colors,
    )
    return recon, report


def _rows_for(odometry, source_indices: list[int]) -> np.ndarray:
    """Positions in odometry.csv of the recording frames the keyframes came from."""
    lookup = {int(frame): row for row, frame in enumerate(odometry.frames)}
    missing = [i for i in source_indices if i not in lookup]
    if missing:
        raise RuntimeError(
            f"odometry.csv has no row for frame(s) {missing[:5]}; the recording and the "
            "extracted frames disagree."
        )
    return np.array([lookup[i] for i in source_indices])


def _lidar_cloud(recording, frames, source_indices, cam_to_world, intrinsics):
    points, colors = [], []
    for frame, index, c2w, K_color in zip(frames, source_indices, cam_to_world, intrinsics):
        depth_path = recording.depth / f"{index:06d}.png"
        confidence_path = recording.confidence / f"{index:06d}.png"
        for path in (depth_path, confidence_path):
            if not path.is_file():
                raise RuntimeError(f"The recording is missing {path.parent.name}/{path.name}.")

        depth = read_depth_m(depth_path)
        confidence = read_confidence(confidence_path)
        image = cv2.imread(str(frame))
        if image is None:
            raise RuntimeError(f"Could not read keyframe {frame}")

        color_size = (image.shape[1], image.shape[0])
        K_depth = scale_intrinsics(K_color, color_size, (depth.shape[1], depth.shape[0]))
        frame_points = unproject(depth, confidence, K_depth, c2w)
        points.append(frame_points)
        colors.append(sample_colors(frame_points, image, K_color, c2w))

    return np.concatenate(points), np.concatenate(colors)


def write_world_up(path: Path, up: np.ndarray = WORLD_UP) -> None:
    """Records the known up direction so `navmesh` can fix the floor's normal to gravity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"world_up": list(map(float, up))}, indent=2))
