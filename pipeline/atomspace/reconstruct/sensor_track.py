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
from scipy.spatial import cKDTree
from PIL import Image, ImageOps

from ..ingest.stray import find_sidecars, read_confidence, read_depth_m, read_odometry
from .align import MIN_POINTS as MIN_POINTS_FOR_FIT
from .align import robust_similarity
from .export import Reconstruction
from .lidar import MAX_RANGE_M, sample_colors, scale_intrinsics, unproject
from .mast3r_sfm import to_network_pixels, voxel_downsample

# ARKit's world frame is gravity-aligned, so up is +Y by construction.
WORLD_UP = np.array([0.0, 1.0, 0.0])
# Above this trajectory residual the two reconstructions disagree enough that the scene is
# not treated as metric: a navmesh carved from it would be confidently wrong.
RESIDUAL_LIMIT_M = 0.15
# RANSAC reports its residual over the inliers, so a reconstruction that drifted can report a
# better number than one that did not: the smaller the consensus, the tighter it looks. The
# size of that consensus is therefore part of the gate, not just the distance.
MIN_INLIER_SHARE = 0.6
# Median distance from a LiDAR point to the nearest mapped MASt3R point. Both describe the
# same surfaces wherever the sensor reached, and unlike the trajectory residual this can see
# a mistaken pose convention or a mis-scaled depth unprojection, neither of which moves a
# camera centre. Reported and warned on, not gated: the threshold is a guess until a real
# capture calibrates it, and refusing a good capture is worse than printing a warning.
AGREEMENT_LIMIT_M = 0.25
AGREEMENT_SAMPLE = 20_000
# The merged cloud is metric, so the voxel is an absolute size rather than a fraction of an
# arbitrary scene scale.
VOXEL_M = 0.01
MAX_DENSE_POINTS = 2_000_000


def reconstruct(
    raw_dir: Path,
    frames: list[Path],
    source_indices: list[int],
    mast3r: Callable[[], Reconstruction] | None = None,
    capture_size: tuple[int, int] | None = None,
) -> tuple[Reconstruction, dict]:
    """Builds a metric reconstruction from the recording in `raw_dir`.

    `source_indices` are the recording's own frame numbers for `frames`, which ingest keeps
    because blur and motion filtering discard most frames. `mast3r`, when given, is called
    to produce the up-to-scale vision reconstruction that fills in beyond the sensor's range.
    `capture_size` is the recording's own (width, height), which the odometry's intrinsics
    are quoted in; without it the keyframes are assumed to be at that resolution.
    """
    recording = find_sidecars(raw_dir)
    if recording is None:
        raise RuntimeError(f"{raw_dir} holds no odometry.csv, so it is not a Stray Scanner recording.")
    if recording.depth is None or recording.confidence is None:
        raise RuntimeError(
            f"{raw_dir} has no depth/ or confidence/ folder, so there is no LiDAR geometry to "
            "ingest. Record with a LiDAR device, or use the plain-video track."
        )

    odometry = read_odometry(recording.odometry)
    rows = rows_for(odometry, source_indices)
    cam_to_world = odometry.cam_to_world[rows]
    intrinsics = odometry.intrinsics[rows]

    points, colors, skipped = _lidar_cloud(
        recording, frames, source_indices, cam_to_world, intrinsics, capture_size
    )
    report = {
        "track": "fast",
        "keyframes": len(frames),
        "frames_without_depth": skipped,
        "lidar_points": len(points),
        "mast3r_points": 0,
        "residual_m": None,
        "inliers": None,
        "cloud_agreement_m": None,
        "scale_source": "sensor",
        "world_up": WORLD_UP.tolist(),
        "warning": None,
    }

    if mast3r is not None:
        vision = mast3r()
        try:
            fit = robust_similarity(vision.cam_to_world[:, :3, 3], cam_to_world[:, :3, 3])
        except RuntimeError as refused:
            # The sensor cloud is metric and gravity-aligned on its own, so a fit that cannot
            # be made costs the far field, not the scene.
            report["warning"] = (
                f"{refused} MASt3R's reconstruction could not be placed, so this scene holds "
                "only what the LiDAR reached."
            )
            fit = None
        if fit is not None:
            mapped = fit.apply(vision.points)
            agreement = cloud_agreement(points, mapped)
            report["residual_m"] = round(fit.residual_m, 6)
            report["inliers"] = int(fit.inliers.sum())
            report["mast3r_points"] = len(vision.points)
            report["cloud_agreement_m"] = round(agreement, 6)
            report["warning"] = _quality_warning(report, len(frames), agreement)
            if _is_unreliable(report, len(frames)):
                report["scale_source"] = "unscaled"
            points = np.concatenate([points, mapped])
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


def _is_unreliable(report: dict, cameras: int) -> bool:
    """Whether the two reconstructions agree closely enough to call the result metric."""
    return (
        report["residual_m"] > RESIDUAL_LIMIT_M
        or report["inliers"] < _needed_inliers(cameras)
    )


def _needed_inliers(cameras: int) -> int:
    return max(MIN_POINTS_FOR_FIT, round(MIN_INLIER_SHARE * cameras))


def _quality_warning(report: dict, cameras: int, agreement: float) -> str | None:
    if report["residual_m"] > RESIDUAL_LIMIT_M:
        return (
            f"The vision and sensor camera paths disagree by {report['residual_m']:.3f} m "
            f"(limit {RESIDUAL_LIMIT_M} m), which is too large to call the scene metric. "
            "Treating it as unscaled; set a scale reference by hand, or recapture."
        )
    if report["inliers"] < _needed_inliers(cameras):
        return (
            f"Only {report['inliers']} of {cameras} cameras agree between the vision and "
            f"sensor paths ({_needed_inliers(cameras)} needed), so the "
            f"{report['residual_m']:.3f} m residual is measured over too few of them to mean "
            "anything. Treating the scene as unscaled."
        )
    if agreement > AGREEMENT_LIMIT_M:
        return (
            f"The two clouds sit {agreement:.3f} m apart where both should see the same "
            f"surfaces (expected under {AGREEMENT_LIMIT_M} m). The camera paths agree, so this "
            "points at the depth unprojection or the pose convention rather than at drift. "
            "The scene is still marked metric; check it in the viewer before trusting it."
        )
    return None


def cloud_agreement(lidar: np.ndarray, mapped: np.ndarray, seed: int = 0) -> float:
    """Median distance from a sample of LiDAR points to the nearest mapped MASt3R point.

    The trajectory residual is computed from camera centres, which neither a transposed pose
    convention nor a mis-scaled depth unprojection disturbs. This does see both, because it
    compares the geometry those produce rather than the path the camera took.
    """
    if not len(lidar) or not len(mapped):
        return float("nan")
    rng = np.random.default_rng(seed)
    sample = lidar[rng.choice(len(lidar), min(AGREEMENT_SAMPLE, len(lidar)), replace=False)]
    distances, _ = cKDTree(mapped).query(sample, k=1)
    return float(np.median(distances))


def rows_for(odometry, source_indices: list[int]) -> np.ndarray:
    """Positions in odometry.csv of the recording frames the keyframes came from."""
    lookup = {int(frame): row for row, frame in enumerate(odometry.frames)}
    missing = [i for i in source_indices if i not in lookup]
    if missing:
        raise RuntimeError(
            f"odometry.csv has no row for frame(s) {missing[:5]}; the recording and the "
            "extracted frames disagree."
        )
    return np.array([lookup[i] for i in source_indices])


def _lidar_cloud(recording, frames, source_indices, cam_to_world, intrinsics, capture_size):
    points, colors, skipped = [], [], 0
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

        # odometry.csv quotes intrinsics in capture pixels, and ingest is free to have written
        # the keyframe smaller, so neither resolution may be assumed to be the other.
        keyframe_size = (image.shape[1], image.shape[0])
        source_size = capture_size or keyframe_size
        K_depth = scale_intrinsics(K_color, source_size, (depth.shape[1], depth.shape[0]))
        K_keyframe = scale_intrinsics(K_color, source_size, keyframe_size)
        try:
            frame_points = unproject(depth, confidence, K_depth, c2w)
        except RuntimeError:
            # Pointed down a corridor or out of a window: this frame measured nothing, which
            # is no reason to discard the frames that did.
            skipped += 1
            continue
        points.append(frame_points)
        colors.append(sample_colors(frame_points, image, K_keyframe, c2w))

    if not points:
        raise RuntimeError(
            f"No depth in range on any of the {len(frames)} keyframes: nothing within "
            f"{MAX_RANGE_M} m at usable confidence. The LiDAR only reaches about 5 m, so a "
            "large space needs passes closer to its surfaces."
        )
    if skipped:
        print(f"{skipped} keyframe(s) measured no depth in range and contributed no points.")
    return np.concatenate(points), np.concatenate(colors), skipped


def write_world_up(path: Path, up: np.ndarray = WORLD_UP) -> None:
    """Records the known up direction so `navmesh` can fix the floor's normal to gravity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"world_up": list(map(float, up))}, indent=2))


def mast3r_init(frames: list[Path], intrinsics: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """MASt3R's per-image `init`, carrying the camera's measured intrinsics.

    `sparse_scene_optimizer` accepts three keys, and only one of them is usable here:

    - `intrinsics` works, and ARKit's are measured rather than solved, so they are a better
      starting point than MASt3R's own estimate. This is what the function returns.
    - `cam2w` reaches `assert False, 'inverse kinematic chain not yet implemented'`
      (sparse_ga.py:247) on the pinned commit and on upstream main, which is why this track
      overrides the poses after alignment instead of seeding them.
    - `depthmap` replaces `core_depth[idx]`, which `condense_data` fills from each image's
      canonical-view *anchors*: a one-dimensional vector over pixels chosen inside
      `sparse_global_alignment`, whose length and positions the caller cannot know in
      advance. Seeding LiDAR depth there would mean substituting it as those anchors are
      built, inside `prepare_canonical_data`, rather than passing a value in. Not attempted.
    """
    out = {}
    for frame, K in zip(frames, intrinsics):
        width, height = ImageOps.exif_transpose(Image.open(frame)).size
        out[str(frame)] = {"intrinsics": to_network_pixels(K, (width, height))}
    return out
