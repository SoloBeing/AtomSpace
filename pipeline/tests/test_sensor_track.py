import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from atomspace.config import ScenePaths
from atomspace.ingest.stray import ARKIT_TO_OPENCV
from atomspace.reconstruct.export import Reconstruction
from atomspace.reconstruct.scale import ScaleReference, metric_factor, save_reference
from atomspace.reconstruct.sensor_track import RESIDUAL_LIMIT_M, WORLD_UP, reconstruct

HEADER = (
    "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
    " distortion_center_x, distortion_center_y\n"
)
DEPTH_SIZE = (32, 24)  # (width, height), the same 4:3 the real 256x192 map has
COLOR_SIZE = (240, 180)
FOCAL = 180.0  # colour-frame pixels


def _poses(n=8):
    """A loop around a room at varying height, so the trajectory is not collinear."""
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.tile(np.eye(4), (n, 1, 1))
    for i, a in enumerate(angle):
        # Face inward, i.e. rotate about ARKit's up by the angle around the circle.
        out[i, :3, :3] = Rotation.from_rotvec(a * np.array([0.0, 1.0, 0.0])).as_matrix()
        out[i, :3, 3] = [2.0 * np.cos(a), 1.4 + 0.1 * (i % 3), 2.0 * np.sin(a)]
    return out


def _build_scene(tmp_path, poses):
    """A recording folder plus its extracted keyframes. Every depth pixel reads 2 m, so the
    cloud is one patch per camera at a known distance."""
    raw = tmp_path / "raw"
    frames_dir = tmp_path / "frames"
    (raw / "depth").mkdir(parents=True)
    (raw / "confidence").mkdir()
    frames_dir.mkdir()

    rows = []
    for i, pose in enumerate(poses):
        position = pose[:3, 3]
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x, y, z, w
        rows.append(
            f"{i / 30:.4f}, {i:06d}, {position[0]}, {position[1]}, {position[2]}, "
            f"{quaternion[0]}, {quaternion[1]}, {quaternion[2]}, {quaternion[3]}, "
            f"{FOCAL}, {FOCAL}, {(COLOR_SIZE[0] - 1) / 2}, {(COLOR_SIZE[1] - 1) / 2}, ,\n"
        )
        cv2.imwrite(str(raw / "depth" / f"{i:06d}.png"), np.full(DEPTH_SIZE[::-1], 2000, np.uint16))
        cv2.imwrite(str(raw / "confidence" / f"{i:06d}.png"), np.full(DEPTH_SIZE[::-1], 2, np.uint8))
        colour = np.zeros((*COLOR_SIZE[::-1], 3), np.uint8)
        colour[:, :] = (i * 10, 100, 200)  # BGR
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), colour)

    (raw / "odometry.csv").write_text(HEADER + "".join(rows))
    (raw / "rgb.mp4").write_bytes(b"")
    frames = sorted(frames_dir.glob("*.jpg"))
    return raw, frames, list(range(len(frames))), poses


@pytest.fixture
def scene(tmp_path):
    return _build_scene(tmp_path, _poses())


def _stub_mast3r(truth_points, poses, scale, seed=0, pose_noise_m=0.0):
    """A MASt3R whose reconstruction is the truth in its own arbitrary frame: rotated,
    translated and scaled, exactly the relationship the alignment has to undo."""
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    offset = np.array([4.0, -2.0, 9.0])
    rng = np.random.default_rng(seed)

    def to_mast3r(points):
        return (points - offset) @ rotation / scale

    def run():
        centres = to_mast3r(poses[:, :3, 3])
        if pose_noise_m:
            centres = centres + rng.normal(0, pose_noise_m / scale, centres.shape)
        cam_to_world = np.tile(np.eye(4), (len(poses), 1, 1))
        cam_to_world[:, :3, 3] = centres
        return Reconstruction(
            image_names=[f"{i:05d}.jpg" for i in range(len(poses))],
            intrinsics=np.tile(np.eye(3), (len(poses), 1, 1)),
            cam_to_world=cam_to_world,
            points=to_mast3r(truth_points).astype(np.float32),
            colors=np.zeros((len(truth_points), 3), np.uint8),
        )

    return run


def test_sensor_track_without_mast3r_uses_lidar_alone(scene):
    raw, frames, indices, poses = scene

    recon, report = reconstruct(raw, frames, indices, mast3r=None)

    assert len(recon.cam_to_world) == len(frames)
    assert len(recon.points) > 0
    assert report["residual_m"] is None
    assert report["mast3r_points"] == 0
    assert report["scale_source"] == "sensor"


def test_lidar_points_land_at_the_measured_depth(scene):
    """One camera, so every point's provenance is known. Depth reads a flat 2 m, so the
    nearest point sits 2 m away along the optical axis and the rest fan out behind it."""
    raw, frames, _, poses = scene

    recon, _ = reconstruct(raw, [frames[0]], [0], mast3r=None)

    distances = np.linalg.norm(recon.points - poses[0, :3, 3], axis=1)
    np.testing.assert_allclose(distances.min(), 2.0, atol=0.01)
    assert distances.min() >= 2.0  # 2 m is the depth along z, so nothing is closer


def test_poses_are_arkits_and_metric(scene):
    """MASt3R's poses here are deliberately wrong — identity rotations in a shrunken frame.
    The output must carry ARKit's."""
    raw, frames, indices, poses = scene
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)

    recon, _ = reconstruct(raw, frames, indices, mast3r=_stub_mast3r(truth.points, poses, scale=3.0))

    np.testing.assert_allclose(recon.cam_to_world, poses @ ARKIT_TO_OPENCV, atol=1e-9)


def test_mast3r_points_are_mapped_into_arkit_metres(scene):
    raw, frames, indices, poses = scene
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)

    recon, report = reconstruct(
        raw, frames, indices, mast3r=_stub_mast3r(truth.points, poses, scale=3.0)
    )

    assert report["residual_m"] < 1e-6
    assert report["scale_source"] == "sensor"
    assert report["mast3r_points"] > 0
    # Every mapped MASt3R point must coincide with a LiDAR point it was built from.
    extent = recon.points.max(axis=0) - recon.points.min(axis=0)
    np.testing.assert_allclose(extent, truth.points.max(axis=0) - truth.points.min(axis=0), atol=0.02)


def test_a_bad_residual_withholds_metric_scale(scene):
    """Quality first: the pipeline does not label geometry metric that it has just measured
    as badly aligned, so the scene stays recoverable by the manual scale tool instead."""
    raw, frames, indices, poses = scene
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)
    noisy = _stub_mast3r(truth.points, poses, scale=3.0, pose_noise_m=0.5)

    _, report = reconstruct(raw, frames, indices, mast3r=noisy)

    assert report["residual_m"] > RESIDUAL_LIMIT_M
    assert report["scale_source"] == "unscaled"
    assert "too large to call the scene metric" in report["warning"]


def test_report_names_arkits_world_up(scene):
    raw, frames, indices, _ = scene
    _, report = reconstruct(raw, frames, indices, mast3r=None)
    np.testing.assert_array_equal(report["world_up"], WORLD_UP)


def test_only_the_keyframes_odometry_rows_are_used(scene):
    """Ingest keeps a fraction of the frames, so the rows are selected by source index and
    not by position in the file."""
    raw, frames, _, poses = scene
    picked = [0, 3, 5, 6]

    recon, report = reconstruct(raw, [frames[i] for i in picked], picked, mast3r=None)

    assert report["keyframes"] == len(picked)
    np.testing.assert_allclose(recon.cam_to_world, poses[picked] @ ARKIT_TO_OPENCV, atol=1e-9)


def test_a_missing_depth_frame_is_named(scene):
    raw, frames, indices, _ = scene
    (raw / "depth" / "000003.png").unlink()

    with pytest.raises(RuntimeError, match="000003"):
        reconstruct(raw, frames, indices, mast3r=None)


def test_a_recording_without_depth_is_refused(scene, tmp_path):
    raw, frames, indices, _ = scene
    for png in (raw / "depth").glob("*.png"):
        png.unlink()
    (raw / "depth").rmdir()

    with pytest.raises(RuntimeError, match="depth"):
        reconstruct(raw, frames, indices, mast3r=None)


def test_intrinsics_come_from_the_odometry(scene):
    raw, frames, indices, _ = scene
    recon, _ = reconstruct(raw, frames, indices, mast3r=None)
    np.testing.assert_allclose(recon.intrinsics[:, 0, 0], FOCAL)
    np.testing.assert_allclose(recon.intrinsics[:, 0, 2], (COLOR_SIZE[0] - 1) / 2)


def test_report_is_json_serialisable(scene):
    raw, frames, indices, _ = scene
    _, report = reconstruct(raw, frames, indices, mast3r=None)
    json.loads(json.dumps(report))


def _fast_scene(tmp_path, report=None):
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    paths.ingest_meta.write_text(json.dumps({"track": "fast", "keyframes": ["00001.jpg"]}))
    if report is not None:
        (paths.logs / "sensor_track.json").write_text(json.dumps(report))
    return paths


def test_metric_factor_trusts_a_sensor_scaled_report(tmp_path):
    paths = _fast_scene(tmp_path, {"scale_source": "sensor"})
    assert metric_factor(paths) == (1.0, "sensor")


def test_metric_factor_withholds_scale_when_the_report_says_unscaled(tmp_path):
    """A tripped residual leaves the scene exactly where a vision-only scene starts, so the
    manual scale tool can still rescue it."""
    paths = _fast_scene(tmp_path, {"scale_source": "unscaled"})
    assert metric_factor(paths) is None


def test_metric_factor_falls_back_to_the_track_without_a_report(tmp_path):
    """The ARKitScenes converter writes a fast-track scene without running this stage."""
    paths = _fast_scene(tmp_path)
    assert metric_factor(paths) == (1.0, "sensor")


def test_a_user_reference_still_wins_over_an_unscaled_report(tmp_path):
    paths = _fast_scene(tmp_path, {"scale_source": "unscaled"})
    save_reference(ScaleReference(a=(0, 0, 0), b=(1, 0, 0), distance_m=2.0), paths.scale)
    assert metric_factor(paths) == (2.0, "user")


def test_reconstruct_reads_a_scene_raw_folder(scene):
    """ingest copies the video in as `input.mp4`, not `rgb.mp4`, and the frames are already
    extracted by the time this stage runs. The sidecars are what it actually needs."""
    raw, frames, indices, poses = scene
    (raw / "rgb.mp4").rename(raw / "input.mp4")

    recon, report = reconstruct(raw, frames, indices, mast3r=None)

    assert len(recon.points) > 0
    assert report["scale_source"] == "sensor"


FIXTURE = Path(__file__).resolve().parents[2] / "test" / "strayroom"


@pytest.mark.skipif(not FIXTURE.is_dir(), reason="fixture not generated")
def test_the_committed_fixture_reconstructs_a_box_room():
    """The room is 4 m by 3 m with a 2.6 m ceiling. Recovering those three numbers from
    millimetre depth at a seventh of the colour resolution is the whole arithmetic chain:
    get the intrinsics rescale or the pose convention wrong and the box comes out the wrong
    size or sheared, while still looking like a room."""
    frames = sorted((FIXTURE / "frames").glob("*.jpg"))
    indices = [int(f.stem) for f in frames]

    recon, report = reconstruct(FIXTURE / "raw", frames, indices, mast3r=None)

    extent = recon.points.max(axis=0) - recon.points.min(axis=0)
    np.testing.assert_allclose(sorted(extent), [2.6, 3.0, 4.0], atol=0.05)
    assert report["scale_source"] == "sensor"


@pytest.mark.skipif(not FIXTURE.is_dir(), reason="fixture not generated")
def test_the_fixture_sits_on_the_floor_at_zero():
    frames = sorted((FIXTURE / "frames").glob("*.jpg"))
    recon, _ = reconstruct(FIXTURE / "raw", frames, [int(f.stem) for f in frames], mast3r=None)
    assert recon.points[:, 1].min() == pytest.approx(0.0, abs=0.03)


def _mostly_wrong_mast3r(truth_points, poses, scale, good=5, seed=1):
    """A MASt3R run that drifted: only `good` cameras keep the true relationship. RANSAC
    finds that small consensus and reports a perfect residual over it."""
    base = _stub_mast3r(truth_points, poses, scale)
    rng = np.random.default_rng(seed)

    def run():
        recon = base()
        recon.cam_to_world[good:, :3, 3] += rng.normal(0, 1.0 / scale, (len(poses) - good, 3))
        return recon

    return run


def test_a_small_inlier_consensus_withholds_metric_scale(scene):
    """RANSAC reports the residual over its inliers, so a worse reconstruction can report a
    better number: five agreeing cameras out of eight give 0.000 m while the rest are a metre
    out. The count has to be gated too, or the alarm is loudest when it is least needed."""
    raw, frames, indices, poses = scene
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)

    _, report = reconstruct(
        raw, frames, indices, mast3r=_mostly_wrong_mast3r(truth.points, poses, scale=3.0, good=4)
    )

    assert report["inliers"] < len(frames)
    assert report["scale_source"] == "unscaled"
    assert "cameras" in report["warning"]


def test_a_refused_fit_keeps_the_lidar_cloud(tmp_path):
    """A corridor walk cannot be fitted, but the LiDAR cloud is metric and gravity-aligned
    on its own. Throwing it away would turn a partial result into no result."""
    straight = np.tile(np.eye(4), (8, 1, 1))
    straight[:, :3, 3] = np.c_[np.linspace(0, 6, 8), np.full(8, 1.4), np.zeros(8)]
    raw, frames, indices, poses = _build_scene(tmp_path, straight)
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)

    recon, report = reconstruct(
        raw, frames, indices, mast3r=_stub_mast3r(truth.points, poses, scale=3.0)
    )

    assert len(recon.points) > 0
    assert report["mast3r_points"] == 0
    assert report["residual_m"] is None
    assert "straight line" in report["warning"]
    assert report["scale_source"] == "sensor"  # the sensor geometry is still metric


def test_one_out_of_range_frame_does_not_abort_the_capture(scene):
    """A real walk has frames pointed down a corridor or out of a window. Those contribute
    nothing; they must not discard the frames that did see something."""
    raw, frames, indices, _ = scene
    cv2.imwrite(str(raw / "depth" / "000003.png"), np.full(DEPTH_SIZE[::-1], 9000, np.uint16))

    recon, report = reconstruct(raw, frames, indices, mast3r=None)

    assert len(recon.points) > 0
    assert report["frames_without_depth"] == 1
    assert len(recon.cam_to_world) == len(frames)  # the pose is still a camera


def test_every_frame_out_of_range_is_still_refused(scene):
    raw, frames, indices, _ = scene
    for i in range(len(frames)):
        cv2.imwrite(str(raw / "depth" / f"{i:06d}.png"), np.full(DEPTH_SIZE[::-1], 9000, np.uint16))

    with pytest.raises(RuntimeError, match="in range"):
        reconstruct(raw, frames, indices, mast3r=None)


def test_the_clouds_are_compared_not_just_the_camera_paths(scene):
    """The trajectory residual is computed from camera centres, which a wrong pose convention
    and a wrong depth-intrinsics rescale both leave untouched. Comparing the two clouds is
    what can see those."""
    raw, frames, indices, poses = scene
    truth, _ = reconstruct(raw, frames, indices, mast3r=None)

    _, good = reconstruct(raw, frames, indices, mast3r=_stub_mast3r(truth.points, poses, scale=3.0))
    assert good["cloud_agreement_m"] == pytest.approx(0.0, abs=0.05)
    assert good["warning"] is None

    displaced = _stub_mast3r(truth.points + np.array([1.5, 0.0, 0.0]), poses, scale=3.0)
    _, bad = reconstruct(raw, frames, indices, mast3r=displaced)
    assert bad["cloud_agreement_m"] > 0.5
    assert "clouds" in bad["warning"]


@pytest.mark.skipif(not FIXTURE.is_dir(), reason="fixture not generated")
def test_a_downscaled_keyframe_does_not_shrink_the_room(tmp_path):
    """odometry.csv quotes intrinsics in capture pixels. ingest caps the long side at
    max_frame_side, so a capture wider than that arrives as smaller keyframes — and reading
    the source resolution off the keyframe would then rescale by the wrong factor and yield a
    plausible room of the wrong size, which is this project's top anti-goal."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame in sorted((FIXTURE / "frames").glob("*.jpg")):
        image = cv2.imread(str(frame))
        half = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        cv2.imwrite(str(frames_dir / frame.name), half)

    frames = sorted(frames_dir.glob("*.jpg"))
    indices = [int(f.stem) for f in frames]
    recon, _ = reconstruct(FIXTURE / "raw", frames, indices, mast3r=None, capture_size=(240, 180))

    extent = recon.points.max(axis=0) - recon.points.min(axis=0)
    np.testing.assert_allclose(sorted(extent), [2.6, 3.0, 4.0], atol=0.05)
