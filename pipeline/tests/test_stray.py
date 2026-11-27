import cv2
import numpy as np
import pytest

from atomspace.ingest.stray import (
    find_recording, find_sidecars, read_confidence, read_depth_m, read_odometry,
)

HEADER = (
    "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
    " distortion_center_x, distortion_center_y\r\n"
)


def _recording(tmp_path, rows=1, depth=True, confidence=True):
    (tmp_path / "rgb.mp4").write_bytes(b"")
    lines = [
        f"{i / 30:.3f}, {i:06d}, 1, 2, 3, 0, 0, 0, 1, 1500, 1500, 960, 720, ,\r\n"
        for i in range(rows)
    ]
    (tmp_path / "odometry.csv").write_text(HEADER + "".join(lines))
    if depth:
        (tmp_path / "depth").mkdir()
    if confidence:
        (tmp_path / "confidence").mkdir()
    return tmp_path


def test_find_recording_needs_odometry_and_video(tmp_path):
    assert find_recording(tmp_path) is None
    (tmp_path / "rgb.mp4").write_bytes(b"")
    assert find_recording(tmp_path) is None
    (tmp_path / "odometry.csv").write_text(HEADER)
    rec = find_recording(tmp_path)
    assert rec is not None
    assert rec.video.name == "rgb.mp4"
    assert rec.odometry.name == "odometry.csv"
    assert rec.depth is None
    assert rec.confidence is None


def test_find_recording_reports_the_optional_folders(tmp_path):
    rec = find_recording(_recording(tmp_path))
    assert rec.depth is not None and rec.depth.is_dir()
    assert rec.confidence is not None and rec.confidence.is_dir()


def test_find_recording_is_none_for_a_plain_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    assert find_recording(video) is None


def test_read_odometry_parses_the_documented_header(tmp_path):
    """Identity rotation at (1, 2, 3). ARKit looks down -Z with +Y up; OpenCV looks down
    +Z with +Y down, so the camera basis is the identity with Y and Z negated."""
    path = tmp_path / "odometry.csv"
    path.write_text(
        HEADER
        + "0.500, 000000, 1, 2, 3, 0, 0, 0, 1, 1500, 1500, 960, 720, ,\r\n"
        + "0.533, 000001, 1, 2, 3, 0, 0, 0, 1, 1500, 1500, 960, 720, 960, 720\r\n"
    )
    odom = read_odometry(path)

    np.testing.assert_allclose(odom.timestamps, [0.5, 0.533])
    assert odom.frames.tolist() == [0, 1]
    assert odom.cam_to_world.shape == (2, 4, 4)
    np.testing.assert_allclose(odom.cam_to_world[0][:3, 3], [1, 2, 3])
    np.testing.assert_allclose(odom.cam_to_world[0][:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(odom.cam_to_world[0][3], [0, 0, 0, 1])
    np.testing.assert_allclose(odom.intrinsics[0], [[1500, 0, 960], [0, 1500, 720], [0, 0, 1]])


def test_read_odometry_applies_rotation_in_opencv_axes(tmp_path):
    """90 degrees about ARKit's +Y (up). ARKit forward is -Z, which maps to -X, so the
    OpenCV camera's +Z column must point along -X."""
    half = np.sqrt(0.5)
    path = tmp_path / "odometry.csv"
    path.write_text(
        HEADER + f"0, 000000, 0, 0, 0, 0, {half}, 0, {half}, 1500, 1500, 960, 720, ,\r\n"
    )
    c2w = read_odometry(path).cam_to_world[0]

    np.testing.assert_allclose(c2w[:3, 2], [-1, 0, 0], atol=1e-9)  # OpenCV forward
    np.testing.assert_allclose(c2w[:3, 1], [0, -1, 0], atol=1e-9)  # OpenCV down
    np.testing.assert_allclose(np.linalg.det(c2w[:3, :3]), 1.0)


def test_read_odometry_rejects_an_unexpected_header(tmp_path):
    path = tmp_path / "odometry.csv"
    path.write_text("time, x, y, z\n0, 1, 2, 3\n")
    with pytest.raises(ValueError, match="odometry.csv"):
        read_odometry(path)


def test_read_depth_converts_millimetres_to_metres(tmp_path):
    path = tmp_path / "000000.png"
    cv2.imwrite(str(path), np.array([[0, 1000], [2500, 65535]], np.uint16))
    np.testing.assert_allclose(read_depth_m(path), [[0, 1.0], [2.5, 65.535]], atol=1e-6)


def test_read_depth_rejects_an_eight_bit_png(tmp_path):
    path = tmp_path / "000000.png"
    cv2.imwrite(str(path), np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="16-bit"):
        read_depth_m(path)


def test_read_confidence_keeps_the_raw_levels(tmp_path):
    path = tmp_path / "000000.png"
    cv2.imwrite(str(path), np.array([[0, 1], [2, 2]], np.uint8))
    got = read_confidence(path)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, [[0, 1], [2, 2]])


def test_find_sidecars_does_not_need_the_video(tmp_path):
    """A scene's raw/ folder holds the sidecars beside a video ingest renamed to input.mp4,
    so the reconstruct stage identifies a recording by its odometry, not its video."""
    recording = _recording(tmp_path)
    (tmp_path / "rgb.mp4").rename(tmp_path / "input.mp4")

    assert find_recording(recording) is None  # strict: ingest needs rgb.mp4
    found = find_sidecars(recording)
    assert found is not None and found.video is None
    assert found.depth is not None and found.confidence is not None


def test_find_sidecars_needs_odometry(tmp_path):
    (tmp_path / "depth").mkdir()
    assert find_sidecars(tmp_path) is None
