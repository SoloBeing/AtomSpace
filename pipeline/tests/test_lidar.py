import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from atomspace.reconstruct.lidar import sample_colors, scale_intrinsics, unproject

# A 256x192 depth map with the colour frame's 1500 px focal scaled down by 7.5.
K_DEPTH = np.array([[200.0, 0, 127.5], [0, 200.0, 95.5], [0, 0, 1]])


def _full(value, shape=(8, 8), dtype=np.float32):
    return np.full(shape, value, dtype)


def _k(fx, width, height):
    return np.array([[fx, 0, (width - 1) / 2], [0, fx, (height - 1) / 2], [0, 0, 1]])


def test_scale_intrinsics_uses_the_pixel_centre_convention():
    """Rescaling must treat a pixel's centre, not its corner, as the sample point, the same
    way mast3r_sfm.to_original_pixels does."""
    K = np.array([[1500.0, 0, 959.5], [0, 1500.0, 719.5], [0, 0, 1]])
    out = scale_intrinsics(K, (1920, 1440), (256, 192))

    np.testing.assert_allclose(out[0, 0], 200.0)
    np.testing.assert_allclose(out[1, 1], 200.0)
    np.testing.assert_allclose(out[:2, 2], [127.5, 95.5])


def test_scale_intrinsics_handles_unequal_axis_ratios():
    K = np.array([[1000.0, 0, 499.5], [0, 1000.0, 249.5], [0, 0, 1]])
    out = scale_intrinsics(K, (1000, 500), (100, 250))
    np.testing.assert_allclose([out[0, 0], out[1, 1]], [100.0, 500.0])


def test_unproject_recovers_a_plane_at_a_known_distance():
    """A depth map of a constant 2 m, seen from the identity pose: every point sits at
    z = 2, and the patch spans the frustum's width at that distance."""
    K = _k(100.0, 8, 8)
    points = unproject(_full(2.0), _full(2, dtype=np.uint8), K, np.eye(4))

    assert len(points) == 64
    np.testing.assert_allclose(points[:, 2], 2.0)
    half_width = 2.0 * 3.5 / 100.0  # (centre pixel -> edge pixel) * z / fx
    np.testing.assert_allclose(points[:, 0].min(), -half_width)
    np.testing.assert_allclose(points[:, 0].max(), half_width)


def test_unproject_places_points_with_the_camera_pose():
    """Camera moved to (5, 0, 0) and yawed 90 degrees about +Y. By the right-hand rule that
    sends OpenCV's forward axis +Z onto +X, so a plane 2 m ahead lands at x = 7 and the
    patch now spreads across z rather than x."""
    K = _k(100.0, 8, 8)
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_rotvec(np.radians(90) * np.array([0.0, 1.0, 0.0])).as_matrix()
    c2w[:3, 3] = [5.0, 0.0, 0.0]

    points = unproject(_full(2.0), _full(2, dtype=np.uint8), K, c2w)
    np.testing.assert_allclose(points[:, 0], 7.0, atol=1e-9)
    half_width = 2.0 * 3.5 / 100.0
    np.testing.assert_allclose(points[:, 2].min(), -half_width, atol=1e-9)
    np.testing.assert_allclose(points[:, 2].max(), half_width, atol=1e-9)


def test_unproject_drops_low_confidence_and_out_of_range():
    depth = np.array([[2.0, 2.0], [9.0, 0.0]], np.float32)
    confidence = np.array([[2, 0], [2, 2]], np.uint8)

    points = unproject(depth, confidence, _k(100.0, 2, 2), np.eye(4))

    assert len(points) == 1  # confidence 0, beyond range, and unmeasured are all gone
    np.testing.assert_allclose(points[0, 2], 2.0)


def test_unproject_can_keep_medium_confidence_only():
    depth = _full(2.0, (2, 2))
    confidence = np.array([[0, 1], [2, 2]], np.uint8)

    assert len(unproject(depth, confidence, _k(100.0, 2, 2), np.eye(4), min_confidence=1)) == 3
    assert len(unproject(depth, confidence, _k(100.0, 2, 2), np.eye(4), min_confidence=2)) == 2


def test_unproject_raises_when_nothing_is_in_range():
    """A hall deeper than the sensor reaches must fail by name here, not as an empty cloud
    that breaks much later inside occupancy_grid."""
    with pytest.raises(RuntimeError, match="in range"):
        unproject(_full(20.0), _full(2, dtype=np.uint8), _k(100.0, 8, 8), np.eye(4))


def test_unproject_rejects_a_mismatched_confidence_map():
    with pytest.raises(ValueError, match="shape"):
        unproject(_full(2.0, (8, 8)), _full(2, (4, 4), np.uint8), _k(100.0, 8, 8), np.eye(4))


def test_sample_colors_reads_the_pixel_the_point_projects_to():
    """The image is BGR as OpenCV reads it; the colours come back RGB as the PLY wants."""
    image = np.zeros((4, 4, 3), np.uint8)
    image[1, 2] = (20, 40, 60)  # BGR
    K = _k(100.0, 4, 4)

    point = np.array([[(2 - 1.5) * 3.0 / 100.0, (1 - 1.5) * 3.0 / 100.0, 3.0]])
    np.testing.assert_array_equal(sample_colors(point, image, K, np.eye(4)), [[60, 40, 20]])


def test_sample_colors_follows_the_camera_pose():
    image = np.zeros((4, 4, 3), np.uint8)
    image[:, :] = (10, 20, 30)
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, 0.0, -5.0]

    point = np.array([[0.0, 0.0, -2.0]])  # 3 m in front of a camera sitting at z = -5
    np.testing.assert_array_equal(sample_colors(point, image, _k(100.0, 4, 4), c2w), [[30, 20, 10]])
