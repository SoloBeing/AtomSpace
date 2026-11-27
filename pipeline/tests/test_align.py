import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from atomspace.reconstruct.align import robust_similarity, umeyama

SCALE = 2.5
ROTATION = Rotation.from_rotvec(np.radians(30) * np.array([1.0, 2.0, 3.0]) / np.sqrt(14)).as_matrix()
TRANSLATION = np.array([0.4, -1.2, 7.0])


def _transform(points: np.ndarray) -> np.ndarray:
    return SCALE * points @ ROTATION.T + TRANSLATION


def _walk(n: int, seed: int = 0) -> np.ndarray:
    """A loop around a room: spread in all three axes, as a real capture is."""
    rng = np.random.default_rng(seed)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(angle), rng.uniform(-0.3, 0.3, n), np.sin(angle)]


def test_umeyama_recovers_a_known_similarity():
    source = _walk(40)
    scale, rotation, translation = umeyama(source, _transform(source))

    assert scale == pytest.approx(SCALE, rel=1e-9)
    np.testing.assert_allclose(rotation, ROTATION, atol=1e-9)
    np.testing.assert_allclose(translation, TRANSLATION, atol=1e-9)


def test_umeyama_never_returns_a_reflection():
    """A mirrored cloud has no similarity transform onto the original, so the fit must stay
    a rotation rather than flipping an axis to drive the error down."""
    source = _walk(40)
    target = source * [1, 1, -1]
    _, rotation, _ = umeyama(source, target)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_robust_similarity_ignores_outliers():
    source = _walk(40)
    target = _transform(source)
    target[::5] += 3.0  # 8 cameras displaced by metres

    fit = robust_similarity(source, target, threshold_m=0.05)

    assert fit.inliers.sum() == 32
    assert fit.residual_m < 1e-6
    assert fit.scale == pytest.approx(SCALE, rel=1e-6)
    np.testing.assert_allclose(fit.rotation, ROTATION, atol=1e-6)


def test_residual_is_reported_in_metres():
    """Every target pushed 0.10 m along +X after the transform: no similarity fits, and the
    leftover error is exactly that displacement."""
    source = _walk(40)
    target = _transform(source) + np.array([0.10, 0, 0])

    fit = robust_similarity(source, target, threshold_m=1.0)
    assert fit.residual_m == pytest.approx(0.0, abs=1e-6)

    target[::2] -= np.array([0.20, 0, 0])  # half pushed the other way, so nothing cancels
    fit = robust_similarity(source, target, threshold_m=1.0)
    assert fit.residual_m == pytest.approx(0.10, abs=1e-3)


def test_apply_maps_the_source_onto_the_target():
    source = _walk(40)
    fit = robust_similarity(source, _transform(source))
    np.testing.assert_allclose(fit.apply(source), _transform(source), atol=1e-9)


def test_matrix_agrees_with_apply():
    source = _walk(40)
    fit = robust_similarity(source, _transform(source))
    homogeneous = np.c_[source, np.ones(len(source))] @ fit.matrix.T
    np.testing.assert_allclose(homogeneous[:, :3], fit.apply(source), atol=1e-9)


def test_collinear_trajectory_is_refused():
    """A straight corridor walk leaves the rotation about the walking axis unconstrained,
    so a fit would be confident and wrong."""
    line = np.c_[np.linspace(0, 5, 30), np.zeros(30), np.zeros(30)]
    with pytest.raises(RuntimeError, match="straight line"):
        robust_similarity(line, _transform(line))


def test_too_few_cameras_is_refused():
    with pytest.raises(RuntimeError, match="at least 3"):
        robust_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


def test_a_planar_loop_is_accepted():
    """A similarity transform needs three non-collinear correspondences, which a plane
    supplies. Walking a loop at a steady height is the capture this feature exists for, and
    the fixture's own trajectory is exactly that."""
    loop = _walk(40)
    loop[:, 1] = 1.4  # perfectly flat: the out-of-plane extent is zero

    fit = robust_similarity(loop, _transform(loop))

    assert fit.scale == pytest.approx(SCALE, rel=1e-6)
    assert fit.residual_m < 1e-6


def test_a_loop_with_only_handheld_wobble_is_accepted():
    rng = np.random.default_rng(7)
    loop = _walk(60)
    loop[:, 1] = 1.4 + rng.normal(0, 0.03, 60)  # 3 cm of hand wobble
    fit = robust_similarity(loop, _transform(loop))
    assert fit.scale == pytest.approx(SCALE, rel=1e-6)
