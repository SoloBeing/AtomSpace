import numpy as np
import pytest

from atomspace.navmesh.floor import (
    fit_floor, floor_at_known_normal, world_from_floor,
)

Y = np.array([0.0, 1.0, 0.0])


def _room(rng, n_floor=5000, n_ceiling=3000, n_walls=0) -> np.ndarray:
    floor = np.c_[rng.uniform(-3, 3, n_floor), np.zeros(n_floor), rng.uniform(-3, 3, n_floor)]
    table = np.c_[rng.uniform(0, 1, 800), rng.uniform(0, 0.75, 800), rng.uniform(0, 1, 800)]
    ceiling = np.c_[rng.uniform(-3, 3, n_ceiling), np.full(n_ceiling, 2.6), rng.uniform(-3, 3, n_ceiling)]
    wall = np.c_[np.full(n_walls, 3.0), rng.uniform(0, 2.6, n_walls), rng.uniform(-3, 3, n_walls)]
    return np.vstack([floor, table, ceiling, wall])


def _random_rigid(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = q, rng.normal(size=3)
    return t


def _floor_height_after_alignment(points, pose, hint=Y):
    moved = points @ pose[:3, :3].T + pose[:3, 3]
    aligned = world_from_floor(fit_floor(moved, pose[:3, :3] @ hint))
    assert np.isclose(np.linalg.det(aligned[:3, :3]), 1.0)
    return moved @ aligned[:3, :3].T + aligned[:3, 3]


def test_floor_is_found_under_tilted_room():
    rng = np.random.default_rng(0)
    restored = _floor_height_after_alignment(_room(rng), _random_rigid(rng))
    assert np.abs(restored[:5000, 1]).max() < 0.02
    assert np.allclose(restored[-3000:, 1], 2.6, atol=0.02)


def test_floor_beats_a_bigger_ceiling_and_wall():
    rng = np.random.default_rng(1)
    points = _room(rng, n_floor=2000, n_ceiling=6000, n_walls=8000)
    restored = _floor_height_after_alignment(points, _random_rigid(rng))
    assert np.abs(restored[:2000, 1]).max() < 0.02


def test_rough_up_hint_is_enough():
    # A walk that mostly faces one way leaves the mean camera up tilted towards that direction.
    rng = np.random.default_rng(2)
    tilted = np.array([np.sin(np.radians(35)), np.cos(np.radians(35)), 0.0])
    points = _room(rng, n_walls=6000)
    restored = _floor_height_after_alignment(points, _random_rigid(rng), hint=tilted)
    assert np.abs(restored[:5000, 1]).max() < 0.02


def _apply(plane, points):
    world = world_from_floor(plane)
    return points @ world[:3, :3].T + world[:3, 3]


def test_known_normal_is_kept_exactly():
    """Gravity is measured, not inferred, so noise on the floor points must not tilt it."""
    rng = np.random.default_rng(3)
    points = _room(rng)
    points[:5000, 1] += rng.normal(0, 0.01, 5000)

    plane = floor_at_known_normal(points, Y)

    np.testing.assert_array_equal(plane.normal, Y)


def test_known_normal_finds_the_lowest_horizontal_surface():
    # Floor at 0, a table top at 0.75 and a ceiling at 2.6: the floor is the one wanted.
    rng = np.random.default_rng(4)
    restored = _apply(floor_at_known_normal(_room(rng), Y), _room(rng))
    assert np.abs(restored[:5000, 1]).max() < 0.03


def test_known_normal_survives_an_unnormalised_direction():
    rng = np.random.default_rng(5)
    plane = floor_at_known_normal(_room(rng), Y * 7.0)
    np.testing.assert_allclose(plane.normal, Y)


def test_a_sparsely_seen_floor_is_still_found():
    """300 floor points under 8000 of wall: far too few to be a plane the SVD would pick out
    of that, but enough to be the lowest surface once gravity has settled the orientation.
    Height still has to be discovered from floor points, so this does not rescue a capture
    that never looked down at all."""
    rng = np.random.default_rng(6)
    points = _room(rng, n_floor=300, n_walls=8000)
    pose = _random_rigid(rng)
    moved = points @ pose[:3, :3].T + pose[:3, 3]

    gravity = _apply(floor_at_known_normal(moved, pose[:3, :3] @ Y), moved)

    assert np.abs(gravity[:300, 1]).max() < 0.05


def test_known_normal_raises_without_a_populated_surface():
    with pytest.raises(RuntimeError, match="floor"):
        floor_at_known_normal(np.zeros((2, 3)), Y)
