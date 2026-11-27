import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_POINTS = 50_000
RANSAC_ITERS = 1000
MAX_PLANES = 8
# MapAnything's floor is a few cm thick; at 2 cm it missed the 3% share and desk tops became the floor.
INLIER_DIST_M = 0.05
MIN_PLANE_SHARE = 0.03  # of the sampled points; anything smaller is furniture or noise
MIN_UP_ALIGNMENT = 0.7  # |normal·up| of a horizontal plane; camera tilt can skew up_hint by ~35°, walls stay under this


@dataclass(frozen=True)
class FloorPlane:
    """Plane n·p + offset = 0, with `normal` a unit vector pointing up into the room."""

    normal: np.ndarray
    offset: float


def camera_poses(transforms: Path) -> np.ndarray:
    """(N, 4, 4) camera-to-world matrices from transforms.json, OpenGL axes (column 1 is camera up)."""
    return np.array([f["transform_matrix"] for f in json.loads(transforms.read_text())["frames"]])


def fit_floor(points: np.ndarray, up_hint: np.ndarray, seed: int = 0) -> FloorPlane:
    """RANSAC over large planes; the floor is the lowest roughly horizontal one.

    Geometry alone can't tell floor from ceiling or wall (the whole room lies on one side of
    each), so `up_hint` gives the rough up direction, e.g. the mean camera up over the walk.
    Expects points in meters.
    """
    rng = np.random.default_rng(seed)
    up = up_hint / np.linalg.norm(up_hint)
    keep = rng.choice(len(points), min(SAMPLE_POINTS, len(points)), replace=False)
    remaining = points[keep]
    min_inliers = max(3, int(MIN_PLANE_SHARE * len(remaining)))

    horizontal = []
    for _ in range(MAX_PLANES):
        if len(remaining) < min_inliers:
            break
        inliers = _ransac_inliers(remaining, rng)
        if inliers.sum() < min_inliers:
            break
        plane = _least_squares_plane(remaining[inliers], up)
        if plane.normal @ up >= MIN_UP_ALIGNMENT:
            horizontal.append(plane)
        remaining = remaining[~inliers]

    if not horizontal:
        raise RuntimeError("No floor found: the capture needs more of the floor in view.")
    return min(horizontal, key=lambda p: -p.offset * (p.normal @ up))


def _ransac_inliers(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    best = np.zeros(len(points), dtype=bool)
    for _ in range(RANSAC_ITERS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        inliers = np.abs((points - a) @ (normal / length)) < INLIER_DIST_M
        if inliers.sum() > best.sum():
            best = inliers
    return best


def _least_squares_plane(points: np.ndarray, up: np.ndarray) -> FloorPlane:
    center = points.mean(axis=0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    if normal @ up < 0:
        normal = -normal
    return FloorPlane(normal, float(-normal @ center))


def world_from_floor(plane: FloorPlane) -> np.ndarray:
    """4x4 rigid transform mapping the floor normal to +Y and the floor onto y = 0."""
    y = plane.normal
    x = np.cross(np.eye(3)[np.argmin(np.abs(y))], y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    world = np.eye(4)
    world[:3, :3] = np.stack([x, y, z])
    world[1, 3] = plane.offset
    return world
