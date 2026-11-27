import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_POINTS = 50_000
RANSAC_ITERS = 1000
MAX_PLANES = 8
INLIER_DIST_M = 0.02
MIN_PLANE_SHARE = 0.03  # of the sampled points; anything smaller is furniture or noise
MIN_BAND_SHARE = 0.05  # of the densest height band, when the up direction is already known
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


def floor_at_known_normal(points: np.ndarray, normal: np.ndarray) -> FloorPlane:
    """The floor plane when the up direction is already known, as it is on the sensor track.

    ARKit reports gravity, so only the floor's height is left to find. `fit_floor` would
    solve the normal from the points as well, and a sparse floor under a lot of wall lets
    that tilt the whole world — which the ankle-to-head occupancy slabs are sensitive to.

    Heights along `normal` are histogrammed at the plane-fit tolerance, and the lowest bin
    holding a real surface's worth of points is the floor. Expects points in meters.
    """
    up = normal / np.linalg.norm(normal)
    heights = points @ up
    edges = np.arange(heights.min(), heights.max() + 2 * INLIER_DIST_M, INLIER_DIST_M)
    counts, _ = np.histogram(heights, bins=edges)
    # Measured against the densest band rather than the whole cloud: a floor glimpsed
    # briefly under a lot of wall is a small share of the points but still a real surface,
    # whereas a wall spreads itself thinly across every band it crosses.
    surfaces = np.flatnonzero(counts >= max(3, MIN_BAND_SHARE * counts.max()))
    if not len(surfaces):
        raise RuntimeError("No floor found: the capture needs more of the floor in view.")

    # Centre of mass of the lowest populated band, so the plane sits on the points rather
    # than on a bin edge.
    low, high = edges[surfaces[0]], edges[surfaces[0] + 1]
    band = heights[(heights >= low) & (heights < high)]
    return FloorPlane(up, float(-band.mean()))


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
