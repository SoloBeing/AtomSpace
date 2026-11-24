import cv2
import mapbox_earcut
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..config import NavmeshConfig
from .obstacles import OccupancyGrid

SIMPLIFY_CELLS = 0.75  # outline simplification tolerance


def safe_cells(grid: OccupancyGrid, agent_radius_m: float) -> np.ndarray:
    """(H, W) bool: cells at least `agent_radius_m` from anything unwalkable, keeping only the
    largest connected region so every point of the navmesh can be reached from every other."""
    walkable = grid.observed & ~grid.blocked
    walkable[[0, -1], :] = False
    walkable[:, [0, -1]] = False
    clearance = distance_transform_edt(walkable) * grid.cell_size_m
    safe = (clearance > agent_radius_m).astype(np.uint8)
    # One-cell slivers have no area to triangulate.
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(safe, connectivity=4)
    if count < 2:
        raise RuntimeError(
            f"No walkable area left after keeping {agent_radius_m} m from obstacles. "
            "Check the scale reference, or capture more of the floor."
        )
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def walkable_mesh(safe: np.ndarray, grid: OccupancyGrid) -> tuple[np.ndarray, np.ndarray]:
    """Turns one connected region of safe cells into a triangle mesh at y = 0.

    Returns (vertices (V, 3), triangles (T, 3)), counter-clockwise seen from above.
    Outlines run through the centres of boundary cells, so they stay inside the safe area.
    """
    contours, hierarchy = cv2.findContours(safe.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer = [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]
    holes = [c for c, h in zip(contours, hierarchy[0]) if h[3] != -1]
    rings = [cv2.approxPolyDP(c, SIMPLIFY_CELLS, True)[:, 0] for c in outer + holes]
    rings = [r for r in rings if len(r) >= 3]

    outline = np.concatenate(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    triangles = mapbox_earcut.triangulate_float64(outline, ends).reshape(-1, 3).astype(np.int64)

    x, z = (outline + 0.5).T * grid.cell_size_m + np.array(grid.origin_xz)[:, None]
    vertices = np.stack([x, np.zeros_like(x), z], axis=1)

    # With +Y up, x × z points down, so counter-clockwise from above is negative area in (x, z).
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    clockwise = (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0]) > 0
    triangles[clockwise] = triangles[clockwise][:, ::-1]
    return vertices, triangles


def build_navmesh(vertices: np.ndarray, triangles: np.ndarray, cfg: NavmeshConfig) -> dict:
    """Returns a dict matching schemas/navmesh.schema.json, including the polygon neighbours the
    viewer's A* route search walks across."""
    edge_owner: dict[tuple[int, int], tuple[int, int]] = {}
    neighbours = [[-1, -1, -1] for _ in range(len(triangles))]
    for poly, tri in enumerate(triangles.tolist()):
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            if edge in edge_owner:
                other, other_i = edge_owner.pop(edge)
                neighbours[poly][i] = other
                neighbours[other][other_i] = poly
            else:
                edge_owner[edge] = (poly, i)

    return {
        "format_version": 1,
        "units": "m",
        "agent": {
            "radius_m": cfg.agent_radius_m,
            "height_m": cfg.agent_height_m,
            "max_climb_m": cfg.max_climb_m,
        },
        "vertices": np.round(vertices, 3).tolist(),
        "polygons": [{"v": tri, "n": n} for tri, n in zip(triangles.tolist(), neighbours)],
    }
