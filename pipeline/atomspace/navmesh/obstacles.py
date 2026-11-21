from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from ..config import NavmeshConfig

MAX_GRID_SIDE = 4000  # 200 m at the default cell size; beyond this the scale is almost certainly wrong


@dataclass
class OccupancyGrid:
    origin_xz: tuple[float, float]
    cell_size_m: float
    observed: np.ndarray  # (H, W) bool: floor was seen here
    blocked: np.ndarray  # (H, W) bool: something between ankle and head height


def occupancy_grid(points_world: np.ndarray, cfg: NavmeshConfig, walked_xz: np.ndarray | None = None) -> OccupancyGrid:
    """Projects world points (y up, floor at 0) onto the XZ plane. Points between
    `ankle_height_m` and `agent_height_m` mark cells blocked; floor points mark cells observed.
    Unobserved cells stay unwalkable so holes in the capture don't become paths, except a patch
    entirely enclosed by observed floor: anything standing there would show up as blocked.

    `walked_xz` (N, 2) are camera positions. Whoever filmed stood there, so those cells are floor
    and free even though a phone held ahead of the body rarely sees the ground under its owner.
    """
    y = points_world[:, 1]
    floor = points_world[np.abs(y) < cfg.ankle_height_m][:, [0, 2]]
    obstacles = points_world[(y >= cfg.ankle_height_m) & (y <= cfg.agent_height_m)][:, [0, 2]]
    if len(floor) == 0:
        raise RuntimeError("No points near the floor plane; check the scale reference.")

    # Bounds come from the floor, trimmed so a few stray points can't stretch the grid.
    cell = cfg.cell_size_m
    low = np.percentile(floor, 0.1, axis=0) - 2 * cell
    high = np.percentile(floor, 99.9, axis=0) + 2 * cell
    width, height = np.ceil((high - low) / cell).astype(int)
    if max(width, height) > MAX_GRID_SIDE:
        raise RuntimeError(f"Scene spans {max(width, height) * cell:.0f} m; check the scale reference.")

    observed = _point_counts(floor, low, cell, height, width) > 0
    blocked = _point_counts(obstacles, low, cell, height, width) >= cfg.min_blocked_points
    # A blocked cell with at most one blocked neighbour is a sliver (a mesh fragment, a seat edge) that
    # would otherwise cost a whole agent-sized circle of floor.
    neighbours = cv2.filter2D(blocked.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    blocked &= neighbours > 2  # the count includes the cell itself

    # A floor seen a little patchily is still floor; blocked cells are applied after this.
    radius = int(np.ceil(cfg.floor_gap_fill_m / (2 * cell)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    observed = cv2.morphologyEx(observed.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if walked_xz is not None and len(walked_xz) > 0:
        walked = np.zeros((height, width), np.uint8)
        pixels = np.floor((walked_xz - low) / cell).astype(int)
        for col, row in pixels:
            cv2.circle(walked, (int(col), int(row)), int(round(cfg.walked_radius_m / cell)), 1, -1)
        observed |= walked.astype(bool)
        blocked &= ~walked.astype(bool)
    observed = binary_fill_holes(observed)
    return OccupancyGrid((float(low[0]), float(low[1])), cell, observed, blocked)


def _point_counts(xz: np.ndarray, low: np.ndarray, cell: float, height: int, width: int) -> np.ndarray:
    cols, rows = np.floor((xz - low) / cell).astype(int).T
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    counts = np.bincount(rows[inside] * width + cols[inside], minlength=height * width)
    return counts.reshape(height, width)
