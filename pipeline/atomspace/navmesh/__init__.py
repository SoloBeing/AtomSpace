import json

import numpy as np

from ..config import NavmeshConfig, ScenePaths
from ..reconstruct.export import read_ply
from ..reconstruct.scale import metric_factor
from .floor import camera_poses, fit_floor, world_from_floor
from .obstacles import occupancy_grid
from .polygonize import build_navmesh, safe_cells, walkable_mesh


def run(paths: ScenePaths, cfg: NavmeshConfig = NavmeshConfig()) -> None:
    metric = metric_factor(paths)
    if metric is None:
        raise RuntimeError("No metric scale yet. Set it with the viewer's scale tool or `atomspace scale`.")
    factor, _ = metric
    paths.ensure()

    points, _ = read_ply(paths.dense_points)
    points = points.astype(np.float64) * factor
    poses = camera_poses(paths.transforms)
    plane = fit_floor(points, poses[:, :3, 1].mean(axis=0))
    world_from_scaled = world_from_floor(plane)
    world_from_recon = world_from_scaled @ np.diag([factor, factor, factor, 1.0])
    points_world = points @ world_from_scaled[:3, :3].T + world_from_scaled[:3, 3]
    cameras_world = (poses[:, :3, 3] * factor) @ world_from_scaled[:3, :3].T + world_from_scaled[:3, 3]

    grid = occupancy_grid(points_world, cfg, walked_xz=cameras_world[:, [0, 2]])
    safe = safe_cells(grid, cfg.agent_radius_m)
    vertices, triangles = walkable_mesh(safe, grid)
    navmesh = build_navmesh(vertices, triangles, cfg)

    _move_pois(paths, world_from_recon)
    # Column-major so the viewer can pass it straight to THREE.Matrix4.fromArray.
    paths.world.write_text(json.dumps({"world_from_recon": world_from_recon.T.flatten().tolist()}, indent=2))
    paths.navmesh.write_text(json.dumps(navmesh))
    area_m2 = safe.sum() * grid.cell_size_m**2
    print(f"Navmesh: {len(navmesh['polygons'])} polygons, {area_m2:.1f} m2 walkable")


def _move_pois(paths: ScenePaths, world_from_recon: np.ndarray) -> None:
    """POIs are stored in world space; keep them on the same spot of the splat when the frame changes."""
    if not (paths.world.exists() and paths.pois.exists()):
        return
    old = np.array(json.loads(paths.world.read_text())["world_from_recon"]).reshape(4, 4).T
    move = world_from_recon @ np.linalg.inv(old)

    def moved(p: list[float]) -> list[float]:
        return (move[:3, :3] @ p + move[:3, 3]).tolist()

    data = json.loads(paths.pois.read_text())
    for poi in data["pois"]:
        poi["position"] = moved(poi["position"])
        if "view" in poi:
            poi["view"] = {k: moved(v) for k, v in poi["view"].items()}
    paths.pois.write_text(json.dumps(data, indent=2))
