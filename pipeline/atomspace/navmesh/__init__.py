import json

import numpy as np

from ..config import NavmeshConfig, ScenePaths
from ..reconstruct.export import read_ply
from ..reconstruct.scale import metric_factor
from .floor import camera_poses, fit_floor, floor_at_known_normal, world_from_floor
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
    # The sensor track records ARKit's gravity, which beats inferring up from camera tilt.
    world_up = paths.recon / "world_up.json"
    if world_up.exists():
        plane = floor_at_known_normal(points, np.array(json.loads(world_up.read_text())["world_up"]))
    else:
        plane = fit_floor(points, poses[:, :3, 1].mean(axis=0))
    world_from_scaled = world_from_floor(plane)
    world_from_recon = world_from_scaled @ np.diag([factor, factor, factor, 1.0])
    points_world = points @ world_from_scaled[:3, :3].T + world_from_scaled[:3, 3]
    cameras_world = (poses[:, :3, 3] * factor) @ world_from_scaled[:3, :3].T + world_from_scaled[:3, 3]

    grid = occupancy_grid(points_world, cfg, walked_xz=cameras_world[:, [0, 2]])
    safe = safe_cells(grid, cfg.agent_radius_m)
    vertices, triangles = walkable_mesh(safe, grid)
    navmesh = build_navmesh(vertices, triangles, cfg)

    # Column-major so the viewer can pass it straight to THREE.Matrix4.fromArray.
    paths.world.write_text(json.dumps({"world_from_recon": world_from_recon.T.flatten().tolist()}, indent=2))
    paths.navmesh.write_text(json.dumps(navmesh))
    area_m2 = safe.sum() * grid.cell_size_m**2
    print(f"Navmesh: {len(navmesh['polygons'])} polygons, {area_m2:.1f} m2 walkable")
