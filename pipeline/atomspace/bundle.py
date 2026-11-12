import json
from datetime import datetime, timezone

import jsonschema
import numpy as np

from .config import SCHEMAS_DIR, ScenePaths
from .navmesh.floor import FloorPlane, camera_poses, world_from_floor
from .reconstruct.scale import load_reference, metric_factor

IDENTITY = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def _validate(name: str, data: dict) -> None:
    schema = json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text())
    jsonschema.validate(data, schema)


def _upright_from_cameras(paths: ScenePaths) -> list[float]:
    """Before a scale is set there is no floor yet, and MASt3R's y axis points down. This stands
    the scene upright around the first camera so the viewer's scale tool has a usable view."""
    poses = camera_poses(paths.transforms)
    up = poses[:, :3, 1].mean(axis=0)
    world = world_from_floor(FloorPlane(up / np.linalg.norm(up), 0.0))
    world[:3, 3] = -world[:3, :3] @ poses[0, :3, 3]
    return world.T.flatten().tolist()


def run(paths: ScenePaths) -> None:
    ingest_meta = json.loads(paths.ingest_meta.read_text())

    metric = metric_factor(paths)
    if metric is None:
        scale = {"source": "unscaled", "factor": 1.0}
    else:
        factor, source = metric
        scale = {"source": source, "factor": factor}
        if source == "user":
            ref = load_reference(paths.scale)
            scale["reference"] = {"a": list(ref.a), "b": list(ref.b), "distance_m": ref.distance_m}

    if paths.world.exists():
        world_from_recon = json.loads(paths.world.read_text())["world_from_recon"]
    elif paths.transforms.exists():
        world_from_recon = _upright_from_cameras(paths)
    else:
        world_from_recon = IDENTITY

    if not paths.pois.exists():
        paths.pois.write_text(json.dumps({"format_version": 1, "pois": []}, indent=2))
    _validate("pois", json.loads(paths.pois.read_text()))

    has_navmesh = paths.navmesh.exists()
    if has_navmesh:
        _validate("navmesh", json.loads(paths.navmesh.read_text()))

    manifest = {
        "format_version": 1,
        "scene": paths.root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": ingest_meta["track"],
        "keyframes": len(ingest_meta["keyframes"]),
        "world_from_recon": world_from_recon,
        "scale": scale,
        "files": {
            "splat": paths.splat.name,
            "navmesh": paths.navmesh.name if has_navmesh else None,
            "pois": paths.pois.name,
        },
    }
    if paths.splat_meta.exists() and json.loads(paths.splat_meta.read_text()).get("antialiased"):
        manifest["splat_antialiased"] = True
    _validate("manifest", manifest)
    paths.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Bundle ready: {paths.bundle}")
