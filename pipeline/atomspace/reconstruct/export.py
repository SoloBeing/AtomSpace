import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..config import ScenePaths

# OpenCV camera axes (x right, y down, looking +z) <-> OpenGL/nerfstudio (x right, y up, looking -z).
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
PLY_VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])


@dataclass
class Reconstruction:
    image_names: list[str]
    intrinsics: np.ndarray  # (N, 3, 3), pixels of the original keyframes
    cam_to_world: np.ndarray  # (N, 4, 4), OpenCV camera convention, recon space
    points: np.ndarray  # (M, 3) float32, recon space
    colors: np.ndarray  # (M, 3) uint8


def write_reconstruction(recon: Reconstruction, paths: ScenePaths) -> None:
    """Writes recon/transforms.json (nerfstudio-style, per-frame intrinsics) and recon/dense_points.ply.

    transform_matrix follows nerfstudio's OpenGL camera convention; file paths are relative to recon/.
    """
    paths.recon.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, K, c2w in zip(recon.image_names, recon.intrinsics, recon.cam_to_world):
        w, h = ImageOps.exif_transpose(Image.open(paths.frames / name)).size
        frames.append({
            "file_path": f"../{paths.frames.name}/{name}",
            "transform_matrix": (c2w @ CV_TO_GL).tolist(),
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": w,
            "h": h,
        })
    paths.transforms.write_text(json.dumps(
        {"camera_model": "OPENCV", "ply_file_path": paths.dense_points.name, "frames": frames}, indent=2
    ))
    write_ply(paths.dense_points, recon.points, recon.colors)

    # A new reconstruction has its own arbitrary frame, so a scale reference picked in the old one
    # is wrong, and so is everything derived from it.
    stale = [p for p in (paths.scale, paths.world, paths.navmesh) if p.exists()]
    for p in stale:
        p.unlink()
    if stale:
        print(f"Removed {', '.join(p.name for p in stale)}: the scale must be set again for the new reconstruction.")
    # POIs can't be mapped into the new frame, but they're hand-made, so set them aside instead of deleting.
    if paths.pois.exists():
        old_pois = paths.pois.replace(paths.pois.with_name("pois.old.json"))
        print(f"Moved POIs to {old_pois}: place them again once the scale is set.")


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY with float xyz and uchar rgb (what splat trainers read)."""
    vertices = np.empty(len(points), dtype=PLY_VERTEX)
    for i, axis in enumerate("xyz"):
        vertices[axis] = points[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        vertices[channel] = colors[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads back a file from write_ply; returns (points (M, 3) float32, colors (M, 3) uint8)."""
    header, body = path.read_bytes().split(b"end_header\n", 1)
    if b"binary_little_endian" not in header:
        raise ValueError(f"{path.name} is not a binary little-endian PLY")
    vertices = np.frombuffer(body, dtype=PLY_VERTEX)
    points = np.stack([vertices[axis] for axis in "xyz"], axis=1)
    colors = np.stack([vertices[channel] for channel in ("red", "green", "blue")], axis=1)
    return points, colors
