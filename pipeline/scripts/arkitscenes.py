"""Turn an ARKitScenes capture into an AtomSpace scene, to test the navmesh on real rooms.

ARKitScenes (Apple, https://github.com/apple/ARKitScenes) has iPad walk-throughs of real rooms with
ARKit's LiDAR mesh and camera path, both in meters. Its licence is Apple's own, so nothing from it is
committed here: `fetch` downloads one capture outside the repo, `convert` writes a scene folder.

    python pipeline/scripts/arkitscenes.py fetch 45663150 --out ../datasets/arkitscenes
    python pipeline/scripts/arkitscenes.py convert ../datasets/arkitscenes/45663150 --scene arkit-45663150

`convert` stands in for ingest + reconstruct + splat: surface samples of the mesh become
`dense_points.ply`, the camera path becomes `transforms.json`, and the same samples drawn as flat
discs become `scene.splat` so the viewer has something to show. Then the real navmesh and bundle
steps run. The scene is marked as the sensor track, so it counts as metric without a scale reference.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from atomspace import bundle, navmesh
from atomspace.config import ScenePaths
from atomspace.reconstruct.export import write_ply
from atomspace.splat.compress import SPLAT_DTYPE

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
POINTS_PER_M2 = 2500  # one sample every 2 cm, close to what MASt3R's dense points give
DISC_THICKNESS = 0.1  # of the disc radius
# ARKitScenes cameras look down +z with y pointing down (OpenCV); transforms.json wants OpenGL.
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def fetch(video_id: str, out: Path, split: str, video: bool) -> None:
    names = [f"{video_id}_3dod_mesh.ply", "lowres_wide.traj", f"{video_id}_3dod_annotation.json"]
    if video:
        names.append(f"{video_id}.mov")
    folder = out / video_id
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = folder / name
        if target.exists():
            print(f"Have {target}")
            continue
        print(f"Downloading {name}")
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl", "--fail", "-sS", "-o", str(partial), f"{BASE_URL}/{split}/{video_id}/{name}"], check=True)
        partial.rename(target)


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices (V, 3), colours (V, 3) uint8, triangles (T, 3)) from ARKitScenes' binary PLY."""
    header, body = path.read_bytes().split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    count = {line.split()[1]: int(line.split()[2]) for line in lines if line.startswith("element")}
    vertex = np.dtype([("xyz", "<f4", 3), ("rgba", "u1", 4)])
    face = np.dtype([("n", "u1"), ("v", "<i4", 3)])
    if "format binary_little_endian 1.0" not in lines or "property list uchar int vertex_indices" not in lines:
        raise ValueError(f"{path.name}: not the binary PLY layout ARKitScenes uses")
    vertices = np.frombuffer(body, vertex, count["vertex"])
    faces = np.frombuffer(body, face, count["face"], offset=count["vertex"] * vertex.itemsize)
    if np.any(faces["n"] != 3):
        raise ValueError(f"{path.name}: expected triangles only")
    return vertices["xyz"].astype(np.float64), vertices["rgba"][:, :3], faces["v"].astype(np.int64)


def sample_surface(vertices, colours, triangles, per_m2: float, seed: int = 0):
    """Evenly spread points over the mesh. ARKit keeps few vertices on flat floors, so its vertices
    alone would leave the floor looking unseen. Returns points, colours, unit normals."""
    rng = np.random.default_rng(seed)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    count = int(area.sum() * per_m2)
    picked = rng.choice(len(triangles), count, p=area / area.sum())
    u, v = rng.random((2, count))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = 1 - u - v
    tri = triangles[picked]
    points = w[:, None] * vertices[tri[:, 0]] + u[:, None] * vertices[tri[:, 1]] + v[:, None] * vertices[tri[:, 2]]
    rgb = w[:, None] * colours[tri[:, 0]] + u[:, None] * colours[tri[:, 1]] + v[:, None] * colours[tri[:, 2]]
    normals = cross[picked] / np.maximum(np.linalg.norm(cross[picked], axis=1, keepdims=True), 1e-12)
    return points, rgb.round().astype(np.uint8), normals


def camera_to_world(traj: Path) -> np.ndarray:
    """(N, 4, 4) OpenGL camera-to-world matrices. Each .traj line is time, then the world-to-camera
    rotation (axis-angle) and translation in meters."""
    t = np.loadtxt(traj)
    world_from_camera = np.tile(np.eye(4), (len(t), 1, 1))
    rotation = Rotation.from_rotvec(t[:, 1:4]).as_matrix().transpose(0, 2, 1)
    world_from_camera[:, :3, :3] = rotation
    world_from_camera[:, :3, 3] = -np.einsum("nij,nj->ni", rotation, t[:, 4:7])
    return world_from_camera @ OPENCV_TO_OPENGL


def disc_splat(points, colours, normals, per_m2: float) -> np.ndarray:
    """Flat discs lying in the surface, sized to just overlap their neighbours."""
    radius = 1.0 / np.sqrt(per_m2)
    out = np.empty(len(points), SPLAT_DTYPE)
    out["position"] = points
    out["scale"] = [radius, radius, radius * DISC_THICKNESS]
    out["rgba"] = np.c_[colours, np.full(len(points), 255, np.uint8)]
    # Rotate the disc's thin z axis onto the surface normal; .splat stores w, x, y, z.
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, normals)
    sin = np.linalg.norm(axis, axis=1)
    angle = np.arctan2(sin, normals @ z)
    axis = np.where(sin[:, None] > 1e-9, axis / np.maximum(sin, 1e-12)[:, None], [1.0, 0.0, 0.0])
    xyzw = Rotation.from_rotvec(axis * angle[:, None]).as_quat()
    wxyz = np.c_[xyzw[:, 3], xyzw[:, :3]]
    out["rotation"] = np.clip(wxyz * 128 + 128, 0, 255).round().astype(np.uint8)
    return out


def convert(capture: Path, scenes_root: Path, scene: str, per_m2: float) -> None:
    video_id = capture.name
    paths = ScenePaths(scenes_root / scene)
    paths.ensure()

    vertices, colours, triangles = read_mesh(capture / f"{video_id}_3dod_mesh.ply")
    points, rgb, normals = sample_surface(vertices, colours, triangles, per_m2)
    write_ply(paths.dense_points, points.astype(np.float32), rgb)
    paths.splat.write_bytes(disc_splat(points, rgb, normals, per_m2).tobytes())

    poses = camera_to_world(capture / "lowres_wide.traj")
    frames = [{"file_path": f"{i:05d}", "transform_matrix": m.tolist()} for i, m in enumerate(poses)]
    paths.transforms.write_text(json.dumps({"source": f"ARKitScenes {video_id}", "frames": frames}))
    # The sensor (fast) track means metric poses, which ARKit's are.
    paths.ingest_meta.write_text(
        json.dumps({"source": f"ARKitScenes {video_id}", "track": "fast", "keyframes": [f["file_path"] for f in frames]})
    )
    print(f"{len(vertices):,} mesh vertices -> {len(points):,} surface samples, {len(poses)} cameras")

    navmesh.run(paths)
    bundle.run(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download one capture's mesh, camera path and furniture boxes")
    f.add_argument("video_id")
    f.add_argument("--out", type=Path, required=True, help="folder outside the repo")
    f.add_argument("--split", choices=["Training", "Validation"], default="Validation")
    f.add_argument("--video", action="store_true", help="also the .mov (hundreds of MB), for a full pipeline run")
    c = sub.add_parser("convert", help="write a scene folder with navmesh and bundle")
    c.add_argument("capture", type=Path, help="folder made by fetch")
    c.add_argument("--scene", required=True)
    c.add_argument("--scenes-root", type=Path, default=Path("scenes"))
    c.add_argument("--points-per-m2", type=float, default=POINTS_PER_M2)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.video_id, args.out, args.split, args.video)
    else:
        convert(args.capture, args.scenes_root, args.scene, args.points_per_m2)


if __name__ == "__main__":
    main()
