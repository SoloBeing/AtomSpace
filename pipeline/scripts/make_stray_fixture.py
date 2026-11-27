"""Writes a synthetic Stray Scanner recording of a plain box room, for the tests.

A real capture is the only thing that can confirm the index correspondence, the rotation
flag handling or that LiDAR plus MASt3R beats vision alone — a synthetic recording is
consistent by construction and proves none of that. What it does prove is the arithmetic:
that depth in millimetres at the depth map's resolution, intrinsics quoted for the colour
frame, and ARKit's pose convention together put points where they belong. That is the one
defect that would otherwise look like a room and silently be the wrong size.

    python pipeline/scripts/make_stray_fixture.py --out test/strayroom

The result holds `raw/` (odometry.csv and the depth and confidence PNGs) beside pre-extracted
`frames/`, the shape a scene has once ingest has run. No video, so nothing lands in Git LFS.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Room spanning x in [0, 4], y in [0, 2.6], z in [0, 3]: 4 m by 3 m with a 2.6 m ceiling.
ROOM = np.array([[0.0, 0.0, 0.0], [4.0, 2.6, 3.0]])
DEPTH_SIZE = (32, 24)  # (width, height); the real sensor is 256x192, the same 4:3
COLOR_SIZE = (240, 180)  # 7.5x the depth map, as 1920x1440 is to 256x192
COLOR_FOCAL = 150.0  # colour-frame pixels, giving a wide-ish 77 degree horizontal view
MAX_RANGE_M = 5.0
# One colour per face, so a point's colour says which surface it was measured on:
# -x, +x, -y (floor), +y (ceiling), -z, +z. BGR, as OpenCV writes it.
FACE_COLORS = np.array([
    [60, 60, 180], [60, 180, 60], [40, 40, 40], [220, 220, 220], [180, 60, 60], [60, 180, 180],
], np.uint8)


def poses(count: int) -> np.ndarray:
    """A loop near the middle of the room, looking outward, pitching down and up so the
    floor and ceiling are both seen. ARKit convention: +Y up, -Z forward."""
    out = np.tile(np.eye(4), (count, 1, 1))
    centre = ROOM.mean(axis=0)
    for i in range(count):
        angle = 2 * np.pi * i / count
        pitch = np.radians([-35.0, 0.0, 35.0][i % 3])
        # Yaw so that -Z points outward along the radius, then pitch about the camera's own x.
        yaw = Rotation.from_rotvec((np.pi - angle) * np.array([0.0, 1.0, 0.0]))
        out[i, :3, :3] = (yaw * Rotation.from_rotvec(pitch * np.array([1.0, 0.0, 0.0]))).as_matrix()
        out[i, :3, 3] = [centre[0] + 0.8 * np.sin(angle), 1.4, centre[2] + 0.8 * np.cos(angle)]
    return out


def _rays(size: tuple[int, int], focal: float) -> np.ndarray:
    """(H, W, 3) camera-frame directions with z = 1, in OpenCV axes."""
    width, height = size
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([(u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal, np.ones_like(u)], axis=-1)


def cast(pose_arkit: np.ndarray, size: tuple[int, int], focal: float) -> tuple[np.ndarray, np.ndarray]:
    """Depth along the camera's z axis to the room's surfaces, and which face was hit.

    `pose_arkit` is ARKit's camera-to-world; rays are built in OpenCV axes, so the pose is
    converted the same way the reader converts it.
    """
    pose = pose_arkit @ np.diag([1.0, -1.0, -1.0, 1.0])
    directions = _rays(size, focal) @ pose[:3, :3].T
    origin = pose[:3, 3]

    # Slab method: the exit distance is the nearest positive bound crossing on any axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (ROOM[0] - origin) / directions
        t_high = (ROOM[1] - origin) / directions
    t_axis = np.where(directions > 0, t_high, t_low)
    t_axis[~np.isfinite(t_axis) | (t_axis <= 0)] = np.inf

    hit_axis = np.argmin(t_axis, axis=-1)
    depth = np.take_along_axis(t_axis, hit_axis[..., None], axis=-1)[..., 0]
    positive = np.take_along_axis(directions, hit_axis[..., None], axis=-1)[..., 0] > 0
    return depth, 2 * hit_axis + positive


def write(out: Path, count: int) -> None:
    raw, frames = out / "raw", out / "frames"
    (raw / "depth").mkdir(parents=True, exist_ok=True)
    (raw / "confidence").mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    depth_focal = COLOR_FOCAL * DEPTH_SIZE[0] / COLOR_SIZE[0]
    rows = []
    for i, pose in enumerate(poses(count)):
        depth, _ = cast(pose, DEPTH_SIZE, depth_focal)
        millimetres = np.where(depth <= MAX_RANGE_M, depth * 1000, 0).round().astype(np.uint16)
        cv2.imwrite(str(raw / "depth" / f"{i:06d}.png"), millimetres)
        cv2.imwrite(
            str(raw / "confidence" / f"{i:06d}.png"),
            np.where(depth <= MAX_RANGE_M, 2, 0).astype(np.uint8),
        )

        _, faces = cast(pose, COLOR_SIZE, COLOR_FOCAL)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), FACE_COLORS[faces], [cv2.IMWRITE_JPEG_QUALITY, 95])

        position = pose[:3, 3]
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x, y, z, w
        rows.append(
            f"{i / 30:.4f}, {i:06d}, {position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}, "
            f"{quaternion[0]:.9f}, {quaternion[1]:.9f}, {quaternion[2]:.9f}, {quaternion[3]:.9f}, "
            f"{COLOR_FOCAL}, {COLOR_FOCAL}, {(COLOR_SIZE[0] - 1) / 2}, {(COLOR_SIZE[1] - 1) / 2}, ,\n"
        )

    header = (
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
        " distortion_center_x, distortion_center_y\n"
    )
    (raw / "odometry.csv").write_text(header + "".join(rows))
    print(f"Wrote {count} frames to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    write(args.out, args.frames)


if __name__ == "__main__":
    main()
