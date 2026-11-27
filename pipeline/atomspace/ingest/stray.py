"""Reads a Stray Scanner recording (https://github.com/strayrobots/scanner).

A recording is a folder holding `rgb.mp4` (HEVC), `odometry.csv`, `imu.csv` and per-frame
`depth/` and `confidence/` PNGs. Only the colour video and the odometry are required here;
without `depth/` the capture still has metric poses, just no measured geometry.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

VIDEO_NAME = "rgb.mp4"
ODOMETRY_NAME = "odometry.csv"

# ARKit cameras are +X right, +Y up, -Z forward; OpenCV is +X right, +Y down, +Z forward.
# export.py expects OpenCV and applies its own flip on the way out to OpenGL.
ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

# docs/format.md, in order. The last two may be empty when the device reports no distortion.
ODOMETRY_COLUMNS = (
    "timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw",
    "fx", "fy", "cx", "cy", "distortion_center_x", "distortion_center_y",
)
DEPTH_SCALE = 1000.0  # depth PNGs hold millimetres


@dataclass(frozen=True)
class Recording:
    """A Stray Scanner capture folder. `depth` and `confidence` are None when absent, and
    `video` is None for a scene's own raw/ folder, where ingest has already renamed it."""

    root: Path
    video: Path | None
    odometry: Path
    depth: Path | None
    confidence: Path | None


@dataclass(frozen=True)
class Odometry:
    """Per-frame camera path. `cam_to_world` is OpenCV, metres, in ARKit's world frame."""

    timestamps: np.ndarray  # (N,) seconds
    frames: np.ndarray  # (N,) int, indexes depth/<frame>.png
    cam_to_world: np.ndarray  # (N, 4, 4)
    intrinsics: np.ndarray  # (N, 3, 3), colour-frame pixels


def find_recording(path: Path) -> Recording | None:
    """The recording to ingest at `path`, or None if it isn't one. A plain video file is not
    one, and neither is a folder without the capture's own `rgb.mp4`."""
    found = find_sidecars(path)
    if found is None or not (path / VIDEO_NAME).is_file():
        return None
    return Recording(
        root=found.root,
        video=path / VIDEO_NAME,
        odometry=found.odometry,
        depth=found.depth,
        confidence=found.confidence,
    )


def find_sidecars(path: Path) -> Recording | None:
    """The same folder identified by its odometry alone, for stages that run after ingest.

    A scene's `raw/` holds the copied sidecars beside a video ingest renamed to `input.<ext>`,
    and by then the keyframes have already been extracted, so the video is of no further use.
    """
    if not path.is_dir():
        return None
    odometry = path / ODOMETRY_NAME
    if not odometry.is_file():
        return None
    optional = [path / name if (path / name).is_dir() else None for name in ("depth", "confidence")]
    return Recording(root=path, video=None, odometry=odometry, depth=optional[0], confidence=optional[1])


def read_odometry(path: Path) -> Odometry:
    """Parses odometry.csv. Positions and quaternions are the camera's pose in ARKit's world
    frame, converted here to the OpenCV camera convention."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path.name} is empty")
    header = tuple(name.strip() for name in lines[0].split(","))
    if header[:len(ODOMETRY_COLUMNS)] != ODOMETRY_COLUMNS:
        raise ValueError(
            f"{path.name} does not have Stray Scanner's columns; found {header}"
        )

    rows = [line.split(",") for line in lines[1:] if line.strip()]
    if not rows:
        raise ValueError(f"{path.name} has a header but no rows")
    values = np.array([[_float(cell) for cell in row[:13]] for row in rows])

    quaternions = Rotation.from_quat(values[:, 5:9])  # scipy takes x, y, z, w — the file's order
    cam_to_world = np.tile(np.eye(4), (len(values), 1, 1))
    cam_to_world[:, :3, :3] = quaternions.as_matrix()
    cam_to_world[:, :3, 3] = values[:, 2:5]

    intrinsics = np.tile(np.eye(3), (len(values), 1, 1))
    intrinsics[:, 0, 0], intrinsics[:, 1, 1] = values[:, 9], values[:, 10]
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = values[:, 11], values[:, 12]

    return Odometry(
        timestamps=values[:, 0],
        frames=values[:, 1].astype(np.int64),
        cam_to_world=cam_to_world @ ARKIT_TO_OPENCV,
        intrinsics=intrinsics,
    )


def read_depth_m(path: Path) -> np.ndarray:
    """(H, W) float32 metres from a 16-bit depth PNG. Unmeasured pixels come back as 0."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read depth frame {path}")
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError(f"{path.name} is not a single-channel 16-bit depth PNG")
    return image.astype(np.float32) / DEPTH_SCALE


def read_confidence(path: Path) -> np.ndarray:
    """(H, W) uint8 of ARKit's 0/1/2 confidence levels. 0 is unreliable."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read confidence frame {path}")
    if image.ndim != 2:
        raise ValueError(f"{path.name} is not a single-channel confidence PNG")
    return image.astype(np.uint8)


def _float(cell: str) -> float:
    """Empty cells (an absent distortion centre) read as NaN rather than failing."""
    text = cell.strip()
    return float(text) if text else float("nan")
