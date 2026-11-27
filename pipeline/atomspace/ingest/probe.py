import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}


@dataclass(frozen=True)
class VideoMeta:
    width: int  # as extracted (upright, after rotation)
    height: int
    fps: float
    duration_s: float
    rotation_deg: int  # 0/90/180/270 container flag; ffmpeg applies it on extraction
    codec: str
    hdr: bool
    device_make: str | None
    device_model: str | None
    focal_length_35mm: float | None
    has_ar_poses: bool
    has_depth: bool


def _ffprobe(video: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {video}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return round(float(side["rotation"])) % 360
    tag = stream.get("tags", {}).get("rotate")
    return int(tag) % 360 if tag else 0


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key, "0/0")
        num, _, den = value.partition("/")
        if den and int(den) != 0 and int(num) != 0:
            return float(Fraction(int(num), int(den)))
    raise RuntimeError("Could not determine the video's frame rate.")


def _tag(tags: dict[str, str], *suffixes: str) -> str | None:
    for key, value in tags.items():
        if key.endswith(suffixes):
            return value.strip() or None
    return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def probe_video(video: Path, recording: "Recording | None" = None) -> VideoMeta:
    """Reads container metadata with ffprobe.

    Plain .mp4 files never carry AR poses or depth; those come from capture apps that
    export sidecar files next to the video, so has_ar_poses/has_depth check for them.
    `recording`, when given, is an already-identified Stray Scanner folder and answers both
    questions directly, which is what the fast track routes on.
    """
    info = _ffprobe(video)
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise RuntimeError(f"{video} has no video stream.")

    rotation = _rotation(stream)
    width, height = int(stream["width"]), int(stream["height"])
    if rotation in (90, 270):
        width, height = height, width

    fmt = info.get("format", {})
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    tags = {k.lower(): v for k, v in {**fmt.get("tags", {}), **stream.get("tags", {})}.items()}

    return VideoMeta(
        width=width,
        height=height,
        fps=_fps(stream),
        duration_s=duration,
        rotation_deg=rotation,
        codec=stream.get("codec_name", "unknown"),
        hdr=stream.get("color_transfer") in HDR_TRANSFERS,
        device_make=_tag(tags, "make", "manufacturer"),
        device_model=_tag(tags, ".model"),
        focal_length_35mm=_float_or_none(_tag(tags, "35mm_equivalent")),
        has_ar_poses=recording is not None or (video.parent / "odometry.csv").is_file(),
        has_depth=(recording.depth is not None) if recording else (video.parent / "depth").is_dir(),
    )
