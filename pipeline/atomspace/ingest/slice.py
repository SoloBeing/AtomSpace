import shutil
import subprocess
from pathlib import Path


def _scale_filter(max_side: int) -> str:
    return (
        f"scale='if(gte(iw,ih),min({max_side},iw),-2)'"
        f":'if(gte(iw,ih),-2,min({max_side},ih))'"
    )


def extract_strided_frames(
    video: Path, out_dir: Path, stride: int, max_side: int
) -> list[tuple[int, Path]]:
    """Extracts every `stride`-th frame, paired with the index it had in the recording.

    The sensor track matches keyframes to `odometry.csv` rows and `depth/NNNNNN.png` files by
    that index, so frames are selected by position (`select` on n) rather than resampled in
    time the way `extract_frames` does: an fps filter would resample and leave nothing exact
    to match against. `-fps_mode passthrough` keeps ffmpeg from adding or dropping frames of
    its own, so the nth output is frame n * stride.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    select = "1" if stride == 1 else f"not(mod(n\\,{stride}))"
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-vf", f"select='{select}',{_scale_filter(max_side)}",
            "-fps_mode", "passthrough", "-q:v", "2",
            str(out_dir / "%06d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "fps_mode" in result.stderr:
            raise RuntimeError(
                "This ffmpeg is too old for the sensor track: -fps_mode arrived in ffmpeg 5. "
                "Without it ffmpeg pads the output to a constant frame rate, and every keyframe "
                "would be paired with the wrong pose. Install ffmpeg 5 or newer, or use the "
                f"plain-video track. ffmpeg said: {result.stderr.strip()}"
            )
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {video}.")
    return [(i * stride, path) for i, path in enumerate(frames)]


def extract_frames(video: Path, out_dir: Path, fps: float, max_side: int) -> list[Path]:
    """Extracts JPEG frames at `fps`, upright (ffmpeg applies the rotation flag) and with
    the long side capped at `max_side`. Returns paths in time order; any previous contents
    of `out_dir` are removed first.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"fps={fps},{_scale_filter(max_side)}",
        "-q:v", "2",
        str(out_dir / "%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {video}.")
    return frames
