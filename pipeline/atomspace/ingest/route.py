from enum import Enum
from pathlib import Path

from .probe import VideoMeta


class Track(str, Enum):
    FALLBACK = "fallback"
    FAST = "fast"


def choose_track(meta: VideoMeta) -> Track:
    """The sensor fast track whenever the capture brought ARKit poses with it.

    Depth is not required here: ARKit's poses are metric with or without a LiDAR sensor, and
    that alone removes the manual scale step. The reconstruct stage is what needs depth, and
    says so if it is missing.
    """
    return Track.FAST if meta.has_ar_poses else Track.FALLBACK


def stride_for(fps: float, target_fps: float) -> int:
    """How many recorded frames to step over to land near `target_fps`.

    The fast track extracts by frame index so keyframes can be matched to odometry rows, but
    the blur and motion filters are tuned in units of frames at `target_fps`: at 60 fps a
    15-frame sharpness window spans 0.12 s, over which motion blur is constant, so the local
    median goes blurry too and the filter stops rejecting anything.
    """
    return max(1, round(fps / target_fps)) if fps > 0 else 1


def reconcile_frames(
    pairs: list[tuple[int, Path]], odometry_rows: int, stride: int
) -> list[tuple[int, Path]]:
    """Drops extracted frames that the odometry has no pose for, or refuses the pair outright.

    The whole fast track rests on frame N of the video being row N of `odometry.csv` and
    `depth/NNNNNN.png`. A recorder that stops writing poses a frame or two before it stops
    writing video is ordinary, and the overhang is simply trimmed. A larger disagreement
    means these are not two views of the same recording, and carrying on would attach every
    pose to the wrong image — a scene that still looks like a room and is wrong throughout.
    """
    if not pairs:
        return pairs
    implied = pairs[-1][0] + 1  # frames the video is known to have reached
    if abs(implied - odometry_rows) > stride:
        raise RuntimeError(
            f"The video and the odometry disagree: the video reaches frame {implied} but "
            f"odometry.csv has {odometry_rows} rows. They are not the same recording, so "
            "every pose would be matched to the wrong frame. Re-export the capture."
        )
    kept = [pair for pair in pairs if pair[0] < odometry_rows]
    if len(kept) != len(pairs):
        print(
            f"Trimming {len(pairs) - len(kept)} frame(s) past the end of odometry.csv "
            f"({odometry_rows} rows)."
        )
    return kept
