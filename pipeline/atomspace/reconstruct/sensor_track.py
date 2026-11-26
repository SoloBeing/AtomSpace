from pathlib import Path

from .export import Reconstruction


def reconstruct(raw_dir: Path, frames: list[Path]) -> Reconstruction:
    """Fast track: takes ARKit/ARCore poses (already metric) from the capture's sidecar files
    and matches them to keyframes by timestamp. Dense points come from LiDAR depth when present,
    otherwise from MapAnything with poses held fixed.
    """
    raise NotImplementedError
