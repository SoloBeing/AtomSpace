from pathlib import Path

import cv2
import numpy as np

from .blur import PROGRESS_EVERY, load_gray

ANALYSIS_WIDTH = 320
# Below this peak response the phase-correlation shift is noise (e.g. a featureless wall).
MIN_PHASE_RESPONSE = 0.05


def _small(path: Path) -> np.ndarray:
    gray = load_gray(path)
    h, w = gray.shape
    size = (ANALYSIS_WIDTH, max(1, round(h * ANALYSIS_WIDTH / w)))
    return np.ascontiguousarray(cv2.resize(gray, size, interpolation=cv2.INTER_AREA))


def step_motion(a: np.ndarray, b: np.ndarray) -> float:
    """How far the view moved between two consecutive frames, as a fraction of image width.

    Phase correlation measures panning/turning exactly even for large shifts; dense optical
    flow catches walking forward (radial flow, no net shift) but is only reliable for small
    steps. Taking the larger covers both.
    """
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(np.float32(a), np.float32(b), window)
    shift = float(np.hypot(dx, dy)) if response >= MIN_PHASE_RESPONSE else 0.0
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=4, winsize=21, iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    local = float(np.median(np.linalg.norm(flow, axis=2)))
    return max(shift, local) / a.shape[1]


def select_keyframes(frames: list[Path], min_motion: float, max_keyframes: int) -> list[Path]:
    """Keeps a frame once the view has moved `min_motion` (fraction of image width, summed
    over consecutive frames) since the last kept frame. If more than `max_keyframes` survive,
    they are subsampled evenly; since keyframes are about the same motion apart, coverage
    stays uniform.
    """
    if not frames:
        return []
    kept = [frames[0]]
    previous = _small(frames[0])
    travelled = 0.0
    for i, frame in enumerate(frames[1:], 2):
        if i % PROGRESS_EVERY == 0:
            print(f"Motion: {i}/{len(frames)}", flush=True)
        current = _small(frame)
        travelled += step_motion(previous, current)
        previous = current
        if travelled >= min_motion:
            kept.append(frame)
            travelled = 0.0
    if len(kept) > max_keyframes:
        print(f"Motion: {len(kept)} distinct views, thinning evenly to {max_keyframes}", flush=True)
        idx = np.linspace(0, len(kept) - 1, max_keyframes).round().astype(int)
        kept = [kept[i] for i in idx]
    return kept
