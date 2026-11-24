from pathlib import Path

import cv2
import numpy as np

# Scores are computed at a fixed size so they're comparable across videos.
ANALYSIS_SIDE = 640
PROGRESS_EVERY = 500


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read frame {path}")
    return image


def _fit(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    s = ANALYSIS_SIDE / max(h, w)
    if s >= 1:
        return gray
    return cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian on the grayscale image. Higher means sharper."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(_fit(gray), cv2.CV_64F).var())


def filter_sharp(frames: list[Path], min_relative: float, window: int) -> list[Path]:
    """Keeps frames at least `min_relative` times as sharp as the median of the `window`
    frames around them. Relative, because a sharp shot of a plain wall scores lower than
    a blurry shot of a bookshelf.
    """
    if not frames:
        return []
    scores = []
    for i, f in enumerate(frames, 1):
        scores.append(sharpness(load_gray(f)))
        if i % PROGRESS_EVERY == 0:
            print(f"Sharpness: {i}/{len(frames)}", flush=True)
    scores = np.array(scores)
    half = window // 2
    kept = []
    for i, frame in enumerate(frames):
        local = scores[max(0, i - half) : i + half + 1]
        if scores[i] >= min_relative * np.median(local):
            kept.append(frame)
    return kept
