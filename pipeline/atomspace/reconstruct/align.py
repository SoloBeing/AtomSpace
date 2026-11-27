"""Fits a similarity transform between two camera trajectories over the same keyframes.

MASt3R cannot be given known poses — `init['cam2w']` hits an unimplemented assert in
mast3r.cloud_opt.sparse_ga — so the sensor track lets it solve poses on its own and then
maps its whole reconstruction onto ARKit's metric frame afterwards. Both trajectories
describe the same cameras in the same order, so the correspondence is already known and
only the transform has to be found.
"""

from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3
# Ratio of the trajectory's thinnest to thickest principal extent. A walk down a corridor
# leaves rotation about the walking axis unconstrained, so a fit would be confident and wrong.
MIN_SPREAD = 0.05


@dataclass(frozen=True)
class Similarity:
    """`scale * rotation @ p + translation`, with the fit's quality beside it."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    residual_m: float  # RMS distance over the inliers, in the target's units
    inliers: np.ndarray  # (N,) bool

    @property
    def matrix(self) -> np.ndarray:
        """The same transform as a 4x4, for composing with other poses."""
        out = np.eye(4)
        out[:3, :3] = self.scale * self.rotation
        out[:3, 3] = self.translation
        return out

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping `source` onto `target` (Umeyama 1991).

    Returns (scale, rotation, translation). The rotation is always proper: a reflection
    would fit a mirrored cloud better, but no rigid camera path is mirrored, so a negative
    determinant means the data is wrong rather than the transform.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean

    covariance = target_centred.T @ source_centred / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1] = -1
    rotation = u @ np.diag(correction) @ vt

    variance = (source_centred**2).sum() / len(source)
    scale = float((singular * correction).sum() / variance) if variance > 0 else 1.0
    return scale, rotation, target_mean - scale * rotation @ source_mean


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    threshold_m: float = 0.05,
    iters: int = 200,
    seed: int = 0,
) -> Similarity:
    """RANSAC similarity fit, so a few badly-placed cameras can't drag the whole transform.

    `source` is the up-to-scale trajectory and `target` the metric one, so the residual
    comes out in metres.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) != len(target):
        raise ValueError(f"{len(source)} source points against {len(target)} target points")
    if len(source) < MIN_POINTS:
        raise RuntimeError(f"A trajectory fit needs at least {MIN_POINTS} cameras, got {len(source)}")
    _refuse_degenerate(target)

    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), bool)
    for _ in range(iters):
        sample = rng.choice(len(source), MIN_POINTS, replace=False)
        try:
            candidate = _fit(source, target, sample)
        except np.linalg.LinAlgError:
            continue
        inliers = _errors(candidate, source, target) < threshold_m
        if inliers.sum() > best.sum():
            best = inliers

    if best.sum() < MIN_POINTS:
        best = np.ones(len(source), bool)  # nothing agreed; fit everything and let the residual say so
    fit = _fit(source, target, np.flatnonzero(best))
    residual = float(np.sqrt((_errors(fit, source, target)[best] ** 2).mean()))
    return Similarity(fit[0], fit[1], fit[2], residual, best)


def _refuse_degenerate(target: np.ndarray) -> None:
    singular = np.linalg.svd(target - target.mean(axis=0), compute_uv=False)
    if singular[0] <= 0 or singular[-1] / singular[0] < MIN_SPREAD:
        raise RuntimeError(
            "The camera path is too close to a straight line to fit a transform against. "
            "Capture a loop that covers the space rather than a single pass through it."
        )


def _fit(source, target, index) -> tuple[float, np.ndarray, np.ndarray]:
    return umeyama(source[index], target[index])


def _errors(fit, source, target) -> np.ndarray:
    scale, rotation, translation = fit
    return np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
