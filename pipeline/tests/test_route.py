from pathlib import Path

import pytest

from atomspace.ingest.probe import VideoMeta
from atomspace.ingest.route import Track, choose_track, reconcile_frames, stride_for


def _meta(**kw) -> VideoMeta:
    defaults = dict(
        width=1920, height=1440, fps=30.0, duration_s=4.0, rotation_deg=0, codec="hevc",
        hdr=False, device_make="Apple", device_model="iPhone", focal_length_35mm=None,
        has_ar_poses=False, has_depth=False,
    )
    return VideoMeta(**{**defaults, **kw})


def test_choose_track_is_fast_with_ar_poses():
    assert choose_track(_meta(has_ar_poses=True, has_depth=True)) is Track.FAST
    assert choose_track(_meta(has_ar_poses=False)) is Track.FALLBACK


def test_poses_without_depth_still_take_the_fast_track():
    """ARKit's poses are metric with or without a LiDAR sensor, and that alone removes the
    manual scale step. The reconstruct stage is what insists on depth."""
    assert choose_track(_meta(has_ar_poses=True, has_depth=False)) is Track.FAST


def test_stride_keeps_the_filters_tuned_rate():
    assert stride_for(fps=60.0, target_fps=6.0) == 10
    assert stride_for(fps=30.0, target_fps=6.0) == 5
    assert stride_for(fps=29.97, target_fps=6.0) == 5


def test_stride_is_never_below_one():
    """A video slower than the target rate must not ask for every 0th frame."""
    assert stride_for(fps=4.0, target_fps=6.0) == 1
    assert stride_for(fps=0.0, target_fps=6.0) == 1


def _pairs(count, stride):
    return [(i * stride, Path(f"{i:06d}.jpg")) for i in range(count)]


def test_reconcile_keeps_everything_when_the_counts_agree():
    pairs = _pairs(24, stride=5)  # frames 0..115 of a 120-frame video
    assert reconcile_frames(pairs, odometry_rows=120, stride=5) == pairs


def test_reconcile_truncates_a_short_overhang():
    """A video one frame longer than the odometry is normal: the recorder stops writing poses
    a frame before it stops writing video."""
    pairs = _pairs(24, stride=5)

    got = reconcile_frames(pairs, odometry_rows=114, stride=5)

    assert [i for i, _ in got] == list(range(0, 114, 5))
    assert len(got) == 23


def test_reconcile_refuses_a_large_mismatch():
    """Beyond one stride the two streams are not the same recording, and every pose would
    attach to the wrong image."""
    with pytest.raises(RuntimeError, match="odometry"):
        reconcile_frames(_pairs(24, stride=5), odometry_rows=60, stride=5)


def test_reconcile_refuses_odometry_far_longer_than_the_video():
    with pytest.raises(RuntimeError, match="odometry"):
        reconcile_frames(_pairs(24, stride=5), odometry_rows=400, stride=5)


def test_reconcile_names_both_counts_so_the_mismatch_is_diagnosable():
    with pytest.raises(RuntimeError, match="116.*60|60.*116"):
        reconcile_frames(_pairs(24, stride=5), odometry_rows=60, stride=5)
