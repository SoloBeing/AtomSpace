import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from atomspace import ingest
from atomspace.config import ScenePaths
from atomspace.ingest.probe import probe_video
from atomspace.ingest import slice as slice_module
from atomspace.ingest.slice import extract_strided_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
# probe_video shells out to ffprobe, which some ffmpeg distributions ship separately.
needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")


@pytest.fixture
def panning_video(tmp_path):
    """4 s, 30 fps, 640x360 window panning across a 1280x720 test pattern at 150 px/s."""
    path = tmp_path / "pan.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
            "-vf", "crop=640:360:'min(t*150,640)':180", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


@needs_ffprobe
def test_probe_reads_basic_metadata(panning_video):
    meta = probe_video(panning_video)
    assert (meta.width, meta.height) == (640, 360)
    assert meta.fps == pytest.approx(30)
    assert meta.duration_s == pytest.approx(4, abs=0.2)
    assert meta.rotation_deg == 0
    assert not meta.hdr
    assert not meta.has_ar_poses


@needs_ffprobe
def test_ingest_end_to_end(panning_video, tmp_path):
    paths = ScenePaths(tmp_path / "scenes" / "pan")
    ingest.run(panning_video, paths, max_keyframes=150)

    data = json.loads(paths.ingest_meta.read_text())
    assert data["track"] == "fallback"
    assert data["candidates"] == pytest.approx(24, abs=1)
    assert 3 <= len(data["keyframes"]) < data["candidates"]
    assert sorted(p.name for p in paths.frames.glob("*.jpg")) == data["keyframes"]
    assert not (paths.frames / "candidates").exists()
    assert (paths.raw / "input.mp4").is_file()


@needs_ffprobe
def test_ingest_rejects_static_video(tmp_path):
    video = tmp_path / "still.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=3",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    with pytest.raises(RuntimeError, match="distinct views"):
        ingest.run(video, ScenePaths(tmp_path / "scenes" / "still"), max_keyframes=150)


def test_strided_extraction_reports_true_source_indices(panning_video, tmp_path):
    """4 s at 30 fps is 120 frames; stride 5 keeps every fifth, and each kept frame has to
    report the index it had in the recording, because that is what indexes odometry.csv."""
    got = extract_strided_frames(panning_video, tmp_path / "out", stride=5, max_side=1920)

    assert [i for i, _ in got] == list(range(0, 120, 5))
    assert all(path.is_file() for _, path in got)
    assert [path.name for _, path in got] == sorted(path.name for _, path in got)


def test_stride_of_one_keeps_every_frame(panning_video, tmp_path):
    got = extract_strided_frames(panning_video, tmp_path / "out", stride=1, max_side=1920)
    assert [i for i, _ in got] == list(range(120))


def test_strided_extraction_caps_the_long_side(panning_video, tmp_path):
    got = extract_strided_frames(panning_video, tmp_path / "out", stride=20, max_side=320)
    assert max(Image.open(got[0][1]).size) == 320


def _recording(tmp_path, frames=120, rotate=0, odometry_rows=None):
    """A Stray Scanner folder: a real panning rgb.mp4 plus the sidecars for every frame."""
    root = tmp_path / "capture"
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=size=1280x720:rate=30:duration={frames / 30}",
         "-vf", "crop=640:360:'min(t*150,640)':180", "-pix_fmt", "yuv420p",
         str(root / "unrotated.mp4")],
        check=True,
    )
    # A real Display Matrix, which is what probe.py reads; the legacy `rotate` tag is
    # ignored by current ffmpeg, so writing one would test nothing.
    subprocess.run(
        ["ffmpeg", "-v", "error", *(["-display_rotation", str(rotate)] if rotate else []),
         "-i", str(root / "unrotated.mp4"), "-c", "copy", str(root / "rgb.mp4")],
        check=True,
    )
    (root / "unrotated.mp4").unlink()
    header = ("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
              " distortion_center_x, distortion_center_y\n")
    rows = odometry_rows if odometry_rows is not None else frames
    lines = [f"{i / 30:.4f}, {i:06d}, {i * 0.01}, 1.4, 0, 0, 0, 0, 1, 320, 320, 319.5, 179.5, ,\n"
             for i in range(rows)]
    (root / "odometry.csv").write_text(header + "".join(lines))
    for i in range(rows):
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((24, 32), 2000, np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((24, 32), 2, np.uint8))
    return root


@needs_ffprobe
def test_ingest_of_a_recording_copies_sidecars_and_records_indices(tmp_path):
    paths = ScenePaths(tmp_path / "scenes" / "room")

    ingest.run(_recording(tmp_path), paths, max_keyframes=150)

    data = json.loads(paths.ingest_meta.read_text())
    assert data["track"] == "fast"
    assert len(data["source_indices"]) == len(data["keyframes"])
    assert data["source_indices"] == sorted(data["source_indices"])
    assert (paths.raw / "odometry.csv").is_file()
    assert (paths.raw / "depth").is_dir()
    assert (paths.raw / "input.mp4").is_file()


@needs_ffprobe
def test_source_indices_are_multiples_of_the_stride(tmp_path):
    """30 fps against a 6 fps target is a stride of 5, and index is position times stride."""
    paths = ScenePaths(tmp_path / "scenes" / "room")
    ingest.run(_recording(tmp_path), paths, max_keyframes=150)

    indices = json.loads(paths.ingest_meta.read_text())["source_indices"]
    assert all(i % 5 == 0 for i in indices)
    assert max(indices) < 120


@needs_ffprobe
def test_every_keyframe_has_a_depth_frame_to_match(tmp_path):
    """The point of carrying source indices: each one must name a file that exists."""
    paths = ScenePaths(tmp_path / "scenes" / "room")
    ingest.run(_recording(tmp_path), paths, max_keyframes=150)

    data = json.loads(paths.ingest_meta.read_text())
    for index in data["source_indices"]:
        assert (paths.raw / "depth" / f"{index:06d}.png").is_file()


@needs_ffprobe
def test_ingest_refuses_a_rotated_recording(tmp_path):
    with pytest.raises(RuntimeError, match="rotation"):
        ingest.run(_recording(tmp_path, rotate=90), ScenePaths(tmp_path / "s"), max_keyframes=150)


@needs_ffprobe
def test_ingest_refuses_mismatched_frame_counts(tmp_path):
    recording = _recording(tmp_path, frames=120, odometry_rows=40)
    with pytest.raises(RuntimeError, match="odometry"):
        ingest.run(recording, ScenePaths(tmp_path / "s"), max_keyframes=150)


@needs_ffprobe
def test_a_plain_video_still_takes_the_fallback_track(panning_video, tmp_path):
    paths = ScenePaths(tmp_path / "scenes" / "pan")
    ingest.run(panning_video, paths, max_keyframes=150)
    data = json.loads(paths.ingest_meta.read_text())
    assert data["track"] == "fallback"
    assert "source_indices" not in data


@needs_ffprobe
def test_pointing_at_the_video_inside_a_recording_is_explained(tmp_path):
    """`atomspace run myroom capture/rgb.mp4` is the natural variation of the documented
    command, and it used to reach an AttributeError deep in the fast track."""
    recording = _recording(tmp_path)
    with pytest.raises(RuntimeError, match="folder"):
        ingest.run(recording / "rgb.mp4", ScenePaths(tmp_path / "s"), max_keyframes=150)


def test_an_ffmpeg_too_old_for_fps_mode_is_diagnosed(tmp_path, monkeypatch):
    """-fps_mode arrived in ffmpeg 5.0, and env/environment.yml used to leave the version
    open, so the whole fast track died on a raw 'Unrecognized option' complaint."""
    class Failed:
        returncode = 1
        stderr = "Unrecognized option 'fps_mode'.\nError splitting the argument list."

    monkeypatch.setattr(slice_module.subprocess, "run", lambda *a, **kw: Failed())
    with pytest.raises(RuntimeError, match="ffmpeg 5"):
        extract_strided_frames(tmp_path / "x.mp4", tmp_path / "out", stride=5, max_side=1920)
