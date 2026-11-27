import json
import shutil
import subprocess

import pytest
from PIL import Image

from atomspace import ingest
from atomspace.config import ScenePaths
from atomspace.ingest.probe import probe_video
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
