import json
import shutil
import subprocess

import pytest

from atomspace import ingest
from atomspace.config import ScenePaths
from atomspace.ingest.probe import probe_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")


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


def test_probe_reads_basic_metadata(panning_video):
    meta = probe_video(panning_video)
    assert (meta.width, meta.height) == (640, 360)
    assert meta.fps == pytest.approx(30)
    assert meta.duration_s == pytest.approx(4, abs=0.2)
    assert meta.rotation_deg == 0
    assert not meta.hdr
    assert not meta.has_ar_poses


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


def test_ingest_rejects_static_video_and_keeps_the_previous_ingest(panning_video, tmp_path):
    paths = ScenePaths(tmp_path / "scenes" / "pan")
    ingest.run(panning_video, paths, max_keyframes=150)
    frames_before = sorted(p.name for p in paths.frames.glob("*.jpg"))

    video = tmp_path / "still.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=3",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    with pytest.raises(RuntimeError, match="distinct views"):
        ingest.run(video, paths, max_keyframes=150)

    assert sorted(p.name for p in paths.frames.glob("*.jpg")) == frames_before
    assert (paths.raw / "input.mp4").read_bytes() == panning_video.read_bytes()
    assert not (paths.frames / "candidates").exists()
