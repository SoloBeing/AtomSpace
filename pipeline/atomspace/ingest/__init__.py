import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ..config import IngestConfig, ScenePaths
from .blur import filter_sharp
from .dedupe import select_keyframes
from .probe import probe_video
from .route import Track, choose_track, reconcile_frames, stride_for
from .slice import extract_frames, extract_strided_frames
from .stray import find_recording, read_odometry

MIN_KEYFRAMES = 3
SIDECARS = ("odometry.csv", "imu.csv", "camera_matrix.csv")
SIDECAR_DIRS = ("depth", "confidence", "distortion")


def run(video: Path, paths: ScenePaths, max_keyframes: int, cfg: IngestConfig = IngestConfig()) -> None:
    """Slices a capture into keyframes. `video` is a video file, or a Stray Scanner folder."""
    recording = find_recording(video)
    source = recording.video if recording else video
    if not source.is_file():
        raise FileNotFoundError(
            f"Not a video or a Stray Scanner recording: {video}" if video.is_dir()
            else f"Video not found: {video}"
        )
    paths.ensure()
    stored = paths.raw / f"input{source.suffix.lower()}"
    if source.resolve() != stored.resolve():
        shutil.copy2(source, stored)
    if recording:
        _copy_sidecars(recording, paths.raw)

    meta = probe_video(source, recording)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, {meta.duration_s:.1f} s, {meta.codec}"
        + (f", {meta.device_make} {meta.device_model}" if meta.device_model else "")
    )
    if meta.hdr:
        print(
            "WARNING: this is an HDR video; extracted frames may look washed out. "
            "For best results record in SDR (turn off HDR Video in the phone's camera settings)."
        )
    track = choose_track(meta)

    for old in paths.frames.glob("*.jpg"):
        old.unlink()
    candidates_dir = paths.frames / "candidates"

    if track is Track.FAST:
        if recording is None:
            raise RuntimeError(
                f"{video} sits beside a sensor recording's sidecars but is not one itself. "
                "Point at the recording folder (the one holding rgb.mp4 and odometry.csv), "
                "not at a file inside it."
            )
        candidates, source_index = _fast_candidates(recording, stored, candidates_dir, meta, cfg)
    else:
        candidates = extract_frames(stored, candidates_dir, min(cfg.extract_fps, meta.fps), cfg.max_frame_side)
        source_index = None

    sharp = filter_sharp(candidates, cfg.min_relative_sharpness, cfg.sharpness_window)
    keyframes = select_keyframes(sharp, cfg.min_motion, max_keyframes)
    if len(keyframes) < MIN_KEYFRAMES:
        shutil.rmtree(candidates_dir)
        raise RuntimeError(
            f"Only {len(keyframes)} distinct views found. Record a longer, slower walk that covers the space."
        )
    for frame in keyframes:
        shutil.copy2(frame, paths.frames / frame.name)
    shutil.rmtree(candidates_dir)

    meta_out = {
        "video": stored.name,
        "meta": asdict(meta),
        "track": track.value,
        "extract_fps": min(cfg.extract_fps, meta.fps),
        "candidates": len(candidates),
        "sharp": len(sharp),
        "keyframes": [f.name for f in keyframes],
    }
    if source_index is not None:
        # Which recording frame each keyframe was, so the sensor track can find its pose and
        # its depth map. Blur and motion filtering drop most frames, so position is not index.
        meta_out["source_indices"] = [source_index[f] for f in keyframes]
    paths.ingest_meta.write_text(json.dumps(meta_out, indent=2))
    print(f"Ingest: {len(candidates)} frames -> {len(sharp)} sharp -> {len(keyframes)} keyframes ({track.value} track)")


def _copy_sidecars(recording, raw: Path) -> None:
    """The sensor track reads these from the scene's own raw/ folder, so they travel with it."""
    for name in SIDECARS:
        if (recording.root / name).is_file():
            shutil.copy2(recording.root / name, raw / name)
    for name in SIDECAR_DIRS:
        if (recording.root / name).is_dir():
            shutil.copytree(recording.root / name, raw / name, dirs_exist_ok=True)


def _fast_candidates(recording, stored: Path, candidates_dir: Path, meta, cfg):
    """Extracts by frame index and returns the candidates with a path-to-index mapping."""
    if meta.rotation_deg:
        raise RuntimeError(
            f"This recording carries a {meta.rotation_deg} degree rotation flag, which ffmpeg "
            "applies to the extracted frames while odometry.csv's intrinsics and the depth maps "
            "stay in the sensor's own orientation, so the two would no longer line up. Use the "
            "plain-video track for this capture, which handles rotation correctly."
        )
    stride = stride_for(meta.fps, cfg.extract_fps)
    pairs = extract_strided_frames(stored, candidates_dir, stride, cfg.max_frame_side)
    rows = len(read_odometry(recording.odometry).frames)
    pairs = reconcile_frames(pairs, rows, stride)
    return [path for _, path in pairs], {path: index for index, path in pairs}
