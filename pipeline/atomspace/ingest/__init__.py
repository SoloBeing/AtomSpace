import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ..config import IngestConfig, ScenePaths
from .blur import filter_sharp
from .dedupe import select_keyframes
from .probe import probe_video
from .route import choose_track
from .slice import extract_frames

MIN_KEYFRAMES = 3


def run(video: Path, paths: ScenePaths, max_keyframes: int, cfg: IngestConfig = IngestConfig()) -> None:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    paths.ensure()
    meta = probe_video(video)
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

    # Nothing of the previous ingest is touched until the new video has proved usable.
    candidates_dir = paths.frames / "candidates"
    candidates = extract_frames(video, candidates_dir, min(cfg.extract_fps, meta.fps), cfg.max_frame_side)
    sharp = filter_sharp(candidates, cfg.min_relative_sharpness, cfg.sharpness_window)
    keyframes = select_keyframes(sharp, cfg.min_motion, max_keyframes)
    if len(keyframes) < MIN_KEYFRAMES:
        shutil.rmtree(candidates_dir)
        raise RuntimeError(
            f"Only {len(keyframes)} distinct views found. Record a longer, slower walk that covers the space."
        )

    stored = paths.raw / f"input{video.suffix.lower()}"
    if video.resolve() != stored.resolve():
        shutil.copy2(video, stored)
    for old in paths.frames.glob("*.jpg"):
        old.unlink()
    for frame in keyframes:
        shutil.copy2(frame, paths.frames / frame.name)
    shutil.rmtree(candidates_dir)

    paths.ingest_meta.write_text(
        json.dumps(
            {
                "video": stored.name,
                "meta": asdict(meta),
                "track": track.value,
                "extract_fps": min(cfg.extract_fps, meta.fps),
                "candidates": len(candidates),
                "sharp": len(sharp),
                "keyframes": [f.name for f in keyframes],
            },
            indent=2,
        )
    )
    print(f"Ingest: {len(candidates)} frames -> {len(sharp)} sharp -> {len(keyframes)} keyframes ({track.value} track)")
