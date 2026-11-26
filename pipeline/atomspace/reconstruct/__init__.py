import json

from ..config import ScenePaths
from ..gpu import gpu_stage
from ..ingest.route import Track
from . import mapanything_sfm, sensor_track
from .export import write_reconstruction


def run(paths: ScenePaths) -> None:
    ingest_meta = json.loads(paths.ingest_meta.read_text())
    frames = [paths.frames / name for name in ingest_meta["keyframes"]]

    with gpu_stage("reconstruct", paths.logs):
        if ingest_meta["track"] == Track.FAST.value:
            recon = sensor_track.reconstruct(paths.raw, frames)
        else:
            recon = mapanything_sfm.reconstruct(frames, work_dir=paths.recon / "ba")

    write_reconstruction(recon, paths)
    print(f"Reconstruct: {len(recon.image_names)} cameras, {len(recon.points)} points")
