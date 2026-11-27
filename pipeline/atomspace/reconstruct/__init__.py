import json

from ..config import Profile, ScenePaths
from ..gpu import gpu_stage
from ..ingest.route import Track
from . import mast3r_sfm, sensor_track
from .export import write_reconstruction


def run(paths: ScenePaths, profile: Profile) -> None:
    ingest_meta = json.loads(paths.ingest_meta.read_text())
    frames = [paths.frames / name for name in ingest_meta["keyframes"]]

    report = None
    with gpu_stage("reconstruct", paths.logs):
        if ingest_meta["track"] == Track.FAST.value:
            recon, report = sensor_track.reconstruct(
                paths.raw,
                frames,
                ingest_meta["source_indices"],
                mast3r=lambda: mast3r_sfm.reconstruct(frames, profile, cache_dir=paths.recon / "cache"),
            )
        else:
            recon = mast3r_sfm.reconstruct(frames, profile, cache_dir=paths.recon / "cache")

    write_reconstruction(recon, paths)
    if report is not None:
        # write_reconstruction clears anything derived from the previous reconstruction, so
        # the report and the up direction are written after it, not before.
        paths.logs.mkdir(parents=True, exist_ok=True)
        (paths.logs / "sensor_track.json").write_text(json.dumps(report, indent=2))
        sensor_track.write_world_up(paths.recon / "world_up.json")
        print(
            f"Sensor track: {report['lidar_points']} LiDAR points, "
            f"{report['mast3r_points']} from MASt3R"
            + (f", trajectories agree to {report['residual_m']:.3f} m "
               f"over {report['inliers']}/{report['keyframes']} cameras"
               if report["residual_m"] is not None else "")
        )
        if report["warning"]:
            print(f"WARNING: {report['warning']}")

    print(f"Reconstruct: {len(recon.image_names)} cameras, {len(recon.points)} points")
