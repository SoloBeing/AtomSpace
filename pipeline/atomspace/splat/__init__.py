import json

from ..config import Profile, ScenePaths, SplatOptions
from ..gpu import gpu_stage
from .compress import ply_to_splat
from .train import train


def run(paths: ScenePaths, profile: Profile, options: SplatOptions = SplatOptions()) -> None:
    with gpu_stage("splat", paths.logs):
        train(paths.frames, paths.transforms, paths.dense_points, paths.splat_ply, profile, options)
    # The bundle step reads this: an antialiased splat needs the viewer's matching render mode.
    paths.splat_meta.write_text(json.dumps({"antialiased": options.antialiased}, indent=2))
    ply_to_splat(paths.splat_ply, paths.splat)
    print(f"Splat: {paths.splat} ({paths.splat.stat().st_size / 1e6:.1f} MB)")
