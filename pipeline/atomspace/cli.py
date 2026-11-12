import argparse
from pathlib import Path

from . import bundle, ingest, navmesh, reconstruct, splat
from .config import PROFILES, ScenePaths, SplatOptions
from .gpu import resolve_profile
from .reconstruct.scale import ScaleReference, metric_factor, save_reference


def _paths(args: argparse.Namespace) -> ScenePaths:
    return ScenePaths(Path(args.scenes_root) / args.scene)


def cmd_run(args: argparse.Namespace) -> None:
    paths = _paths(args)
    profile = resolve_profile(args.profile)
    print(f"Profile: {profile.name}")
    ingest.run(Path(args.video), paths, profile.max_keyframes)
    reconstruct.run(paths, profile)
    splat.run(paths, profile)
    if metric_factor(paths) is not None:
        navmesh.run(paths)
    else:
        print(
            "Skipping navmesh: no metric scale yet. Set it with the viewer's scale tool "
            f"or `atomspace scale {args.scene} ...`, then run `atomspace navmesh` and `atomspace bundle`."
        )
    bundle.run(paths)


def cmd_ingest(args: argparse.Namespace) -> None:
    ingest.run(Path(args.video), _paths(args), resolve_profile(args.profile).max_keyframes)


def cmd_reconstruct(args: argparse.Namespace) -> None:
    reconstruct.run(_paths(args), resolve_profile(args.profile))


def cmd_splat(args: argparse.Namespace) -> None:
    options = SplatOptions(
        max_aspect=args.max_aspect, random_background=args.random_background, antialiased=args.antialiased
    )
    splat.run(_paths(args), resolve_profile(args.profile), options)


def cmd_scale(args: argparse.Namespace) -> None:
    ref = ScaleReference(a=tuple(args.a), b=tuple(args.b), distance_m=args.distance)
    save_reference(ref, _paths(args).scale)
    print(f"Scale factor {ref.factor():.4f} saved.")
    print(f"Next: atomspace navmesh {args.scene} && atomspace bundle {args.scene}")


def cmd_navmesh(args: argparse.Namespace) -> None:
    navmesh.run(_paths(args))


def cmd_bundle(args: argparse.Namespace) -> None:
    bundle.run(_paths(args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atomspace", description="Video to 3D indoor scene bundle.")
    parser.add_argument("--scenes-root", default="scenes", help="folder holding one subfolder per scene")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="override the auto-detected VRAM profile")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help: str, video: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("scene", help="scene name (folder under --scenes-root)")
        if video:
            p.add_argument("video", help="path to the captured .mp4/.mov")
        p.set_defaults(fn=fn)
        return p

    add("run", cmd_run, "full pipeline: ingest -> reconstruct -> splat -> navmesh -> bundle", video=True)
    add("ingest", cmd_ingest, "slice frames, drop blurry/duplicate ones", video=True)
    add("reconstruct", cmd_reconstruct, "camera poses + dense points (GPU)")
    p = add("splat", cmd_splat, "train 3D Gaussian splat (GPU)")
    p.add_argument("--max-aspect", type=float, help="penalise needles longer than this many times their width")
    p.add_argument("--random-background", action="store_true", help="random background colour while training")
    p.add_argument("--antialiased", action="store_true", help="antialiased rasterisation (viewer follows)")
    p = add("scale", cmd_scale, "set metric scale from two points and their real distance")
    p.add_argument("--a", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"), help="point A, recon space")
    p.add_argument("--b", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"), help="point B, recon space")
    p.add_argument("--distance", type=float, required=True, help="real distance between A and B in meters")
    add("navmesh", cmd_navmesh, "carve walkable navmesh (needs metric scale)")
    add("bundle", cmd_bundle, "write manifest and validate the scene bundle")

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
