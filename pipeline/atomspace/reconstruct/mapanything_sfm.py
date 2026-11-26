import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..gpu import vram_gib
from .export import Reconstruction

MODEL = "facebook/map-anything-apache"
REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"  # pinned so a later upload can't change results
# 101 views peaked at 8.7-11.2 GiB on a 15 GB T4; 200 ran out of memory there.
MIN_VRAM_GIB = 11.5
MAX_DENSE_POINTS = 3_000_000
ENCODER_CHUNK = 8

# Bundle adjustment (pycolmap). Default SIFT (0.0067) found only ~1-1.7k features per frame of a dim room.
SIFT_PEAK_THRESHOLD = 0.002
# MapAnything's poses are off by more than COLMAP's default 4 px, which threw out good matches.
# Triangulate loose, refine, re-triangulate tighter.
THRESHOLDS_PX = (16.0, 8.0, 4.0)
# A view with fewer triangulated points than this isn't held by BA; it keeps MapAnything's pose.
MIN_OBSERVATIONS = 30


def reconstruct(frames: list[Path], work_dir: Path) -> Reconstruction:
    """MapAnything (Apache-2.0 weights) gives poses, intrinsics and a dense cloud in one pass; COLMAP
    bundle adjustment then sharpens the poses, which a feed-forward model can't do (the splat streaks).

    MapAnything's output is roughly metric; the user's scale reference still decides the final scale.
    """
    vram = vram_gib()
    if vram is not None and vram < MIN_VRAM_GIB:
        raise RuntimeError(
            f"Reconstruct needs a GPU with at least 12 GB (this one has {vram:.1f} GiB). Run it on a free "
            "Kaggle T4 with env/kaggle_reconstruct.ipynb, then continue here with `atomspace splat`."
        )
    names, c2w, K_model, model_hw, points, colors = infer(frames)
    W, H = Image.open(frames[0]).size
    K = np.stack([to_frame_pixels(k, model_hw, (W, H)) for k in K_model])

    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    try:
        c2w, K = bundle_adjust(frames[0].parent, names, c2w, K, (W, H), work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return Reconstruction(names, K, c2w, points, colors)


def infer(frames: list[Path]):
    """Images-only MapAnything inference, patched to fit a 15 GB T4 at ~100 views."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model = MapAnything.from_pretrained(MODEL, revision=REVISION).to("cuda").eval()
    # MapAnything sends every view through DINOv2 in one batch (5.41 GiB of attention at 100 views).
    # The encoder is per-image, so chunking changes nothing but memory. Its ~4.4 GB of weights are
    # idle after encoding, so they wait on the CPU, leaving room for the multi-view transformer.
    encode = model.encoder.forward

    def chunked_encode(inp):
        model.encoder.to("cuda")
        outs = [encode(type(inp)(image=inp.image[i:i + ENCODER_CHUNK], data_norm_type=inp.data_norm_type))
                for i in range(0, len(inp.image), ENCODER_CHUNK)]
        model.encoder.to("cpu")
        torch.cuda.empty_cache()
        out = outs[0]
        out.features = torch.cat([o.features for o in outs])
        if out.registers is not None:
            out.registers = torch.cat([o.registers for o in outs])
        return out

    model.encoder.forward = chunked_encode
    # MapAnything takes its device from its first parameter, the encoder's; pin it to CUDA.
    type(model).device = property(lambda self: torch.device("cuda"))
    model.encoder.to("cpu")
    torch.cuda.empty_cache()

    views = load_images([str(f) for f in frames])
    t = time.monotonic()
    # The math SDPA kernel would build the full global attention (541 GiB at 100 views on a T4).
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                            amp_dtype="fp16", apply_mask=True, mask_edges=True)
    print(f"MapAnything: {len(frames)} views in {time.monotonic() - t:.0f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)

    c2w = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds]).astype(np.float64)
    K = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds]).astype(np.float64)
    model_hw = preds[0]["pts3d"].shape[1:3]
    points, colors = [], []
    for p in preds:
        m = p["mask"][0, ..., 0].bool().cpu().numpy()
        points.append(p["pts3d"][0].float().cpu().numpy()[m])
        colors.append((p["img_no_norm"][0].float().cpu().numpy()[m] * 255).clip(0, 255).astype(np.uint8))
    points, colors = np.concatenate(points), np.concatenate(colors)
    if len(points) > MAX_DENSE_POINTS:
        keep = np.sort(np.random.default_rng(0).choice(len(points), MAX_DENSE_POINTS, replace=False))
        points, colors = points[keep], colors[keep]
    del model, preds
    torch.cuda.empty_cache()
    return [f.name for f in frames], c2w, K, model_hw, points.astype(np.float32), colors


def to_frame_pixels(K: np.ndarray, model_hw: tuple[int, int], frame_size: tuple[int, int]) -> np.ndarray:
    """Maps intrinsics of MapAnything's input back to keyframe pixels. load_images ("fixed_mapping")
    scales each frame to cover the model size, s = max(mw / W, mh / H), then centre-crops."""
    (mh, mw), (W, H) = model_hw, frame_size
    s = max(mw / W, mh / H)
    out = K.copy()
    out[:2, :2] /= s
    out[0, 2] = W / 2 + (K[0, 2] - mw / 2) / s
    out[1, 2] = H / 2 + (K[1, 2] - mh / 2) / s
    return out


def bundle_adjust(frames_dir: Path, names: list[str], c2w: np.ndarray, K: np.ndarray,
                  frame_size: tuple[int, int], work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + exhaustive matching, then triangulation with MapAnything's poses held fixed, so matches
    that disagree with them (repeated desks) fail to triangulate, which is what broke plain COLMAP.
    Then BA of poses and ONE shared focal (one phone video = one lens), robust loss, loose to tight.

    Returns poses in MapAnything's frame (its dense cloud must still fit) and the shared K per view.
    """
    import pycolmap as p

    W, H = frame_size
    f0 = float(np.median(K[:, 0, 0]))
    db_path = work_dir / "database.db"
    t = time.monotonic()
    ext = p.FeatureExtractionOptions()
    ext.sift.peak_threshold = SIFT_PEAK_THRESHOLD
    p.extract_features(db_path, frames_dir, image_names=names, camera_mode=p.CameraMode.SINGLE,
                       reader_options=p.ImageReaderOptions(camera_model="PINHOLE",
                                                           camera_params=f"{f0},{f0},{W / 2},{H / 2}"),
                       extraction_options=ext)
    p.match_exhaustive(db_path)
    print(f"BA: features and matches in {time.monotonic() - t:.0f} s", flush=True)

    db = p.Database.open(db_path)
    rec = p.Reconstruction()
    for cam in db.read_all_cameras():
        rec.add_camera_with_trivial_rig(cam)
    index = {n: i for i, n in enumerate(names)}
    for im in db.read_all_images():
        rec.add_image_with_trivial_frame(im, p.Rigid3d(np.linalg.inv(c2w[index[im.name]])[:3, :4]))
    db.close()

    opts = p.BundleAdjustmentOptions(refine_focal_length=True, refine_principal_point=False,
                                     refine_extra_params=False, print_summary=False)
    opts.ceres.loss_function_type = p.LossFunctionType.CAUCHY
    for rnd, thr in enumerate(THRESHOLDS_PX, 1):
        tri = p.IncrementalPipelineOptions()
        tri.triangulation.ignore_two_view_tracks = False  # sparse keyframes: many points are in two views only
        tri.triangulation.create_max_angle_error = tri.triangulation.continue_max_angle_error = thr / 2
        tri.triangulation.merge_max_reproj_error = tri.triangulation.complete_max_reproj_error = thr
        tri.mapper.filter_max_reproj_error = thr
        rec = p.triangulate_points(rec, db_path, frames_dir, work_dir / f"tri{rnd}", clear_points=True,
                                   options=tri, refine_intrinsics=False)
        p.bundle_adjustment(rec, opts)
        rec.update_point_3d_errors()
        for pid in [pid for pid, pt in rec.points3D.items() if pt.error > thr]:
            rec.delete_point3D(pid)
        print(f"BA round {rnd} ({thr:g} px): {rec.num_points3D():,} points, "
              f"{rec.compute_mean_reprojection_error():.2f} px", flush=True)

    by_name = {im.name: im for im in rec.images.values()}
    obs = np.array([sum(pt.has_point3D() for pt in by_name[n].points2D) for n in names])
    held = obs >= MIN_OBSERVATIONS
    if held.sum() < 3:
        raise RuntimeError(f"Bundle adjustment held only {held.sum()} of {len(names)} views; the capture has too little texture.")
    ba_c2w = np.stack([np.linalg.inv(np.vstack([by_name[n].cam_from_world().matrix(), [0, 0, 0, 1]])) for n in names])

    # BA's frame drifts (shift, rotation, scale). Map it back with the similarity that best fits the
    # views BA held; views it didn't hold never moved, so they keep MapAnything's pose as it was.
    s, R, t_ = similarity(ba_c2w[held, :3, 3], c2w[held, :3, 3])
    out = c2w.copy()
    out[held, :3, :3] = R @ ba_c2w[held, :3, :3]
    out[held, :3, 3] = s * ba_c2w[held, :3, 3] @ R.T + t_
    moved = np.linalg.norm(out[held, :3, 3] - c2w[held, :3, 3], axis=1)
    cam = next(iter(rec.cameras.values()))
    fx, fy = (float(v) for v in cam.params[:2])
    print(f"BA: focal {fx:.1f}/{fy:.1f} px, {held.sum()}/{len(names)} views held (min {obs.min()} points), "
          f"camera shift median {np.median(moved):.3f} max {moved.max():.3f}, frame scale {s:.3f}", flush=True)
    K_out = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
    return out, np.repeat(K_out[None], len(names), axis=0)


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / ((src - ms) ** 2).sum())
    return s, R, md - s * R @ ms
