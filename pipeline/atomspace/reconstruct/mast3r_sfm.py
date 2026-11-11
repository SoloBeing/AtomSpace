import argparse
import gc
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..config import CHECKPOINTS_DIR, MAST3R_DIR, MAST3R_IMAGE_SIZE, Profile
from .export import Reconstruction

# A folder: Hugging Face hosts the model as config.json + model.safetensors.
MODEL_CKPT = CHECKPOINTS_DIR / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
# Retriever finds the codebook next to this file (..._retrieval_codebook.pkl).
RETRIEVAL_CKPT = CHECKPOINTS_DIR / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
PATCH_SIZE = 16

# Optimiser settings from MASt3R's demo (coarse 3D matching, then 2D reprojection refinement).
LR_COARSE, ITERS_COARSE = 0.07, 300
LR_FINE, ITERS_FINE = 0.01, 300
MIN_CONF = 1.5  # dense points below this MASt3R confidence are dropped (demo default)
MAX_DENSE_POINTS = 2_000_000
VOXEL_FRACTION = 0.005  # dedupe voxel = this fraction of the median camera-to-point distance
# Part of MASt3R's cache file paths, so matching and alignment must be given the same values.
SUBSAMPLE, DESC_CONF = 8, "desc_conf"


def reconstruct(frames: list[Path], profile: Profile, cache_dir: Path) -> Reconstruction:
    """Runs MASt3R-SfM on the keyframes.

    Pair graph: each frame is paired with its `profile.pair_window` temporal neighbours plus
    `profile.pair_retrieval_k` retrieval matches (retrieval checkpoint + ASMK), which keeps
    memory roughly linear in frame count. Then sparse global alignment
    (mast3r.cloud_opt.sparse_ga) solves poses, intrinsics and dense depth.

    Output is up-to-scale; metric scale is applied later from the user's reference.
    """
    for ckpt in (MODEL_CKPT, RETRIEVAL_CKPT):
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing {ckpt.name}. Run env/download_weights.sh.")
    sys.path.insert(0, str(MAST3R_DIR))
    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import convert_dust3r_pairs_naming, forward_mast3r, sparse_global_alignment
    from mast3r.model import AsymmetricMASt3R
    from mast3r.retrieval.processor import Retriever

    # MASt3R caches by a hash of the image path, so frames from an earlier ingest that reused
    # the same file names would be picked up as if current.
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)

    image_paths = [str(f) for f in frames]
    model = AsymmetricMASt3R.from_pretrained(str(MODEL_CKPT)).to("cuda").eval()

    # The retrieval checkpoint stores its arguments as a Namespace, which Retriever's torch.load
    # refuses by default since PyTorch 2.6.
    torch.serialization.add_safe_globals([argparse.Namespace])
    retriever = Retriever(str(RETRIEVAL_CKPT), backbone=model, device="cuda")
    with torch.no_grad():
        similarity = retriever(image_paths)
    del retriever
    torch.cuda.empty_cache()
    log_vram("after retrieval")

    edges = pair_graph(len(frames), profile.pair_window, profile.pair_retrieval_k, similarity)
    print(f"MASt3R: {len(frames)} frames, {len(edges)} pairs")
    imgs = load_images(image_paths, size=MAST3R_IMAGE_SIZE, verbose=False)
    pairs = [(imgs[i], imgs[j]) for i, j in edges]
    pairs += [(b, a) for a, b in pairs]
    pairs = convert_dust3r_pairs_naming(image_paths, pairs)

    try:
        # Matching: the only part that needs the network. Results go to the disk cache.
        cache_encoder(model)
        faster_nn_search()
        forward_mast3r(pairs, model, cache_path=str(cache_dir), subsample=SUBSAMPLE,
                       desc_conf=DESC_CONF, device="cuda")
        log_vram("after matching")

        # Alignment reads only the cache, so the weights (~2.6 GiB) can leave the GPU first.
        # forward_mast3r's own `del model` can't free them while this function holds a reference.
        del model
        gc.collect()  # cache_encoder's closure and the model refer to each other
        torch.cuda.empty_cache()
        log_vram("model freed")
        # With no model, forward_mast3r silently skips pairs missing from the cache.
        cached, _ = forward_mast3r(pairs, None, cache_path=str(cache_dir), subsample=SUBSAMPLE,
                                   desc_conf=DESC_CONF, device="cuda")
        if len(cached) != len(pairs):
            raise RuntimeError(f"Only {len(cached)} of {len(pairs)} pairs are in the match cache.")

        # One phone lens with focus locked during capture, so all frames share intrinsics.
        scene = sparse_global_alignment(
            image_paths, pairs, str(cache_dir), None, subsample=SUBSAMPLE, desc_conf=DESC_CONF,
            lr1=LR_COARSE, niter1=ITERS_COARSE, lr2=LR_FINE, niter2=ITERS_FINE,
            device="cuda", opt_depth=True, shared_intrinsics=True, matching_conf_thr=0.0,
        )
        log_vram("after alignment")
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)

        cam_to_world = scene.get_im_poses().cpu().numpy()
        centers = cam_to_world[:, :3, 3]
        points, colors = [], []
        for i in range(len(frames)):
            keep = (confs[i] > MIN_CONF).cpu().numpy().ravel()
            points.append(pts3d[i].cpu().numpy().reshape(-1, 3)[keep])
            colors.append(scene.imgs[i].reshape(-1, 3)[keep])
        points = np.concatenate(points).astype(np.float32)
        colors = (np.concatenate(colors) * 255).round().astype(np.uint8)
        finite = np.isfinite(points).all(axis=1)
        points, colors = points[finite], colors[finite]

        # Recon space has no fixed unit, so the voxel size follows the scene's own scale.
        sample = points[:: max(1, len(points) // 100_000)]
        nearest_cam = np.linalg.norm(sample[:, None] - centers[None], axis=2).min(axis=1)
        voxel = VOXEL_FRACTION * float(np.median(nearest_cam))
        points, colors = voxel_downsample(points, colors, voxel, MAX_DENSE_POINTS)

        intrinsics = scene.intrinsics.cpu().numpy()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    sizes = [ImageOps.exif_transpose(Image.open(f)).size for f in frames]
    return Reconstruction(
        image_names=[f.name for f in frames],
        intrinsics=np.stack([to_original_pixels(K, size) for K, size in zip(intrinsics, sizes)]),
        cam_to_world=cam_to_world,
        points=points,
        colors=colors,
    )


def cache_encoder(model) -> None:
    """Encodes each image once. MASt3R only reuses encoder output within a pair, so otherwise an
    image goes through the encoder again for every pair it's in, always with the same result."""
    cache = {}

    def encoded(image, true_shape):
        key = hashlib.md5(image.cpu().numpy().tobytes()).digest()
        if key not in cache:
            features, positions, _ = model._encode_image(image, true_shape)
            cache[key] = (features.cpu(), positions.cpu())  # main memory: VRAM is the scarce one
        features, positions = cache[key]
        return features.to(image.device), positions.to(image.device)

    def encode_pair(image1, image2, true_shape1, true_shape2):
        features1, positions1 = encoded(image1, true_shape1)
        features2, positions2 = encoded(image2, true_shape2)
        return features1, features2, positions1, positions2

    model._encode_image_pairs = encode_pair


def faster_nn_search() -> None:
    """Speeds up MASt3R's matching-point search without changing a single match.

    Each search query compares a few hundred to 2,304 pixels' descriptors against every pixel of
    the other image. Queries large enough to be split into blocks (fast_nn.py:43-67) took ~84 ms,
    because every block also found the reverse-direction match, which the query then discards
    (fast_nn.py:82-84), and wrote results with boolean masks, each of which makes the CPU wait for
    the GPU. Those queries now go to nearest_in_blocks. Smaller queries keep MASt3R's own single-pass
    path, which is already fast.
    """
    import torch
    from mast3r import fast_nn

    original = fast_nn.cdistMatcher.query
    if getattr(original, "atomspace", False):
        return

    def query(self, queries, k=1, **kw):
        block = kw.get("block_size")
        if (k != 1 or block is None or not isinstance(queries, torch.Tensor) or queries.numel() == 0
                or len(queries) * len(self.db_pts) <= block**2):
            return original(self, queries, k=k, **kw)
        return None, nearest_in_blocks(queries.to(self.device), self.db_pts, block, kw.get("dist", "l2"))

    query.atomspace = True
    fast_nn.cdistMatcher.query = query


def nearest_in_blocks(A, B, block_size: int, dist: str) -> np.ndarray:
    """Index of each row of A's nearest row of B: largest dot product (dist "dot") or smallest
    Euclidean distance ("l2"). Same blocks, same arithmetic and same tie rule (strict <, so the
    earlier block wins) as the A side of MASt3R's bruteforce_reciprocal_nns, so the result is
    identical; torch.where keeps the best so far without making the CPU wait."""
    import torch

    out = []
    for i in range(0, len(A), block_size):
        A_i = A[i:i + block_size]
        best = torch.full((len(A_i),), float("inf"), device=A.device, dtype=A.dtype)
        nearest = torch.full((len(A_i),), -1, device=A.device, dtype=torch.int64)
        for j in range(0, len(B), block_size):
            B_j = B[j:j + block_size]
            if dist == "dot":
                similarity, index = torch.max(A_i @ B_j.T, dim=1)
                distance = similarity.neg()
            elif dist == "l2":
                distance, index = torch.min(torch.cdist(A_i, B_j), dim=1)
            else:
                raise ValueError(f"Unknown {dist=}")
            better = distance < best
            best = torch.where(better, distance, best)
            nearest = torch.where(better, index + j, nearest)
        out.append(nearest)
    return torch.cat(out).cpu().numpy()


def log_vram(phase: str) -> None:
    """Prints memory held now and the stage's peak so far (the peak never resets mid-stage)."""
    import torch

    gib = 1024**3
    print(f"VRAM {phase}: {torch.cuda.memory_allocated() / gib:.2f} GiB held, "
          f"peak so far {torch.cuda.max_memory_allocated() / gib:.2f} GiB")


def pair_graph(n: int, window: int, retrieval_k: int, similarity: np.ndarray | None) -> list[tuple[int, int]]:
    """Unordered image pairs (i < j): every frame with its next `window` frames, plus its
    `retrieval_k` most similar frames outside that window (revisits and loop closures)."""
    edges = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}
    if similarity is not None and retrieval_k > 0:
        sim = (similarity + similarity.T) / 2
        for i in range(n):
            far = [j for j in np.argsort(-sim[i]) if abs(int(j) - i) > window]
            edges.update((min(i, int(j)), max(i, int(j))) for j in far[:retrieval_k])
    return sorted(edges)


def to_original_pixels(K: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Maps intrinsics of MASt3R's network input back to the keyframe's own pixels.

    Mirrors dust3r's load_images: resize so the long side is MAST3R_IMAGE_SIZE, then a centred
    crop to a multiple of the patch size (square inputs are cropped to 4:3).
    """
    w0, h0 = original_size
    s = MAST3R_IMAGE_SIZE / max(w0, h0)
    w, h = round(w0 * s), round(h0 * s)
    half_w = ((2 * (w // 2)) // PATCH_SIZE) * PATCH_SIZE / 2
    half_h = ((2 * (h // 2)) // PATCH_SIZE) * PATCH_SIZE / 2
    if w == h:
        half_h = 3 * half_w / 4
    left, top = w // 2 - half_w, h // 2 - half_h
    sx, sy = w / w0, h / h0
    out = np.eye(3)
    out[0, 0] = K[0, 0] / sx
    out[1, 1] = K[1, 1] / sy
    out[0, 2] = (K[0, 2] + left + 0.5) / sx - 0.5
    out[1, 2] = (K[1, 2] + top + 0.5) / sy - 0.5
    return out


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float, max_points: int,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Keeps one point per voxel (overlapping views see the same surface many times), then
    randomly thins to max_points."""
    cells = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    first.sort()
    if len(first) > max_points:
        first = np.sort(np.random.default_rng(seed).choice(first, max_points, replace=False))
    return points[first], colors[first]
