import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..config import SPLAT_ITERATIONS, Profile
from ..reconstruct.export import CV_TO_GL, read_ply
from .compress import SH_C0

# Learning rates and loss weight from gsplat's simple_trainer, which follows the 3DGS paper.
LR_MEANS, LR_SCALES, LR_QUATS, LR_OPACITIES, LR_COLORS = 1.6e-4, 5e-3, 1e-3, 5e-2, 2.5e-3
MEANS_LR_FINAL_RATIO = 0.01
SSIM_WEIGHT = 0.2
INITIAL_OPACITY = 0.1
INIT_SHARE = 0.5  # of the Gaussian budget; the rest is left for densification to spend on detail
SAVE_EVERY = 5000
LOG_EVERY = 500


@dataclass
class Camera:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    intrinsics: np.ndarray  # (3, 3), pixels of `image`
    cam_to_world: np.ndarray  # (4, 4), OpenCV camera convention


def load_cameras(frames_dir: Path, transforms: Path, downscale: int) -> list[Camera]:
    """Reads transforms.json, whose matrices are in OpenGL axes, and the matching frames."""
    cameras = []
    for frame in json.loads(transforms.read_text())["frames"]:
        path = frames_dir / Path(frame["file_path"]).name
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read {path}")
        height, width = bgr.shape[:2]
        small = cv2.resize(bgr, (width // downscale, height // downscale), interpolation=cv2.INTER_AREA)
        sx, sy = small.shape[1] / width, small.shape[0] / height
        intrinsics = np.array([
            [frame["fl_x"] * sx, 0, frame["cx"] * sx],
            [0, frame["fl_y"] * sy, frame["cy"] * sy],
            [0, 0, 1],
        ])
        cameras.append(Camera(
            image=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
            intrinsics=intrinsics,
            cam_to_world=np.array(frame["transform_matrix"]) @ CV_TO_GL,
        ))
    return cameras


def initial_gaussians(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeds one round Gaussian per point, sized by the spacing to its nearest neighbours."""
    if len(points) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
        points, colors = points[keep], colors[keep]
    distances, _ = cKDTree(points).query(points, k=4, workers=-1)
    spacing = np.sqrt((distances[:, 1:] ** 2).mean(axis=1)).clip(min=1e-6)
    n = len(points)
    rng = np.random.default_rng(seed)
    return {
        "means": points.astype(np.float32),
        "scales": np.tile(np.log(spacing)[:, None], (1, 3)).astype(np.float32),
        "quats": rng.random((n, 4), dtype=np.float32),
        "opacities": np.full(n, np.log(INITIAL_OPACITY / (1 - INITIAL_OPACITY)), np.float32),
        "sh0": ((colors / 255 - 0.5) / SH_C0)[:, None].astype(np.float32),  # (N, 1, 3)
    }


def write_gaussians_ply(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    """Standard 3DGS .ply, written atomically since it's also saved during training.

    Not gsplat's export_splats: it fails on base-colour-only models (it reshapes the empty
    higher-order colours with -1).
    """
    n = len(gaussians["means"])
    data = np.concatenate([
        gaussians["means"], gaussians["sh0"].reshape(n, 3), gaussians["opacities"][:, None],
        gaussians["scales"], gaussians["quats"],
    ], axis=1).astype("<f4")
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"

    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    os.replace(partial, path)


def train(frames_dir: Path, transforms: Path, init_points: Path, out_ply: Path, profile: Profile) -> None:
    """Trains 3D Gaussians with gsplat, seeded from MASt3R's dense points.

    Memory limits from the profile: images downscaled by `splat_downscale`, densification
    stops growing at `splat_max_gaussians`. Training itself peaks well under 1 GiB, so the
    frames stay on the GPU and rasterization runs unpacked (faster for one camera per step).
    Output is a standard 3DGS .ply in recon space.

    Base colour only (sh_degree 0): the .splat format can't store view-dependent colour, so
    training it would use memory for something the viewer never shows.
    """
    import torch
    import torch.nn.functional as F
    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import reset_opa

    @dataclass
    class BudgetedStrategy(DefaultStrategy):
        """gsplat's densification with a hard Gaussian budget: one round can nearly double the
        count, so past the budget only the highest-gradient Gaussians may still grow."""

        max_gaussians: int = 1_500_000

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            room = self.max_gaussians - len(params["means"])
            if room <= 0:
                return 0, 0
            grads = state["grad2d"] / state["count"].clamp_min(1)
            threshold = self.grow_grad2d
            if int((grads > threshold).sum()) > room:  # each duplicate or split adds exactly one
                self.grow_grad2d = float(torch.topk(grads, room + 1).values[-1])
            try:
                return super()._grow_gs(params, optimizers, state, step)
            finally:
                self.grow_grad2d = threshold

        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            super().step_post_backward(params, optimizers, state, step, info, packed=packed)
            # gsplat 1.5.3 means to do this too, but `... == 0 & step > 0` is never true.
            if 0 < step < self.refine_stop_iter and step % self.reset_every == 0:
                reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)

    device = "cuda"
    cameras = load_cameras(frames_dir, transforms, profile.splat_downscale)
    points, colors = read_ply(init_points)
    start = initial_gaussians(points, colors, int(profile.splat_max_gaussians * INIT_SHARE))

    centers = np.stack([c.cam_to_world[:3, 3] for c in cameras])
    scene_scale = max(1.1 * float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()), 1e-3)

    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.from_numpy(array)) for name, array in start.items()}
    ).to(device)
    rates = {
        "means": LR_MEANS * scene_scale, "scales": LR_SCALES, "quats": LR_QUATS,
        "opacities": LR_OPACITIES, "sh0": LR_COLORS,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15, fused=True)
        for name, lr in rates.items()
    }
    means_decay = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=MEANS_LR_FINAL_RATIO ** (1 / SPLAT_ITERATIONS)
    )
    strategy = BudgetedStrategy(max_gaussians=profile.splat_max_gaussians)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    images = [torch.from_numpy(c.image).to(device) for c in cameras]  # uint8, ~1.5 MB each at 960x540
    Ks = [torch.from_numpy(c.intrinsics).float().to(device)[None] for c in cameras]
    viewmats = [torch.from_numpy(np.linalg.inv(c.cam_to_world)).float().to(device)[None] for c in cameras]

    def save() -> None:
        write_gaussians_ply(out_ply, {name: p.detach().cpu().numpy() for name, p in splats.items()})

    print(f"Splat: {len(cameras)} views, {len(start['means'])} starting Gaussians, scene scale {scene_scale:.2f}")
    began = time.monotonic()
    order: list[int] = []
    for step in range(SPLAT_ITERATIONS):
        if not order:
            order = torch.randperm(len(cameras)).tolist()
        view = order.pop()
        target = images[view].float()[None] / 255.0  # (1, H, W, 3)
        height, width = target.shape[1:3]

        render, _, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=splats["sh0"],
            viewmats=viewmats[view],
            Ks=Ks[view],
            width=width,
            height=height,
            sh_degree=0,
            packed=False,
        )
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)

        l1 = F.l1_loss(render, target)
        loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - _ssim(render, target))
        loss.backward()

        # gsplat's order: densification runs after the optimizers, and it replaces their parameters.
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_decay.step()
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)

        done = step + 1
        if done % LOG_EVERY == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"  step {done}/{SPLAT_ITERATIONS}: loss {loss.item():.4f}, "
                f"{len(splats['means'])} Gaussians, peak {peak:.2f} GiB, {time.monotonic() - began:.0f} s",
                flush=True,
            )
        if done % SAVE_EVERY == 0:
            save()
    save()


@lru_cache
def _gaussian_1d(dtype, device, window_size: int = 11, sigma: float = 1.5):
    """Rows and columns halves of the 11x11 SSIM window, made once per dtype and device."""
    import torch

    x = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum()).repeat(3, 1, 1, 1)  # (3, 1, 1, window)
    return g, g.transpose(2, 3)


def _ssim(a, b):
    """Mean structural similarity of two (1, H, W, 3) images in 0..1."""
    import torch.nn.functional as F

    # The Gaussian window is separable: two 1-D passes give the same blur as one 11x11 pass, with 22 not 121 taps.
    rows, cols = _gaussian_1d(a.dtype, a.device)
    a, b = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    blur = lambda image: F.conv2d(F.conv2d(image, rows, groups=3), cols, groups=3)
    mean_a, mean_b = blur(a), blur(b)
    var_a = blur(a * a) - mean_a**2
    var_b = blur(b * b) - mean_b**2
    covar = blur(a * b) - mean_a * mean_b
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mean_a * mean_b + c1) * (2 * covar + c2)) / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))
    ).mean()
