"""Verifies the GPU stack end to end: CUDA arch, VRAM profile, gsplat kernels, MASt3R inference."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def check_cuda():
    if not torch.cuda.is_available():
        sys.exit("FAIL: torch cannot see a CUDA GPU.")
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    print(f"GPU      : {props.name}")
    print(f"Arch     : {arch} (torch built for {torch.cuda.get_arch_list()})")
    print(f"VRAM     : {props.total_memory / 1024**3:.1f} GiB")
    # A kernel built for the same major version and an equal or lower minor runs on this card.
    built = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    compatible = any(
        int(a[3:]) // 10 == cap[0] and int(a[3:]) % 10 <= cap[1] for a in built
    )
    if not compatible:
        sys.exit(f"FAIL: this PyTorch build has no kernels for {arch}. Reinstall from env/requirements-gpu.txt.")

    from atomspace.gpu import resolve_profile
    print(f"Profile  : {resolve_profile().name}")


def check_gsplat():
    from gsplat import rasterization

    torch.manual_seed(0)
    n, dev = 10_000, "cuda"
    means = (torch.rand(n, 3, device=dev) - 0.5) * 2
    means[:, 2] += 4
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
    scales = torch.full((n, 3), -4.0, device=dev).exp()
    opacities = torch.full((n,), 0.8, device=dev)
    colors = torch.rand(n, 3, device=dev, requires_grad=True)
    viewmats = torch.eye(4, device=dev)[None]
    Ks = torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device=dev)

    target = torch.rand(1, 240, 320, 3, device=dev)
    opt = torch.optim.Adam([colors], lr=0.05)
    for _ in range(50):
        img, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 320, 240)
        loss = (img - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"gsplat   : OK (forward + backward, loss {loss.item():.3f})")


def check_mast3r():
    if not CKPT.exists():
        sys.exit(f"FAIL: missing {CKPT.name}. Run env/download_weights.sh.")
    sys.path.insert(0, str(ROOT / "third_party" / "mast3r"))
    from mast3r.model import AsymmetricMASt3R
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    model = AsymmetricMASt3R.from_pretrained(str(CKPT)).to("cuda").eval()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        rng = np.random.default_rng(0)
        for i in range(2):
            p = Path(tmp) / f"{i}.png"
            Image.fromarray(rng.integers(0, 255, (384, 512, 3), dtype=np.uint8)).save(p)
            paths.append(str(p))
        images = load_images(paths, size=512, verbose=False)
        out = inference([tuple(images)], model, "cuda", batch_size=1, verbose=False)
    print(f"MASt3R   : OK (pointmap {tuple(out['pred1']['pts3d'].shape)})")


if __name__ == "__main__":
    check_cuda()
    torch.cuda.reset_peak_memory_stats()
    check_gsplat()
    check_mast3r()
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print("All checks passed.")
