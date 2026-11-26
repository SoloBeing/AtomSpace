"""Verifies the GPU stack: CUDA arch, VRAM profile, gsplat kernels, MapAnything and pycolmap installed."""
import sys

import torch


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


def check_reconstruct():
    import mapanything  # noqa: F401
    import pycolmap

    from atomspace.reconstruct.mapanything_sfm import MIN_VRAM_GIB

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    where = "here" if vram >= MIN_VRAM_GIB else "on Kaggle (env/kaggle_reconstruct.ipynb): this card is too small"
    print(f"Recon    : MapAnything + pycolmap {pycolmap.__version__} installed; reconstruct runs {where}")


if __name__ == "__main__":
    check_cuda()
    torch.cuda.reset_peak_memory_stats()
    check_gsplat()
    check_reconstruct()
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print("All checks passed.")
