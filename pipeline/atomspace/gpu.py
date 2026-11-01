import json
import time
from contextlib import contextmanager
from pathlib import Path

from .config import PROFILES, Profile

MIN_PROFILE = PROFILES["vram_4gb"]
RECOMMENDED_PROFILE = PROFILES["vram_8gb"]


def vram_gib() -> float | None:
    # torch is imported lazily so CPU-only stages work on machines without the GPU env.
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def resolve_profile(name: str | None = None) -> Profile:
    if name:
        return PROFILES[name]
    vram = vram_gib()
    if vram is None:
        return RECOMMENDED_PROFILE  # CPU-only stages: nothing here is limited by this machine
    eligible = [p for p in PROFILES.values() if vram >= p.min_vram_gib]
    if not eligible:
        raise RuntimeError(f"GPU has {vram:.1f} GiB VRAM; AtomSpace needs at least 4 GB.")
    if vram < RECOMMENDED_PROFILE.min_vram_gib:
        print(f"Warning: {vram:.1f} GiB VRAM is below the recommended 8 GB. Using reduced limits; expect it to be slow.")
    return max(eligible, key=lambda p: p.min_vram_gib)


def require_gpu() -> None:
    vram = vram_gib()
    if vram is None:
        raise RuntimeError(
            "This stage needs an NVIDIA GPU (RTX 20 series or newer, 4 GB+, 8 GB recommended). See env/README.md."
        )
    if vram < MIN_PROFILE.min_vram_gib:
        raise RuntimeError(f"GPU has {vram:.1f} GiB VRAM; AtomSpace needs at least 4 GB.")


@contextmanager
def gpu_stage(name: str, log_dir: Path):
    """Runs a GPU stage with a clean cache and records duration and peak VRAM to logs/<name>.json."""
    require_gpu()
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        yield
    finally:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": name,
            "gpu": torch.cuda.get_device_name(0),
            "duration_s": round(time.monotonic() - start, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
        (log_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
        torch.cuda.empty_cache()
