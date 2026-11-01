from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MAST3R_DIR = REPO_ROOT / "third_party" / "mast3r"
SCHEMAS_DIR = REPO_ROOT / "schemas"

MAST3R_IMAGE_SIZE = 512
SPLAT_ITERATIONS = 30_000


@dataclass(frozen=True)
class Profile:
    """GPU memory limits. Picked by detected VRAM, so any card at or above min_vram_gib works."""

    name: str
    min_vram_gib: float
    max_keyframes: int
    pair_window: int
    pair_retrieval_k: int
    splat_downscale: int
    splat_max_gaussians: int


# Thresholds sit slightly under the marketed size: an "8 GB" card reports ~7.6-8.0 GiB.
# vram_4gb holds the settings tested on a 4 GB RTX 2050: 101 keyframes, 735 pairs, 4.41 GiB peak.
PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("vram_4gb", 3.5, 101, 5, 3, 2, 1_500_000),
        Profile("vram_8gb", 7.5, 150, 5, 10, 2, 1_500_000),
        Profile("vram_12gb", 11.5, 250, 8, 15, 2, 3_000_000),
        Profile("vram_24gb", 23.0, 400, 10, 20, 1, 6_000_000),
    )
}


@dataclass(frozen=True)
class IngestConfig:
    extract_fps: float = 6.0
    max_frame_side: int = 1920
    min_relative_sharpness: float = 0.6
    sharpness_window: int = 15
    min_motion: float = 0.15


@dataclass(frozen=True)
class NavmeshConfig:
    agent_radius_m: float = 0.25
    agent_height_m: float = 1.7
    ankle_height_m: float = 0.15
    max_climb_m: float = 0.2
    cell_size_m: float = 0.05
    min_blocked_points: int = 3  # fewer than this in a cell is floating noise, not an obstacle
    floor_gap_fill_m: float = 0.15
    walked_radius_m: float = 0.3


@dataclass(frozen=True)
class ScenePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def recon(self) -> Path:
        return self.root / "recon"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def bundle(self) -> Path:
        return self.root / "bundle"

    @property
    def ingest_meta(self) -> Path:
        return self.root / "ingest.json"

    @property
    def transforms(self) -> Path:
        return self.recon / "transforms.json"

    @property
    def dense_points(self) -> Path:
        return self.recon / "dense_points.ply"

    @property
    def scale(self) -> Path:
        return self.recon / "scale.json"

    @property
    def world(self) -> Path:
        return self.recon / "world.json"

    @property
    def splat_ply(self) -> Path:
        return self.recon / "splat.ply"

    @property
    def manifest(self) -> Path:
        return self.bundle / "manifest.json"

    @property
    def splat(self) -> Path:
        return self.bundle / "scene.splat"

    @property
    def navmesh(self) -> Path:
        return self.bundle / "navmesh.json"

    @property
    def pois(self) -> Path:
        return self.bundle / "pois.json"

    def ensure(self) -> None:
        for d in (self.raw, self.frames, self.recon, self.logs, self.bundle):
            d.mkdir(parents=True, exist_ok=True)
