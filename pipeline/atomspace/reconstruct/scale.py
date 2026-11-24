import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..config import ScenePaths
from ..ingest.route import Track

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ScaleReference:
    """Two points in recon space and the real-world distance between them."""

    a: Vec3
    b: Vec3
    distance_m: float

    def factor(self) -> float:
        d = math.dist(self.a, self.b)
        if d <= 0 or self.distance_m <= 0:
            raise ValueError("Scale reference needs two distinct points and a positive distance.")
        return self.distance_m / d


def save_reference(ref: ScaleReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"a": list(ref.a), "b": list(ref.b), "distance_m": ref.distance_m}, indent=2))


def load_reference(path: Path) -> ScaleReference:
    d = json.loads(path.read_text())
    return ScaleReference(a=tuple(d["a"]), b=tuple(d["b"]), distance_m=d["distance_m"])


def metric_factor(paths: ScenePaths) -> tuple[float, str] | None:
    """Recon-to-meters factor and its source, or None if the scene isn't metric yet."""
    if paths.scale.exists():
        return load_reference(paths.scale).factor(), "user"
    if paths.ingest_meta.exists() and json.loads(paths.ingest_meta.read_text())["track"] == Track.FAST.value:
        return 1.0, "sensor"
    return None
