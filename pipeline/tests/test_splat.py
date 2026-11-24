import numpy as np
import pytest
from PIL import Image

from atomspace.config import ScenePaths
from atomspace.reconstruct.export import Reconstruction, write_reconstruction
from atomspace.splat.compress import SH_C0, SPLAT_DTYPE, ply_to_splat
from atomspace.splat.train import initial_gaussians, load_cameras, write_gaussians_ply


def _gaussians(**overrides):
    n = len(overrides.get("means", np.zeros((2, 3))))
    g = {
        "means": np.array([[1, 2, 3], [4, 5, 6]], np.float32)[:n],
        "scales": np.log(np.full((n, 3), 0.25, np.float32)),
        "quats": np.tile(np.array([2, 0, 0, 0], np.float32), (n, 1)),  # unnormalised, as training leaves them
        "opacities": np.zeros(n, np.float32),  # sigmoid(0) = 0.5
        "sh0": np.zeros((n, 1, 3), np.float32),  # mid grey
    }
    return g | overrides


def _convert(tmp_path, gaussians):
    ply, out = tmp_path / "splat.ply", tmp_path / "scene.splat"
    write_gaussians_ply(ply, gaussians)
    ply_to_splat(ply, out)
    return np.frombuffer(out.read_bytes(), dtype=SPLAT_DTYPE)


def test_each_splat_is_32_bytes_and_decodes_the_training_values(tmp_path):
    splats = _convert(tmp_path, _gaussians())
    assert SPLAT_DTYPE.itemsize == 32
    assert len(splats) == 2
    np.testing.assert_array_equal(splats["position"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(splats["scale"], 0.25)
    np.testing.assert_array_equal(splats["rgba"][0], [128, 128, 128, 128])
    np.testing.assert_array_equal(splats["rotation"][0], [255, 128, 128, 128])  # w first, normalised


def test_colour_comes_from_the_base_harmonic(tmp_path):
    sh0 = np.array([[[0.5 / SH_C0, -0.5 / SH_C0, 0]]], np.float32)
    splats = _convert(tmp_path, _gaussians(means=np.zeros((1, 3), np.float32), sh0=sh0))
    np.testing.assert_array_equal(splats["rgba"][0][:3], [255, 0, 128])


def test_faint_and_broken_gaussians_are_dropped(tmp_path):
    g = _gaussians(opacities=np.array([-10, 0], np.float32))
    assert len(_convert(tmp_path, g)) == 1

    g = _gaussians(means=np.array([[np.nan, 0, 0], [1, 1, 1]], np.float32))
    splats = _convert(tmp_path, g)
    np.testing.assert_array_equal(splats["position"], [[1, 1, 1]])

    with pytest.raises(ValueError, match="no visible"):
        _convert(tmp_path, _gaussians(opacities=np.array([-10, -10], np.float32)))


def test_needles_are_dropped_but_flat_discs_stay(tmp_path):
    scales = np.log(np.array([[0.1, 0.1, 0.002], [0.1, 0.001, 0.001]], np.float32))  # disc, needle
    splats = _convert(tmp_path, _gaussians(scales=scales))
    np.testing.assert_array_equal(splats["position"], [[1, 2, 3]])


def test_cameras_round_trip_from_export_at_half_size(tmp_path):
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    pixels = np.zeros((360, 640, 3), np.uint8)
    pixels[..., 0] = 255  # pure red, to catch swapped channels
    Image.fromarray(pixels).save(paths.frames / "00001.png")

    c2w = np.eye(4)
    c2w[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    c2w[:3, 3] = [1, 2, 3]
    K = np.array([[500.0, 0, 320], [0, 400.0, 180], [0, 0, 1]])
    points, colors = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8)
    write_reconstruction(Reconstruction(["00001.png"], K[None], c2w[None], points, colors), paths)

    camera = load_cameras(paths.frames, paths.transforms, downscale=2)[0]
    assert camera.image.shape == (180, 320, 3)
    assert tuple(camera.image[90, 160]) == (255, 0, 0)
    np.testing.assert_allclose(camera.intrinsics, [[250, 0, 160], [0, 200, 90], [0, 0, 1]])
    np.testing.assert_allclose(camera.cam_to_world, c2w)  # back to OpenCV axes


def test_initial_gaussians_stay_within_the_budget_and_survive_duplicates():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, (1000, 3)).astype(np.float32)
    points[1] = points[0]
    points[2] = points[0]
    points[3] = points[0]
    colors = np.full((1000, 3), 255, np.uint8)

    g = initial_gaussians(points, colors, max_points=2000)
    assert len(g["means"]) == 1000
    assert np.isfinite(g["scales"]).all()  # four identical points have zero spacing
    np.testing.assert_allclose(g["sh0"][0, 0], 0.5 / SH_C0)
    assert g["sh0"].shape == (1000, 1, 3) and g["opacities"].shape == (1000,)

    assert len(initial_gaussians(points, colors, max_points=100)["means"]) == 100


def test_separable_ssim_matches_the_full_11x11_window():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from atomspace.splat.train import _ssim

    gen = torch.Generator().manual_seed(0)
    a, b = torch.rand(1, 40, 50, 3, generator=gen, dtype=torch.float64), torch.rand(1, 40, 50, 3, generator=gen, dtype=torch.float64)
    x = torch.arange(11, dtype=a.dtype) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = torch.outer(g, g)[None, None].repeat(3, 1, 1, 1)
    p, q = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    blur = lambda image: F.conv2d(image, window, groups=3)
    ma, mb = blur(p), blur(q)
    va, vb, cov = blur(p * p) - ma**2, blur(q * q) - mb**2, blur(p * q) - ma * mb
    c1, c2 = 0.01**2, 0.03**2
    full = (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))).mean()

    assert torch.allclose(_ssim(a, b), full, rtol=0, atol=1e-12)
    assert _ssim(a, a).item() == pytest.approx(1.0)
