import json

import numpy as np
import pytest
from PIL import Image

from atomspace.config import ScenePaths
from atomspace.reconstruct.export import Reconstruction, read_ply, write_reconstruction
from atomspace.reconstruct.mast3r_sfm import (
    cache_encoder, faster_nn_search, nearest_in_blocks, pair_graph, to_original_pixels, voxel_downsample,
)


def _connected(n, edges):
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for a, b in edges:
            for j in ((b,) if a == i else (a,) if b == i else ()):
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
    return len(seen) == n


def test_pair_graph_window_only():
    edges = pair_graph(6, window=2, retrieval_k=0, similarity=None)
    assert edges == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]


def test_pair_graph_adds_loop_closure():
    # Last frame looks like the first one (the walk came back to the start).
    sim = np.eye(10)
    sim[0, 9] = sim[9, 0] = 0.9
    edges = pair_graph(10, window=2, retrieval_k=1, similarity=sim)
    assert (0, 9) in edges
    assert all(a < b for a, b in edges) and len(set(edges)) == len(edges)
    assert _connected(10, edges)


def test_intrinsics_map_back_to_keyframe_pixels():
    # 1920x1080 -> 512x288, no crop: focal and centre scale by 3.75.
    K = np.array([[400.0, 0, 255.5], [0, 400.0, 143.5], [0, 0, 1]])
    out = to_original_pixels(K, (1920, 1080))
    np.testing.assert_allclose(out[0, 0], 1500.0)
    np.testing.assert_allclose(out[:2, 2], [959.5, 539.5])

    # 1000x700 -> 512x358, cropped to 512x352: centre of the crop is still the image centre.
    K = np.array([[300.0, 0, 255.5], [0, 300.0, 175.5], [0, 0, 1]])
    out = to_original_pixels(K, (1000, 700))
    np.testing.assert_allclose(out[:2, 2], [499.5, 349.5], atol=0.5)

    # Square 800x800 -> 512x512, cropped to 512x384 (4:3).
    K = np.array([[300.0, 0, 255.5], [0, 300.0, 191.5], [0, 0, 1]])
    out = to_original_pixels(K, (800, 800))
    np.testing.assert_allclose(out[:2, 2], [399.5, 399.5], atol=0.5)


def test_voxel_downsample_merges_repeated_views():
    rng = np.random.default_rng(0)
    surface = rng.uniform(0, 1, (1000, 3)).astype(np.float32)
    points = np.concatenate([surface + rng.normal(0, 1e-4, surface.shape) for _ in range(5)]).astype(np.float32)
    colors = np.zeros((len(points), 3), np.uint8)
    kept, _ = voxel_downsample(points, colors, voxel=0.01, max_points=10_000)
    assert len(kept) < 1500
    kept, _ = voxel_downsample(points, colors, voxel=1e-6, max_points=300)
    assert len(kept) == 300


def test_write_reconstruction(tmp_path):
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    Image.new("RGB", (640, 360)).save(paths.frames / "00001.jpg")
    paths.scale.write_text("{}")
    paths.navmesh.write_text("{}")

    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    K = np.array([[500.0, 0, 320], [0, 500.0, 180], [0, 0, 1]])
    points = np.array([[0, 0, 1], [1, 2, 3.5]], np.float32)
    colors = np.array([[255, 0, 0], [0, 128, 255]], np.uint8)
    write_reconstruction(Reconstruction(["00001.jpg"], K[None], c2w[None], points, colors), paths)

    t = json.loads(paths.transforms.read_text())
    frame = t["frames"][0]
    assert (paths.recon / frame["file_path"]).resolve() == (paths.frames / "00001.jpg").resolve()
    assert (frame["w"], frame["h"], frame["fl_x"], frame["cx"]) == (640, 360, 500.0, 320.0)
    m = np.array(frame["transform_matrix"])
    np.testing.assert_allclose(m[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(m[:3, :3], np.diag([1, -1, -1]))  # OpenGL camera axes
    assert t["ply_file_path"] == "dense_points.ply"

    data = paths.dense_points.read_bytes()
    header, body = data.split(b"end_header\n")
    assert b"element vertex 2" in header
    v = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    np.testing.assert_array_equal(v["p"], points)
    np.testing.assert_array_equal(v["c"], colors)

    read_points, read_colors = read_ply(paths.dense_points)
    np.testing.assert_array_equal(read_points, points)
    np.testing.assert_array_equal(read_colors, colors)

    assert not paths.scale.exists()
    assert not paths.navmesh.exists()


def test_each_image_is_encoded_once_however_many_pairs_it_is_in():
    torch = pytest.importorskip("torch")

    class Model:
        calls = 0

        def _encode_image(self, image, true_shape):
            self.calls += 1
            return image * 2, image * 3, None

    model = Model()
    cache_encoder(model)
    a, b, c = (torch.full((1, 3, 2, 2), float(value)) for value in (1, 2, 4))
    shape = torch.tensor([[2, 2]])

    feat_a, feat_b, pos_a, pos_b = model._encode_image_pairs(a, b, shape, shape)
    feat_b2, feat_c, pos_b2, _ = model._encode_image_pairs(b.clone(), c, shape, shape)

    assert model.calls == 3
    assert torch.equal(feat_a, a * 2) and torch.equal(pos_a, a * 3)
    assert torch.equal(feat_b, b * 2) and torch.equal(pos_b, b * 3)
    assert torch.equal(feat_b2, feat_b) and torch.equal(pos_b2, pos_b)
    assert torch.equal(feat_c, c * 2)


def _mast3r_fast_nn():
    """MASt3R's own fast_nn module, the reference for the faster search."""
    import sys

    from atomspace.config import MAST3R_DIR

    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    if not (MAST3R_DIR / "mast3r" / "fast_nn.py").exists():
        pytest.skip("third_party/mast3r not fetched")
    sys.path.insert(0, str(MAST3R_DIR))
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r import fast_nn

    return fast_nn


@pytest.mark.parametrize("dist", ["dot", "l2"])
def test_block_search_gives_mast3rs_matches(dist):
    torch = pytest.importorskip("torch")
    fast_nn = _mast3r_fast_nn()
    gen = torch.Generator().manual_seed(0)
    A = torch.nn.functional.normalize(torch.randn(50, 8, generator=gen), dim=1)
    B = torch.nn.functional.normalize(torch.randn(300, 8, generator=gen), dim=1)
    # Planted tie across blocks: the same row in block 0 and block 12; the earlier one must win.
    B[200] = B[5]
    A[0] = B[5]
    expected, _ = fast_nn.bruteforce_reciprocal_nns(A, B, device="cpu", block_size=16, dist=dist)
    got = nearest_in_blocks(A, B, 16, dist)
    np.testing.assert_array_equal(got, expected)
    assert got[0] == 5


def test_faster_search_leaves_every_match_unchanged(monkeypatch):
    torch = pytest.importorskip("torch")
    fast_nn = _mast3r_fast_nn()
    monkeypatch.setattr(fast_nn.cdistMatcher, "query", fast_nn.cdistMatcher.query)  # restored after the test
    gen = torch.Generator().manual_seed(1)
    pts1 = torch.nn.functional.normalize(torch.randn(32, 48, 8, generator=gen), dim=-1)
    pts2 = torch.nn.functional.normalize(torch.randn(32, 48, 8, generator=gen), dim=-1)
    # block_size 64: the 24 starting points x 1,536 pixels exceed 64**2, so the blocked path runs;
    # later, smaller queries take MASt3R's own single-pass path.
    kw = dict(subsample_or_initxy1=8, ret_xy=False, device="cpu", dist="dot", block_size=64)
    before = fast_nn.fast_reciprocal_NNs(pts1, pts2, **kw)
    faster_nn_search()
    assert getattr(fast_nn.cdistMatcher.query, "atomspace", False)
    after = fast_nn.fast_reciprocal_NNs(pts1, pts2, **kw)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(a, b)
