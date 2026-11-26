import json

import numpy as np
import pytest
from PIL import Image

from atomspace.config import ScenePaths
from atomspace.reconstruct import mapanything_sfm
from atomspace.reconstruct.export import Reconstruction, read_ply, write_reconstruction
from atomspace.reconstruct.mapanything_sfm import similarity, to_frame_pixels


def test_intrinsics_map_back_to_keyframe_pixels():
    # 1920x1080 -> 518x294: scaled by s = 294 / 1080 to cover, width 522.7 centre-cropped to 518.
    K = np.array([[300.0, 0, 259.0], [0, 300.0, 147.0], [0, 0, 1]])
    out = to_frame_pixels(K, (294, 518), (1920, 1080))
    s = 294 / 1080
    np.testing.assert_allclose(out[0, 0], 300 / s)
    np.testing.assert_allclose(out[:2, 2], [960.0, 540.0])
    # An off-centre principal point keeps its offset, scaled.
    K[0, 2] += 10
    np.testing.assert_allclose(to_frame_pixels(K, (294, 518), (1920, 1080))[0, 2], 960 + 10 / s)


def test_similarity_recovers_scale_rotation_shift():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    dst = 1.53 * src @ R.T + [1, -2, 3]
    s, R_, t = similarity(src, dst)
    np.testing.assert_allclose([s, *t], [1.53, 1, -2, 3], atol=1e-9)
    np.testing.assert_allclose(R_, R, atol=1e-9)


def test_reconstruct_refuses_a_small_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(mapanything_sfm, "vram_gib", lambda: 3.7)
    with pytest.raises(RuntimeError, match="kaggle_reconstruct.ipynb"):
        mapanything_sfm.reconstruct([tmp_path / "00001.jpg"], tmp_path / "ba")


def test_write_reconstruction(tmp_path):
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    Image.new("RGB", (640, 360)).save(paths.frames / "00001.jpg")
    paths.scale.write_text("{}")
    paths.navmesh.write_text("{}")
    paths.pois.write_text('{"pois": ["hand-made"]}')

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
    assert not paths.pois.exists()
    assert "hand-made" in paths.pois.with_name("pois.old.json").read_text()
