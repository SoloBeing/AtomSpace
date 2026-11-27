import json

import numpy as np

from atomspace import bundle, navmesh
from atomspace.config import NavmeshConfig, ScenePaths
from atomspace.navmesh.obstacles import occupancy_grid
from atomspace.navmesh.polygonize import build_navmesh, safe_cells, walkable_mesh
from atomspace.reconstruct.export import write_ply
from atomspace.reconstruct.scale import ScaleReference, save_reference

CFG = NavmeshConfig(agent_radius_m=0.4)  # the room sizes below are written around this margin
STEP = 0.02


def _floor(x0, x1, z0, z1):
    x, z = np.meshgrid(np.arange(x0, x1, STEP), np.arange(z0, z1, STEP))
    return np.c_[x.ravel(), np.zeros(x.size), z.ravel()]


def _box(x0, x1, z0, z1, top=1.9):
    """Solid column from the floor up: a wall or a piece of furniture."""
    x, y, z = np.meshgrid(np.arange(x0, x1, STEP), np.arange(0.2, top, 0.1), np.arange(z0, z1, STEP))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def _walled_room(*extra):
    walls = [_box(-0.1, 5.1, -0.1, 0), _box(-0.1, 5.1, 4, 4.1), _box(-0.1, 0, 0, 4), _box(5, 5.1, 0, 4)]
    return np.vstack([_floor(0, 5, 0, 4), *walls, *extra])


def _mesh(points):
    grid = occupancy_grid(points, CFG)
    return walkable_mesh(safe_cells(grid, CFG.agent_radius_m), grid)


def _area(vertices, triangles):
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return np.abs(np.cross(b - a, c - a)[:, 1]).sum() / 2


def _inside(vertices, triangles, x, z):
    p = np.array([x, 0.0, z])
    for tri in triangles:
        a, b, c = vertices[tri]
        signs = [np.cross(q - o, p - o)[1] for o, q in ((a, b), (b, c), (c, a))]
        if all(s >= 0 for s in signs):
            return True
    return False


def _room_area_with_margin(m):
    """5 x 4 m room less a margin m, less the 1 m table grown by m (rounded corners)."""
    return (5 - 2 * m) * (4 - 2 * m) - ((1 + 2 * m) ** 2 - (4 - np.pi) * m**2)


def _assert_margin_is_tight(vertices, triangles):
    # Never closer than the agent radius; outlines through cell centres and grid snapping
    # may leave up to two cells more, on the safe side.
    assert _room_area_with_margin(0.5) < _area(vertices, triangles) < _room_area_with_margin(0.4)


def test_room_with_table_keeps_clear_of_everything():
    vertices, triangles = _mesh(_walled_room(_box(2, 3, 1.5, 2.5, top=0.8)))
    _assert_margin_is_tight(vertices, triangles)

    assert _inside(vertices, triangles, 1.0, 1.0)
    assert not _inside(vertices, triangles, 2.5, 2.0)  # on the table
    assert not _inside(vertices, triangles, 1.75, 2.0)  # 0.25 m from the table
    assert not _inside(vertices, triangles, 0.2, 2.0)  # 0.2 m from the wall
    assert not _inside(vertices, triangles, 2.5, 3.8)


def test_unobserved_floor_is_not_walkable():
    # Floor only seen in the left half; the right half is a hole in the capture.
    points = np.vstack([_floor(0, 2.5, 0, 4), _box(-0.1, 5.1, -0.1, 0), _box(-0.1, 5.1, 4, 4.1), _box(-0.1, 0, 0, 4)])
    vertices, triangles = _mesh(points)
    assert vertices[:, 0].max() < 2.5


def test_floor_under_the_walked_path_counts_even_if_never_seen():
    # A phone held ahead of the body leaves a hole in the floor exactly where its owner walked.
    x, z = np.meshgrid(np.arange(0, 5, STEP), np.arange(0, 4, STEP))
    keep = ~((np.abs(x - 2.5) < 1.2) & (np.abs(z - 2) < 1.0))
    floor = np.c_[x[keep], np.zeros(keep.sum()), z[keep]]
    walls = [_box(-0.1, 5.1, -0.1, 0), _box(-0.1, 5.1, 4, 4.1), _box(-0.1, 0, 0, 4), _box(5, 5.1, 0, 4)]
    vertices, triangles = _mesh(np.vstack([floor, *walls]))
    assert _inside(vertices, triangles, 2.5, 2.0)


def test_walked_cells_are_free_of_noise_and_connect_to_the_floor():
    # A strip the camera crossed joins two seen floors, past a smear of noise that reads as obstacle.
    floor = np.vstack([_floor(0, 2, 0, 4), _floor(3, 5, 0, 4)])
    smear = _box(2.0, 3.0, 1.8, 2.2, top=0.5)
    walls = [_box(-0.1, 5.1, -0.1, 0), _box(-0.1, 5.1, 4, 4.1), _box(-0.1, 0, 0, 4), _box(5, 5.1, 0, 4)]
    points = np.vstack([floor, smear, *walls])

    path = np.c_[np.linspace(1.5, 3.5, 30), np.full(30, 2.0)]
    grid = occupancy_grid(points, NavmeshConfig(walked_radius_m=0.6), walked_xz=path)
    vertices, triangles = walkable_mesh(safe_cells(grid, CFG.agent_radius_m), grid)
    assert _inside(vertices, triangles, 1.0, 2.0) and _inside(vertices, triangles, 4.0, 2.0)


def test_stray_points_do_not_block_the_floor():
    stray = np.array([[2.5, 1.0, 2.0], [2.5, 1.1, 2.0]])
    vertices, triangles = _mesh(_walled_room(stray))
    assert _inside(vertices, triangles, 2.5, 2.0)


def _two_rooms(gap):
    """Rooms joined through a wall at x = 3, with an opening `gap` meters wide around z = 2."""
    wall = [_box(3, 3.1, 0, 2 - gap / 2), _box(3, 3.1, 2 + gap / 2, 4)]
    bigger = _floor(3, 8, 0, 4)
    outer = [_box(-0.1, 8.1, -0.1, 0), _box(-0.1, 8.1, 4, 4.1), _box(-0.1, 0, 0, 4), _box(8, 8.1, 0, 4)]
    return np.vstack([_floor(0, 3, 0, 4), bigger, *wall, *outer])


def test_doorway_wider_than_the_agent_joins_rooms():
    vertices, triangles = _mesh(_two_rooms(gap=1.2))
    assert _inside(vertices, triangles, 1.5, 2.0) and _inside(vertices, triangles, 6.0, 2.0)


def test_doorway_narrower_than_the_agent_keeps_the_larger_room_only():
    vertices, triangles = _mesh(_two_rooms(gap=0.6))
    assert _inside(vertices, triangles, 6.0, 2.0)
    assert not _inside(vertices, triangles, 1.5, 2.0)


def test_navmesh_topology():
    vertices, triangles = _mesh(_walled_room(_box(2, 3, 1.5, 2.5)))
    mesh = build_navmesh(vertices, triangles, CFG)
    polygons = mesh["polygons"]

    # Counter-clockwise from above (+Y): the face normal points up.
    for poly in polygons:
        a, b, c = np.array(mesh["vertices"])[poly["v"]]
        assert np.cross(b - a, c - a)[1] > 0

    # Neighbours agree in both directions, and across a shared edge.
    for index, poly in enumerate(polygons):
        for i, other in enumerate(poly["n"]):
            if other == -1:
                continue
            assert index in polygons[other]["n"]
            assert {poly["v"][i], poly["v"][(i + 1) % 3]} <= set(polygons[other]["v"])
    assert any(-1 in p["n"] for p in polygons)


def test_unscaled_bundle_stands_upright_around_the_first_camera(tmp_path):
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    camera = np.eye(4)
    camera[:3, :3] = q
    camera[:3, 3] = [1.0, 2.0, 3.0]
    paths.transforms.write_text(json.dumps({"frames": [{"transform_matrix": camera.tolist()}]}))
    paths.ingest_meta.write_text(json.dumps({"track": "fallback", "keyframes": ["00001.jpg"]}))

    bundle.run(paths)

    manifest = json.loads(paths.manifest.read_text())
    assert manifest["scale"]["source"] == "unscaled"
    world_from_recon = np.array(manifest["world_from_recon"]).reshape(4, 4).T
    np.testing.assert_allclose(world_from_recon @ camera[:, 3], [0, 0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(world_from_recon[:3, :3] @ q[:, 1], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(world_from_recon[:3, :3]), 1.0)


def test_run_end_to_end_from_recon_space(tmp_path):
    """A half-size, tilted reconstruction comes out as a metric, floor-aligned, valid bundle."""
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    world_points = _walled_room(_box(2, 3, 1.5, 2.5))

    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    offset = np.array([1.0, -2.0, 0.5])
    recon_points = (world_points @ q.T) * 0.5 + offset

    write_ply(paths.dense_points, recon_points.astype(np.float32), np.zeros((len(recon_points), 3), np.uint8))
    camera = np.eye(4)
    camera[:3, :3] = q
    paths.transforms.write_text(json.dumps({"frames": [{"transform_matrix": camera.tolist()}]}))
    paths.ingest_meta.write_text(json.dumps({"track": "fallback", "keyframes": ["00001.jpg"]}))
    save_reference(ScaleReference((0, 0, 0), (1, 0, 0), 2.0), paths.scale)

    navmesh.run(paths, CFG)
    bundle.run(paths)

    manifest = json.loads(paths.manifest.read_text())
    assert manifest["files"]["navmesh"] == "navmesh.json"
    assert manifest["scale"]["factor"] == 2.0

    world_from_recon = np.array(manifest["world_from_recon"]).reshape(4, 4).T
    homogeneous = np.c_[recon_points, np.ones(len(recon_points))]
    aligned = (world_from_recon @ homogeneous.T).T[:, :3]
    floor_height = aligned[: 250 * 200, 1]
    assert np.abs(floor_height).max() < 0.02
    np.testing.assert_allclose(np.ptp(aligned[:, 1]), 1.9, atol=0.1)  # metric again

    mesh = json.loads(paths.navmesh.read_text())
    vertices = np.array(mesh["vertices"])
    assert np.all(vertices[:, 1] == 0)
    triangles = np.array([p["v"] for p in mesh["polygons"]])
    _assert_margin_is_tight(vertices, triangles)


def _tilted_fast_scene(tmp_path, write_world_up: bool):
    """A room whose recon frame is rolled 12 degrees, with gravity recorded (or not). The
    floor points are deliberately sparse under a lot of wall, which is where a point fit
    drifts and a measured up direction does not."""
    roll = np.radians(12)
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    points = _walled_room()
    paths = ScenePaths(tmp_path / "scene")
    paths.ensure()
    write_ply(paths.recon / "dense_points.ply", (points @ rotation.T).astype(np.float32),
              np.zeros((len(points), 3), np.uint8))
    paths.ingest_meta.write_text(json.dumps({"track": "fast", "keyframes": ["00001.jpg"]}))
    (paths.logs / "sensor_track.json").write_text(json.dumps({"scale_source": "sensor"}))

    cameras = np.c_[np.linspace(1, 4, 12), np.full(12, 1.5), np.full(12, 2.0)] @ rotation.T
    frames = [{"transform_matrix": _looking_down_from(c).tolist()} for c in cameras]
    paths.transforms.write_text(json.dumps({"frames": frames}))
    if write_world_up:
        (paths.recon / "world_up.json").write_text(json.dumps({"world_up": list(rotation @ [0, 1, 0])}))
    return paths, rotation


def _looking_down_from(centre):
    """An OpenGL camera pose at `centre` with an up axis that is deliberately unhelpful."""
    pose = np.eye(4)
    pose[:3, 3] = centre
    return pose


def test_navmesh_prefers_recorded_gravity_over_a_point_fit(tmp_path):
    """The recorded direction is deliberately 3 degrees off the synthetic floor, which is
    perfectly planar and which a point fit therefore recovers exactly. If world_up.json is
    being read, the world rotation follows the recorded direction and not the fit."""
    paths, rotation = _tilted_fast_scene(tmp_path, write_world_up=False)
    nudge = np.radians(3)
    recorded = rotation @ [0.0, np.cos(nudge), np.sin(nudge)]
    (paths.recon / "world_up.json").write_text(json.dumps({"world_up": list(recorded)}))

    navmesh.run(paths, CFG)

    world = np.array(json.loads(paths.world.read_text())["world_from_recon"]).reshape(4, 4).T
    recovered_up = np.linalg.inv(world[:3, :3])[:, 1]
    np.testing.assert_allclose(recovered_up, recorded, atol=1e-9)


def test_navmesh_still_fits_the_floor_without_recorded_gravity(tmp_path):
    paths, _ = _tilted_fast_scene(tmp_path, write_world_up=False)
    navmesh.run(paths, CFG)
    assert paths.navmesh.exists()
