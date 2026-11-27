# LiDAR Ingestion Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second capture pathway that ingests a Stray Scanner recording (ARKit poses + LiDAR depth) and produces a metric, gravity-aligned scene bundle without a manual scale step.

**Architecture:** A new `Track.FAST` route through the existing stages. Ingest copies the recording's sidecars and extracts frames on a stride of the native frame rate so every keyframe keeps its true source index. Reconstruct unprojects the LiDAR depth into ARKit's metric frame, runs MASt3R-SfM as today, fits a RANSAC similarity transform between the two camera trajectories, maps MASt3R's cloud through it, and emits ARKit's poses with the merged cloud. Downstream stages are unchanged except that the floor plane takes its normal from gravity.

**Tech Stack:** Python 3.10+, numpy, scipy (`Rotation`, `ndimage`), opencv-python, Pillow, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-27-lidar-ingestion-design.md`

## Global Constraints

- Priorities: quality first, efficiency second; neither traded against correctness.
- The plain-video path must keep working bit-for-bit unchanged. Fast-track behaviour is additive.
- Depth maps: 16-bit single-channel PNG, 256x192, values in **millimetres**. Confidence: PNG, values 0/1/2, 0 is unreliable.
- Colour video: 1920x1440 HEVC. `odometry.csv` intrinsics are in **colour-frame pixels** and must be rescaled to the depth resolution before unprojection. Never hardcode either resolution; derive both from the files.
- `odometry.csv` header, verbatim: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y`. Timestamps in seconds, positions in metres, distortion centre may be empty.
- ARKit cameras are +Y up, -Z forward. `export.py` applies `CV_TO_GL` itself, so readers must produce **OpenCV** (+Y down, +Z forward) `cam_to_world`.
- MASt3R's `init` accepts `intrinsics` and `depthmap`; `cam2w` hits `assert False` at `sparse_ga.py:247`. Never pass `cam2w`.
- Pixel-centre convention when rescaling a principal point: `c' = (c + 0.5) * s - 0.5`, matching `mast3r_sfm.to_original_pixels`.
- All tests in this plan run without a GPU and without `third_party/mast3r` fetched.

## Review Focus

Input classes the spec implies but which no task's happy path exercises. Each has its test added to the task that owns the code.

1. **Frame counts disagree** — the colour stream, `odometry.csv` rows and `depth/` files do not all have the same length. Every pose then attaches to the wrong image. A reasonable person expects a loud failure, not a plausible scene. *(Task 5)*
2. **A pose convention guessed wrong** — if Stray Scanner's quaternion is world-to-camera rather than camera-to-world, geometry comes out inside-out. A reasonable person expects the reader to be pinned by a test with hand-computed matrices so the real capture confirms rather than discovers it. *(Task 1)*
3. **A degenerate trajectory** — a straight corridor walk leaves camera centres near-collinear, so the Umeyama rotation is ill-conditioned. A reasonable person expects a refusal with a clear reason, not a silently wrong transform. *(Task 2)*
4. **Everything out of range** — a capture of a large hall where almost every depth pixel is confidence 0 or beyond the sensor's reach. A reasonable person expects a named error, not an empty point cloud that fails much later inside `occupancy_grid`. *(Task 3)*
5. **A rotated capture** — a portrait-locked recording with a non-zero container rotation flag, which `extract_frames` applies and which would rotate keyframes out from under the odometry intrinsics. A reasonable person expects a refusal on the fast track. *(Task 5)*

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/atomspace/ingest/stray.py` | **Create.** Everything that knows Stray Scanner's on-disk format: recording discovery, `odometry.csv`, depth and confidence PNGs. |
| `pipeline/atomspace/reconstruct/lidar.py` | **Create.** Depth unprojection into world points and their colours. |
| `pipeline/atomspace/reconstruct/align.py` | **Create.** Umeyama similarity fit with RANSAC. |
| `pipeline/atomspace/reconstruct/sensor_track.py` | **Replace.** Orchestration of the fast track. |
| `pipeline/atomspace/ingest/slice.py` | **Modify.** Add strided native-rate extraction. |
| `pipeline/atomspace/ingest/probe.py` | **Modify.** Probe a recording directory. |
| `pipeline/atomspace/ingest/route.py` | **Modify.** Switch on `has_ar_poses`. |
| `pipeline/atomspace/ingest/__init__.py` | **Modify.** Fast-track branch: copy sidecars, strided extraction, `source_indices`. |
| `pipeline/atomspace/navmesh/floor.py` | **Modify.** Add a known-normal floor fit. |
| `pipeline/atomspace/navmesh/__init__.py` | **Modify.** Use gravity on the fast track. |
| `pipeline/atomspace/reconstruct/scale.py` | **Modify.** `metric_factor` consults the sensor-track report, not the track alone. |
| `pipeline/atomspace/reconstruct/mast3r_sfm.py` | **Modify.** Accept an optional `init` for depth and intrinsics seeding. |
| `pipeline/scripts/make_stray_fixture.py` | **Create.** Writes a synthetic recording for tests and manual runs. |
| `pipeline/tests/test_stray.py`, `test_align.py`, `test_lidar.py`, `test_sensor_track.py` | **Create.** |
| `pipeline/tests/test_ingest.py`, `test_floor.py` | **Modify.** Fast-track cases. |

---

### Task 1: Stray Scanner format reader

**Files:**
- Create: `pipeline/atomspace/ingest/stray.py`
- Test: `pipeline/tests/test_stray.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Recording(root: Path, video: Path, odometry: Path, depth: Path | None, confidence: Path | None)`
  - `find_recording(path: Path) -> Recording | None` — `None` when `path` is not a Stray Scanner recording directory.
  - `@dataclass(frozen=True) Odometry(timestamps: np.ndarray, frames: np.ndarray, cam_to_world: np.ndarray, intrinsics: np.ndarray)` — shapes `(N,)`, `(N,)` int, `(N, 4, 4)` OpenCV camera-to-world, `(N, 3, 3)`.
  - `read_odometry(path: Path) -> Odometry`
  - `read_depth_m(path: Path) -> np.ndarray` — `(H, W)` float32 metres, 0 where unmeasured.
  - `read_confidence(path: Path) -> np.ndarray` — `(H, W)` uint8.
  - `ARKIT_TO_OPENCV: np.ndarray` — `np.diag([1.0, -1.0, -1.0, 1.0])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_find_recording_needs_odometry_and_video(tmp_path):
    assert find_recording(tmp_path) is None
    (tmp_path / "rgb.mp4").write_bytes(b"")
    assert find_recording(tmp_path) is None
    (tmp_path / "odometry.csv").write_text("timestamp, frame\n")
    rec = find_recording(tmp_path)
    assert rec is not None and rec.video.name == "rgb.mp4" and rec.depth is None

def test_find_recording_is_none_for_a_plain_video(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"")
    assert find_recording(video) is None

def test_read_odometry_parses_the_documented_header(tmp_path):
    # Identity rotation at (1, 2, 3): ARKit -Z forward becomes OpenCV +Z forward.
    path = tmp_path / "odometry.csv"
    path.write_text(
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
        " distortion_center_x, distortion_center_y\r\n"
        "0.500, 000000, 1, 2, 3, 0, 0, 0, 1, 1500, 1500, 960, 720, ,\r\n"
        "0.533, 000001, 1, 2, 3, 0, 0, 0, 1, 1500, 1500, 960, 720, 960, 720\r\n"
    )
    odom = read_odometry(path)
    assert odom.timestamps.tolist() == [0.5, 0.533]
    assert odom.frames.tolist() == [0, 1]
    np.testing.assert_allclose(odom.cam_to_world[0][:3, 3], [1, 2, 3])
    np.testing.assert_allclose(odom.cam_to_world[0][:3, :3], np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(odom.intrinsics[0], [[1500, 0, 960], [0, 1500, 720], [0, 0, 1]])

def test_read_odometry_applies_rotation_in_opencv_axes():
    # 90 deg about ARKit +Y (up): ARKit forward -Z maps to -X, so OpenCV +Z maps to -X.
    ...  # quaternion (0, sin(45), 0, cos(45)); assert cam_to_world[0][:3, 2] == [-1, 0, 0]

def test_read_depth_converts_millimetres_to_metres(tmp_path):
    path = tmp_path / "000000.png"
    cv2.imwrite(str(path), np.array([[0, 1000], [2500, 65535]], np.uint16))
    np.testing.assert_allclose(read_depth_m(path), [[0, 1.0], [2.5, 65.535]], atol=1e-6)

def test_read_depth_rejects_an_eight_bit_png(tmp_path):
    path = tmp_path / "000000.png"
    cv2.imwrite(str(path), np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="16-bit"):
        read_depth_m(path)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_stray.py -v`
Expected: FAIL, `ModuleNotFoundError: atomspace.ingest.stray`

- [ ] **Step 3: Implement `stray.py`**

`find_recording` returns `None` unless `path` is a directory containing both `odometry.csv` and `rgb.mp4`; `depth`/`confidence` are set only when those directories exist.

`read_odometry` reads with `np.genfromtxt(..., delimiter=",", names=True, dtype=None)` or equivalent; tolerate CRLF, surrounding spaces in the header and empty distortion fields. Build rotations with `scipy.spatial.transform.Rotation.from_quat` (which takes `x, y, z, w` — the file's order). Treat `(x, y, z)` and the quaternion as **camera-to-world in ARKit axes**, then right-multiply by `ARKIT_TO_OPENCV`.

`read_depth_m` reads with `cv2.imread(str(path), cv2.IMREAD_UNCHANGED)`, raises `ValueError` mentioning "16-bit" unless the result is 2-D `uint16`, and divides by 1000.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/test_stray.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/ingest/stray.py pipeline/tests/test_stray.py
git commit -m "Read Stray Scanner recordings"
```

---

### Task 2: Trajectory alignment

**Files:**
- Create: `pipeline/atomspace/reconstruct/align.py`
- Test: `pipeline/tests/test_align.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Similarity(scale: float, rotation: np.ndarray, translation: np.ndarray, residual_m: float, inliers: np.ndarray)` with a `matrix` property returning the `(4, 4)` transform and an `apply(points)` method.
  - `umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]` — least-squares similarity mapping `source` onto `target`.
  - `robust_similarity(source, target, threshold_m: float = 0.05, iters: int = 200, seed: int = 0) -> Similarity`
  - `MIN_SPREAD = 0.05` — smallest acceptable ratio of the trajectory's thinnest to thickest principal extent.

- [ ] **Step 1: Write the failing tests**

```python
def test_umeyama_recovers_a_known_similarity():
    # source -> 2.5 * R * source + t, with R a 30 deg rotation about an arbitrary axis.
    ...  # assert scale == approx(2.5), R and t to 1e-9

def test_robust_similarity_ignores_outliers():
    # 40 points on a known transform, 8 of them displaced by metres.
    fit = robust_similarity(source, target, threshold_m=0.05)
    assert fit.inliers.sum() == 32
    assert fit.residual_m < 1e-6
    np.testing.assert_allclose(fit.scale, 2.5, rtol=1e-6)

def test_residual_is_reported_in_metres():
    # Every target displaced 0.10 m along +X: no transform fits, residual is 0.10.
    assert fit.residual_m == pytest.approx(0.10, abs=1e-3)

def test_collinear_trajectory_is_refused():
    # A straight corridor walk: camera centres on one line.
    with pytest.raises(RuntimeError, match="straight line"):
        robust_similarity(line, line * 2.0)

def test_too_few_cameras_is_refused():
    with pytest.raises(RuntimeError, match="at least 3"):
        robust_similarity(np.zeros((2, 3)), np.zeros((2, 3)))
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_align.py -v`
Expected: FAIL, `ModuleNotFoundError: atomspace.reconstruct.align`

- [ ] **Step 3: Implement `align.py`**

`umeyama` is the standard closed form: centre both clouds, SVD of the cross-covariance, flip the last singular vector's sign when `det < 0`, scale as the ratio of the correlation trace to the source variance.

`robust_similarity` raises `RuntimeError` mentioning "at least 3" below three points; checks the **target** trajectory's spread by comparing the smallest and largest singular values of its centred coordinates and raises `RuntimeError` mentioning "straight line" when the ratio is under `MIN_SPREAD`; then samples minimal sets of 3, scores inliers by residual under `threshold_m`, keeps the largest consensus, refits `umeyama` on the inliers and reports the inlier RMS residual in metres.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/test_align.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/reconstruct/align.py pipeline/tests/test_align.py
git commit -m "Fit a similarity transform between two camera trajectories"
```

---

### Task 3: Depth unprojection and colouring

**Files:**
- Create: `pipeline/atomspace/reconstruct/lidar.py`
- Test: `pipeline/tests/test_lidar.py`

**Interfaces:**
- Consumes: `read_depth_m`, `read_confidence` (Task 1).
- Produces:
  - `scale_intrinsics(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray` — sizes are `(width, height)`.
  - `unproject(depth_m, confidence, K_depth, cam_to_world, min_confidence=1, max_range_m=5.0) -> np.ndarray` — `(M, 3)` float64 world points.
  - `sample_colors(points, image: np.ndarray, K_color, cam_to_world) -> np.ndarray` — `(M, 3)` uint8, BGR-to-RGB handled here.
  - `MAX_RANGE_M = 5.0`

- [ ] **Step 1: Write the failing tests**

```python
def test_scale_intrinsics_uses_the_pixel_centre_convention():
    K = np.array([[1500.0, 0, 959.5], [0, 1500.0, 719.5], [0, 0, 1]])
    out = scale_intrinsics(K, (1920, 1440), (256, 192))
    np.testing.assert_allclose(out[0, 0], 200.0)       # 1500 / 7.5
    np.testing.assert_allclose(out[:2, 2], [127.5, 95.5])

def test_unproject_recovers_a_plane_at_a_known_distance():
    # Depth map filled with 2.0 m, identity pose: every point lands at z = 2.0,
    # and the frustum's width at 2 m equals 2 * 2.0 * (w / 2) / fx.
    ...

def test_unproject_places_points_with_the_camera_pose():
    # Camera translated to (5, 0, 0) and yawed 90 deg: the plane lands on the expected axis.
    ...

def test_unproject_drops_low_confidence_and_out_of_range():
    # One pixel at confidence 0, one at 9 m, one good.
    assert len(unproject(depth, conf, K, np.eye(4))) == 1

def test_unproject_raises_when_nothing_is_in_range():
    with pytest.raises(RuntimeError, match="in range"):
        unproject(np.full((4, 4), 20.0), np.full((4, 4), 2, np.uint8), K, np.eye(4))

def test_sample_colors_reads_the_pixel_the_point_projects_to():
    # A 4x4 image with one red pixel; a point placed on that ray comes back red.
    ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_lidar.py -v`
Expected: FAIL, `ModuleNotFoundError: atomspace.reconstruct.lidar`

- [ ] **Step 3: Implement `lidar.py`**

`unproject` builds a pixel grid, keeps `(confidence >= min_confidence) & (0 < depth <= max_range_m)`, back-projects with `K_depth` in OpenCV axes (`x = (u - cx) * z / fx`), transforms by `cam_to_world`, and raises `RuntimeError` mentioning "in range" when the mask is empty.

`sample_colors` projects each world point back through `inv(cam_to_world)` and `K_color`, clips to the image bounds and takes nearest-neighbour samples.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/test_lidar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/reconstruct/lidar.py pipeline/tests/test_lidar.py
git commit -m "Unproject LiDAR depth into coloured world points"
```

---

### Task 4: Strided native-rate frame extraction

**Files:**
- Modify: `pipeline/atomspace/ingest/slice.py`
- Test: `pipeline/tests/test_ingest.py`

**Interfaces:**
- Produces: `extract_strided_frames(video: Path, out_dir: Path, stride: int, max_side: int) -> list[tuple[int, Path]]` — `(source_index, path)` with `source_index = ordinal * stride`, in time order.
- `extract_frames` keeps its current signature and behaviour.

- [ ] **Step 1: Write the failing test**

```python
def test_strided_extraction_reports_true_source_indices(panning_video, tmp_path):
    # 4 s at 30 fps = 120 frames; stride 5 keeps every fifth.
    got = extract_strided_frames(panning_video, tmp_path / "out", stride=5, max_side=1920)
    assert [i for i, _ in got] == list(range(0, 120, 5))[: len(got)]
    assert all(p.is_file() for _, p in got)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd pipeline && python -m pytest tests/test_ingest.py -k strided -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `extract_strided_frames`**

Select frames by index rather than resampling time, so the mapping to `odometry.csv` rows stays exact: `-vf "select='not(mod(n\,{stride}))',{scale}" -fps_mode passthrough`. Reuse the existing `scale` filter expression and `-q:v 2`. Pair each output with `ordinal * stride`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd pipeline && python -m pytest tests/test_ingest.py -k strided -v`
Expected: PASS (skipped without ffmpeg)

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/ingest/slice.py pipeline/tests/test_ingest.py
git commit -m "Extract frames on a stride of the native frame rate"
```

---

### Task 5: Ingest fast track

**Files:**
- Modify: `pipeline/atomspace/ingest/probe.py`, `route.py`, `__init__.py`
- Test: `pipeline/tests/test_ingest.py`

**Interfaces:**
- Consumes: `find_recording`, `read_odometry` (Task 1); `extract_strided_frames` (Task 4).
- Produces:
  - `probe_video(video: Path, recording: Recording | None = None) -> VideoMeta` — `has_ar_poses`/`has_depth` come from `recording` when given.
  - `choose_track(meta: VideoMeta) -> Track` — `Track.FAST` when `meta.has_ar_poses`.
  - `ingest.run(video: Path, ...)` accepts a recording directory as `video`.
  - `ingest.json` gains `"source_indices": [int, ...]`, parallel to `"keyframes"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_choose_track_is_fast_with_ar_poses():
    assert choose_track(_meta(has_ar_poses=True)) is Track.FAST
    assert choose_track(_meta(has_ar_poses=False)) is Track.FALLBACK

def test_ingest_of_a_recording_copies_sidecars_and_records_indices(stray_recording, tmp_path):
    paths = ScenePaths(tmp_path / "scenes" / "room")
    ingest.run(stray_recording, paths, max_keyframes=150)
    data = json.loads(paths.ingest_meta.read_text())
    assert data["track"] == "fast"
    assert len(data["source_indices"]) == len(data["keyframes"])
    assert data["source_indices"] == sorted(data["source_indices"])
    assert (paths.raw / "odometry.csv").is_file()
    assert (paths.raw / "depth").is_dir()

def test_ingest_refuses_a_rotated_recording(stray_recording_rotated, tmp_path):
    with pytest.raises(RuntimeError, match="rotat"):
        ingest.run(stray_recording_rotated, ScenePaths(tmp_path / "s"), max_keyframes=150)

def test_ingest_refuses_mismatched_frame_counts(stray_recording_short_odometry, tmp_path):
    with pytest.raises(RuntimeError, match="odometry"):
        ingest.run(stray_recording_short_odometry, ScenePaths(tmp_path / "s"), max_keyframes=150)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_ingest.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the fast-track branch**

In `ingest.run`: call `find_recording(video)`; when it returns a recording, use its `video` for probing and slicing, copy `odometry.csv`, `depth/` and `confidence/` into `paths.raw`, and take the fast-track path. Stride is `round(meta.fps / cfg.extract_fps)`, at least 1, so the filters keep their 6 fps-equivalent tuning. Refuse a non-zero `meta.rotation_deg`, naming it. Compare the odometry row count against the extracted frame count scaled by the stride: raise `RuntimeError` naming "odometry" when they differ by more than one stride, and otherwise truncate to the common prefix with a printed warning. Carry `source_index` through `filter_sharp` and `select_keyframes` by keeping a path-to-index mapping, and write `source_indices` alongside `keyframes`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/ -v`
Expected: PASS, existing tests unchanged

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/ingest pipeline/tests/test_ingest.py
git commit -m "Route a Stray Scanner recording through the fast track"
```

---

### Task 6: Gravity-aligned floor

**Files:**
- Modify: `pipeline/atomspace/navmesh/floor.py`, `pipeline/atomspace/navmesh/__init__.py`
- Test: `pipeline/tests/test_floor.py`

**Interfaces:**
- Produces: `floor_at_known_normal(points: np.ndarray, normal: np.ndarray) -> FloorPlane` — keeps `normal` exactly and solves only the offset.
- `fit_floor` is untouched, so the fallback track is unaffected. Wiring it into `navmesh.run` belongs to Task 7, which writes the file holding gravity.

- [ ] **Step 1: Write the failing tests**

```python
def test_known_normal_is_kept_exactly():
    # A room whose floor points carry noise that would tilt an SVD fit.
    plane = floor_at_known_normal(points, Y)
    np.testing.assert_allclose(plane.normal, Y)

def test_known_normal_finds_the_lowest_horizontal_surface():
    # Floor at y = 0, table at 0.75, ceiling at 2.6, all with gravity known.
    restored = _apply(floor_at_known_normal(points, Y), points)
    assert np.abs(restored[:5000, 1]).max() < 0.02

def test_gravity_beats_a_point_fit_on_a_sparse_floor():
    # 300 floor points against 8000 wall points: fit_floor tilts, gravity does not.
    ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_floor.py -v`
Expected: FAIL, `ImportError: floor_at_known_normal`

- [ ] **Step 3: Implement `floor_at_known_normal` in `pipeline/atomspace/navmesh/floor.py`**

Project the points onto `normal`, histogram the heights at `INLIER_DIST_M` resolution and take the lowest bin holding at least `MIN_PLANE_SHARE` of the points as the floor, returning `FloorPlane(normal, -height)`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/test_floor.py tests/test_navmesh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/navmesh/floor.py pipeline/tests/test_floor.py
git commit -m "Fit the floor to a known gravity normal"
```

---

### Task 7: Sensor track orchestration

**Files:**
- Replace: `pipeline/atomspace/reconstruct/sensor_track.py`
- Modify: `pipeline/atomspace/reconstruct/__init__.py`, `pipeline/atomspace/reconstruct/scale.py`, `pipeline/atomspace/navmesh/__init__.py`
- Test: `pipeline/tests/test_sensor_track.py`

**Interfaces:**
- Consumes: `Odometry`/`read_odometry` (Task 1), `robust_similarity` (Task 2), `unproject`/`sample_colors` (Task 3), `source_indices` (Task 5).
- Produces:
  - `reconstruct(raw_dir: Path, frames: list[Path], source_indices: list[int], mast3r: Callable[[], Reconstruction] | None) -> tuple[Reconstruction, dict]` — the dict is the quality report written to `logs/`.
  - `RESIDUAL_LIMIT_M = 0.15` — above this the scene is not treated as metric.
  - `recon/world_up.json` holding ARKit's up direction for Task 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_sensor_track_without_mast3r_uses_lidar_alone(stray_scene):
    recon, report = reconstruct(raw, frames, indices, mast3r=None)
    assert len(recon.cam_to_world) == len(frames)
    assert report["residual_m"] is None and len(recon.points) > 0

def test_mast3r_points_are_mapped_into_arkit_metres(stray_scene):
    # A stub MASt3R whose reconstruction is the truth scaled by 1/3 and rotated:
    # after alignment its points must land on the LiDAR points.
    recon, report = reconstruct(raw, frames, indices, mast3r=stub)
    assert report["residual_m"] < 1e-6
    assert report["scale_source"] == "sensor"

def test_a_bad_residual_withholds_metric_scale(stray_scene):
    # A stub whose trajectory disagrees by 0.5 m.
    recon, report = reconstruct(raw, frames, indices, mast3r=noisy_stub)
    assert report["residual_m"] > RESIDUAL_LIMIT_M
    assert report["scale_source"] == "unscaled"
    assert report["warning"]

def test_poses_are_arkits_not_mast3rs(stray_scene):
    # The stub's poses are deliberately wrong; the output keeps ARKit's.
    ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_sensor_track.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `sensor_track.reconstruct`**

Read `odometry.csv`, select the rows named by `source_indices`, unproject each keyframe's depth with intrinsics scaled from the colour frame to the depth map, colour the points from the keyframe JPEG, and concatenate. When `mast3r` is given, call it, fit `robust_similarity` from MASt3R's camera centres onto ARKit's, map its cloud through the fit and concatenate. Voxel-downsample the merged cloud with `voxel_downsample` at an absolute size in metres. Write `recon/world_up.json`. Return the `Reconstruction` carrying ARKit's `cam_to_world` and ARKit's per-frame intrinsics, plus a report holding `residual_m`, `inliers`, `scale_source` and any `warning`.

In `reconstruct/__init__.py`, pass `source_indices` and a callable that runs `mast3r_sfm.reconstruct`, write the report to `paths.logs / "sensor_track.json"`, and print the residual. When `scale_source` is `"unscaled"`, leave the scene unscaled so `metric_factor` does not claim `sensor` — gate that in `scale.metric_factor` on the report rather than on the track alone. In `navmesh.run`, when `recon/world_up.json` exists, use `floor_at_known_normal` (Task 6) with that direction in place of `fit_floor`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/reconstruct pipeline/tests/test_sensor_track.py
git commit -m "Adopt ARKit poses and merge the LiDAR and MASt3R clouds"
```

---

### Task 8: Synthetic fixture and end-to-end run

**Files:**
- Create: `pipeline/scripts/make_stray_fixture.py`
- Create: `test/strayroom/` (generated recording, committed)
- Test: `pipeline/tests/test_sensor_track.py`

**Interfaces:**
- Produces: `python pipeline/scripts/make_stray_fixture.py --out test/strayroom` writes `odometry.csv`, `depth/`, `confidence/` and pre-extracted `frames/` for a synthetic box room with a known camera path.

- [ ] **Step 1: Write the failing test**

```python
def test_the_committed_fixture_reconstructs_a_box_room():
    # Fixture room is 4 x 3 x 2.6 m; the merged cloud's extent must match within 5 cm.
    recon, report = reconstruct(FIXTURE / "raw", frames, indices, mast3r=None)
    extent = recon.points.max(axis=0) - recon.points.min(axis=0)
    np.testing.assert_allclose(sorted(extent), [2.6, 3.0, 4.0], atol=0.05)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd pipeline && python -m pytest tests/test_sensor_track.py -k fixture -v`
Expected: FAIL, fixture missing

- [ ] **Step 3: Write the generator and commit its output**

Ray-cast a box room from each pose on a circular walk to make depth maps, set confidence 2 within range and 0 beyond, render matching flat-coloured JPEGs per wall, and write the documented `odometry.csv`. Keep it to ~12 frames at 256x192 so the fixture stays small and out of Git LFS. Add a `test/README.md` note and a `THIRD_PARTY_NOTICES.md` line stating the fixture is ours.

- [ ] **Step 4: Run to verify it passes**

Run: `cd pipeline && python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/scripts/make_stray_fixture.py test/strayroom test/README.md THIRD_PARTY_NOTICES.md
git commit -m "Add a synthetic Stray Scanner fixture"
```

---

### Task 9: Seed MASt3R with LiDAR depth and ARKit intrinsics

The spec's one genuinely open technical question. Everything before this task stands without
it, so if it cannot be made to fit, close the task and say so rather than forcing it.

**Files:**
- Modify: `pipeline/atomspace/reconstruct/mast3r_sfm.py`, `pipeline/atomspace/reconstruct/sensor_track.py`
- Test: `pipeline/tests/test_reconstruct.py`

**Interfaces:**
- Produces: `mast3r_sfm.reconstruct(frames, profile, cache_dir, init: dict | None = None)` — `init` is passed through to `sparse_global_alignment`, never containing `cam2w`.
- `depth_init(odometry, frames, source_indices, raw_dir) -> dict` in `sensor_track.py`, building MASt3R's per-image `init` entries from the recording.

- [ ] **Step 1: Write the failing tests**

```python
def test_init_never_carries_cam2w():
    # cam2w hits assert False at sparse_ga.py:247, so it must never be built.
    init = depth_init(odometry, frames, indices, raw)
    assert init and all("cam2w" not in v for v in init.values())
    assert all(set(v) <= {"depthmap", "intrinsics"} for v in init.values())

def test_init_is_keyed_by_the_image_paths_mast3r_is_given():
    assert set(init) == {str(f) for f in frames}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd pipeline && python -m pytest tests/test_reconstruct.py -k init -v`
Expected: FAIL, `ImportError: depth_init`

- [ ] **Step 3: Implement the seeding**

Add the `init` parameter to `mast3r_sfm.reconstruct` and forward it to `sparse_global_alignment`. Build each entry's `intrinsics` from the odometry row, resized to MASt3R's network input by inverting `to_original_pixels`. For `depthmap`, resample the keyframe's depth map to the canonical-view layout `core_depth` expects.

**On the GPU machine**, run the full pipeline on the committed fixture and on a real recording with seeding on and off, and compare the resulting clouds. `core_depth` is subsampled canonical-view data, not a raw 256x192 map, so the resampling has to be confirmed against MASt3R's own shapes rather than assumed. If it will not fit cleanly, revert this task's changes to `sensor_track.py`, keep the `init` parameter on `mast3r_sfm.reconstruct` (it is harmless and correct), and record the finding in the spec.

- [ ] **Step 4: Run to verify they pass**

Run: `cd pipeline && python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atomspace/reconstruct pipeline/tests/test_reconstruct.py
git commit -m "Seed MASt3R with LiDAR depth and ARKit intrinsics"
```

---

### Task 10: Documentation

**Files:**
- Modify: `docs/architecture.md`, `docs/file_structure.md`, `README.md`

- [ ] **Step 1: Update the docs**

Move the sensor track from "Long-term" to "Built today" in `architecture.md`'s scope table, replace the `(planned)` markers on the capture and reconstruct rows with the real flow, and add the fast track to the current-pipeline diagram including the residual gate. Add the new modules to `file_structure.md`. Add a README section on capturing with Stray Scanner and running `atomspace run <scene> <recording-folder>`, stating that ingest and reconstruct must run on the same machine because `scenes/*/raw/` is gitignored.

- [ ] **Step 2: Verify**

Run: `cd pipeline && python -m pytest -q && cd ../viewer && npx tsc --noEmit`
Expected: all tests pass, no type errors

- [ ] **Step 3: Commit**

```bash
git add docs README.md
git commit -m "Document the sensor capture track"
```
