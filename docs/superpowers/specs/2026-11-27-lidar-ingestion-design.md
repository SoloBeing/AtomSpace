# LiDAR ingestion path — design

Date: 2026-11-27
Status: approved for planning

## Intent

AtomSpace's headline claim stays what it is: point an ordinary phone at a room, run one
command, walk around it in a browser. That is the capability being offered and nothing here
competes with it.

This adds a second, optional capture pathway for anyone holding an iPhone with LiDAR. Where
vision has to infer geometry and scale, the sensor measures them: ARKit's poses are metric
and gravity-aligned, and depth within the sensor's reach is observed rather than solved.
The pathway exists because it produces a **better reconstruction**, not because it produces a
faster one.

It also removes the manual metric-scale step for scenes captured this way. That step is
currently the weakest link in the whole pipeline: after `reconstruct`, a scene is
`unscaled`, and someone must open the viewer's admin tool, click two points, measure the real
distance between them and type it in before `navmesh` will run. It is error-prone — a
mis-measured door scales the entire scene — and in practice it gets skipped. All three
scenes on the live deployment are `"source": "unscaled"` with `"navmesh": null`.

**Priorities, in order: quality first, efficiency second.** Where the two conflict, take the
better reconstruction. Neither is traded against correctness.

## Goals

1. Dense points are the union of the confidence-filtered LiDAR unprojection and the MASt3R
   cloud, the latter mapped into ARKit's frame — measured geometry where the sensor reaches,
   inferred geometry beyond it.
2. Final camera poses in `transforms.json` are ARKit's: metric, gravity-aligned.
3. The floor plane's normal comes from gravity rather than from a point fit, so world space
   is exactly upright.
4. `atomspace` ingests a Stray Scanner recording folder and produces a bundle whose
   `scale.source` is `sensor`, with no human scale step in the run.
5. The run measures and reports the ARKit↔MASt3R trajectory residual in metres, as a
   quality signal and an automatic alarm when a capture has gone wrong.
6. The plain-video path keeps working, bit-for-bit unchanged, as the default.

## Non-goals

- Android / ARCore. One capture format.
- A LiDAR-only track that skips MASt3R. Considered and rejected: the sensor reaches about
  5 m, so far walls and high ceilings would come back empty. It would be the faster path and
  the worse reconstruction.
- Importing finished meshes or splats from Polycam, Scaniverse or similar. The
  reconstruction stays ours.
- AR navigation, VPS, multi-floor navmeshes and stairs.
- Forking MASt3R to implement the missing inverse kinematic chain.
- Fixing the unscaled state of the three already-deployed scenes. Real, but separate work.
- Making the LiDAR pathway the default, or presenting it as the primary way to capture.

## Anti-goals

Things that would make this a failure even if it runs green:

- **A plausible but metrically wrong point cloud.** The depth intrinsics must be rescaled
  from the colour frame's resolution to the depth map's. Get that wrong and the cloud still
  looks like a room — it is simply the wrong size or sheared. It then silently poisons the
  navmesh, and nobody notices, because every downstream stage keeps working. This is the
  highest-risk defect in the design and most of the test plan exists to catch it.
- **A pathway that is only better on paper.** If the merged cloud is not visibly better than
  what vision alone produces, there is no reason for this to exist.
- **Breaking the plain-video path.** It is the product; this is an accessory to it.
- **A path that only works on the one phone we tested on.**
- **Any new manual step.** The pathway removes one; it must not add another.

## Constraints

**Capture.** Stray Scanner 1.4 (free, last updated 7 April 2026), iOS 18.6+, LiDAR device
required. Export layout, per its `docs/format.md`:

```
<hash>/
├── camera_matrix.csv   3x3 intrinsics of the final frame; superseded by odometry.csv
├── odometry.csv
├── imu.csv
├── depth/              000000.png, 000001.png, ...
├── confidence/         000000.png, 000001.png, ...
├── distortion/         000000.bin, ... (optional)
└── rgb.mp4             HEVC
```

`odometry.csv` has a header row, then columns verbatim:

```
timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y
```

Timestamps in seconds; `x, y, z` in metres from session start; `fx, fy, cx, cy` in pixels
**of the colour frame**; distortion centre may be empty.

**Sensor.** LiDAR range about 5 m, beyond which depth is unreliable. Depth maps are 16-bit
single-channel PNG, 256x192, values in millimetres. Confidence maps are 0/1/2; 0 is
unreliable. Colour video is 1920x1440.

**MASt3R.** `sparse_scene_optimizer` accepts an `init` dict keyed by image path.
`intrinsics` (sparse_ga.py:223) and `depthmap` (sparse_ga.py:232) are honoured.
`cam2w` hits `assert False, 'inverse kinematic chain not yet implemented'`
(sparse_ga.py:247) on both the pinned submodule commit `f5209af` and upstream `main`.
Known poses therefore cannot be given to the optimiser.

**Compute.** RTX 2050, 4 GB VRAM, WSL2 is the floor. Profiles are chosen by detected VRAM.

**Repository.** `.gitignore` excludes `scenes/*/raw/`, so a scene's sidecars never travel
between machines. The fast track therefore runs ingest and reconstruct on one box. Document
it; don't fight it.

## Approach

Seed what MASt3R will accept, then override the poses afterwards.

```
Stray Scanner folder
        │
        ▼
[1] INGEST (fast track)
    probe sees odometry.csv + depth/ → Track.FAST
    copy video AND sidecars into scenes/<name>/raw/
    slice rgb.mp4 at its native frame rate → candidates
    blur + motion filters run over a 6 fps-equivalent stride
    each surviving keyframe carries its ORIGINAL frame index as source_index
        │
        ▼
[2] RECONSTRUCT (sensor track)
    read odometry.csv → per-frame ARKit cam_to_world + per-frame intrinsics
    unproject each keyframe's depth/NNNNNN.png with depth-scaled intrinsics,
        discard confidence 0, colour each point from its own keyframe JPEG
        → LiDAR cloud, metric, in ARKit's frame
    run MASt3R-SfM, seeded via init with LiDAR depth and ARKit intrinsics
        → up-to-scale cloud, MASt3R's own frame
    Umeyama + RANSAC between the two camera trajectories (same keyframes,
        one-to-one correspondence) → similarity transform, and a residual in metres
    map the MASt3R cloud through it
        │
        ▼
    Reconstruction(cam_to_world = ARKit, points = LiDAR ∪ mapped MASt3R)
        │
        ▼
[3..] splat → navmesh → bundle, with scale.source = "sensor"
```

### Decisions already taken

**Capture app: Stray Scanner.** Chosen over Polycam's raw export, which is richer on paper
(full-resolution keyframes, per-frame blur scores, a mesh) but requires Developer Mode
switched on *before* capture and does not apply retroactively — a silent trap that costs a
whole capture session. Stray Scanner needs no account, no cloud upload and no pre-flight
toggle, its format is documented to the column, and it hands over an HEVC video rather than
pre-selected keyframes, which suits a pipeline that slices and filters video itself.

**Input shape.** `atomspace run <scene> <path-to-stray-folder>`. The CLI's `video` argument
accepts a directory; a directory holding `odometry.csv` and `rgb.mp4` selects the fast track.
One command either way.

**Track selection** stays where `route.choose_track` already is, switching on
`meta.has_ar_poses` as its docstring says. `probe_video` keeps looking for `odometry.csv`
and `depth/`; it gains the ability to look inside a passed directory.

**Sidecars get copied.** `ingest.run` currently copies only the video into
`scenes/<name>/raw/`, while `reconstruct` hands `sensor_track` `paths.raw`. The sidecars
must be copied too, or the fast track detects poses at ingest and finds nothing at
reconstruct.

**Correspondence is by index, not timestamp.** `sensor_track`'s docstring proposes matching
poses to keyframes by timestamp, but no timestamp is recorded anywhere: frames are written
`%05d.jpg` at a constant `extract_fps` of 6, and dedupe then discards most of them. Extract
at the video's native rate so ffmpeg frame N corresponds to `odometry.csv` row N and to
`depth/NNNNNN.png`, and carry each surviving keyframe's original index through
`ingest.json` as `source_index`. Correspondence becomes a lookup, not a fuzzy match.

**Filters run on a stride, not on every native-rate frame.** `filter_sharp`'s
`sharpness_window=15` and `select_keyframes`' `min_motion=0.15` are both tuned in units of
frames at 6 fps. At 60 fps the window becomes ±0.12 s, and since motion blur spans many
consecutive frames the local median goes blurry too and the sharpness filter stops rejecting
anything. So index at native rate for correspondence, but run the filters over a
6 fps-equivalent stride, carrying true `source_index` values through. Existing tuning holds,
and cost stays where it is today instead of rising tenfold.

**Correspondence gets verified, not assumed.** `odometry.csv` carries both a `timestamp` and
a `frame` column, neither of which the mapping uses. Cross-check them against the extracted
frame count as a guard: a small mismatch truncates to the common prefix with a warning, a
large one aborts. A one-frame offset between colour and odometry would attach every pose to
the wrong image and produce exactly the top anti-goal, and no synthetic test can catch it.

**Assert on the container rotation flag.** `probe_video` reads it and `extract_frames`
applies it, so a portrait-locked capture would rotate keyframes out from under the odometry
intrinsics and the depth maps. Refuse a non-zero rotation on the fast track rather than
silently mis-handle it.

**Coordinate conversion is the sharp edge.** ARKit cameras are +Y up, -Z forward;
`export.py` expects OpenCV (+Y down, +Z forward) `cam_to_world` and applies `CV_TO_GL`
itself on the way out. The reader must convert ARKit → OpenCV, not ARKit → OpenGL, or the
scene is flipped. Hand-computed unit tests, not eyeballing.

**Gravity fixes the floor's normal, not just its sign.** ARKit's world Y is gravity-up. In
`floor.py` as it stands, `up_hint` only disambiguates the normal's sign and gates which
planes count as horizontal; the normal `world_from_floor` actually aligns to +Y comes from
the SVD in `_least_squares_plane`, so a hint alone leaves the world able to sit tilted — and
the ankle-to-head occupancy slabs in `obstacles.py` are sensitive to that. Add an optional
"normal is known, solve only the offset" path through `fit_floor`, leaving the fallback
track's behaviour untouched.

**`transforms.json` carries ARKit's per-frame intrinsics**, which are measured and
per-frame, rather than MASt3R's optimised shared pair. `export.py` already writes per-frame
`fl_x/fl_y/cx/cy`.

**The residual is written to `logs/`, not to the manifest.** `manifest.json` is
`additionalProperties: false`, schema-validated and consumed by the viewer; putting a
quality metric there would mean editing the schema, the viewer's bundle types and possibly
`format_version`, for something no viewer reads.

**A tripped alarm does not claim sensor scale.** Above the residual threshold, print loudly,
write the residual to `logs/`, and leave the scene unscaled — it then behaves exactly like a
vision-only scene today and stays recoverable with the manual scale tool. Quality comes
first, so the pipeline does not carve a navmesh out of geometry it has just measured as
badly aligned and then label it metric.

**Scale needs no new code.** `metric_factor` already returns `(1.0, "sensor")` for a
fast-track scene, and the manifest schema already permits `"source": "sensor"`.

### Left to the builder

- Voxel size for the merged cloud. `VOXEL_FRACTION` sizes the voxel relative to the scene's
  arbitrary scale; on the fast track the cloud is metric, so an absolute size in metres is
  available and simpler.
- RANSAC threshold and iteration count for the trajectory fit, and the residual value that
  trips the alarm.
- Whether `NavmeshConfig.min_blocked_points` (3) needs retuning for LiDAR point density.
- Whether to discard confidence-1 depth as well as confidence-0. Quality-first suggests
  trying both and keeping whichever reconstructs better.
- Whether unprojecting keyframes alone gives thin coverage, and if so what stride of
  non-keyframe depth frames to add.

The one genuinely open technical question is **whether depth seeding survives contact**:
`init['depthmap']` writes into `core_depth`, which lives at MASt3R's internal subsampled
canonical-view resolution rather than as a raw 256x192 map. Reshaping into that layout must
be verified on the GPU machine. Since quality is the first priority and there is no clock,
this is worth real effort rather than a timebox — it is the one lever that improves MASt3R's
own output instead of merely repositioning it afterwards. If it genuinely cannot be made to
fit, drop it; steps 2 through 4 stand without it.

## Testing

Everything below runs without a GPU, alongside the existing 33 CPU tests.

- `odometry.csv` parsing: the verbatim header, units, empty distortion fields, a short file.
- ARKit → OpenCV conversion, against matrices worked out by hand.
- Depth unprojection against a synthetic plane at a known distance, including the
  colour-to-depth intrinsics rescale. This is the test that guards the top anti-goal.
- Confidence filtering.
- Umeyama with RANSAC: recover a known similarity transform from a synthetic trajectory,
  including one with injected outliers.
- Keyframe index mapping survives blur and motion filtering at native rate with a stride.
- The correspondence guard: mismatched counts truncate or abort as specified.

A small fixture goes under `test/`, matching how `test/testroom/frames/` already provides
sample data: pre-extracted JPEGs plus their sidecars, no video, so it stays out of Git LFS.

**A real capture is required before this is called done.** The CPU tests run against a
synthetic fixture so development is not blocked waiting for one, but a synthetic fixture is
consistent by construction and cannot confirm the index correspondence, the rotation flag
behaviour, or that the merged cloud is actually better than vision alone — which is the whole
premise. Integration is a full `atomspace run` over a real Stray Scanner recording of a real
room, compared side by side against the same room reconstructed from plain video.
