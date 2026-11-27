# LiDAR ingestion path — design

Date: 2026-11-27
Status: approved for planning
Target: Techfest Namma Space Round 3, IIT Bombay, 16 December 2026

## Intent

On Round 3 Day 1 the team is handed an unseen venue on the IIT Bombay campus and must
capture it, reconstruct it, tag POIs and deploy the platform inside a fixed timeframe.
Field capture efficiency is 20% of the score and the judges' Q&A probes processing
turnaround directly.

Today the pipeline cannot produce a navmesh without a human in the loop. After
`reconstruct`, a scene is `unscaled`: someone must open the viewer's admin tool, click two
points, measure the real distance between them and type it in before `navmesh` will run at
all. That step is why all three scenes on the live deployment are `"source": "unscaled"`
with `"navmesh": null` — the manual step was never done, so the Round 2 features have
nothing to run against.

An ARKit + LiDAR capture removes that step entirely. ARKit's poses are already metric and
already gravity-aligned, so scale and floor orientation arrive with the data instead of
being reconstructed or hand-measured. Within the sensor's range the geometry is also
measured rather than inferred.

This is why the work targets Round 3 and not Round 2: nothing in Round 2's scoring
(POI tagging, pathfinding, search UI, repository cleanliness) depends on how the geometry
was captured.

## Goals

1. `atomspace` ingests a Stray Scanner recording folder and produces a scene bundle whose
   `scale.source` is `sensor`, with no human scale step anywhere in the run.
2. Final camera poses in `transforms.json` are ARKit's: metric, gravity-aligned.
3. `dense_points.ply` is the union of the confidence-filtered LiDAR unprojection and the
   MASt3R cloud, the latter mapped into ARKit's frame.
4. `fit_floor` uses ARKit gravity as its up direction instead of the mean camera up.
5. The run reports the ARKit↔MASt3R trajectory residual in metres, as a capture-quality
   signal and an automatic alarm when a capture has gone wrong.
6. The existing plain-video path keeps working, unchanged, as the fallback.

## Non-goals

- Android / ARCore. One capture device, one format.
- A LiDAR-only track that skips MASt3R. Considered and rejected: the sensor reaches about
  5 m, and the Lecture Hall Complex has corridors and halls far deeper than that, so far
  walls and high ceilings would come back empty.
- Importing finished meshes or splats from Polycam, Scaniverse or similar. The
  reconstruction stays ours; Round 3's Q&A cross-examines our pipeline.
- AR navigation, VPS, multi-floor navmeshes and stairs.
- Forking MASt3R to implement the missing inverse kinematic chain.
- Fixing the unscaled state of the three already-deployed scenes. Real and urgent, but a
  separate piece of work on a three-day clock.

## Anti-goals

Things that would make this a failure even if it runs green:

- **A plausible but metrically wrong point cloud.** The depth intrinsics must be rescaled
  from the colour frame's resolution to the depth map's. Get that wrong and the cloud still
  looks like a room — it is simply the wrong size or sheared. It then silently poisons the
  navmesh, and nobody notices until the live demo. This is the single highest-risk defect in
  the whole design and the tests exist mainly to catch it.
- **Any new manual step at the venue.** The entire point is removing one.
- **Breaking the fallback track.** If the LiDAR path fails at the LHC, plain video must
  still work that same afternoon.
- **A path that only works on the one phone we tested on.**

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
unreliable and is discarded. Colour video is 1920x1440.

**MASt3R.** `sparse_scene_optimizer` accepts an `init` dict keyed by image path.
`intrinsics` (sparse_ga.py:223) and `depthmap` (sparse_ga.py:232) are honoured.
`cam2w` hits `assert False, 'inverse kinematic chain not yet implemented'`
(sparse_ga.py:247) on both the pinned submodule commit `f5209af` and upstream `main`.
Known poses therefore cannot be given to the optimiser.

**Compute.** RTX 2050, 4 GB VRAM, WSL2 is the floor. Profiles are chosen by detected VRAM.

**Schedule.** Round 3 Day 1 is time-boxed. Round 2 closes 30 November 2026 and is not
touched by this work.

## Approach

Seed what MASt3R will accept, then override the poses afterwards.

```
Stray Scanner folder
        │
        ▼
[1] INGEST (fast track)
    probe sees odometry.csv + depth/ → Track.FAST
    copy video AND sidecars into scenes/<name>/raw/
    slice rgb.mp4 at native frame rate → candidates
    existing blur + motion filters pick keyframes
    record each keyframe's ORIGINAL frame index in ingest.json
        │
        ▼
[2] RECONSTRUCT (sensor track)
    read odometry.csv → per-frame ARKit cam_to_world + per-frame intrinsics
    unproject depth/NNNNNN.png with depth-scaled intrinsics, drop confidence 0
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
[3..] splat → navmesh → bundle, unchanged, with scale.source = "sensor"
```

### Decisions already taken

**Capture app: Stray Scanner.** Chosen over Polycam's raw export, which is richer on paper
(full-resolution keyframes, per-frame blur scores, a mesh) but requires Developer Mode
switched on *before* capture and does not apply retroactively — a run-ending trap at a venue
with one shot at it. Stray Scanner needs no account, no cloud upload and no pre-flight
toggle, its format is documented to the column, and it hands over an HEVC video rather than
pre-selected keyframes, which suits a pipeline that already slices and filters video itself.

**Input shape.** `atomspace run <scene> <path-to-stray-folder>`. The CLI's `video` argument
accepts a directory; a directory holding `odometry.csv` and `rgb.mp4` selects the fast track.
One command either way.

**Track selection** stays where `route.choose_track` already is, switching on
`meta.has_ar_poses` as its docstring says. `probe_video` keeps looking for `odometry.csv`
and `depth/`; it gains the ability to look inside a passed directory.

**Sidecars get copied.** `ingest.run` currently copies only the video into
`scenes/<name>/raw/`, while `reconstruct` hands `sensor_track` `paths.raw`. The sidecars
must be copied too or the fast track detects poses at ingest and finds nothing at
reconstruct.

**Correspondence is by index, not timestamp.** `sensor_track`'s current docstring proposes
matching poses to keyframes by timestamp, but no timestamp is recorded anywhere: frames are
written `%05d.jpg` at a constant `extract_fps` of 6, and dedupe then discards most of them.
On the fast track, extract at the video's native rate so that ffmpeg frame N corresponds to
`odometry.csv` row N and to `depth/NNNNNN.png`, and carry each surviving keyframe's original
index through `ingest.json` as `source_index`. Correspondence becomes a lookup rather than a
fuzzy match.

**Blur and motion filtering stay on.** Stray Scanner gives raw video, the existing filters
are good, and MASt3R still consumes the keyframes.

**Coordinate conversion is the sharp edge.** ARKit cameras are +Y up, -Z forward;
`export.py` expects OpenCV (`+Y` down, `+Z` forward) `cam_to_world` and applies `CV_TO_GL`
itself on the way out. The reader must therefore convert ARKit → OpenCV, not ARKit →
OpenGL, or the scene is flipped. Hand-computed unit tests, not eyeballing.

**Gravity.** ARKit's world Y is gravity-up, so the floor plane's *normal* is known exactly.
Keep the RANSAC fit for the floor's *height* — to put the floor at y = 0 — but pass ARKit up
as `up_hint` rather than the mean camera up. `floor.py:12` notes the current hint can be off
by about 35 degrees.

**Scale needs no new code.** `metric_factor` already returns `(1.0, "sensor")` for a
fast-track scene, and the manifest schema already permits `"source": "sensor"`.

### Left to the builder

- **Whether depth seeding survives contact.** `init['depthmap']` writes into `core_depth`,
  which lives at MASt3R's internal subsampled canonical-view resolution, not as a raw
  256x192 map. Reshaping into that layout must be verified on the GPU machine. If it does not
  fit cleanly, drop the seeding and keep everything else — the Umeyama step does not depend
  on it. Do not spend more than a short timebox on this.
- Voxel size for the merged cloud. `VOXEL_FRACTION` currently sizes the voxel relative to
  the scene's own arbitrary scale; on the fast track the cloud is metric, so an absolute
  size in metres is available and simpler.
- RANSAC threshold and iteration count for the trajectory fit, and what residual should be
  treated as "this capture is bad, warn the operator".
- Whether `NavmeshConfig.min_blocked_points` (3) needs retuning for LiDAR point density,
  which differs from MASt3R's.
- Whether to keep or discard confidence-1 depth as well as confidence-0.

## Testing

Everything below runs without a GPU, alongside the existing 33 CPU tests.

- `odometry.csv` parsing: the verbatim header, units, empty distortion fields, a short file.
- ARKit → OpenCV conversion, against matrices worked out by hand.
- Depth unprojection against a synthetic plane at a known distance, including the
  colour-to-depth intrinsics rescale. This is the test that guards the top anti-goal.
- Confidence filtering.
- Umeyama with RANSAC: recover a known similarity transform from a synthetic trajectory,
  including one with injected outliers.
- Keyframe index mapping survives blur and motion filtering.

A small fixture recording — a handful of frames with their sidecars — goes under `test/`,
matching how `test/testroom/frames/` already provides sample data. Licensing of a
self-captured fixture is ours, so it can be committed.

Integration on the GPU machine is a full `atomspace run` of a real Stray Scanner capture of
a room, confirming a `sensor`-scaled bundle with a navmesh and a reported residual.
