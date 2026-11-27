# Architecture

## Goal

Floor plans are hard to read, and they don't look anything like the place you're standing in. So what if a building came with a link instead? Open it and you're inside a walkable, photorealistic 3D copy. Look around in first person, search for a room or an item, and fly straight to it.

That's where AtomSpace is today. The bigger goal is Google Maps for indoors: hold up your phone and follow walking directions through its camera to wherever you're headed.

## Scope

| Built today | Long-term |
|---|---|
| Video ingestion with blur and duplicate filtering | ARCore (Android) sensor capture |
| Sensor capture: ARKit poses + LiDAR depth, via Stray Scanner | AI auto-tagging of POIs |
| MASt3R-SfM reconstruction (poses + dense points) | Multi-level navmesh, including stairs |
| 3D Gaussian splat for visuals | A* pathfinding + floor route ribbon |
| Navmesh for collision, with gravity on the sensor track | VPS indexer (`vps_index.bin`) |
| Free-roam viewer with dead-stop collision | In-person AR HUD |
| Manual POI tagging + fuzzy search + fly-to | |

Ordinary phone video stays the headline capability. The sensor track is a second, optional
pathway for anyone holding an iPhone with LiDAR: where vision infers geometry and scale, the
sensor measures them. It exists because it reconstructs better, not because it is faster.

## Long-term architecture

This is the complete product we're aiming for. Everything marked `planned` hasn't been built yet, and its design may change.

```
CAPTURE    any phone video  ·  sensor capture: ARKit + LiDAR  ·  ARCore (planned)
    │
    ▼
BUILD      ingest → reconstruct → splat
           scale (vision only; sensor captures arrive metric)
           navmesh, multi-level with stairs (planned)
           POIs: manual, AI auto-tagging (planned)
           VPS indexer (planned)
    │
    ▼
BUNDLE     manifest.json · scene.splat · navmesh.json · pois.json
           vps_index.bin (planned)
    │
    ├──────────────────────────────┐
    ▼                              ▼
WEB VIEWER                     IN-PERSON AR (planned)
explore from anywhere          navigate on site
free roam · search · fly-to    VPS: find the phone's position
A* route ribbon (planned)      A* route to the destination
                               AR HUD: directions through the camera
```

The same bundle serves both sides, so a scene built for the web viewer can later guide people through the real building. Navmesh polygons already store their neighbours for this reason: A* can be added without rebuilding existing scenes.

## Current pipeline

```
[0] CAPTURE        smartphone video, 1080p, slow layered loops, finish where you started
        │ .mp4
        ▼
[1] INGEST         probe metadata → choose track → slice frames → drop blurry → drop near-duplicates
        │          fast track also copies the recording's sidecars into raw/, slices by frame
        │          index (filters still run on a 6 fps-equivalent stride) and records each
        │          keyframe's source_index, which names its pose and its depth map
        │ frames/*.jpg (≤ profile.max_keyframes) + ingest.json
        ▼
[2] RECONSTRUCT    fallback (default): MASt3R-SfM, sparse pair graph (window + retrieval)
        │          fast track: ARKit poses + LiDAR depth from the recording's sidecars.
        │          MASt3R still runs, seeded with ARKit's measured intrinsics, and its cloud
        │          is mapped onto ARKit's metric frame by a RANSAC similarity fit between the
        │          two camera trajectories. Output poses are ARKit's; points are LiDAR within
        │          about 5 m plus MASt3R's beyond it. Residual and counts go to
        │          logs/sensor_track.json, gravity to recon/world_up.json.
        │ recon/transforms.json + recon/dense_points.ply   (recon space, up-to-scale on fallback)
        ├───────────────────────────────┐
        ▼                               ▼
[3] SPLAT                         [4] SCALE (user)
    gsplat, seeded from               two points + real distance → recon/scale.json
    dense points                        │
        │ bundle/scene.splat            ▼
        │                         [5] NAVMESH
        │                             floor RANSAC (camera up picks the floor), or
        │                             the recorded gravity on the sensor track → align
        │                             +Y up → occupancy grid (ankle → head height) →
        │                             eroded by 0.25 m agent radius → outlines → triangles
        │                               │ bundle/navmesh.json + recon/world.json
        └───────────────┬───────────────┘
                        ▼
[6] BUNDLE         manifest.json + empty pois.json if new; everything validated against schemas/
                        │
                        ▼
[7] VIEWER         Three.js + gaussian-splats-3d
                   WASD free-roam · navmesh dead stop · POI pins · search · fly-to
                   admin (?admin): scale tool, POI editor
```

## Coordinate spaces

- **Recon space:** whatever MASt3R produced. Arbitrary orientation, arbitrary scale on the fallback track. The splat stays in this space.
- **World space:** meters, +Y up, floor at y = 0. Navmesh and POIs live here.
- **`manifest.world_from_recon`:** a 4x4 matrix (column-major) holding the scale factor and floor alignment. The viewer applies it to the splat so everything lines up. Changing the scale never requires retraining the splat.

## Metric scale

MASt3R output is only near-metric, so scale comes from the user: click two points with a known real distance (a door is about 2 m) in the viewer's scale tool, or pass them to `atomspace scale`. Until then the scene is `unscaled`, has no navmesh, and can still be viewed (`bundle` stands it upright around the first camera, so the scale tool has a usable view). On the sensor track the poses are already metric, so no reference is needed and the scene
is `source: "sensor"` from the start — unless the stage's own check fails it. The vision and
sensor camera paths are fitted against each other, and a residual above `RESIDUAL_LIMIT_M`
(0.15 m) means the two reconstructions disagree too much to call the result metric: the scene
is left `unscaled` rather than have a navmesh carved from geometry already measured as badly
aligned, and the manual scale tool can still rescue it.

## Collision

There's no physics engine. Each frame the controls propose a move, and `NavmeshGuard.tryMove` checks whether the new position is still on the navmesh. If it isn't, the camera doesn't move at all: a dead stop, with no sliding along walls. On the navmesh, the camera is snapped to floor height + 1.6 m.

## GPU profiles

Processing targets NVIDIA RTX 20-series or newer cards with at least 4 GB of VRAM; 8 GB is recommended. `gpu.resolve_profile()` picks limits by detected VRAM:

| Profile | Keyframes | Pair window / retrieval | Splat resolution | Max Gaussians |
|---|---|---|---|---|
| `vram_4gb` | 101 | 5 / 3 | 1/2 | 1.5M |
| `vram_8gb` | 150 | 5 / 10 | 1/2 | 1.5M |
| `vram_12gb` | 250 | 8 / 15 | 1/2 | 3M |
| `vram_24gb` | 400 | 10 / 20 | full | 6M |

GPU stages write duration and peak VRAM to `scenes/<name>/logs/` so the limits can be tuned on real scenes. So far the full pipeline has run on one laptop RTX 2050 under WSL2, with reduced settings.

## Decisions

| Decision | Reason |
|---|---|
| MASt3R-SfM instead of COLMAP or plain DUSt3R | Handles blank interior walls; DUSt3R alone doesn't scale past ~30 frames |
| User-supplied metric scale | Simple and reliable; no dependency on phone sensors |
| Pure-vision track is the default | Works with any phone; the sensor track will be a speedup when data exists |
| Dead-stop collision instead of wall sliding | Predictable, trivial to implement, no physics engine |
| Profiles by VRAM, not by GPU model | Limits follow available memory, so bigger cards get better quality |
| WSL2, no Docker | CUDA extensions build cleanly with gcc; no Docker GPU setup on Windows |
| Navmesh polygons store neighbours | A* can be added later without rebuilding existing scenes |
| Stray Scanner for sensor capture | Documented to the column, free, and needs no account or pre-flight toggle. Polycam's raw export is richer but requires Developer Mode set *before* capture, and does not apply retroactively |
| ARKit poses adopted after the fact, not seeded into MASt3R | `init['cam2w']` reaches `assert False, 'inverse kinematic chain not yet implemented'` (sparse_ga.py:247); `init['intrinsics']` is accepted and is seeded |
| LiDAR depth is not seeded into MASt3R either | `init['depthmap']` replaces `core_depth[idx]`, a per-anchor vector built inside `sparse_global_alignment`, whose length and pixel positions a caller cannot know in advance |
| A bad trajectory residual withholds metric scale | No navmesh beats a confidently wrong one |
| Fast track slices by frame index, filters on a stride | Index is what matches odometry rows and depth maps; the blur and motion filters are tuned in frames at 6 fps and stop rejecting anything at 60 |
| A rotation flag is refused on the fast track | ffmpeg applies it to the frames while the odometry intrinsics and depth maps stay unrotated |
