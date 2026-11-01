# Architecture

## Goal

Floor plans are hard to read, and they don't look anything like the place you're standing in. So what if a building came with a link instead? Open it and you're inside a walkable, photorealistic 3D copy. Look around in first person, search for a room or an item, and fly straight to it.

That's where AtomSpace is today. The bigger goal is Google Maps for indoors: hold up your phone and follow walking directions through its camera to wherever you're headed.

## Scope

| Built today | Long-term |
|---|---|
| Video ingestion with blur and duplicate filtering | ARKit/ARCore and LiDAR fast track for capture |
| MASt3R-SfM reconstruction (poses + dense points) | AI auto-tagging of POIs |
| 3D Gaussian splat for visuals | Multi-level navmesh, including stairs |
| Navmesh for collision | A* pathfinding + floor route ribbon |
| Free-roam viewer with dead-stop collision | VPS indexer (`vps_index.bin`) |
| Manual POI tagging + fuzzy search + fly-to | In-person AR HUD |

## Long-term architecture

This is the complete product we're aiming for. Everything marked `planned` hasn't been built yet, and its design may change.

```
CAPTURE    any phone video  ·  sensor capture: ARKit/ARCore + LiDAR (planned)
    │
    ▼
BUILD      ingest → reconstruct → splat
           scale → navmesh, multi-level with stairs (planned)
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
        │ frames/*.jpg (≤ profile.max_keyframes) + ingest.json
        ▼
[2] RECONSTRUCT    fallback (default): MASt3R-SfM, sparse pair graph (window + retrieval)
        │          fast track (planned): ARKit/ARCore poses + LiDAR depth from capture sidecars
        │ recon/transforms.json + recon/dense_points.ply   (recon space, up-to-scale on fallback)
        ├───────────────────────────────┐
        ▼                               ▼
[3] SPLAT                         [4] SCALE (user)
    gsplat, seeded from               two points + real distance → recon/scale.json
    dense points                        │
        │ bundle/scene.splat            ▼
        │                         [5] NAVMESH
        │                             floor RANSAC (camera up picks the floor) → align
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

MASt3R output is only near-metric, so scale comes from the user: click two points with a known real distance (a door is about 2 m) in the viewer's scale tool, or pass them to `atomspace scale`. Until then the scene is `unscaled`, has no navmesh, and can still be viewed (`bundle` stands it upright around the first camera, so the scale tool has a usable view). On the fast track, sensor poses will already be metric (`source: "sensor"`).

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
