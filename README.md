# AtomSpace

Got a phone and a room? Walk around filming for a minute or two, run one command, and step inside your own space in the browser. AtomSpace is still in its early stages, so expect a few rough edges!

**[Try it now :D!](https://atomspace-deploy.vercel.app/?scene=friendroom-r30)**

| Level view | Tilted-down view |
|---|---|
| ![A reconstructed room in the web viewer, seen at eye level](docs/images/room-level.png) | ![The same room seen from a downward angle](docs/images/room-down.png) |

## How it works

![Pipeline: capture, ingest, MASt3R, then the 3DGS engine and scale plus navmesh, into a static scene bundle, the client runtime and the web viewer](docs/images/architecture.png)

1. **Ingest:** ffmpeg extracts frames, OpenCV drops blurred and near-duplicate ones.
2. **Reconstruct:** MASt3R-SfM estimates camera poses and dense 3D points from a sparse graph of image pairs, including loop-closure pairs. COLMAP isn't used.
3. **Splat:** gsplat trains 3D Gaussians for 30,000 steps, seeded from the dense points.
4. **Compress:** the model becomes a 32-byte-per-Gaussian `.splat`, without near-transparent and needle-shaped Gaussians.
5. **Scale and navmesh:** the user gives one real distance, then a Python builder turns the dense points into walkable floor polygons with a 0.25 m buffer around obstacles.
6. **Bundle and view:** static files are validated against `schemas/`, and the Three.js viewer renders them in the browser with [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D).

Loop closure is what removes drift. Below, each real frame (top) sits above the generated point cloud projected into the same view (bottom):

![Test room frames above the generated point cloud projected into the same views](docs/images/reprojection.jpg)

The navmesh comes from an occupancy grid of those dense points. These are from the test room at a trial scale:

| Occupancy grid | Navmesh |
|---|---|
| ![Occupancy grid with camera positions in yellow, blocked cells in red and observed cells in grey](docs/images/occupancy-grid.png) | ![Walkable area in green with camera positions in yellow](docs/images/navmesh.png) |

More detail: [docs/architecture.md](docs/architecture.md).

## Status

Capture, reconstruction and free roaming work today. Here's where everything stands:

| Feature | Status |
|---|---|
| Video to walkable splat in the browser | Working, used for the live demo |
| Laptop and phone controls | Working on the live demo; phones run it less smoothly |
| Scale tool, navmesh collision | Built and tested on the test room or synthetic data; not applied to the demo scene, so you can walk through its walls |
| POI search, fly-to and POI editor | Built and tested on synthetic data; the demo scene has no POIs |
| ARKit/LiDAR fast track | Designed, not implemented (`sensor_track.py` raises `NotImplementedError`) |
| Pathfinding, phone-camera AR navigation | Planned |

## Results

A plain `atomspace run` took a 44-second iPhone video to a viewable scene in 80 minutes, on a laptop RTX 2050 with 4 GB of VRAM under WSL2. It picked the `vram_4gb` profile by itself and used real ASMK retrieval, with no wrapper or skipped checks.

| Stage | Result | Time | Peak GPU memory |
|---|---|---|---|
| Ingest | 263 frames down to 101 keyframes | Under 1 min | None |
| Reconstruct | 715 image pairs from ASMK, 101 of 101 cameras, 2M dense points | 42 min | 5.0 GiB (PyTorch's figure; `nvidia-smi` read at most 3,955 MiB) |
| Splat | 30,000 steps, 1.5M Gaussians, about 1.09M kept, 34.9 MB file | 33 min | 0.54 GiB |
| Bundle | Static scene files | Seconds | None |

The GPU averaged 51°C and peaked at 61°C. A few things to know about this run:

- The video is the one behind Room-1 in the live demo, but the live scene was made separately on Google Colab. Our 4 GB run looks worse than it, and we haven't pinned down why. The 4 GB profile matches only 3 retrieval pairs per frame, against 10 with 8 GB, which is a likely factor.
- It ran on a machine that was already set up, so it doesn't show a first-try install.
- Navmesh was skipped because it needs a metric scale from the user.

An earlier run on a 90-second video, with the same settings but a stand-in for ASMK, made Room-2 in the live demo in 75 minutes.

We haven't measured image quality (PSNR, SSIM) yet.

8 GB of VRAM is recommended and 4 GB is the minimum.

## Setup

### Requirements

- **Viewer and CPU stages:** Node 20+, Python 3.10+, any modern browser.
- **GPU stages (`reconstruct`, `splat`):** an NVIDIA GPU from the RTX 20 series or newer on WSL2 or Linux, with 8 GB of VRAM recommended and 4 GB the minimum.

### Clone

```bash
sudo apt install -y git-lfs    # do this first, or large files arrive as small text pointers
git clone --recursive https://github.com/SoloBeing/AtomSpace.git
cd AtomSpace
```

### GPU machine

Follow [env/README.md](env/README.md). In short:

```bash
bash env/setup_wsl.sh          # conda environment, ASMK, curope, gsplat
bash env/download_weights.sh   # about 3 GB from Hugging Face; safe to interrupt and re-run
conda activate atomspace
python env/smoke_test.py
```

`setup_wsl.sh` needs conda, so it doesn't work on Google Colab. There, gsplat installed with:

```bash
TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=2 pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
```

### Viewer

```bash
cd viewer
npm install
npm run dev
```

Open `http://localhost:5173/?scene=<name>`. Add `&admin` for the scale and POI tools, which only work on the dev server.

## Reconstruct the sample scene

[`test/testroom/frames/`](test/README.md) holds the 101 keyframes from our test room, so you can skip capture and ingest:

```bash
cp -r test/testroom/frames scenes/testroom/
atomspace reconstruct testroom
atomspace splat testroom
atomspace bundle testroom
cd viewer && npm install && npm run dev
```

Then open `http://localhost:5173/?scene=testroom`. `scenes/testroom/ingest.json` is already in the repository and lists the keyframes.

Expect a result like Room-2 in the live demo, including the eye-level black band, because this capture was tilted down throughout. Room-1 shows what a better capture can do.

## Reconstruct your own space

Walk the room slowly in layered loops at different heights, finish where you started, and add close-ups of important objects. Use 1080p, with HDR, beauty modes and heavy stabilisation turned off.

```bash
atomspace run myroom path/to/video.mp4   # ingest, reconstruct, splat, bundle

# optional: metric scale, then collision
atomspace scale myroom --a 0.1 0.0 2.3 --b 0.1 1.1 2.3 --distance 2.0
atomspace navmesh myroom && atomspace bundle myroom
```

`--a` and `--b` are two points in reconstruction space, and `--distance` is their real separation in metres. The viewer's scale tool (`&admin`) picks them by clicking. Each stage reruns on its own: `atomspace {ingest,reconstruct,splat,scale,navmesh,bundle} <scene>`.

## Controls

| Action | Laptop | Phone |
|---|---|---|
| Walk | W A S D | Left thumb |
| Sprint | Hold Shift | Not available |
| Look | Click the scene, then move the mouse | Right thumb |
| Free the cursor | Esc | Not needed |

## Deploy

The viewer is static, so any static host works, and no special headers are needed.

```bash
cd viewer && npm run build
```

Serve `viewer/dist/` with each scene at `/scenes/<name>/bundle/`, or set `VITE_SCENES_BASE_URL` at build time to host scenes elsewhere.

To offer a choice of worlds, add `worlds.json` beside the scene folders. Visitors to the bare address then get a carousel, while `?scene=<name>` still opens a scene directly:

```json
{
  "format_version": 1,
  "worlds": [{ "scene": "testroom", "name": "Test Room" }]
}
```

A world's card shows `bundle/thumbnail.jpg` from its scene, or a plain background if there isn't one. Without `worlds.json`, visitors need `?scene=<name>` in the URL.

## Tests

```bash
cd pipeline && pip install -e ".[dev]" && pytest -q     # 33 tests, no GPU needed
cd viewer && npx tsc --noEmit && npx vitest run         # 9 tests
```

## License

AtomSpace's own code is under the [MIT License](LICENSE).

**MASt3R, DUSt3R and CroCo, including their weights, are CC BY-NC-SA 4.0, so using this pipeline to reconstruct a scene is non-commercial only**, whatever the license on our code. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which also covers the images and sample data.
