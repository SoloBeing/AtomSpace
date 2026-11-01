# Third-party notices

The [MIT License](LICENSE) covers AtomSpace's own code only. Everything below keeps its own license.

## Non-commercial components

The `reconstruct` stage depends on these, so reconstructing scenes with this pipeline is **non-commercial only**.

| Component | Copyright | License | Where it's used |
|---|---|---|---|
| [MASt3R](https://github.com/naver/mast3r) code | Naver Corporation | CC BY-NC-SA 4.0 | `third_party/mast3r` submodule |
| MASt3R model and retrieval weights | Naver Corporation | CC BY-NC-SA 4.0 | Downloaded from Hugging Face to `checkpoints/`, not in this repository |
| [DUSt3R](https://github.com/naver/dust3r) | Naver Corporation | CC BY-NC-SA 4.0 | Inside the MASt3R submodule |
| [CroCo](https://github.com/naver/croco) | Naver Corporation | CC BY-NC-SA 4.0 | Inside the DUSt3R submodule |

Our changes to these components: none. License summary: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

## Viewer

| Component | License |
|---|---|
| [three.js](https://github.com/mrdoob/three.js) | MIT |
| [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) | MIT |
| [Fuse.js](https://github.com/krisk/Fuse) | Apache-2.0 |

## Pipeline

| Component | License |
|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 |
| [PyTorch and torchvision](https://github.com/pytorch/pytorch) | BSD-style |
| [ASMK](https://github.com/jenicek/asmk) | MIT |
| [OpenCV](https://github.com/opencv/opencv-python) | Apache-2.0 for OpenCV; the wheels bundle other libraries under their own terms |
| [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU |
| [mapbox-earcut](https://github.com/skogler/mapbox_earcut_python) | ISC |
| [jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT |
| [FFmpeg](https://ffmpeg.org/) | LGPL or GPL depending on the build; installed separately and run as a command-line tool, not distributed here |

## Images and sample data

The files in `docs/images/` and `test/` show real rooms and are **not** covered by the MIT License. They are included only so this project can be evaluated and its results reproduced. The architecture diagram in `docs/images/` is the exception and may be reused with the code.
