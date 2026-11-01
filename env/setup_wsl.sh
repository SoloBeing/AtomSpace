#!/usr/bin/env bash
# One-time setup of the GPU environment inside WSL2 Ubuntu (or native Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="atomspace"
BUILD_DIR="$ROOT/third_party/_build"
TORCH_PINS="$BUILD_DIR/torch-constraints.txt"
GSPLAT_REF="v1.5.3"
ASMK_REF="2a96d9c03a841dffdfddabc699a20512dcd09363"
APT_HINT="sudo apt update && sudo apt install -y build-essential git git-lfs wget libgl1 libglib2.0-0"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "$ROOT" == /mnt/[a-z]/* ]]; then
  echo "WARNING: repo is at $ROOT, on the Windows filesystem. Builds and frame I/O will be very slow." >&2
  echo "         Clone inside the Linux filesystem instead, e.g. ~/AtomSpace." >&2
fi

echo "==> Checking GPU"
command -v nvidia-smi >/dev/null \
  || die "nvidia-smi not found. Install the latest NVIDIA driver on Windows; WSL2 picks it up automatically."
SMI_QUERY=(--query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits)
# stdout only: WSL's nvidia-smi can print warnings on stderr that would otherwise be parsed as GPU data.
if ! GPU_INFO="$(nvidia-smi "${SMI_QUERY[@]}" 2>/dev/null)"; then
  die "nvidia-smi failed: $(nvidia-smi "${SMI_QUERY[@]}" 2>&1 | head -n3 || true)
       Update the Windows NVIDIA driver, then run 'wsl --shutdown' from PowerShell and reopen Ubuntu."
fi
GPU_INFO="${GPU_INFO%%$'\n'*}"
IFS=',' read -r GPU_NAME CAP MEM_MIB <<<"$GPU_INFO"
CAP="${CAP// /}"
MEM_MIB="${MEM_MIB// /}"
[[ "$CAP" =~ ^[0-9]+\.[0-9]+$ ]] \
  || die "Could not read the GPU's compute capability (got '$CAP'). The driver is probably too old; update it on Windows."
[[ "$MEM_MIB" =~ ^[0-9]+$ ]] || die "Could not read the GPU's memory size (got '$MEM_MIB')."
CAP_INT=$(( ${CAP%%.*} * 10 + ${CAP#*.} ))
echo "    $GPU_NAME | compute capability $CAP | ${MEM_MIB} MiB"

(( CAP_INT >= 75 )) || die "Compute capability $CAP is too old. Minimum is 7.5 (RTX 20 series / Turing)."
(( MEM_MIB >= 3500 )) || die "Only ${MEM_MIB} MiB VRAM. Minimum is 4 GB; 8 GB is recommended."
(( MEM_MIB >= 7600 )) || echo "    WARNING: ${MEM_MIB} MiB VRAM is below the recommended 8 GB." >&2
export TORCH_CUDA_ARCH_LIST="$CAP"
# Parallel nvcc jobs can exhaust WSL2's RAM share and get OOM-killed.
export MAX_JOBS="${MAX_JOBS:-4}"

echo "==> Checking system packages"
for tool in gcc g++ git wget; do
  command -v "$tool" >/dev/null || die "Missing $tool. Run: $APT_HINT"
done
# No grep -q: it exits early and pipefail would turn the match into a failure.
ldconfig -p | grep 'libGL\.so\.1' >/dev/null || die "Missing libGL (needed by OpenCV). Run: $APT_HINT"
ldconfig -p | grep 'libgthread-2\.0\.so\.0' >/dev/null || die "Missing GLib (needed by OpenCV). Run: $APT_HINT"
git lfs version >/dev/null 2>&1 \
  || die "Missing git-lfs (scene bundles are stored in LFS). Run: $APT_HINT
       then, in the repo: git lfs install && git lfs pull"

echo "==> Checking conda"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  for base in miniconda3 miniforge3 mambaforge anaconda3; do
    if [[ -x "$HOME/$base/bin/conda" ]]; then
      CONDA_BIN="$HOME/$base/bin/conda"
      break
    fi
  done
fi
[[ -n "$CONDA_BIN" ]] \
  || die "conda not found. Install Miniconda (https://docs.anaconda.com/miniconda/), open a new terminal, and re-run."
CONDA_BASE="$("$CONDA_BIN" info --base)"

# Conda's activation hooks reference unset variables, so nounset is off around them.
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

echo "==> Creating conda env '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
  conda env update -n "$ENV_NAME" -f "$ROOT/env/environment.yml"
else
  conda env create -n "$ENV_NAME" -f "$ROOT/env/environment.yml"
fi
set +u
conda activate "$ENV_NAME"
set -u

export CUDA_HOME="$CONDA_PREFIX"
# Conda's CUDA packages keep headers here rather than in $CUDA_HOME/include.
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
# Check the nvcc the extension builds will use, not whichever one happens to be first on PATH.
NVCC="$CUDA_HOME/bin/nvcc"
[[ -x "$NVCC" ]] || die "nvcc not found at $NVCC. environment.yml should install cuda-toolkit 12.8 into the env."
"$NVCC" --version | grep 'release 12\.8' >/dev/null \
  || die "Conda env has the wrong CUDA version: $("$NVCC" --version | tail -n1). Expected 12.8."

check_torch() {
  python - <<'EOF' || die "PyTorch is no longer the CUDA 12.8 build. Fix: pip install --force-reinstall -r env/requirements-gpu.txt"
import sys
import torch
cuda = torch.version.cuda or "none"
print(f"    torch {torch.__version__} | CUDA {cuda} | GPU visible: {torch.cuda.is_available()}")
sys.exit(0 if cuda.startswith("12.8") and torch.cuda.is_available() else 1)
EOF
}

echo "==> Installing PyTorch (cu128)"
pip install -r "$ROOT/env/requirements-gpu.txt"
check_torch
mkdir -p "$BUILD_DIR"
# Every later install is constrained to this exact torch, so nothing can swap it for a PyPI build.
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" \
  >"$TORCH_PINS"

echo "==> Fetching MASt3R submodule"
git -C "$ROOT" submodule update --init --recursive
pip install -c "$TORCH_PINS" -r "$ROOT/third_party/mast3r/requirements.txt"
pip install -c "$TORCH_PINS" -r "$ROOT/third_party/mast3r/dust3r/requirements.txt"
check_torch

echo "==> Building ASMK @ ${ASMK_REF:0:7} (retrieval for MASt3R-SfM pair selection)"
if [[ ! -d "$BUILD_DIR/asmk" ]]; then
  git clone https://github.com/jenicek/asmk "$BUILD_DIR/asmk"
fi
git -C "$BUILD_DIR/asmk" checkout -q "$ASMK_REF" 2>/dev/null || {
  git -C "$BUILD_DIR/asmk" fetch -q origin
  git -C "$BUILD_DIR/asmk" checkout -q "$ASMK_REF"
}
(cd "$BUILD_DIR/asmk/cython" && cythonize ./*.pyx)
# No build isolation: ASMK's installer imports faiss, which is only in this env.
pip install --no-build-isolation -c "$TORCH_PINS" "$BUILD_DIR/asmk"

echo "==> Building curope (optional speedup, MAX_JOBS=$MAX_JOBS)"
if ! (cd "$ROOT/third_party/mast3r/dust3r/croco/models/curope" && python setup.py build_ext --inplace); then
  echo "    curope build failed; MASt3R will fall back to the slower PyTorch RoPE." >&2
fi

echo "==> Building gsplat $GSPLAT_REF for sm_${CAP_INT} (MAX_JOBS=$MAX_JOBS, takes a while)"
pip install --no-build-isolation -c "$TORCH_PINS" \
  "git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_REF}"

echo "==> Installing AtomSpace pipeline"
pip install -c "$TORCH_PINS" -e "$ROOT/pipeline[dev]"
check_torch

echo
echo "Done. Next:"
echo "  bash env/download_weights.sh"
echo "  conda activate $ENV_NAME && python env/smoke_test.py"
