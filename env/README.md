# GPU environment

The GPU stages (`reconstruct`, `splat`) need an NVIDIA card from the RTX 20 series (Turing) or newer with 8 GB of VRAM recommended and 4 GB the minimum. They run in WSL2 on Windows, or on native Linux. The viewer and CPU stages don't need any of this.

## Setup (once per machine)

1. **Windows:** install the latest NVIDIA Game Ready or Studio driver. Don't install a driver inside WSL.
2. **Windows PowerShell:** `wsl --install -d Ubuntu-24.04`, then restart.
3. **Ubuntu:**
   ```bash
   nvidia-smi                                   # must list your GPU
   sudo apt update && sudo apt install -y build-essential git git-lfs wget libgl1 libglib2.0-0
   # install Miniconda: https://docs.anaconda.com/miniconda/, then open a new terminal
   ```
4. **Clone inside the Linux filesystem** (`~/AtomSpace`, not `/mnt/c/...`). Builds and frame I/O are much slower across the Windows mount.
   ```bash
   git clone --recursive https://github.com/SoloBeing/AtomSpace.git ~/AtomSpace && cd ~/AtomSpace
   bash env/setup_wsl.sh
   bash env/download_weights.sh
   conda activate atomspace
   python env/smoke_test.py
   ```

`setup_wsl.sh` reads the card's compute capability and builds gsplat for it (`7.5` Turing, `8.6` Ampere, `8.9` Ada, `12.0` Blackwell). gsplat and ASMK are pinned to fixed versions, and the script checks after each install step that PyTorch is still the CUDA 12.8 build.

`download_weights.sh` can be interrupted and re-run safely. It resumes partial downloads, checks every file's size against a pinned value, and only moves a file into `checkpoints/` once it passes. On the first run it records each file's sha256 in `env/weights.sha256`. **After `smoke_test.py` passes, commit that file** (`git add env/weights.sha256 && git commit -m "Record MASt3R weight checksums"`), so every later download on any machine is checked against the same hashes. If Naver replaces a file upstream, the script stops instead of quietly downloading a different model.

## Common failures

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch isn't the cu128 build. Re-run `pip install -r env/requirements-gpu.txt`. |
| `nvidia-smi` not found in WSL | Update the Windows NVIDIA driver, then run `wsl --shutdown` and reopen. |
| gsplat build fails with an `nvcc` error | Run `conda activate atomspace` first so `CUDA_HOME` points at CUDA 12.8. |
| curope build fails | Safe to ignore. MASt3R falls back to a slower PyTorch path. |
| gsplat build dies with `Killed` | Out of RAM during compilation. Re-run with fewer jobs: `MAX_JOBS=2 bash env/setup_wsl.sh`. |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |
| `.splat` / `.ply` / video files are tiny text files starting with `version https://git-lfs` | git-lfs wasn't installed when you cloned. `sudo apt install -y git-lfs && git lfs install && git lfs pull` |
