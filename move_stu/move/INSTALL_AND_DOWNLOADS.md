# Required Downloads and Environment

This task is implemented as a full PCT-policy-driven Isaac Gym pipeline. Missing
model or simulator assets are treated as hard errors.

## Required downloads

1. NVIDIA Isaac Gym Preview 4
   - Official page: https://developer.nvidia.com/isaac-gym
   - Download archive: https://developer.nvidia.com/isaac-gym/download
   - Required file name is typically `IsaacGym_Preview_4_Package.tar.gz`.
   - NVIDIA requires accepting the Isaac Gym license in browser before download.

2. Online-3D-BPP-PCT pretrained model
   - Repo: https://github.com/alexfrom0815/Online-3D-BPP-PCT
   - README pretrained model folder: https://drive.google.com/drive/folders/14PC3aVGiYZU5AaGdNM9YOVdp8pPiZ3fe
   - Use the discrete EMS model trained for bin size `(10, 10, 10)` and item sizes `1..5`.
   - Put the `.pt`/`.pth` weight at:
     `/2024233240/external/Online-3D-BPP-PCT/pretrained/ems_10x10x10.pt`

## Environment target

Isaac Gym Preview 4 officially lists Ubuntu 18.04/20.04, Python 3.6/3.7/3.8,
NVIDIA driver 470.74+, and Pascal-or-newer GPU with at least 8 GB VRAM.

Recommended local env:

```bash
/2024233240/miniconda3/condabin/conda create -y -n move_bpp python=3.8
/2024233240/miniconda3/condabin/conda run -n move_bpp python -m pip install numpy==1.23.5 gym==0.13.0 imageio imageio-ffmpeg
```

After extracting Isaac Gym:

```bash
tar -xf IsaacGym_Preview_4_Package.tar.gz -C /2024233240/external
/2024233240/miniconda3/condabin/conda run -n move_bpp python -m pip install -e /2024233240/external/isaacgym/python
```

Then install a CUDA-compatible PyTorch build for the machine driver/CUDA stack.

## Plan generation

```bash
/2024233240/miniconda3/condabin/conda run -n move_bpp python -m move.online_palletizing \
  --model-path /2024233240/external/Online-3D-BPP-PCT/pretrained/ems_10x10x10.pt \
  --cases 5 \
  --items-per-case 12
```


## Installed workspace environment

The working environment currently installed in this workspace is:

```bash
/2024233240/move/run_move_bpp_env.sh -c "import isaacgym; from isaacgym import gymapi, gymtorch; import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `run_move_bpp_env.sh` instead of calling Python directly, because Isaac Gym needs the conda env `bin/` on `PATH` and `lib/` on `LD_LIBRARY_PATH`.
