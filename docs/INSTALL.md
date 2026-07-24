# Installation

## Frozen environment

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| CUDA userspace / nvcc | 12.1 |
| Habitat-Sim | 0.2.4, headless + Bullet |
| Habitat-Lab | v0.2.4 (`1639e1ae732ba1e84199a1a04b79c7243c3f8586`) |
| GA-VLN | `cc6086b7081a346695abecf6821f827e2db44a43` |

CUDA 12.1 matches the official GA-VLN PyTorch wheels. The host driver may be
newer; do not downgrade a working cluster driver merely to match the userspace
toolkit.

Conda or Micromamba is required because Habitat-Sim 0.2.4 is installed from
the `aihabitat`/`conda-forge` channels. A plain Python `venv` is not the
supported deployment path.

## Commands

```bash
bash scripts/create_env.sh
conda activate evimem
bash scripts/bootstrap_ga_vln.sh --workspace /local_nvme/$USER/evimem-runtime
```

`bootstrap_ga_vln.sh` refuses to change a dirty existing baseline checkout. It
installs GA-VLN requirements only after PyTorch is present and installs
FlashAttention with `--no-build-isolation`. A host C++ compiler (`g++`) is also
required; `cuda-nvcc=12.1` is included in the Conda environment.

## Driver check

Before downloading large assets:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Expected PyTorch/CUDA output is `2.1.2` and `12.1`. Both A40s should be visible.

## Why not install the latest stack?

Habitat, Transformers, FlashAttention, and GA-VLN are tightly coupled. Upgrading
one component before G0 would change the baseline and make a reproduction
failure hard to diagnose. Modernization belongs in a separate, post-G0 branch.
