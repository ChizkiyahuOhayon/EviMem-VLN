# Installation

## Frozen environment

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| CUDA userspace / nvcc | 12.1 |
| Habitat-Sim | 0.2.4, headless + Bullet |
| Habitat-Lab/Baselines | commit `1639e1ae732ba1e84199a1a04b79c7243c3f8586` |
| GA-VLN source | internal fork of `cc6086b7081a346695abecf6821f827e2db44a43` |

CUDA 12.1 matches the upstream GA-VLN PyTorch wheels. A newer compatible host
driver is expected. Conda or Micromamba is used because Habitat-Sim 0.2.4 is
provided through the `aihabitat`/`conda-forge` channels. NumPy, PyTorch, and
torchvision are installed by `create_env.sh` with pip after the Conda solve.
This avoids dependence on Conda mirrors carrying the complete
`pytorch-cuda`/`cuda-libraries` package chain.

## Commands

```bash
bash scripts/create_env.sh
conda activate evimem
python -m pip install -r requirements.txt
python -m pip install \
  "habitat-lab @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-lab" \
  "habitat-baselines @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-baselines"
python -m pip install -e .
```

No GA-VLN checkout is created by these commands. FlashAttention 2.5.8 builds
against the PyTorch/CUDA environment. Its installer may use a compatible
prebuilt wheel; a source-build fallback requires `nvcc`, `ninja`, and a C++
compiler. If build isolation fails, first verify `nvcc --version`, then install
it after the rest:

```bash
python -m pip install -r requirements.txt --no-deps
MAX_JOBS=4 python -m pip install --no-build-isolation flash-attn==2.5.8
```

Before downloading large assets:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m gavln.gavln_eval --help
```

Expected PyTorch/CUDA versions are `2.1.2` and `12.1`. Both A40s should be
visible. Stage assets from slow NAS to local NVMe before Habitat evaluation.

If a previous environment solve failed before this fix, rerun with:

```bash
bash scripts/create_env.sh --update
```

If no `evimem` environment was created, use the normal command without
`--update`.

The runtime pins PyAV 14.0.0 because it provides a CPython 3.9 manylinux wheel.
Newer PyAV 14.4.0 falls back to a local FFmpeg 7 source build on Python 3.9 and
is not required by the Habitat navigation evaluator.
