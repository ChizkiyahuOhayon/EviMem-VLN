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
python -m pip install \
  "habitat-lab @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-lab" \
  "habitat-baselines @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-baselines"
```

No GA-VLN checkout is created by these commands. FlashAttention 2.5.8 builds
against the PyTorch/CUDA environment. Its installer may use a compatible
prebuilt wheel; a source-build fallback requires `nvcc`, `ninja`, and a C++
compiler. It is installed with the upstream-required `--no-build-isolation`
flag by `scripts/install_runtime.sh`. To repair or update an existing
environment, run:

```bash
bash scripts/install_runtime.sh
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

WaveDrom, svgwrite, and latex2mathml were removed from the runtime requirements:
they are documentation tools and are not imported by the GA-VLN or EviMem code.
