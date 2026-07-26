# EviMem-VLN

EviMem-VLN is a self-contained fork of
[GA-VLN](https://github.com/jahhaoyang/GA-VLN) with a persistent, fixed-budget
sparse world-memory backend. The complete source tree required by the GA-VLN
model and evaluator is included. Users do **not** clone GA-VLN, use a
submodule, set `GA_VLN_ROOT`, or modify `PYTHONPATH`.

The imported baseline is fixed to:

```text
cc6086b7081a346695abecf6821f827e2db44a43
```

See [UPSTREAM.md](UPSTREAM.md) for provenance, authorization, modifications,
and the upstream license caveat.

> Research status: the internal code path and asset-free tests are implemented.
> Full R2R-CE/RxR-CE scores have not been produced in this checkout, so this
> repository makes no SOTA or oral-acceptance claim.

## Install

The frozen stack targets Linux, Python 3.9, PyTorch 2.1.2, CUDA 12.1,
Habitat-Sim 0.2.4, and two A40 GPUs.

```bash
git clone https://github.com/ChizkiyahuOhayon/EviMem-VLN.git
cd EviMem-VLN

bash scripts/create_env.sh
conda activate evimem

# Habitat-Lab/Baselines v0.2.4 may be installed from the official source:
python -m pip install \
  "habitat-lab @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-lab" \
  "habitat-baselines @ git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#subdirectory=habitat-baselines"

```

`create_env.sh` installs the runtime requirements, FlashAttention 2.5.8, and
the repository packages. For an existing environment, run
`bash scripts/install_runtime.sh`. See [docs/INSTALL.md](docs/INSTALL.md) for
FlashAttention and driver notes.

## External assets

Weights and datasets are deliberately excluded from Git:

```text
checkpoints/gavln_official/
model/siglip-so400m-patch14-384/
model/VGGT-1B/
vln_data/datasets/r2r/
vln_data/datasets/RxR_VLNCE_v0/
vln_data/scene_datasets/mp3d/
```

Download the public model and VLN-CE assets to the NAS, obtain MP3D through its
official licensed process, then stage them to the repository on local NVMe:

```bash
python scripts/download_models.py --asset-root /nas/evimem-assets
python scripts/download_vlnce.py --asset-root /nas/evimem-assets
bash scripts/download_mp3d.sh \
  --asset-root /nas/evimem-assets \
  --download-script /secure/path/download_mp.py \
  --i-accept-mp3d-terms

bash scripts/stage_assets.sh --asset-root /nas/evimem-assets
evimem doctor
```

The evaluator fails with a detailed asset report before importing
Torch/Habitat when assets are missing. See [docs/DATA.md](docs/DATA.md).

## Run

Original GA-VLN memory:

```bash
bash scripts/run_gavln_baseline.sh --nproc 2 --split val_unseen
```

EviMem sparse world memory:

```bash
bash scripts/run_evimem.sh --nproc 2 --split val_unseen
```

Equivalent module commands are:

```bash
torchrun --standalone --nproc_per_node=2 -m gavln.gavln_eval \
  --config configs/gavln.yaml

torchrun --standalone --nproc_per_node=2 -m gavln.gavln_eval \
  --config configs/evimem.yaml
```

The original GA-VLN multimodal training modules remain under `llava/train/`.
The checked-in production integration in this revision targets navigation
evaluation; a stateful multi-episode EviMem training recipe is not yet claimed.

## Memory configuration

| Key | Meaning |
|---|---|
| `memory_backend` | `gavln` for original windowed BEV; `evimem` for persistent sparse world memory |
| `memory_horizon` | Evidence retention in executed actions: `8`, `32`, `64`, or `route` |
| `memory_resident_slots` | Fixed sparse-ledger capacity |
| `memory_token_budget` | Maximum selected BEV memory tokens; `null` means all occupied cells |
| `memory_seed` | Deterministic bottom-k admission seed |

Both modes share the same vision encoders, position encoding,
`mm_projector`, LLM, prompts, action parser, and Habitat evaluator.

## Architecture

```text
RGB/depth + pose
  ├─ SigLIP patch features ─┐
  └─ VGGT patch features ───┤
                            ▼
               MemoryBackend.build_tokens()
               ├─ gavln: windowed agent-BEV scatter mean
               └─ evimem: persistent sparse world ledger
                            │
                            ▼
          token selection → GA-VLN position encoding
          → mm_projector → <memory> tokens → Qwen policy
```

EviMem state is allocated per evaluator environment. Episode reset clears both
the ledger and language cache; periodic dialogue reset clears only the language
cache. Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

```text
config/                 Habitat task configuration
configs/                GA-VLN/EviMem mode configuration
evimem/                 EviMem reference and production Torch backend
gavln/                  evaluator, model, utilities, memory interface/backends
llava/ trl/ vggt/       baseline model dependencies carried in the fork
scripts/                environment, assets, baseline, and EviMem launchers
tests/                  asset-free and optional Torch backend tests
```

## Tests

Without datasets or checkpoints:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m gavln.gavln_eval --help
```

The suite checks internal imports, missing-asset diagnostics, reset isolation,
ring/ledger capacity, retention, token budgets, determinism, and absence of
runtime GA-VLN cloning or external path dependencies. Torch-specific backend
tests run in the full CUDA environment and skip in lightweight CI without
PyTorch.

## Migration from the overlay release

- Delete any old `runtime/GA-VLN` checkout; it is no longer used.
- Remove `--ga-vln-root`, `GA_VLN_ROOT`, and custom GA-VLN `PYTHONPATH`.
- Place or stage assets under this repository using the paths above.
- Replace `bootstrap_ga_vln.sh` with `pip install -e .`.
- Select `configs/gavln.yaml` or `configs/evimem.yaml`.

## Implemented and not yet implemented

Implemented: complete internal GA-VLN source, dual memory backends, world-cell
ledger, fixed-capacity deterministic admission/eviction, action-horizon
retention, token budgeting, per-environment episode isolation, and the original
projection/LLM path.

Not claimed: published benchmark scores for this fork, EviMem-specific training,
Gaussian-splat rendering, or AwareVLN-specific learned objectives. Those are
research extensions and should not be conflated with the verified integration.

## License and attribution

Original EviMem code is MIT licensed under the boundary stated in
[LICENSE](LICENSE). GA-VLN did not provide a standard license at the pinned
commit; its code is included based on the project authorization recorded in
[UPSTREAM.md](UPSTREAM.md). That authorization should be retained as legal
provenance and is not replaced by the EviMem MIT license.

GA-VLN is by Jiahao Yang, Zihan Wang, Xiangyang Li, Xing Zhu, Yujun Shen,
Yinghao Xu, and Shuqiang Jiang. External models, datasets, Habitat, LLaVA,
StreamVLN, TRL, and VGGT retain their own licenses and terms.
