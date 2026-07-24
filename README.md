# EviMem-VLN

**Future- and dependence-aware evidence admission for budgeted spatial memory in continuous vision-language navigation.**

EviMem-VLN is a research overlay for the official GA-VLN baseline. It keeps the
navigation policy, action parser, prompt, front-view path, and evaluator fixed,
and introduces a testable memory layer at one interface: after BEV
scatter-mean and before positional encoding / multimodal projection.

> **Research status:** Phase-0 infrastructure is implemented and unit-tested.
> No navigation result, SOTA claim, or Gaussian advantage is claimed yet.

## Why an overlay repository?

The inspected GA-VLN revision does not contain a visible license. This
repository therefore contains only original EviMem code, tests, environment
definitions, and deployment scripts. It does **not** redistribute GA-VLN,
checkpoints, or datasets. The bootstrap script clones and verifies the official
baseline separately at:

```text
cc6086b7081a346695abecf6821f827e2db44a43
```

## Reproducible server setup

Recommended host: Linux, two NVIDIA A40 48 GB GPUs, a recent NVIDIA driver,
local NVMe scratch, and NAS for canonical archives.

```bash
git clone https://github.com/ChizkiyahuOhayon/EviMem-VLN.git
cd EviMem-VLN

# Python 3.9, PyTorch 2.1.2, CUDA 12.1, Habitat-Sim 0.2.4
bash scripts/create_env.sh
conda activate evimem

# Clone pinned GA-VLN and Habitat-Lab, then install dependencies
bash scripts/bootstrap_ga_vln.sh \
  --workspace /local_nvme/$USER/evimem-runtime

# Download immutable model revisions to the NAS archive
python scripts/download_models.py \
  --asset-root /nas/evimem-assets

# Download R2R-CE and RxR-CE episode annotations
python scripts/download_vlnce.py \
  --asset-root /nas/evimem-assets

# MP3D requires prior acceptance of the Matterport3D Terms and the official script
bash scripts/download_mp3d.sh \
  --asset-root /nas/evimem-assets \
  --download-script /secure/path/download_mp.py \
  --i-accept-mp3d-terms

# Stage immutable assets from slow NAS to local NVMe
bash scripts/stage_assets.sh \
  --asset-root /nas/evimem-assets \
  --ga-vln-root /local_nvme/$USER/evimem-runtime/GA-VLN

# Verify source revision and every required asset
evimem doctor \
  --ga-vln-root /local_nvme/$USER/evimem-runtime/GA-VLN

# Print the exact two-GPU baseline command before executing it
bash scripts/run_gavln_baseline.sh \
  --ga-vln-root /local_nvme/$USER/evimem-runtime/GA-VLN \
  --nproc 2 \
  --split val_unseen \
  --dry-run
```

See [installation](docs/INSTALL.md), [data and licenses](docs/DATA.md), and
[architecture](docs/ARCHITECTURE.md) before launching G0.

## Implemented Phase-0 contracts

- 33-record observation ring with legacy gather and explicit reset lifecycle;
- fixed-shape unique-world-cell ledger and inclusive action TTL;
- BLAKE2b-63 deterministic bottom-k admission;
- observation-world-cell incidence rasterization and stable token selector;
- exact-quota fallback with fail-fast residual underfill;
- matched iid/AR noise traces, local SE(2), yaw wrap, and clipped depth noise;
- source / asset manifests, worktree hashes, and resume rejection;
- A0 trace recorder and a dependency-free reference adapter.

Run the asset-free suite:

```bash
python -m unittest discover -s tests -v
```

## Scientific gate order

```text
G0 baseline → A0 refactor/interface parity → G1 horizon reversal
→ G2 future retention → G3 dependence support → G4 carrier gate
→ G5 full mechanism → G6 frozen R2R/RxR evaluation
```

Gaussian splatting is not implemented in Phase 0. It is authorized only if the
earlier scientific gates pass.

## License

Original EviMem-VLN code is MIT licensed. External repositories, models, and
datasets retain their own licenses. In particular, VGGT is CC-BY-NC-4.0 and
Matterport3D requires acceptance of its Terms of Use.
