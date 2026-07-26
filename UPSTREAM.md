# Upstream provenance

## Imported baseline

- Project: GA-VLN
- Source: <https://github.com/jahhaoyang/GA-VLN>
- Fixed commit: `cc6086b7081a346695abecf6821f827e2db44a43`
- Imported into EviMem-VLN: 2026-07-24
- Import method: tracked source/configuration from the fixed Git tree, excluding
  generated `__pycache__` bytecode; the root README and launch script were
  merged with EviMem's versions. There is no submodule or runtime dependency on
  an external checkout.

The imported tree supplies `gavln/`, `llava/`, `trl/`, `vggt/`, the Habitat
configuration, runtime requirements, assets, and original evaluation scripts.

## Authorization and license status

The fixed GA-VLN commit contains no `LICENSE`, `COPYING`, or `NOTICE` file, and
the upstream GitHub repository did not declare a detected license when checked
on 2026-07-24. EviMem-VLN therefore does not assign an invented license to
GA-VLN code.

The repository owner/user explicitly represented on 2026-07-24 that
authorization had been obtained to clone, modify, and upload the GA-VLN code.
The import and public-fork work proceeded on that representation. Maintainers
should preserve the underlying written permission with the project records;
this file is provenance, not independent legal verification.

The MIT terms in `LICENSE` apply only to original EviMem-VLN work. Upstream and
third-party components retain their own copyright and license status.

## Main EviMem modifications

- `gavln/gavln_eval.py`: lightweight installed entry point and early asset
  validation.
- `gavln/eval_runtime.py`: package-safe imports, backend configuration, episode
  reset, and observation/action metadata delivery.
- `gavln/model/model_gavln.py`: unified memory interface connected between
  vision/VGGT features and the original position encoding/projector/LLM path.
- `gavln/memory/`: backend protocol plus the original GA-VLN implementation.
- `evimem/torch_backend.py`: persistent fixed-capacity sparse world ledger and
  deterministic token readout.
- `configs/`, `scripts/`, packaging, documentation, and tests: standalone
  installation and launch support.

## Future upstream synchronization

Do not replace this tree with an unpinned checkout. To synchronize:

1. fetch the desired upstream revision in a separate maintainer workspace;
2. record its commit and inspect its license;
3. diff it against the fixed commit above;
4. merge changes file by file while preserving both memory backends;
5. run the full asset-free suite and R2R-CE baseline parity evaluation;
6. update this provenance record.
