#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install -r "${REPO_ROOT}/build-constraints.txt"
python -m pip install \
  -c "${REPO_ROOT}/constraints-cu121.txt" \
  -r "${REPO_ROOT}/requirements.txt"
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
  --no-build-isolation \
  -r "${REPO_ROOT}/requirements-flash-attn.txt"
python -m pip install -e "${REPO_ROOT}"

python -c 'import flash_attn, torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "flash-attn", flash_attn.__version__)'
