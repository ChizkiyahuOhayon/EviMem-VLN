#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment/environment-cu121.yml"
ENV_NAME="evimem"
UPDATE=0

usage() {
  echo "Usage: $0 [--name ENV_NAME] [--update]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v micromamba >/dev/null 2>&1; then
  RUNNER=(micromamba)
elif command -v conda >/dev/null 2>&1; then
  RUNNER=(conda)
else
  echo "Install Micromamba or Conda before running this script." >&2
  exit 1
fi

if [[ "${UPDATE}" -eq 1 ]]; then
  "${RUNNER[@]}" env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  "${RUNNER[@]}" env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

"${RUNNER[@]}" run --name "${ENV_NAME}" \
  python -m pip install --upgrade "pip==24.0" wheel "setuptools==69.5.1"
"${RUNNER[@]}" run --name "${ENV_NAME}" \
  python -m pip install \
    "numpy==1.26.1" \
    "gdown==5.2.0" \
    "huggingface-hub==0.23.2" \
    "hf-transfer==0.1.9"
"${RUNNER[@]}" run --name "${ENV_NAME}" \
  python -m pip install \
    "torch==2.1.2" \
    "torchvision==0.16.2" \
    --index-url https://download.pytorch.org/whl/cu121
"${RUNNER[@]}" run --name "${ENV_NAME}" \
  python -c 'import numpy, torch, torchvision; assert numpy.__version__ == "1.26.1"; assert torch.__version__.split("+")[0] == "2.1.2"; assert torch.version.cuda == "12.1"; assert torchvision.__version__.split("+")[0] == "0.16.2"; print("numpy", numpy.__version__, "torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda)'
"${RUNNER[@]}" run --name "${ENV_NAME}" \
  bash "${REPO_ROOT}/scripts/install_runtime.sh"

echo "Environment ready: ${ENV_NAME}"
echo "Activate it with:"
echo "  ${RUNNER[0]} activate ${ENV_NAME}"
