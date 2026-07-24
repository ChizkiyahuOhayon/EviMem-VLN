#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${REPO_ROOT}/runtime"
GA_VLN_COMMIT="cc6086b7081a346695abecf6821f827e2db44a43"
GA_VLN_URL="https://github.com/jahhaoyang/GA-VLN.git"
HABITAT_TAG="v0.2.4"
HABITAT_COMMIT="1639e1ae732ba1e84199a1a04b79c7243c3f8586"
MAX_JOBS="${MAX_JOBS:-4}"

usage() {
  echo "Usage: $0 [--workspace PATH]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
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

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit(f"Python 3.9 is required, found {sys.version.split()[0]}")
PY

mkdir -p "${WORKSPACE}"
GA_VLN_DIR="${WORKSPACE}/GA-VLN"
HABITAT_DIR="${WORKSPACE}/habitat-lab"

if [[ ! -d "${GA_VLN_DIR}/.git" ]]; then
  git clone "${GA_VLN_URL}" "${GA_VLN_DIR}"
fi
if [[ -n "$(git -C "${GA_VLN_DIR}" status --porcelain)" ]]; then
  echo "Refusing to change an existing dirty GA-VLN checkout: ${GA_VLN_DIR}" >&2
  exit 1
fi
git -C "${GA_VLN_DIR}" fetch --all --tags
git -C "${GA_VLN_DIR}" checkout --detach "${GA_VLN_COMMIT}"
ACTUAL_COMMIT="$(git -C "${GA_VLN_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${GA_VLN_COMMIT}" ]]; then
  echo "GA-VLN commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 1
fi

if [[ ! -d "${HABITAT_DIR}/.git" ]]; then
  git clone --branch "${HABITAT_TAG}" --depth 1 \
    https://github.com/facebookresearch/habitat-lab.git "${HABITAT_DIR}"
fi
if [[ -n "$(git -C "${HABITAT_DIR}" status --porcelain)" ]]; then
  echo "Refusing to change an existing dirty Habitat-Lab checkout: ${HABITAT_DIR}" >&2
  exit 1
fi
ACTUAL_HABITAT_COMMIT="$(git -C "${HABITAT_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_HABITAT_COMMIT}" != "${HABITAT_COMMIT}" ]]; then
  echo "Habitat-Lab commit mismatch: ${ACTUAL_HABITAT_COMMIT}" >&2
  echo "Expected ${HABITAT_TAG} at ${HABITAT_COMMIT}; remove the clean checkout and retry." >&2
  exit 1
fi

python -m pip install --upgrade "pip==24.0" wheel setuptools
python -m pip install -e "${REPO_ROOT}[download]"
python -m pip install -e "${HABITAT_DIR}/habitat-lab"
python -m pip install -e "${HABITAT_DIR}/habitat-baselines"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build flash-attn; recreate the environment with cuda-nvcc=12.1." >&2
  exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "A C++ compiler (g++) is required to build flash-attn." >&2
  exit 1
fi

FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${FILTERED_REQUIREMENTS}"' EXIT
grep -vE '^flash-attn==' "${GA_VLN_DIR}/requirements.txt" > "${FILTERED_REQUIREMENTS}"
python -m pip install -r "${FILTERED_REQUIREMENTS}"
MAX_JOBS="${MAX_JOBS}" python -m pip install \
  --no-build-isolation "flash-attn==2.5.8"

python - <<'PY'
import habitat
import habitat_sim
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("habitat", getattr(habitat, "__version__", "unknown"))
print("habitat_sim", getattr(habitat_sim, "__version__", "unknown"))
if torch.__version__.split("+")[0] != "2.1.2":
    raise SystemExit("Unexpected PyTorch version")
if torch.version.cuda != "12.1":
    raise SystemExit(f"Expected CUDA 12.1 runtime, found {torch.version.cuda}")
PY

echo "GA-VLN checkout: ${GA_VLN_DIR}"
echo "Habitat-Lab checkout: ${HABITAT_DIR} (${HABITAT_COMMIT})"
echo "Next: download assets to the canonical asset root, then run scripts/stage_assets.sh."
