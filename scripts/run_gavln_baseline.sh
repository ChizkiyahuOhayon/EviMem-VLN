#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC=2
SPLIT="val_unseen"
OUTPUT=""
DRY_RUN=0

usage() {
  echo "Usage: $0 [--nproc 2] [--split val_unseen] [--output PATH] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

OUTPUT="${OUTPUT:-${REPO_ROOT}/results/gavln/${SPLIT}}"
python -m evimem.cli doctor

COMMAND=(
  torchrun
  --standalone
  --nnodes=1
  "--nproc_per_node=${NPROC}"
  -m
  gavln.gavln_eval
  --config
  "${REPO_ROOT}/configs/gavln.yaml"
  --eval_split "${SPLIT}"
  --output_path "${OUTPUT}"
)

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export HYDRA_FULL_ERROR=1
cd "${REPO_ROOT}"
"${COMMAND[@]}"
