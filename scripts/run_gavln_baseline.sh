#!/usr/bin/env bash
set -euo pipefail

GA_VLN_ROOT=""
NPROC=2
SPLIT="val_unseen"
OUTPUT=""
DRY_RUN=0

usage() {
  echo "Usage: $0 --ga-vln-root PATH [--nproc 2] [--split val_unseen] [--output PATH] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ga-vln-root)
      GA_VLN_ROOT="$2"
      shift 2
      ;;
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

if [[ ! -d "${GA_VLN_ROOT}/.git" ]]; then
  usage >&2
  exit 2
fi

GA_VLN_ROOT="$(cd "${GA_VLN_ROOT}" && pwd)"
OUTPUT="${OUTPUT:-${GA_VLN_ROOT}/results/g0/${SPLIT}}"
python -m evimem.cli doctor --ga-vln-root "${GA_VLN_ROOT}"

COMMAND=(
  torchrun
  --standalone
  --nnodes=1
  "--nproc_per_node=${NPROC}"
  gavln/gavln_eval.py
  --model_path ./checkpoints/gavln_official
  --vision_tower_path ./model/siglip-so400m-patch14-384
  --vggt_path ./model/VGGT-1B
  --habitat_config_path config/vln_r2r.yaml
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
cd "${GA_VLN_ROOT}"
"${COMMAND[@]}"
