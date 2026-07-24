#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=""
GA_VLN_ROOT=""

usage() {
  echo "Usage: $0 --asset-root /nas/evimem-assets --ga-vln-root /local_nvme/GA-VLN"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root)
      ASSET_ROOT="$2"
      shift 2
      ;;
    --ga-vln-root)
      GA_VLN_ROOT="$2"
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

if [[ ! -d "${ASSET_ROOT}" || ! -d "${GA_VLN_ROOT}/.git" ]]; then
  usage >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for NAS-to-local staging." >&2
  exit 1
fi

ASSET_ROOT="$(cd "${ASSET_ROOT}" && pwd)"
GA_VLN_ROOT="$(cd "${GA_VLN_ROOT}" && pwd)"

RELATIVE_PATHS=(
  checkpoints/gavln_official
  model/siglip-so400m-patch14-384
  model/VGGT-1B
  vln_data/datasets/r2r
  vln_data/datasets/RxR_VLNCE_v0
  vln_data/scene_datasets/mp3d
)

for relative in "${RELATIVE_PATHS[@]}"; do
  source_path="${ASSET_ROOT}/${relative}"
  target_path="${GA_VLN_ROOT}/${relative}"
  if [[ ! -d "${source_path}" ]]; then
    echo "Skip missing optional/unavailable asset: ${source_path}"
    continue
  fi
  mkdir -p "${target_path}"
  rsync --archive --human-readable --info=progress2 \
    "${source_path}/" "${target_path}/"
done

python -m evimem.cli verify-assets --ga-vln-root "${GA_VLN_ROOT}"
