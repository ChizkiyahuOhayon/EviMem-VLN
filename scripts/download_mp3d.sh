#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_SCRIPT=""
ASSET_ROOT=""
ACCEPTED=0

usage() {
  echo "Usage: $0 --asset-root PATH --download-script /path/to/download_mp.py --i-accept-mp3d-terms"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root)
      ASSET_ROOT="$2"
      shift 2
      ;;
    --download-script)
      DOWNLOAD_SCRIPT="$2"
      shift 2
      ;;
    --i-accept-mp3d-terms)
      ACCEPTED=1
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

if [[ "${ACCEPTED}" -ne 1 ]]; then
  echo "Matterport3D requires prior acceptance of its Terms of Use." >&2
  echo "Obtain the official download_mp.py, then pass --i-accept-mp3d-terms." >&2
  exit 3
fi
if [[ ! -f "${DOWNLOAD_SCRIPT}" || -z "${ASSET_ROOT}" ]]; then
  usage >&2
  exit 2
fi

if command -v python2.7 >/dev/null 2>&1; then
  PYTHON_BIN="python2.7"
elif command -v python2 >/dev/null 2>&1; then
  PYTHON_BIN="python2"
else
  echo "The official Matterport3D downloader requires Python 2.7." >&2
  exit 1
fi

OUTPUT="${ASSET_ROOT}/vln_data/scene_datasets/mp3d"
mkdir -p "${OUTPUT}"
"${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}" --task habitat -o "${OUTPUT}"

SCENE_COUNT="$(find "${OUTPUT}" -mindepth 2 -maxdepth 2 -name '*.glb' -type f | wc -l | tr -d ' ')"
if [[ "${SCENE_COUNT}" != "90" ]]; then
  echo "Expected 90 MP3D Habitat scenes, found ${SCENE_COUNT}." >&2
  exit 1
fi
echo "Matterport3D verified: ${SCENE_COUNT}/90 scenes"
