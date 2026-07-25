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

echo "Environment ready: ${ENV_NAME}"
echo "Activate it, then run:"
echo "  python -m pip install -r requirements.txt"
echo "  python -m pip install -e ."
