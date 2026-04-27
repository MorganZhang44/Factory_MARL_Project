#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${NAVDP_CONDA_ENV:-navdp}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  echo "NavDP must run in its own environment, default: navdp." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_ROOT}"
if [[ "$#" -eq 0 ]]; then
  exec python navdp/navdp_service.py --planner "${NAVDP_PLANNER:-real}" --device "${NAVDP_DEVICE:-cuda:0}"
fi

exec python navdp/navdp_service.py "$@"
