#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${PERCEPTION_CONDA_ENV:-perception}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  echo "Create or update it with: conda env create -f perception/environment.yml" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
CONDA_NO_PLUGINS=true env -u PYTHONPATH -u PYTHONHOME conda run --no-capture-output -n "${CONDA_ENV}" \
  python perception/perception_service.py "$@"
