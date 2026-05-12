#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${SIM2REAL_CONDA_ENV:-sim2real}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Removing existing conda environment: ${ENV_NAME}"
  conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda environment '${ENV_NAME}' from sim2real/environment.yml"
conda env create -f "${PROJECT_ROOT}/sim2real/environment.yml"

echo "Installing sim2real Python requirements"
set +u
conda activate "${ENV_NAME}"
set -u
python -m pip install -r "${PROJECT_ROOT}/sim2real/requirements.txt"

echo
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
