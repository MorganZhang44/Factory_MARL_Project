#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CORE_CONDA_ENV:-core}"
WORKSPACE="${CORE_WORKSPACE:-${PROJECT_ROOT}/ros2/workspace}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  echo "Create it with: conda env create -f core/environment.yml" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"
if [[ "${CORE_CLEAN_WORKSPACE:-0}" == "1" ]]; then
  rm -rf build install log
fi
colcon build \
  --symlink-install \
  --base-paths \
  "${PROJECT_ROOT}/core/ros2" \
  "${PROJECT_ROOT}/ros2"

set +u
source install/setup.bash
set -u

exec ros2 launch factory_bringup core_dashboard.launch.py "$@"
