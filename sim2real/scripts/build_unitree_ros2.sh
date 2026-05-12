#!/usr/bin/env bash
set -euo pipefail

SIM2REAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_ROOT="${SIM2REAL_ROOT}/unitree_ros2"
CONDA_ENV="${SIM2REAL_CONDA_ENV:-sim2real}"

if [[ ! -d "${UNITREE_ROOT}" ]]; then
  echo "unitree_ros2 was not found at ${UNITREE_ROOT}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
if [[ -f "${CONDA_PREFIX}/setup.bash" ]]; then
  source "${CONDA_PREFIX}/setup.bash"
fi
set -u

echo "Building Unitree ROS2 message workspace from ${UNITREE_ROOT}/cyclonedds_ws"
cd "${UNITREE_ROOT}/cyclonedds_ws"
if [[ "${SIM2REAL_CLEAN_WORKSPACE:-0}" == "1" ]]; then
  rm -rf build install log
fi

colcon build --symlink-install

echo
echo "unitree_ros2 message workspace built successfully."
echo "Recommended source path:"
echo "  ${UNITREE_ROOT}/cyclonedds_ws/install/setup.bash"
