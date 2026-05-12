#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_ROOT="${PROJECT_ROOT}/sim2real/unitree_ros2"
CONDA_ENV="${SIM2REAL_CONDA_ENV:-sim2real}"

if [[ ! -d "${UNITREE_ROOT}" ]]; then
  echo "unitree_ros2 was not found at ${UNITREE_ROOT}" >&2
  echo "Expected local clone under sim2real/." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
source "/home/yyz/miniconda3/envs/${CONDA_ENV}/setup.bash"
set -u

echo "Building Unitree ROS2 message workspace from ${UNITREE_ROOT}/cyclonedds_ws"
cd "${UNITREE_ROOT}/cyclonedds_ws"
if [[ "${SIM2REAL_CLEAN_WORKSPACE:-0}" == "1" ]]; then
  rm -rf build install log
fi

if ! colcon build --symlink-install; then
  cat <<'EOF' >&2

unitree_ros2 build did not complete.

Current known blocker on this machine:
  missing ROS package config: rosidl_generator_dds_idl

The local clone is installed under:
  sim2real/unitree_ros2

Once that dependency is available, rerun:
  ./scripts/build_sim2real_unitree_ros2.sh

EOF
  exit 1
fi

echo
echo "unitree_ros2 message workspace built successfully."
echo "Recommended source path:"
echo "  ${UNITREE_ROOT}/cyclonedds_ws/install/setup.bash"
