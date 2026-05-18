#!/usr/bin/env bash
set -euo pipefail

SIM2REAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${SIM2REAL_CONDA_ENV:-sim2real}"
WORKSPACE="${SIM2REAL_WORKSPACE:-${SIM2REAL_ROOT}/ros2/workspace}"
SIM2REAL_NET_IFACE="${SIM2REAL_NET_IFACE:-eno1}"
SIM2REAL_RMW_IMPLEMENTATION="${SIM2REAL_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
UNITREE_ROS2_SETUP="${UNITREE_ROS2_SETUP:-}"

needs_workspace_clean() {
  local active_python script_path script_first_line
  active_python="${CONDA_PREFIX:-}/bin/python"
  script_path="${WORKSPACE}/install/factory_core_sim2real/lib/factory_core_sim2real/core_control_node"

  if [[ ! -e "${WORKSPACE}/install" ]]; then
    return 1
  fi

  if [[ -d "${WORKSPACE}/install/factory_core_sim2real/lib/python3.10" ]]; then
    return 0
  fi

  if [[ ! -f "${script_path}" ]]; then
    return 0
  fi

  script_first_line="$(head -n 1 "${script_path}" || true)"
  if [[ "${script_first_line}" != "#!${active_python}"* ]]; then
    return 0
  fi

  return 1
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  conda env list >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
if [[ -f "${CONDA_PREFIX}/setup.bash" ]]; then
  source "${CONDA_PREFIX}/setup.bash"
fi
set -u

export RMW_IMPLEMENTATION="${SIM2REAL_RMW_IMPLEMENTATION}"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${SIM2REAL_NET_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"
if [[ "${SIM2REAL_CLEAN_WORKSPACE:-0}" == "1" ]] || needs_workspace_clean; then
  echo "Cleaning sim2real workspace build/install artifacts for the active Python environment"
  rm -rf build install log
fi

if [[ -z "${UNITREE_ROS2_SETUP}" && -f "${SIM2REAL_ROOT}/unitree_ros2/cyclonedds_ws/install/setup.bash" ]]; then
  UNITREE_ROS2_SETUP="${SIM2REAL_ROOT}/unitree_ros2/cyclonedds_ws/install/setup.bash"
fi

if [[ -z "${UNITREE_ROS2_SETUP}" && -d "${SIM2REAL_ROOT}/unitree_ros2/cyclonedds_ws" ]]; then
  echo "Unitree ROS2 install/setup.bash was not found. Building the local message workspace first."
  "${SIM2REAL_ROOT}/scripts/build_unitree_ros2.sh"
  if [[ -f "${SIM2REAL_ROOT}/unitree_ros2/cyclonedds_ws/install/setup.bash" ]]; then
    UNITREE_ROS2_SETUP="${SIM2REAL_ROOT}/unitree_ros2/cyclonedds_ws/install/setup.bash"
  fi
fi

if [[ -n "${UNITREE_ROS2_SETUP}" ]]; then
  echo "Sourcing Unitree ROS2 environment: ${UNITREE_ROS2_SETUP}"
  set +u
  source "${UNITREE_ROS2_SETUP}"
  set -u
fi

colcon build --base-paths "${SIM2REAL_ROOT}/ros2"

set +u
source install/setup.bash
set -u

hash -r
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/ros2" ]]; then
  echo "Expected ros2 inside the active conda environment, but it was not found." >&2
  exit 1
fi

exec "${CONDA_PREFIX}/bin/ros2" launch factory_bringup_sim2real core_dashboard.launch.py "$@"
