#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${SIMULATION_CONDA_ENV:-isaaclab51}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  echo "Simulation must run in its own Isaac Sim environment, default: isaaclab51." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

SITE_PACKAGES="$(conda run -n "${CONDA_ENV}" python -c 'import site; print(site.getsitepackages()[0])')"
ISAACLAB_SOURCE="${SITE_PACKAGES}/isaaclab/source"
if [[ -d "${ISAACLAB_SOURCE}" ]]; then
  export PYTHONPATH="${ISAACLAB_SOURCE}/isaaclab:${ISAACLAB_SOURCE}/isaaclab_assets:${ISAACLAB_SOURCE}/isaaclab_tasks:${ISAACLAB_SOURCE}/isaaclab_rl:${ISAACLAB_SOURCE}/isaaclab_mimic:${PYTHONPATH:-}"
fi

ISAAC_ROS2_BRIDGE="${SITE_PACKAGES}/isaacsim/exts/isaacsim.ros2.bridge/humble"
if [[ -d "${ISAAC_ROS2_BRIDGE}" ]]; then
  export PYTHONPATH="${ISAAC_ROS2_BRIDGE}/rclpy:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${ISAAC_ROS2_BRIDGE}/lib:${LD_LIBRARY_PATH:-}"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

ARGS=("$@")
HAS_DEVICE_ARG=0
for arg in "${ARGS[@]}"; do
  if [[ "${arg}" == "--device" || "${arg}" == --device=* ]]; then
    HAS_DEVICE_ARG=1
    break
  fi
done

if [[ "${HAS_DEVICE_ARG}" -eq 0 ]]; then
  ARGS=(--device cuda:0 "${ARGS[@]}")
fi

exec conda run -n "${CONDA_ENV}" python simulation/standalone/validate_slam_scene.py "${ARGS[@]}"
