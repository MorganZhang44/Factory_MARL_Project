#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${SIMULATION_CONDA_ENV:-isaaclab51}"
RUNTIME="${SIMULATION_RUNTIME:-legacy}"

run_conda_clean() {
  CONDA_NO_PLUGINS=true env -u PYTHONPATH -u PYTHONHOME conda run -n "${CONDA_ENV}" "$@"
}

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

unset PYTHONHOME
SITE_PACKAGES="$(run_conda_clean python -c 'import site; print(site.getsitepackages()[0])')"
ISAACLAB_SOURCE="${SITE_PACKAGES}/isaaclab/source"
export PYTHONPATH=""
if [[ -d "${ISAACLAB_SOURCE}" ]]; then
  export PYTHONPATH="${ISAACLAB_SOURCE}/isaaclab:${ISAACLAB_SOURCE}/isaaclab_assets:${ISAACLAB_SOURCE}/isaaclab_tasks:${ISAACLAB_SOURCE}/isaaclab_rl:${ISAACLAB_SOURCE}/isaaclab_mimic"
fi

ISAAC_ROS2_BRIDGE="${SITE_PACKAGES}/isaacsim/exts/isaacsim.ros2.bridge/humble"
if [[ -d "${ISAAC_ROS2_BRIDGE}" ]]; then
  export PYTHONPATH="${ISAAC_ROS2_BRIDGE}/rclpy:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${ISAAC_ROS2_BRIDGE}/lib:${LD_LIBRARY_PATH:-}"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ARGS=("$@")
HAS_DEVICE_ARG=0
ENTRYPOINT="simulation/standalone/validate_slam_scene.py"
FILTERED_ARGS=()
IDX=0
while [[ "${IDX}" -lt "${#ARGS[@]}" ]]; do
  arg="${ARGS[${IDX}]}"
  if [[ "${arg}" == "--runtime" ]]; then
    IDX=$((IDX + 1))
    if [[ "${IDX}" -ge "${#ARGS[@]}" ]]; then
      echo "Missing value for --runtime. Use 'legacy' or 'rewrite'." >&2
      exit 1
    fi
    RUNTIME="${ARGS[${IDX}]}"
    IDX=$((IDX + 1))
    continue
  fi
  if [[ "${arg}" == --runtime=* ]]; then
    RUNTIME="${arg#--runtime=}"
    IDX=$((IDX + 1))
    continue
  fi
  if [[ "${arg}" == "--device" || "${arg}" == --device=* ]]; then
    HAS_DEVICE_ARG=1
  fi
  FILTERED_ARGS+=("${arg}")
  IDX=$((IDX + 1))
done

if [[ "${RUNTIME}" == "rewrite" ]]; then
  ENTRYPOINT="simulation/standalone/run_environment_rewrite.py"
elif [[ "${RUNTIME}" != "legacy" ]]; then
  echo "Unsupported simulation runtime '${RUNTIME}'. Use 'legacy' or 'rewrite'." >&2
  exit 1
fi

if [[ "${HAS_DEVICE_ARG}" -eq 0 ]]; then
  FILTERED_ARGS=(--device cuda:0 "${FILTERED_ARGS[@]}")
fi

echo "[simulation] runtime=${RUNTIME} entrypoint=${ENTRYPOINT}" >&2
CONDA_NO_PLUGINS=true exec env -u PYTHONHOME conda run --no-capture-output -n "${CONDA_ENV}" python "${ENTRYPOINT}" "${FILTERED_ARGS[@]}"
