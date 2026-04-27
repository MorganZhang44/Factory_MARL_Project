#!/usr/bin/env python3
"""Preview the programmatic SlamSceneCfg scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Preview simulation.scenes.slam_scene_cfg.SlamSceneCfg.")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run before exiting. 0 keeps the window open.")
parser.add_argument(
    "--disable_scene_cameras",
    action="store_true",
    help="Skip the robot camera sensors for a faster geometry-only preview.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from simulation.scenes.slam_scene_cfg import SlamSceneCfg  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(eye=[7.0, -8.0, 6.0], target=[0.0, -0.5, 0.5])

    scene_cfg = SlamSceneCfg(num_envs=1, env_spacing=25.0, lazy_sensor_update=False)
    if args_cli.disable_scene_cameras:
        scene_cfg.agent_1_front_camera = None
        scene_cfg.agent_2_front_camera = None

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Programmatic SLAM scene ready.")
    print(f"[INFO] Articulations: {list(scene.articulations.keys())}")
    print(f"[INFO] Sensors: {list(scene.sensors.keys())}")
    print(f"[INFO] Extras: {list(scene.extras.keys())}")

    step_idx = 0
    while simulation_app.is_running():
        if args_cli.steps > 0 and step_idx >= args_cli.steps:
            break
        if sim.is_stopped():
            break

        for articulation in scene.articulations.values():
            articulation.set_joint_position_target(articulation.data.default_joint_pos)
            articulation.write_data_to_sim()

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if args_cli.steps > 0 or sim.is_playing():
            step_idx += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        carb.log_error(f"Programmatic SLAM scene preview failed: {exc}")
        raise
    finally:
        simulation_app.close()
