
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Isaac Lab Scene")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene
from environment.config.scene_cfg import SurveillanceSceneCfg

def main():
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    
    scene_cfg = SurveillanceSceneCfg(num_envs=1, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("\n" + "="*50)
    print("SCENE INSPECTION")
    print("="*50)
    print(f"Articulations: {list(scene.articulations.keys())}")
    print(f"Sensors:       {list(scene.sensors.keys())}")
    # print(f"Rigid Objects: {list(scene.rigid_objects.keys())}")
    print("="*50 + "\n")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
