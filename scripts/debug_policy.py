"""
debug_policy.py
Merged debug script for policy inspection.

Usage:
  python3 scripts/debug_policy.py            # default: moving target
  python3 scripts/debug_policy.py --fixed    # stationary target at [0, 0]
"""
import sys
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd


def load_model(cfg):
    CKPT_PATH = "results/checkpoints/final.pt"
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    obs_norm = RunningMeanStd(shape=(int(cfg["env"].get("obs_dim", 21)),))
    obs_norm.mean  = ckpt["obs_norm_mean"]
    obs_norm.var   = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]

    MAP_HALF = float(cfg["env"]["map_half"])
    actor = Actor(int(cfg["env"].get("obs_dim", 21)), 2, 64, map_half=MAP_HALF)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, obs_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true",
                        help="Fix intruder at [0,0] (stationary target test)")
    parser.add_argument("--steps", type=int, default=40,
                        help="Number of steps to run (default: 40)")
    args = parser.parse_args()

    with open("configs/mappo_config.yaml") as f:
        cfg = yaml.safe_load(f)

    actor, obs_norm = load_model(cfg)
    env = PursuitEnv(cfg, render_mode=None)
    env.reset(seed=42)

    if args.fixed:
        print("=== Fixed Intruder Test (stationary at [0.0, 0.0]) ===")
        env.agent_pos[0] = np.array([2.0, -3.0])
        env.agent_pos[1] = np.array([-1.0, -3.0])
    else:
        print("=== Moving Intruder Test (target moves UP at 1.0 m/s) ===")
        env.target_vel = np.array([0.0, 1.0]) * env.intruder_spd

    print(f"Obs Norm Mean: {obs_norm.mean.round(3)}\n")

    for step in range(1, args.steps + 1):
        if args.fixed:
            env.target_pos = np.array([0.0, 0.0])
            env.target_vel = np.array([0.0, 0.0])
        else:
            # Move target upward at constant speed
            env.target_pos = env.target_pos + np.array([0.0, 1.0]) * 0.1
            env.target_vel = np.array([0.0, 1.0])

        obs   = env._get_obs()
        obs_n = obs_norm.normalize(obs)

        with torch.no_grad():
            act, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
        act_np = act.numpy()

        _, _, done, _, info = env.step(act_np)

        dists = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
        print(f"Step {step:2d}: Target {env.target_pos.round(2)} | Vel {env.target_vel.round(2)}")
        print(f"        A1 {env.agent_pos[0].round(2)}  Act: {act_np[0].round(3)}")
        print(f"        A2 {env.agent_pos[1].round(2)}  Act: {act_np[1].round(3)}")
        print(f"        MinDist: {np.min(dists):.3f}")

        if done:
            print(f"\n>>> CAPTURED in {step} steps! <<<")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
