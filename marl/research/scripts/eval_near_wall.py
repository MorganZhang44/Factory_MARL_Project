"""
eval_near_wall.py — focused test on the failure case: intruder is stationary
AND placed within 1 m of a wall/obstacle. Reports capture rate vs baseline
positions. Use this to compare versions on the specific scenario users see.

Usage:
    python scripts/eval_near_wall.py --checkpoint <path> --episodes 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mappo_role_intruder_300k.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default=str(DEFAULT_CONFIG))
    p.add_argument("--episodes",   type=int, default=30)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force the failure mode every episode:
    cfg["env"]["intruder_speed"] = 0.0                  # stationary
    cfg["env"]["intruder_near_obstacle_prob"] = 1.0     # always near a wall

    env_cfg = cfg["env"]
    obs_dim = int(env_cfg.get("obs_dim", 22))
    action_lim = float(env_cfg.get("agent_max_offset", 3.0))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", obs_dim))

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    obs_norm.mean = ckpt["obs_norm_mean"]
    obs_norm.var = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]

    actor = Actor(obs_dim, 2, 64, map_half=action_lim)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = PursuitEnv(cfg, render_mode=None)

    successes = 0
    min_dists = []
    final_dists = []
    steps_taken = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_min = 99.9
        while not done:
            obs_n = obs_norm.normalize(obs)
            with torch.no_grad():
                action, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
            obs, _, term, trunc, info = env.step(action.numpy())
            done = term or trunc
            ep_min = min(ep_min, info["min_dist"])
        successes += int(info["captured"])
        min_dists.append(ep_min)
        final_dists.append(info["min_dist"])
        steps_taken.append(info["step"])
        print(f"  Ep {ep+1:>2}/{args.episodes}  captured={info['captured']}  "
              f"steps={info['step']:>3}  min={ep_min:.2f}m  fin={info['min_dist']:.2f}m")

    print()
    print(f"NEAR-WALL stationary-intruder eval ({args.episodes} eps):")
    print(f"  Success rate: {successes/args.episodes*100:.1f}%")
    print(f"  Mean min-dist:   {np.mean(min_dists):.2f}m")
    print(f"  Mean final-dist: {np.mean(final_dists):.2f}m")
    print(f"  Mean ep length:  {np.mean(steps_taken):.0f} steps")


if __name__ == "__main__":
    main()
