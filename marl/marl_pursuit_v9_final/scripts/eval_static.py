"""
eval_static.py — diagnose v5 policy on a STATIONARY intruder.

Forces intruder_speed=0 so the env's scripted intruder essentially never moves
(any tiny noise is normalized to 0 magnitude). Reports capture rate and how
close agents get.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/mappo_role_intruder_300k.yaml")
    p.add_argument("--episodes",   type=int, default=20)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # FORCE stationary intruder
    cfg["env"]["intruder_speed"] = 0.0

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
              f"steps={info['step']:>3}  min_dist={ep_min:.2f}m  final_dist={info['min_dist']:.2f}m")

    print()
    print(f"Stationary-intruder eval ({args.episodes} eps):")
    print(f"  Success rate: {successes/args.episodes*100:.1f}%")
    print(f"  Mean min-distance:   {np.mean(min_dists):.2f}m")
    print(f"  Mean final-distance: {np.mean(final_dists):.2f}m")
    print(f"  Mean episode length: {np.mean(steps_taken):.0f} steps")


if __name__ == "__main__":
    main()
