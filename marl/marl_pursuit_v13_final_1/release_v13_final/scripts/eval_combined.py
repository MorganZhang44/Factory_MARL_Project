"""eval_combined.py — eval the demo's full pipeline: v13 policy for the
approach, switch to closing-maneuver fallback when agents are close.
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
from scripts.interactive_demo import compute_closing_actions
from scripts.eval_demo_fallback import step_with_lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/checkpoints_role_300k_v13/final.pt")
    p.add_argument("--config",   default="configs/mappo_role_intruder_300k.yaml")
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["env"]["intruder_speed"] = 0.0
    env = PursuitEnv(cfg, render_mode=None)

    obs_dim = int(cfg["env"].get("obs_dim", 22))
    action_lim = float(cfg["env"].get("agent_max_offset", 3.0))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", obs_dim))
    n = RunningMeanStd(shape=(obs_dim,))
    n.mean = ckpt["obs_norm_mean"]; n.var = ckpt["obs_norm_var"]; n.count = ckpt["obs_norm_count"]
    actor = Actor(obs_dim, 2, 64, map_half=action_lim)
    actor.load_state_dict(ckpt["actor"]); actor.eval()

    successes = 0
    cap_steps = []
    min_dists = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_min = 99.9
        cap_step = -1
        close_frames = 0
        for step in range(env.max_steps):
            d = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
            max_d = float(d.max())
            close_frames = close_frames + 1 if max_d <= 6.0 else 0
            fallback_active = (
                (max_d <= 2.0 and close_frames >= 1)
                or (max_d <= 3.0 and close_frames >= 5)
                or (max_d <= 4.0 and close_frames >= 10)
                or (max_d <= 6.0 and close_frames >= 20)
            )
            if fallback_active:
                actions, lock_flags = compute_closing_actions(
                    env, env.agent_pos.copy(), env.target_pos.copy(),
                    env.capture_radius,
                    agent_yaw=env.agent_yaw.copy(),
                    face_thresh=env.face_capture_thresh,
                )
            else:
                obs_n = n.normalize(obs)
                with torch.no_grad():
                    a, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
                actions = a.numpy()
                lock_flags = [False, False]

            cap, trunc, mind = step_with_lock(env, actions, lock_flags)
            ep_min = min(ep_min, mind)
            obs = env._get_obs()
            if cap:
                cap_step = step; break
            if trunc:
                break

        captured = (cap_step >= 0)
        successes += int(captured)
        if captured: cap_steps.append(cap_step)
        min_dists.append(ep_min)
        print(f"  Ep {ep+1:>2}/{args.episodes}  captured={captured}  step={cap_step:>3}  min_dist={ep_min:.2f}m")

    print()
    print(f"Combined policy + fallback eval ({args.episodes} eps, stationary intruder):")
    print(f"  Success rate: {successes/args.episodes*100:.1f}%")
    print(f"  Mean min-distance: {np.mean(min_dists):.2f}m")
    if cap_steps: print(f"  Mean capture step: {np.mean(cap_steps):.0f}")


if __name__ == "__main__":
    main()
