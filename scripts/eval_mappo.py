"""
eval_mappo.py
Evaluate a trained MAPPO policy with optional rendering.

Usage:
    python scripts/eval_mappo.py --checkpoint results/checkpoints/final.pt --render
    python scripts/eval_mappo.py --checkpoint results/checkpoints/final.pt --episodes 100
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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MAPPO pursuit policy")
    p.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    p.add_argument("--config",     default="configs/mappo_config.yaml")
    p.add_argument("--episodes",   type=int, default=20, help="Number of eval episodes")
    p.add_argument("--render",     action="store_true",  help="Render with matplotlib")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded: {args.checkpoint}  (step {ckpt.get('total_steps', '?'):,})")

    obs_dim = int(env_cfg.get("obs_dim", 13))

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    obs_norm.mean  = ckpt["obs_norm_mean"]
    obs_norm.var   = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]

    actor = Actor(
        obs_dim=obs_dim, action_dim=2, hidden_dim=64,
        map_half=env_cfg["map_half"],
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    render_mode = "human" if args.render else None
    env = PursuitEnv(cfg, render_mode=render_mode)

    # Metrics
    successes, times, ep_rewards = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_r   = 0.0
        done   = False

        while not done:
            obs_n  = obs_norm.normalize(obs)
            obs_t  = torch.FloatTensor(obs_n).to(device)
            with torch.no_grad():
                action, _ = actor.get_action(obs_t, deterministic=True)
            act_np = action.cpu().numpy()

            obs, rewards, terminated, truncated, info = env.step(act_np)
            done   = terminated or truncated
            ep_r  += float(rewards.mean())

        successes.append(info["captured"])
        times.append(info["step"])
        ep_rewards.append(ep_r)
        print(f"  Ep {ep+1:>3}/{args.episodes}  "
              f"captured={info['captured']}  "
              f"steps={info['step']:>3}  "
              f"reward={ep_r:.2f}")

    env.close()

    print("\n── Evaluation Summary ──────────────────────────")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Success rate:   {np.mean(successes)*100:.1f}%")
    print(f"  Mean steps:     {np.mean(times):.1f}")
    print(f"  Mean reward:    {np.mean(ep_rewards):.2f}")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
