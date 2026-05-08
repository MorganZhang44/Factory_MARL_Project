"""
example_inference.py — minimal end-to-end example showing how to load the
trained MAPPO policy and run it from your own code.

Run from the release root:
    python example_inference.py

This script does NOT use pygame. It runs ONE episode in headless mode and
prints the per-step input (observation) and output (action) for the first
3 steps so you can see the data shapes and value ranges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make the local marl/ package importable
sys.path.insert(0, str(Path(__file__).parent))

from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd


# ─── 1. Load config and checkpoint ───────────────────────────────────────
CFG_PATH  = "configs/mappo_role_intruder_300k.yaml"
CKPT_PATH = "checkpoints/v9_final.pt"

with open(CFG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

env_cfg     = cfg["env"]
obs_dim     = int(env_cfg.get("obs_dim", 22))           # 22D agent observation
action_lim  = float(env_cfg.get("agent_max_offset", 3.0))
map_half    = float(env_cfg["map_half"])
cap_radius  = float(env_cfg["capture_radius"])

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print(f"Loaded checkpoint trained for {ckpt.get('total_steps', '?'):,} steps")
print(f"  obs_dim     : {obs_dim}")
print(f"  action_dim  : 2  (relative offset [dx, dy] in metres)")
print(f"  action range: [-{action_lim}, +{action_lim}] per axis")


# ─── 2. Recreate the actor and the obs normaliser ────────────────────────
obs_norm = RunningMeanStd(shape=(obs_dim,))
obs_norm.mean  = ckpt["obs_norm_mean"]
obs_norm.var   = ckpt["obs_norm_var"]
obs_norm.count = ckpt["obs_norm_count"]

actor = Actor(obs_dim=obs_dim, action_dim=2, hidden_dim=64, map_half=action_lim)
actor.load_state_dict(ckpt["actor"])
actor.eval()


# ─── 3. Make an env, run one episode, log first 3 steps ──────────────────
env = PursuitEnv(cfg, render_mode=None)
obs, info = env.reset()

print("\n" + "=" * 72)
print("Step-by-step input / output trace (first 3 frames):")
print("=" * 72)

for step in range(3):
    print(f"\n── Step {step} ──────────────────────────────────")
    print(f"INPUT  obs shape:  {obs.shape}      # (n_agents=2, obs_dim=22)")
    print(f"OBSERVATION breakdown for agent 0:")
    o = obs[0]
    print(f"  obs[0]  agent_id   : {o[0]:.3f}")
    print(f"  obs[1]  role flag  : {o[1]:.3f}   # 0=PURSUER  1=ENCIRCLER")
    print(f"  obs[2:4]  self_pos : ({o[2]:+.3f}, {o[3]:+.3f})")
    print(f"  obs[4:6]  self_vel : ({o[4]:+.3f}, {o[5]:+.3f})")
    print(f"  obs[6:8]  mate_pos : ({o[6]:+.3f}, {o[7]:+.3f})")
    print(f"  obs[8:10] mate_vel : ({o[8]:+.3f}, {o[9]:+.3f})")
    print(f"  obs[10:12] target_pos: ({o[10]:+.3f}, {o[11]:+.3f})")
    print(f"  obs[12:14] target_vel: ({o[12]:+.3f}, {o[13]:+.3f})")
    print(f"  obs[14:22] lidar (8 rays, 0=close 1=clear):")
    print(f"    {np.round(o[14:22], 2).tolist()}")

    # Normalise then forward
    obs_n  = obs_norm.normalize(obs)
    obs_t  = torch.FloatTensor(obs_n)
    with torch.no_grad():
        action, _ = actor.get_action(obs_t, deterministic=True)
    action_np = action.numpy()
    print(f"OUTPUT action shape: {action_np.shape}    # (n_agents=2, 2)")
    print(f"  agent 0 offset [dx, dy] = ({action_np[0,0]:+.3f}, {action_np[0,1]:+.3f})  m")
    print(f"  agent 1 offset [dx, dy] = ({action_np[1,0]:+.3f}, {action_np[1,1]:+.3f})  m")

    # Step env
    obs, reward, terminated, truncated, info = env.step(action_np)
    print(f"REWARD per agent : ({reward[0]:+.3f}, {reward[1]:+.3f})")
    print(f"INFO  captured    : {info['captured']}")
    print(f"      min_dist    : {info['min_dist']:.3f} m")
    print(f"      roles       : {info['roles'].tolist()}  (0=P, 1=E)")
    if terminated:
        print(f"   ✓ Episode terminated (capture)")
        break

print("\n" + "=" * 72)
print(f"Final: captured={info['captured']}  step={info['step']}  min_dist={info['min_dist']:.2f} m")
