# MARL Pursuit-and-Encircle (v9 release)

A two-agent multi-agent reinforcement learning system that learns to encircle and capture a moving intruder in a 2D bounded room with obstacles. Built on MAPPO (Multi-Agent PPO) with role-aware reward shaping and an adversarially co-trained intruder policy.

This is the **v9 release** — the best-performing checkpoint produced during the experimental sweep (v1 → v11). On the hardest test scenario (stationary intruder near a wall, both agents spawned on the same side) it achieves an **83 % capture rate** — up from a 60 % baseline.

---

## 1. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the interactive demo (mouse controls intruder)
python scripts/interactive_demo.py \
    --config configs/mappo_role_intruder_300k.yaml \
    --checkpoint checkpoints/v9_final.pt

# 3. Run a programmatic example (no pygame, prints obs/action for 3 frames)
python example_inference.py

# 4. Headless evaluation (no GUI), 30 episodes
python scripts/eval_static.py    --checkpoint checkpoints/v9_final.pt --episodes 30
python scripts/eval_near_wall.py --checkpoint checkpoints/v9_final.pt --episodes 30
python scripts/eval_demo_case.py --checkpoint checkpoints/v9_final.pt --episodes 30
```

The demo opens a 1280×800 pygame window. Move your mouse to control the red `YOU` star — the two dogs (red P = PURSUER, cyan E = ENCIRCLER) will try to encircle and capture you. Press `R` to reset, `ESC` to quit.

---

## 2. What's in this release

```
release_v9_final/
├── README.md                                ← this file
├── requirements.txt
├── example_inference.py                     ← programmatic API demo
├── checkpoints/
│   └── v9_final.pt                          ← trained policy (220 KB)
├── configs/
│   ├── mappo_role_intruder_300k.yaml        ← main training config
│   └── mappo_role_intruder_smoke.yaml       ← 4k-step smoke config
├── scripts/
│   ├── interactive_demo.py                  ← pygame demo with debug panel
│   ├── train_mappo.py                       ← training entry point
│   ├── eval_static.py                       ← stationary intruder eval
│   ├── eval_near_wall.py                    ← intruder spawned near walls
│   └── eval_demo_case.py                    ← hardest case (combined)
└── marl/
    ├── envs/pursuit_env.py                  ← env (gym API + custom hooks)
    ├── policies/
    │   ├── actor.py                         ← shared agent actor
    │   ├── critic.py                        ← centralized critic (CTDE)
    │   └── intruder_actor.py                ← intruder policy (co-trained)
    ├── rewards/pursuit_reward.py            ← per-role reward, intruder reward
    ├── trainers/mappo_trainer.py            ← MAPPO training loop
    ├── buffers/rollout_buffer.py            ← rollout buffer + GAE-λ
    └── utils/
        ├── astar.py                         ← A* on the obstacle grid
        ├── map_utils.py                     ← obstacle definitions + grid
        └── normalizer.py                    ← Welford running mean/var
```

---

## 3. Dependencies

- Python 3.10+ (tested on 3.11)
- PyTorch 2.0+ (CUDA optional; CPU works for inference)
- numpy, gymnasium, pyyaml
- pygame (interactive demo only)
- matplotlib (env render method only)

Install with `pip install -r requirements.txt`.

UTF-8 locale is required because configs contain non-ASCII characters. On Windows always launch Python with `PYTHONUTF8=1`:

```bash
PYTHONUTF8=1 python scripts/interactive_demo.py ...
```

---

## 4. The task

A 20 m × 20 m walled room (`map_half = 10`) contains two pursuer agents and one intruder. The agents must **simultaneously** satisfy:

1. Both agents are within `capture_radius = 1.5 m` of the intruder
2. The angle between (pursuer→intruder) and (encircler→intruder) is ≥ 90° (i.e. they encircle from opposite sides)

When both conditions hold the episode terminates with `captured = True`. Otherwise the episode runs for up to `max_steps = 500` steps (50 seconds at `dt = 0.1`).

---

## 5. Architecture

### 5.1 Per-agent observation (22 dimensions)

For agent `i`, the observation vector is:

| Index    | Field          | Range                             | Description                           |
|----------|----------------|-----------------------------------|---------------------------------------|
| `[0]`    | `agent_id`     | {0, 1}                            | which dog this is (symmetry-breaking) |
| `[1]`    | `role flag`    | {0, 1}                            | 0 = PURSUER · 1 = ENCIRCLER           |
| `[2:4]`  | `self_pos`     | `[-1, +1]`                        | own position / `map_half`             |
| `[4:6]`  | `self_vel`     | `[-1, +1]`                        | own velocity / `vel_scale`            |
| `[6:8]`  | `mate_pos`     | `[-1, +1]`                        | teammate position / `map_half`        |
| `[8:10]` | `mate_vel`     | `[-1, +1]`                        | teammate velocity / `vel_scale`       |
| `[10:12]`| `target_pos`   | `[-1, +1]`                        | intruder position / `map_half`        |
| `[12:14]`| `target_vel`   | `[-1, +1]`                        | intruder velocity / `vel_scale`       |
| `[14:22]`| `lidar (8 rays)` | `[0, 1]`                        | normalised distance to nearest wall   |

`vel_scale = max(agent_max_speed, intruder_speed) = 1.5`. Lidar shoots 8 rays at 45° spacing; value 1.0 means clear up to 8 m, 0.0 means a wall is right next to the agent.

### 5.2 Action

Each agent outputs a **relative offset subgoal** `(dx, dy)` in metres:

- Type: `np.float32`, shape `(n_agents=2, 2)`
- Range: `[-3, +3]` per axis, then norm-clipped to magnitude ≤ 3 m

The env adds the offset to the agent's current position to get an absolute subgoal, runs A* on the occupancy grid to compute a path, then advances the agent at most `agent_max_speed × dt = 0.15 m` along that path.

### 5.3 Roles (PURSUER / ENCIRCLER)

Each step the env decides which agent is the PURSUER (closer to target) and which is the ENCIRCLER (farther). To prevent oscillation, role switching only happens when:

- ≥ `role_cooldown_steps = 30` steps have passed since the last switch, **AND**
- the distance gap `|d₀ − d₁|` exceeds `role_gap_threshold = 1.0 m`

The current role is fed into the observation (index 1) so a single shared actor network can produce role-conditioned behaviour.

### 5.4 Reward (per agent, different per role)

```
PURSUER:
  + w_dist_pursuer   × Δd_self × 10        (dense distance progress)
  + w_pin_pursuer    × 𝟙[d_self ≤ 1.5]      (pin reward)
  + w_lag_pursuer    × max(0, d_self − 5)   (lag penalty)
  + (shared team terms below)

ENCIRCLER:
  + w_angle_encircler × ((1 − cos θ)/2)² × close_factor   (encirclement angle)
  + w_commit_encircler × commit_quality                   (commit-to-capture bonus)
  + w_dist_encircler  × Δd_self × 10                      (own distance progress)
  + (shared team terms below)

TEAM (added to BOTH roles):
  + w_capture × angle_quality                       (capture bonus)
  + w_time × time_fraction                          (decays with step count)
  + w_proximity × max(0, sep_thresh − d_agents)     (overlap penalty)
  + w_step                                          (constant time pressure)
  + w_copresence × encirclement_quality²            (both inside capture_radius bonus)
```

See `marl/rewards/pursuit_reward.py` for the exact formulae and `configs/mappo_role_intruder_300k.yaml` for the weights.

### 5.5 Intruder co-training

A separate actor + critic learns to evade. During training, episodes alternate between scripted (random walk + 40 % stationary frames + pinned-when-cornered) and policy-driven intruder, mixed by `intruder_hybrid_prob_*` (50 % scripted at start, 10 % at end). The intruder reward rewards survival and distance from the nearest pursuer; capture incurs a `−100` penalty.

### 5.6 Two-agent physical collision

Each dog has `agent_radius = 0.3 m`. The two bodies are kept apart by at least `2 × radius + 0.1 = 0.7 m` via a hard collision check inside the movement step (with X- and Y-axis sliding fallback, mirroring the wall sliding logic).

---

## 6. Performance

Each metric is the success rate over **30 evaluation episodes** with deterministic actor (no exploration noise).

| Scenario                                                           | v9 final |
|--------------------------------------------------------------------|----------|
| Open random spawn, stationary intruder                             | **83 %** |
| Stationary intruder, near walls/obstacles                          | **80 %** |
| Stationary near-wall intruder + agents both spawned on same side   | **83 %** |

Failures are concentrated in the corner geometry where physical encirclement to 90°+ is partially blocked. The interactive demo includes a deterministic fallback controller that drives the agents to ideal encirclement positions when they get stuck near the intruder; this raises practical capture rate close to 100 %.

---

## 7. Programmatic inference (input/output demo)

The simplest possible loop is:

```python
import yaml, torch, numpy as np
from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd

cfg  = yaml.safe_load(open("configs/mappo_role_intruder_300k.yaml", encoding="utf-8"))
ckpt = torch.load("checkpoints/v9_final.pt", map_location="cpu", weights_only=False)

actor = Actor(obs_dim=22, action_dim=2, hidden_dim=64,
              map_half=cfg["env"]["agent_max_offset"])
actor.load_state_dict(ckpt["actor"])
actor.eval()

obs_norm = RunningMeanStd(shape=(22,))
obs_norm.mean, obs_norm.var, obs_norm.count = (
    ckpt["obs_norm_mean"], ckpt["obs_norm_var"], ckpt["obs_norm_count"])

env = PursuitEnv(cfg, render_mode=None)
obs, info = env.reset()           # obs: (2, 22) np.float32

with torch.no_grad():
    obs_n = obs_norm.normalize(obs)
    action, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
action = action.numpy()           # (2, 2) np.float32, dx/dy ∈ [-3, 3]

obs, reward, term, trunc, info = env.step(action)
# reward: (2,) np.float32 — per-agent reward (per-role when use_roles=True)
# info: dict with keys
#   "captured"          (bool)
#   "min_dist"          (float, m)
#   "step"              (int, 1..max_steps)
#   "roles"             ((2,) int64 — 0=PURSUER, 1=ENCIRCLER)
#   "intruder_obs"      ((20,) np.float32 — for the intruder policy)
#   "intruder_reward"   (float)
```

Run `python example_inference.py` to see this in action with concrete numerical traces of the first 3 steps (observation breakdown, action output, reward, info).

### 7.1 Concrete sample input/output

Sample observation for agent 0 at step 0:
```
agent_id   : 0.000
role flag  : 0.000   # PURSUER
self_pos   : (-0.334, +0.153)
self_vel   : (+0.000, +0.000)
mate_pos   : (-0.214, +0.261)
mate_vel   : (+0.000, +0.000)
target_pos : (-0.478, +0.191)
target_vel : (+0.265, +0.755)
lidar      : [0.85, 0.28, 0.20, 0.28, 0.22, 0.32, 0.72, 0.58]
```

Sample action output:
```
agent 0 offset [dx, dy] = (-1.298, +1.590) m
agent 1 offset [dx, dy] = (-0.592, +0.086) m
```

Sample reward + info after step:
```
reward      : (+1.381, +1.456)    # per-agent
captured    : False
min_dist    : 1.386 m
roles       : [0, 1]              # 0 is pursuer, 1 is encircler
```

---

## 8. Training your own checkpoint

```bash
# Smoke test (~30 sec on CPU)
python scripts/train_mappo.py \
    --config configs/mappo_role_intruder_smoke.yaml \
    --device cpu \
    --save-dir results/smoke

# Full training (~25 min on a single RTX, ~50 min on CPU)
python scripts/train_mappo.py \
    --config configs/mappo_role_intruder_300k.yaml \
    --device cuda \
    --save-dir results/my_run

# Resume / fine-tune from an existing checkpoint
python scripts/train_mappo.py \
    --config configs/mappo_role_intruder_300k.yaml \
    --device cuda \
    --save-dir results/my_run_v2 \
    --resume checkpoints/v9_final.pt
```

Training writes intermediate ckpts every `save_interval = 100,000` steps to `--save-dir`. The final ckpt is named `final.pt`.

### 8.1 What's in a checkpoint

```python
{
  "actor":               state_dict,          # shared agent actor
  "critic":              state_dict,          # centralized critic (42→1)
  "obs_norm_mean":       np.array (22,),
  "obs_norm_var":        np.array (22,),
  "obs_norm_count":      float,
  "total_steps":         int,
  "obs_dim":             22,
  "use_roles":           True,
  "intruder_actor":      state_dict,          # intruder evader (only when use_intruder_policy)
  "intruder_critic":     state_dict,
  "intruder_obs_norm_*": ...,
  "intruder_obs_dim":    20,
  "intruder_max_offset": 2.0,
}
```

For inference you only need `actor` + `obs_norm_*` + `obs_dim`.

---

## 9. Evaluation tools

Three diagnostic scripts test the policy under increasing difficulty:

| Script                  | Setup                                                                |
|-------------------------|----------------------------------------------------------------------|
| `eval_static.py`        | intruder stationary, random spawn                                    |
| `eval_near_wall.py`     | intruder stationary AND spawned within 1 m of a wall/obstacle        |
| `eval_demo_case.py`     | the above + agents both spawned in a single 90° wedge from intruder  |

All three accept `--checkpoint`, `--config`, `--episodes`. They print per-episode results and a final summary block:

```
NEAR-WALL stationary-intruder eval (30 eps):
  Success rate: 80.0%
  Mean min-dist:   0.41m
  Mean final-dist: 0.95m
  Mean ep length:  161 steps
```

---

## 10. Interactive demo features

The pygame demo (`scripts/interactive_demo.py`) provides:

- **Mouse-controlled intruder** with EMA-smoothed velocity matched to training distribution
- **Role-coloured agents** (red P / cyan E) with role labels updating live
- **Live debug panel** showing per-frame:
  - Capture conditions (d₁, d₂, cos θ, angle θ — green/red coloured by satisfied/not)
  - Per-agent state (pos, vel, action output, subgoal)
  - Intruder state
  - 8-ray LiDAR mini-radar diagrams per agent
  - Closing-fallback controller status (`RL POLICY` / `FALLBACK (geom)` badge)
- **Closing-maneuver fallback**: when both agents are inside striking range for several frames without capturing, a deterministic geometric controller takes over and drives them to ideal encirclement positions (`d ≈ 1.05 m`, antipodal). Capture is then mathematically guaranteed.
- **A* path visualisation** as dashed lines per agent, plus subgoal markers

Activation thresholds for the fallback (in `interactive_demo.py`):
- `max_d ≤ 2 m` for 5 frames (0.25 s at FPS=20), or
- `max_d ≤ 3 m` for 12 frames (0.6 s), or
- `max_d ≤ 4 m` for 25 frames (1.25 s)

---

## 11. Configuration reference

All knobs live in YAML config files (`configs/`). Key sections:

```yaml
env:
  capture_radius: 1.5         # both agents must be within this for capture
  agent_max_speed: 1.5        # m/s
  agent_max_offset: 3.0       # max per-step relative subgoal magnitude
  intruder_speed: 1.2         # m/s
  intruder_max_offset: 2.0
  agent_radius: 0.3           # body radius for hard collision check
  use_roles: true
  role_cooldown_steps: 30
  role_gap_threshold: 1.0
  intruder_near_obstacle_prob: 0.5   # 50 % of episodes spawn intruder near a wall
  same_side_spawn_prob: 0.4          # 40 % spawn both agents in one 90° wedge

mappo:
  rollout_steps: 1024
  mini_batch_size: 256
  total_timesteps: 800_000
  use_intruder_policy: true
  intruder_hybrid_prob_start: 0.5    # scripted vs RL intruder mix
  intruder_hybrid_prob_end:   0.1
  intruder_hybrid_decay_frac: 0.5

reward:
  w_dist_pursuer:    1.5
  w_pin_pursuer:     1.0
  pin_threshold:     1.5
  w_angle_encircler: 4.0
  w_dist_encircler:  1.2
  w_commit_encircler: 5.0
  commit_radius_mul: 1.5
  w_capture:         400.0
  w_time:            200.0
  w_copresence:      3.0
  w_proximity:      -1.0
  w_step:           -0.12
  ...
```

Tweak any of these and retrain (or fine-tune from `v9_final.pt`).

---

## 12. Known limitations

- **Single map.** Geometry is hard-coded in `marl/utils/map_utils.py` (a 20 m × 20 m room with three internal pillars). Adding new maps requires editing that file.
- **Two pursuers + one intruder only.** The role machinery assumes `n_agents = 2`. More agents would require generalising role assignment.
- **A\* re-plans every step.** Cheap on the 40 × 40 grid but doesn't scale to larger maps without optimisation.
- **Demo capture rule is strict by design** (90° + both within 1.5 m). The closing-maneuver fallback compensates so demo experience is still close to 100 % capture; pure RL alone caps out around 83 %.
- **Same-side approach with intruder in a corner** is geometrically the hardest case. Even with the fallback, very tight corners (intruder pinned in a 60° pocket) may take a few seconds to resolve as A* routes the encircler around obstacles.

---

## 13. Citation / context

This release is the v9 milestone of an experimental sweep that ran v1 → v11. v9 is the best surviving checkpoint; v10 / v11 attempted to make the intruder smarter and failed within the training budget. The full back-history (training curves, ablations) lives in the parent project's `backups/` directory.
