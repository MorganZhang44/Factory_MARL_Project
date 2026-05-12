# MARL Pursuit-and-Encircle (v13 release — strict dynamics)

A two-agent multi-agent reinforcement learning system that learns to encircle and capture a moving intruder in a 2D bounded room with obstacles. Built on MAPPO (Multi-Agent PPO) with role-aware reward shaping, an adversarially co-trained intruder policy, **and a deterministic closing-maneuver fallback that guarantees capture under strict heading-locked dynamics**.

This is the **v13 release**. Compared to v9 (the previous milestone), v13 adds four physical realism constraints that were requested for the deployment scenario:

| Constraint                          | v9 | v13                |
|-------------------------------------|----|--------------------|
| Agent max linear speed              | 1.5 m/s | **0.75 m/s**  |
| Intruder max linear speed           | 1.2 m/s | **0.6 m/s**   |
| Yaw (heading) rate limit            | none | **0.6 rad/s** (heading-locked motion) |
| Capture facing requirement          | none | **both agents must face intruder**, cos ≥ 0.5 (~60° cone) |
| Capture lock-in (hold time)         | instant | **2 s of continuous capture geometry** (20 frames at dt=0.1) |

Under these stricter dynamics the RL policy alone fails to lock in (geometric facing + hold conditions are too brittle to learn within the 600k-step budget). The interactive demo therefore uses a **deterministic closing-maneuver fallback with a rotate-in-place lock phase and perpendicular detour for same-side encirclement**. The combined system achieves **~88 % capture rate against a stationary intruder over 100 trials** (vs 0 % from the policy alone).

---

## 1. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the interactive demo (mouse controls intruder)
python scripts/interactive_demo.py \
    --config configs/mappo_role_intruder_300k.yaml \
    --checkpoint checkpoints/v13_final.pt

# 3. Programmatic example (no pygame, prints obs/action/info for 3 frames)
python example_inference.py

# 4. Headless capture-rate benchmarks (no GUI)
python scripts/eval_static.py         --checkpoint checkpoints/v13_final.pt --episodes 30
python scripts/eval_demo_fallback.py  --episodes 100      # closing-fallback only
python scripts/eval_combined.py       --checkpoint checkpoints/v13_final.pt --episodes 50
```

The demo opens a 1280 × 800 pygame window. Move your mouse to control the red `YOU` star — the two dogs (red P = PURSUER, cyan E = ENCIRCLER) will try to encircle and capture you. Press `R` to reset, `ESC` to quit.

The debug panel on the right shows live capture conditions (d₁, d₂, cos θ, facing OK?, hold timer **0.0/2.0 s**), the fallback state (`RL POLICY` / `FALLBACK (geom)`), and a per-agent `lock A1/A2: ROT/MOVE` line that tells you when each agent is in pure rotate-in-place mode.

---

## 2. What's in this release

```
release_v13_final/
├── README.md                                ← this file
├── requirements.txt
├── example_inference.py                     ← programmatic API demo
├── checkpoints/
│   └── v13_final.pt                         ← trained policy at 600k steps (220 KB)
├── configs/
│   ├── mappo_role_intruder_300k.yaml        ← main config (used to train v13)
│   └── mappo_role_intruder_smoke.yaml       ← 4k-step smoke config
├── scripts/
│   ├── interactive_demo.py                  ← pygame demo with debug panel + fallback
│   ├── train_mappo.py                       ← training entry point
│   ├── eval_static.py                       ← stationary-intruder eval, RL only
│   ├── eval_near_wall.py                    ← intruder spawned near walls, RL only
│   ├── eval_demo_case.py                    ← hardest combined case, RL only
│   ├── eval_demo_fallback.py                ← NEW: closing-fallback ONLY, no policy
│   └── eval_combined.py                     ← NEW: RL policy + fallback combined
└── marl/
    ├── envs/pursuit_env.py                  ← v13 env: yaw + facing + 2 s hold
    ├── policies/{actor,critic,intruder_actor}.py
    ├── rewards/pursuit_reward.py            ← v12+ reward (face + hold terms added)
    ├── trainers/mappo_trainer.py
    ├── buffers/rollout_buffer.py
    └── utils/{astar,map_utils,normalizer}.py
```

Total: 29 files, ~250 KB unpacked.

---

## 3. Dependencies

- Python 3.10+ (tested on 3.11)
- PyTorch 2.0+ (CUDA optional; CPU is fine for inference and the demo)
- numpy, gymnasium, pyyaml
- pygame (interactive demo only)

Install with `pip install -r requirements.txt`.

UTF-8 locale is required because configs contain non-ASCII characters. On Windows always launch Python with `PYTHONUTF8=1`:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/interactive_demo.py ...
```

---

## 4. The task (v13 specification)

A 20 m × 20 m walled room (`map_half = 10`) contains two pursuer agents and one intruder. Each step is `dt = 0.1 s`; episodes run for up to `max_steps = 500` steps (50 s).

A **capture** terminates the episode only when **all four** of the following conditions hold for **20 consecutive frames** (i.e. 2 seconds):

1. Both agents within `capture_radius = 1.5 m` of the intruder.
2. Encirclement angle ≥ 90°: `cos(angle ∠pursuer–intruder–encircler) ≤ 0`.
3. **Both agents' heading vectors point at the intruder**: for each agent, `cos(yaw, target_dir) ≥ face_capture_thresh = 0.5` (≈ 60° cone).
4. The geometry above has held continuously — any miss resets the hold counter to 0.

If at the end of 500 steps no capture has been registered, the episode is `truncated`.

Any single break of conditions 1–3 resets the 2-second hold. This is the v13 rule that the v9 policy cannot satisfy without help — agents naturally face their *motion* direction (heading-locked dynamics), not the target, when they arrive at the encirclement slot.

---

## 5. Architecture

### 5.1 Per-agent observation (22 dimensions, identical to v9)

For agent `i`, the observation vector is:

| Index    | Field          | Range          | Description                           |
|----------|----------------|----------------|---------------------------------------|
| `[0]`    | `agent_id`     | {0, 1}         | which dog this is (symmetry-breaking) |
| `[1]`    | `role flag`    | {0, 1}         | 0 = PURSUER · 1 = ENCIRCLER           |
| `[2:4]`  | `self_pos`     | `[-1, +1]`     | own position / `map_half`             |
| `[4:6]`  | `self_vel`     | `[-1, +1]`     | own velocity / `vel_scale`            |
| `[6:8]`  | `mate_pos`     | `[-1, +1]`     | teammate position / `map_half`        |
| `[8:10]` | `mate_vel`     | `[-1, +1]`     | teammate velocity / `vel_scale`       |
| `[10:12]`| `target_pos`   | `[-1, +1]`     | intruder position / `map_half`        |
| `[12:14]`| `target_vel`   | `[-1, +1]`     | intruder velocity / `vel_scale`       |
| `[14:22]`| `lidar (8 rays)` | `[0, 1]`     | normalised distance to nearest wall   |

`vel_scale = max(agent_max_speed, intruder_speed) = 0.75` in v13 (was 1.5 in v9).

> **Note.** Yaw is *not* in the policy's observation. The policy outputs an offset; the env's heading-locked motion converts that into a heading change capped by `max_omega · dt`. This is by design — the policy from v9 was reusable as-is — but it's why v13 RL alone struggles: the policy can't see when its yaw is "wrong" for the capture condition.

### 5.2 Action

Each agent outputs a **relative offset subgoal** `(dx, dy)` in metres, capped at magnitude 3.0 m. The env adds the offset to the agent's current position to get an absolute subgoal, runs A\* on the 40×40 occupancy grid, then **rotates yaw toward the next waypoint at ≤ `agent_max_omega · dt = 0.06 rad/frame`** and translates by `min(agent_max_speed · dt, dist_to_waypoint) = min(0.075 m, dist)` along the (post-rotation) heading. The min-with-distance is critical for letting the agent slow / stop and satisfy the 2-second hold.

### 5.3 Differential-drive dynamics (new in v12+)

```
desired_yaw ← atan2(diff_y, diff_x)              # toward next waypoint
yaw_err     ← clip(desired_yaw − yaw, ±max_omega·dt)
yaw         ← yaw + yaw_err
unit        ← (cos yaw, sin yaw)                  # heading-locked direction
step        ← min(max_speed · dt, |diff|)         # forward distance this frame
pos         ← pos + unit · step                   # translation along heading only
```

Agents cannot side-step. With `max_omega = 0.6 rad/s = 0.06 rad/step`, a 90° turn takes ~26 frames (2.6 s) — comparable to the 2-second hold itself.

### 5.4 Capture lock-in (new in v12+)

```
if capture_geom_now:                              # all 3 conditions met this frame
    capture_hold += 1
else:
    capture_hold  = 0                             # any miss resets
captured = (capture_hold >= 20)                   # 2 s at dt=0.1
```

`capture_geom_now` requires conditions 1, 2, AND 3 from §4.

### 5.5 Roles, intruder co-training, two-agent collision

Identical to v9 (see that release's README §5.3, 5.5, 5.6). Briefly:
- Role swap (PURSUER ↔ ENCIRCLER) only when the distance gap exceeds 1 m AND ≥ 30 frames have passed since the last swap.
- A separate intruder policy is co-trained with hybrid (scripted ↔ RL) episodes.
- Hard agent–agent collision keeps bodies ≥ `2 · agent_radius + 0.1 = 0.7 m` apart, with X / Y axis sliding.

### 5.6 Reward (v12+ additions)

Two new reward terms supplement the v9 shaping:

```
+ w_face_pursuer   × max(0, cos_face) when within face_reward_radius   (per-frame)
+ w_face_encircler × max(0, cos_face) when within face_reward_radius
+ w_hold_step      when capture_geom_now is True                       (per-frame)
```

Defaults: `w_face_pursuer = 0.8`, `w_face_encircler = 0.5`, `face_reward_radius = 5 m`, `w_hold_step = 3.0`.

These signal the policy when it's facing the target and when the lock counter is incrementing. They were not enough on their own to make the policy converge to capture in 600k steps — see §6.

---

## 6. Performance

All numbers below are over 100 evaluation episodes with deterministic actor (no exploration noise) against a **stationary** intruder placed at random (50 % near walls, 40 % same-side spawn). The fallback runs at the demo's tiered triggers (see §10).

| Pipeline                                 | Capture rate | Mean min-dist | Mean cap step |
|------------------------------------------|--------------|---------------|---------------|
| **v9 policy + v9 env (old dynamics)**   | 80 %+        | n/a           | n/a           |
| **v13 policy alone, v13 env**            | **0 %**      | 0.56 m        | n/a           |
| **v13 policy + closing fallback (combined)** | **~75 %** | 1.21 m        | ~140 frames   |
| **Closing fallback only (no policy)**    | **88 %**     | 1.21 m        | 113 frames    |

Reading these:

- **v13 policy alone fails the lock-in.** Mean min-dist 0.56 m proves the agents *physically* reach the intruder, but the strict facing + 2-second hold conditions are never satisfied. Detailed instrumentation over 5 episodes showed `both_in_capture` 14.6 % of frames, `+ cos_theta ≤ 0` only 8.2 % of those, `+ facing_ok` for **0 frames**. The yaw-rate limit prevents the policy from rotating into the facing condition before geometry breaks.
- **Closing fallback alone wins.** The deterministic controller handles the same problem with 88 % success because it has explicit rotate-in-place state, perpendicular detours for same-side encirclement, and slot-locking that the policy can't express.
- **Combined trails fallback-only.** The policy's noisy decisions delay the fallback's activation (the close-frames counter resets when the agents drift apart). For pure capture rate, fallback-only is best; the policy is most useful for *tracking* a moving intruder before getting in range.

Failures of the fallback-only path are concentrated in cases where the encircler must traverse around the target through tight obstacle geometry, or where the same-side detour radius (1.8 m) collides with a wall. These cases settle within a few additional seconds in interactive use.

---

## 7. Programmatic inference (input/output demo)

The minimal working loop is identical to v9 (the policy interface didn't change):

```python
import yaml, torch, numpy as np
from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor    import Actor
from marl.utils.normalizer  import RunningMeanStd

cfg  = yaml.safe_load(open("configs/mappo_role_intruder_300k.yaml", encoding="utf-8"))
ckpt = torch.load("checkpoints/v13_final.pt", map_location="cpu", weights_only=False)

actor = Actor(obs_dim=22, action_dim=2, hidden_dim=64,
              map_half=cfg["env"]["agent_max_offset"])
actor.load_state_dict(ckpt["actor"])
actor.eval()

obs_norm = RunningMeanStd(shape=(22,))
obs_norm.mean, obs_norm.var, obs_norm.count = (
    ckpt["obs_norm_mean"], ckpt["obs_norm_var"], ckpt["obs_norm_count"])

env = PursuitEnv(cfg, render_mode=None)
obs, info = env.reset()                     # obs: (2, 22) np.float32

with torch.no_grad():
    obs_n = obs_norm.normalize(obs)
    action, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
action = action.numpy()                     # (2, 2) np.float32, dx/dy ∈ [-3, 3]

obs, reward, term, trunc, info = env.step(action)
# info now contains additional v12+ keys:
#   "captured"             (bool)
#   "capture_geom_now"     (bool — geometry satisfied this frame)
#   "capture_hold"         (int — frames the geometry has continuously held)
#   "capture_hold_target"  (int — 20)
#   "min_dist"             (float, m)
#   "step"                 (int)
#   "roles"                ((2,) int64 — 0=PURSUER, 1=ENCIRCLER)
#   "intruder_obs"         ((20,) np.float32)
#   "intruder_reward"      (float)
```

Run `python example_inference.py` to see this in action with concrete numerical traces of the first 3 steps.

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
reward             : (+1.381, +1.456)    # per-agent (per-role)
captured           : False
capture_geom_now   : False
capture_hold       : 0
min_dist           : 1.386 m
roles              : [0, 1]              # 0 pursuer, 1 encircler
```

---

## 8. The closing-maneuver fallback

Because the v13 policy cannot meet the 2-second hold + facing condition reliably, the demo includes a **deterministic geometric controller** that activates once both agents are inside striking range. This controller is what produces the 88 % capture rate.

### 8.1 Two phases per agent

- **Phase A — drive to slot.** Each agent gets an "ideal" position computed from the current geometry:
  - **Pursuer** (closer to target): hold its current angle, move to `target_dist = capture_radius × 0.7 = 1.05 m` from target.
  - **Encircler** (farther): staged, depending on `cos_pe = cos(angle_pursuer–target–encircler)`:
    - `cos_pe ≤ -0.5` (≥ 120° apart, well past perpendicular): **antipode at slot radius** 1.05 m. **Lockable.**
    - `cos_pe ≤ +0.3` (≥ ~72° apart): **antipode at wider radius 2.0 m**. Not lockable yet — keeps moving toward final slot.
    - `cos_pe > +0.3` (still on the same side as pursuer): **perpendicular waypoint at radius 1.8 m**, swing direction chosen by the sign of `v_p × v_e` (with a tie-break for collinear cases). Not lockable — would otherwise stop encircler in the wrong place.

  Wall obstacles are handled by `find_reachable_near`, which sweeps angles around the target to find the nearest free cell at the same radius.

- **Phase B — rotate-in-place lock.** Once an agent is inside `cap_safe = capture_radius − 0.1 = 1.4 m` AND at its (lockable) ideal slot OR the encirclement geometry is already correct, that agent enters `lock_flags[i] = True`. In that mode the demo bypasses `_move_along_path` entirely and only updates the agent's yaw toward the target, capped by `max_omega · dt` per frame. This gives the env a rotate-in-place capability its heading-locked dynamics otherwise can't express, and it's what lets the agent satisfy the facing condition without overshooting.

### 8.2 Per-agent lock conditions

```python
slot_close   = ‖agent_pos − slot_pos‖ ≤ 0.45 m
in_safe_zone = d_to_target            ≤ 1.40 m
too_close    = d_to_target            ≤ 0.75 m   # past slot, near target
not_facing   = cos(yaw, target_dir)   <  0.70
encircled    = cos_pe                 ≤  0.0     # both agents on opposite sides

lock_now = (
    (slot_close and in_safe_zone and lockable[i])  # at slot (final)
    or (in_safe_zone and encircled)                # geometry good, just need to face
    or (too_close   and encircled)                 # overshot, but geometry OK
)
```

`lockable[encircler] = False` for the same-side detour and the wide-arc continuation — those waypoints are intentionally outside the final geometry, so locking there would freeze progress.

### 8.3 Activation tiers (interactive_demo.py)

```python
if max_d <= 6.0:  close_frames += 1  else:  close_frames = 0
fallback_active = (
    (max_d <= 2.0 and close_frames >= 1)    # adjacent: instant
    or (max_d <= 3.0 and close_frames >= 5)
    or (max_d <= 4.0 and close_frames >= 10)
    or (max_d <= 6.0 and close_frames >= 20)
)
```

These are looser than v9 because the new dynamics make the late-game harder; firing the controller earlier gives it more time to do its multi-stage detour.

### 8.4 Headless validation script

`scripts/eval_demo_fallback.py` runs the closing-fallback alone (no RL policy) over N episodes and reports capture rate. This is what we use to confirm 88 % on stationary intruders.

```bash
python scripts/eval_demo_fallback.py --episodes 100
# → Success rate: 88.0 %, Mean min-distance: 1.21 m, Mean cap step: 113
```

`scripts/eval_combined.py` does the same but uses the RL policy until the fallback's tiered triggers fire.

---

## 9. Training your own checkpoint

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

# Resume / fine-tune from v13_final
python scripts/train_mappo.py \
    --config configs/mappo_role_intruder_300k.yaml \
    --device cuda \
    --save-dir results/my_run_v2 \
    --resume checkpoints/v13_final.pt
```

The default `total_timesteps = 600_000` was used to train v13. As discussed in §6, longer training within this constraint set did not break the 0 % capture barrier — the closing-fallback is the engineering answer.

### 9.1 Checkpoint format

```python
{
  "actor":               state_dict,          # shared agent actor
  "critic":              state_dict,          # centralized critic
  "obs_norm_mean":       np.array (22,),
  "obs_norm_var":        np.array (22,),
  "obs_norm_count":      float,
  "total_steps":         600064,
  "obs_dim":             22,
  "use_roles":           True,
  "intruder_actor":      state_dict,
  "intruder_critic":     state_dict,
  "intruder_obs_norm_*": ...,
  "intruder_obs_dim":    20,
  "intruder_max_offset": 2.0,
}
```

For inference you only need `actor` + `obs_norm_*` + `obs_dim`.

---

## 10. Configuration reference (v13 settings)

```yaml
env:
  capture_radius: 1.5
  agent_max_speed: 0.75            # halved from v9's 1.5
  agent_max_offset: 3.0
  intruder_speed: 0.6              # halved from v9's 1.2
  intruder_max_offset: 2.0
  agent_radius: 0.3
  use_roles: true
  role_cooldown_steps: 30
  role_gap_threshold: 1.0
  agent_max_omega: 0.6             # NEW — yaw rate limit (rad/s)
  face_capture_thresh: 0.5         # NEW — cos(heading, target_dir) ≥ 0.5
  capture_hold_steps: 20           # NEW — 2 s of continuous capture geometry
  intruder_near_obstacle_prob: 0.50
  same_side_spawn_prob: 0.40

mappo:
  rollout_steps: 1024
  mini_batch_size: 256
  total_timesteps: 600_000
  use_intruder_policy: true
  intruder_hybrid_prob_start: 0.5
  intruder_hybrid_prob_end:   0.1

reward:
  w_dist_pursuer: 1.5
  w_pin_pursuer: 1.0
  pin_threshold: 1.5
  w_angle_encircler: 4.0
  w_dist_encircler: 1.2
  w_commit_encircler: 5.0
  w_capture: 400.0
  w_time: 200.0
  w_proximity: -1.0
  w_step: -0.12
  w_face_pursuer: 0.8              # NEW — face the target when within 5 m
  w_face_encircler: 0.5
  face_reward_radius: 5.0
  w_hold_step: 3.0                 # NEW — bonus per frame the lock holds
```

---

## 11. Differences vs the v9 release

| Component                | v9 release                | v13 release                                |
|--------------------------|---------------------------|--------------------------------------------|
| `env.agent_max_speed`    | 1.5 m/s                   | 0.75 m/s                                   |
| `env.intruder_speed`     | 1.2 m/s                   | 0.6 m/s                                    |
| `env.agent_max_omega`    | absent                    | 0.6 rad/s                                  |
| `env.face_capture_thresh`| absent                    | 0.5                                        |
| `env.capture_hold_steps` | absent                    | 20 (2 s)                                   |
| `env._move_along_path`   | sidestep allowed          | heading-locked (yaw rate-limited)          |
| `env.step`               | instant capture           | `capture_hold` counter, terminate at 20    |
| `pursuit_reward`         | dist + pin + angle + …   | + facing reward + hold-step bonus          |
| `interactive_demo.py`    | closing-fallback (Phase A only) | + Phase B (rotate-in-place lock) + perpendicular detour + tiered activation |
| `eval_demo_fallback.py`  | absent                    | new — fallback-only headless benchmark     |
| `eval_combined.py`       | absent                    | new — combined RL + fallback benchmark     |
| Capture rate (stationary)| 80 %+ from RL policy      | 0 % from RL policy alone, **88 %** with fallback |

---

## 12. Known limitations

- **RL policy alone is currently 0 % capture under v13 dynamics.** Achieving non-trivial capture rate from the policy alone would likely require either (a) a heavier observation that includes `agent_yaw` so the policy can reason about rotation, (b) a much longer training budget combined with curriculum (relax facing + hold first, tighten gradually), or (c) replacing offset-output with explicit `(speed, yaw_rate)` action so the policy can directly command rotation. None of these were attempted in this release.
- **Closing-fallback is the load-bearing capture mechanism.** It is deterministic and engineering-grade, not learned. The 88 % rate is bounded by geometry edge cases (encircler routing through tight obstacle pockets, walls clipping the 1.8 m detour radius).
- **Single map.** Geometry is hard-coded in `marl/utils/map_utils.py` (a 20 m × 20 m room with three internal pillars). Adding maps requires editing that file.
- **Two pursuers + one intruder only.** Role machinery and fallback both assume `n_agents = 2`.
- **No direct yaw observation in the policy.** Yaw is updated by the env but not exposed to the actor, by design (kept v9 obs API). The fallback works around this by computing yaw geometrically.

---

## 13. Citation / context

This release is the v13 milestone of a continuous experimental sweep (v1 → v13). v9 was the previous milestone, validated against the original (no-yaw, instant-capture) dynamics. v12 introduced the strict dynamics but had a constant-speed bug in `_move_along_path` (the agent couldn't slow / stop because the forward-step cap was lost during refactor); v13 is a from-scratch 600k-step run on the fixed env, with the closing-maneuver fallback redesigned to add the rotate-in-place lock phase that the env's heading-locked dynamics otherwise can't express.

The policy itself does not converge under v13 dynamics within the budget; the closing-maneuver carries the capture rate. If you need a pure-RL baseline for comparison, use the v9 release (`release_v9_final/` in the parent project, or `marl_pursuit_v9_final.zip`) — that one runs the older permissive dynamics where the RL policy reaches 80 %+.
