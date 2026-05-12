"""eval_demo_fallback.py — headless test of the demo's closing-maneuver
fallback (with rotate-in-place lock phase). Replicates the demo's main
loop semantics against a stationary intruder and reports capture rate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.envs.pursuit_env import PursuitEnv
from marl.utils.astar import astar
from scripts.interactive_demo import compute_closing_actions


SG_COMMIT_FRAMES   = 10      # 0.5 s at dt=0.05 (or 1 s at FPS=20)
SG_REACHED_THRESH  = 0.5     # m — within this of subgoal → recommit
SG_TARGET_DRIFT    = 1.5     # m — target moved more than this → recommit
SG_FALLBACK_CAP    = 6.0     # m — closing-fallback subgoal max magnitude


def step_with_lock(env, actions, lock_flags, sg_state=None):
    """Mirrors the demo's main loop with subgoal-commit and lock-flag bypass.

    `sg_state` is a dict carried across steps with keys
        committed_sg / sg_age / sg_committed_target / was_locked
    each a list of length n_agents. If None, fresh state is created.
    """
    env._step_count += 1
    env.target_pos, env.target_vel = env._step_intruder()

    if sg_state is None:
        sg_state = {
            "committed_sg":         [None]*env.n_agents,
            "sg_age":               [0]*env.n_agents,
            "sg_committed_target":  [None]*env.n_agents,
            "was_locked":           [False]*env.n_agents,
        }

    for i in range(env.n_agents):
        if lock_flags[i]:
            to_t = env.target_pos - env.agent_pos[i]
            nt = float(np.linalg.norm(to_t))
            if nt > 1e-3:
                desired = float(np.arctan2(to_t[1], to_t[0]))
                err = (desired - float(env.agent_yaw[i]) + np.pi) % (2.0*np.pi) - np.pi
                max_step = env.agent_max_omega * env.dt
                err = float(np.clip(err, -max_step, max_step))
                env.agent_yaw[i] = (float(env.agent_yaw[i]) + err + np.pi) % (2.0*np.pi) - np.pi
            env.agent_vel[i] = np.zeros(2)
            env._subgoals[i] = env.agent_pos[i].copy()
            env._paths[i] = []
            sg_state["committed_sg"][i]        = None
            sg_state["sg_age"][i]              = 0
            sg_state["sg_committed_target"][i] = None
            sg_state["was_locked"][i]          = True
            continue

        raw = np.array(actions[i], dtype=np.float64)
        mag = float(np.linalg.norm(raw))
        if mag > SG_FALLBACK_CAP:
            raw = (raw / mag) * SG_FALLBACK_CAP
        candidate_sg = env.agent_pos[i] + raw
        candidate_sg = np.clip(candidate_sg, -env.map_half + 0.3, env.map_half - 0.3)

        committed = sg_state["committed_sg"][i]
        should_recommit = False
        if committed is None:
            should_recommit = True
        elif sg_state["was_locked"][i]:
            should_recommit = True
        elif sg_state["sg_age"][i] >= SG_COMMIT_FRAMES:
            should_recommit = True
        elif float(np.linalg.norm(env.agent_pos[i] - committed)) <= SG_REACHED_THRESH:
            should_recommit = True
        else:
            committed_target = sg_state["sg_committed_target"][i]
            if committed_target is None or float(
                np.linalg.norm(env.target_pos - committed_target)
            ) > SG_TARGET_DRIFT:
                should_recommit = True

        if should_recommit:
            sg_state["committed_sg"][i]        = candidate_sg.copy()
            sg_state["sg_age"][i]              = 0
            sg_state["sg_committed_target"][i] = env.target_pos.copy()
            sg_state["was_locked"][i]          = False
        else:
            sg_state["sg_age"][i] += 1

        sg = sg_state["committed_sg"][i]
        env._subgoals[i] = sg
        path = astar(env.obs_map, tuple(env.agent_pos[i].tolist()), tuple(sg.tolist()))
        env._paths[i] = path
        other_pos = env.agent_pos[1 - i].copy() if env.n_agents == 2 else None
        (env.agent_pos[i],
         env.agent_vel[i],
         env.agent_yaw[i]) = env._move_along_path(
            env.agent_pos[i], path, env.agent_max_spd,
            other_pos=other_pos,
            current_yaw=float(env.agent_yaw[i]),
            max_omega=env.agent_max_omega,
        )

    # Capture detection (mirror env.step)
    dists_after = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
    capture_geom_now = False
    both_in_capture = False
    if env.n_agents >= 2:
        d1, d2 = float(dists_after[0]), float(dists_after[1])
        both_in_capture = (d1 <= env.capture_radius and d2 <= env.capture_radius)
        if both_in_capture:
            v1 = env.agent_pos[0] - env.target_pos
            v2 = env.agent_pos[1] - env.target_pos
            cos_th = float(np.dot(v1, v2) / (d1 * d2 + 1e-8))
            if cos_th <= 0.0:
                facing_ok = True
                for i in range(2):
                    head = np.array([np.cos(env.agent_yaw[i]),
                                     np.sin(env.agent_yaw[i])])
                    to_t = env.target_pos - env.agent_pos[i]
                    n_to = float(np.linalg.norm(to_t))
                    if n_to < 1e-3:
                        continue
                    if float(np.dot(head, to_t / n_to)) < env.face_capture_thresh:
                        facing_ok = False; break
                if facing_ok:
                    capture_geom_now = True

    if capture_geom_now:
        env._capture_hold += 1
    else:
        env._capture_hold = 0
    captured = (env._capture_hold >= env.capture_hold_steps)
    truncated = env._step_count >= env.max_steps
    return captured, truncated, float(np.min(dists_after))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/mappo_role_intruder_300k.yaml")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["env"]["intruder_speed"] = 0.0   # stationary

    env = PursuitEnv(cfg, render_mode=None)

    successes = 0
    captures_steps = []
    min_dists = []

    for ep in range(args.episodes):
        env.reset()
        ep_min = 99.9
        cap_step = -1
        sg_state = {
            "committed_sg":         [None]*env.n_agents,
            "sg_age":               [0]*env.n_agents,
            "sg_committed_target":  [None]*env.n_agents,
            "was_locked":           [False]*env.n_agents,
        }
        for step in range(env.max_steps):
            actions, lock_flags = compute_closing_actions(
                env,
                env.agent_pos.copy(),
                env.target_pos.copy(),
                env.capture_radius,
                agent_yaw=env.agent_yaw.copy(),
                face_thresh=env.face_capture_thresh,
            )
            cap, trunc, mind = step_with_lock(env, actions, lock_flags, sg_state)
            ep_min = min(ep_min, mind)
            if cap:
                cap_step = step; break
            if trunc:
                break
        captured = (cap_step >= 0)
        successes += int(captured)
        if captured: captures_steps.append(cap_step)
        min_dists.append(ep_min)
        print(f"  Ep {ep+1:>2}/{args.episodes}  captured={captured}  step={cap_step:>3}  min_dist={ep_min:.2f}m")

    print()
    print(f"Demo-fallback (with lock) eval ({args.episodes} eps, stationary intruder):")
    print(f"  Success rate: {successes/args.episodes*100:.1f}%")
    print(f"  Mean min-distance: {np.mean(min_dists):.2f}m")
    if captures_steps:
        print(f"  Mean capture step (capt eps only): {np.mean(captures_steps):.0f}")


if __name__ == "__main__":
    main()
