#!/usr/bin/env python3
"""Probe joint tracking error for the parallel locomotion setup.

This script boots the parallel Go2 locomotion task against the currently
installed IsaacLab runtime, runs a short rollout, and records:

- commanded base velocity
- observed base linear / angular velocity
- base velocity tracking error
- observed relative joint positions
- observed relative joint velocities
- last applied target relative joint positions
- per-joint absolute error

Outputs:
- `<output_dir>/trace.jsonl`   per-step records
- `<output_dir>/summary.json`  aggregated metrics and run config
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaaclab-source-root",
        default="/home/yyz/projects/IsaacLab/source/extensions",
        help="Root containing the parallel omni.isaac.lab_* source trees.",
    )
    parser.add_argument(
        "--logs-root",
        default="/home/yyz/projects/IsaacLab/logs/rsl_rl/unitree_go2_flat",
        help="Directory containing trained model_*.pt files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint path, or 'latest' to auto-resolve from logs-root.",
    )
    parser.add_argument(
        "--policy-source",
        choices=["parallel", "current_project"],
        default="parallel",
        help="Use the parallel training checkpoint or the current project NPZ policy.",
    )
    parser.add_argument(
        "--current-policy-path",
        default="locomotion/checkpoints/go2_flat_actor_model_499.npz",
        help="Path to the current project's NPZ locomotion policy.",
    )
    parser.add_argument("--output-dir", default="output/parallel_locomotion_probe")
    parser.add_argument("--stiffness", type=float, default=25.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--solver-velocity-iters", type=int, default=0)
    parser.add_argument(
        "--enable-external-forces-every-iteration",
        action="store_true",
        help="Enable PhysX external forces every iteration if supported.",
    )
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--sample-steps", type=int, default=150)
    parser.add_argument("--cmd-vx", type=float, default=0.6)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0", headless=False, enable_cameras=False, disable_fabric=False)
    args = parser.parse_args()
    return args


def ensure_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(log_root: Path) -> str:
    best = None
    best_key = None
    for run_dir in sorted([p for p in log_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for child in run_dir.iterdir():
            match = re.fullmatch(r"model_(\d+)\.pt", child.name)
            if not match:
                continue
            idx = int(match.group(1))
            if idx <= 0:
                continue
            key = (run_dir.name, idx)
            if best_key is None or key > best_key:
                best_key = key
                best = child
    if best is None:
        raise RuntimeError(f"No checkpoint found under {log_root}")
    return str(best)


def bootstrap_parallel_modules(isaaclab_source_root: Path):
    import isaaclab

    submodules = [
        "app",
        "envs",
        "managers",
        "assets",
        "utils",
        "scene",
        "sensors",
        "actuators",
        "sim",
        "markers",
        "terrains",
        "devices",
        "controllers",
    ]
    sys.modules["omni.isaac.lab"] = isaaclab
    for sub in submodules:
        mod = importlib.import_module(f"isaaclab.{sub}")
        sys.modules[f"omni.isaac.lab.{sub}"] = mod

    base_tasks = isaaclab_source_root / "omni.isaac.lab_tasks" / "omni" / "isaac" / "lab_tasks"
    base_assets = isaaclab_source_root / "omni.isaac.lab_assets" / "omni" / "isaac" / "lab_assets"

    ensure_pkg("omni.isaac.lab_assets", base_assets)
    ensure_pkg("omni.isaac.lab_tasks", base_tasks)
    ensure_pkg("omni.isaac.lab_tasks.manager_based", base_tasks / "manager_based")
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion",
        base_tasks / "manager_based" / "locomotion",
    )
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity",
        base_tasks / "manager_based" / "locomotion" / "velocity",
    )
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "mdp",
    )
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "config",
    )
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "config" / "go2",
    )
    ensure_pkg(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.agents",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "config" / "go2" / "agents",
    )
    ensure_pkg("omni.isaac.lab_tasks.utils", base_tasks / "utils")
    ensure_pkg("omni.isaac.lab_tasks.utils.wrappers", base_tasks / "utils" / "wrappers")
    ensure_pkg(
        "omni.isaac.lab_tasks.utils.wrappers.rsl_rl",
        base_tasks / "utils" / "wrappers" / "rsl_rl",
    )

    sys.modules["omni.isaac.lab_assets.unitree"] = load_module(
        "omni.isaac.lab_assets.unitree", base_assets / "unitree.py"
    )
    load_module(
        "omni.isaac.lab_tasks.utils.wrappers.rsl_rl",
        base_tasks / "utils" / "wrappers" / "rsl_rl" / "__init__.py",
    )
    load_module(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "mdp" / "__init__.py",
    )
    load_module(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg",
        base_tasks / "manager_based" / "locomotion" / "velocity" / "velocity_env_cfg.py",
    )
    load_module(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg",
        base_tasks
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "go2"
        / "agents"
        / "rsl_rl_ppo_cfg.py",
    )
    load_module(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg",
        base_tasks
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "go2"
        / "rough_env_cfg.py",
    )
    flat_mod = load_module(
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg",
        base_tasks
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "go2"
        / "flat_env_cfg.py",
    )
    agent_mod = sys.modules[
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg"
    ]
    return flat_mod, agent_mod


def make_robot_cfg(stiffness: float, damping: float):
    from isaaclab.actuators import DCMotorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=stiffness,
                damping=damping,
                friction=0.0,
            ),
        },
    )


def build_env(args: argparse.Namespace, flat_mod, agent_mod, checkpoint_path: str):
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    unitree_cfg_cls = flat_mod.UnitreeGo2FlatEnvCfg_PLAY
    runner_cfg_cls = agent_mod.UnitreeGo2FlatPPORunnerCfg

    cfg = unitree_cfg_cls()
    cfg.scene.num_envs = 1
    cfg.scene.robot = make_robot_cfg(args.stiffness, args.damping).replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.sim.device = args.device
    if hasattr(cfg.sim, "physx") and cfg.sim.physx is not None:
        if hasattr(cfg.sim.physx, "solver_velocity_iteration_count"):
            cfg.sim.physx.solver_velocity_iteration_count = args.solver_velocity_iters
        if hasattr(cfg.sim.physx, "enable_external_forces_every_iteration"):
            cfg.sim.physx.enable_external_forces_every_iteration = (
                args.enable_external_forces_every_iteration
            )

    base_env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(base_env)
    env.get_observations = lambda: base_env.observation_manager.compute()

    runner_cfg = runner_cfg_cls()
    runner_cfg.device = args.device
    train_cfg = runner_cfg.to_dict()
    train_cfg.setdefault("obs_groups", {"policy": ["policy"]})

    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=runner_cfg.device)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed = ppo_runner.alg.policy.load_state_dict(loaded["model_state_dict"])
    if getattr(ppo_runner, "empirical_normalization", False):
        if resumed:
            if "obs_norm_state_dict" in loaded:
                ppo_runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
            if "privileged_obs_norm_state_dict" in loaded and hasattr(
                ppo_runner, "privileged_obs_normalizer"
            ):
                ppo_runner.privileged_obs_normalizer.load_state_dict(
                    loaded["privileged_obs_norm_state_dict"]
                )
        elif "obs_norm_state_dict" in loaded:
            ppo_runner.privileged_obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])

    policy = ppo_runner.get_inference_policy(device=base_env.device)
    return base_env, policy


class NpzActorPolicy:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NPZ locomotion policy not found: {path}")
        weights = np.load(path)
        self.layers = [
            (weights["w0"].astype(np.float32), weights["b0"].astype(np.float32)),
            (weights["w1"].astype(np.float32), weights["b1"].astype(np.float32)),
            (weights["w2"].astype(np.float32), weights["b2"].astype(np.float32)),
            (weights["w3"].astype(np.float32), weights["b3"].astype(np.float32)),
        ]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for weight, bias in self.layers[:-1]:
            y = weight @ x + bias
            x = np.where(y > 0.0, y, np.exp(y) - 1.0).astype(np.float32)
        weight, bias = self.layers[-1]
        return (weight @ x + bias).astype(np.float32)


def run_probe(args: argparse.Namespace) -> dict:
    isaaclab_source_root = Path(args.isaaclab_source_root)
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"

    flat_mod, agent_mod = bootstrap_parallel_modules(isaaclab_source_root)
    if args.policy_source == "parallel":
        checkpoint_path = args.checkpoint
        if checkpoint_path == "latest":
            checkpoint_path = resolve_checkpoint(logs_root)
        base_env, policy = build_env(args, flat_mod, agent_mod, checkpoint_path)
        policy_label = "parallel_checkpoint"
    else:
        checkpoint_path = str(Path(args.current_policy_path).resolve())
        base_env, _ = build_env(args, flat_mod, agent_mod, resolve_checkpoint(logs_root))
        npz_policy = NpzActorPolicy(checkpoint_path)

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            obs = obs_dict["policy"].squeeze(0).detach().cpu().numpy()
            action = npz_policy(obs)
            return torch.tensor(action, device=base_env.device, dtype=torch.float32).unsqueeze(0)

        policy_label = "current_project_npz"

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.set_debug_vis(False)
    command_term.is_heading_env[:] = False
    command_term.is_standing_env[:] = False
    command_term.time_left[:] = 1.0e9
    command_term.heading_target[:] = 0.0

    obs_dict, _ = base_env.reset(seed=args.seed)
    cmd = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_wz]], device=base_env.device, dtype=torch.float32)

    step_records = []
    total_steps = args.warmup_steps + args.sample_steps
    with torch.inference_mode():
        for step_idx in range(total_steps):
            command_term.vel_command_b[:] = cmd
            command_term.is_heading_env[:] = False
            command_term.is_standing_env[:] = False
            command_term.heading_target[:] = 0.0
            actions = policy(obs_dict)
            obs_dict, _, _, _, _ = base_env.step(actions)

            if step_idx < args.warmup_steps:
                continue

            policy_obs = obs_dict["policy"]
            base_lin_vel = policy_obs[:, 0:3].squeeze(0).cpu()
            base_ang_vel = policy_obs[:, 3:6].squeeze(0).cpu()
            joint_pos_rel = policy_obs[:, 12:24].squeeze(0).cpu()
            joint_vel_rel = policy_obs[:, 24:36].squeeze(0).cpu()
            last_action = policy_obs[:, 36:48].squeeze(0).cpu()
            target_pos_rel = last_action * args.action_scale
            abs_error = (joint_pos_rel - target_pos_rel).abs()
            cmd_cpu = cmd.squeeze(0).cpu()
            base_lin_vel_xy_abs_error = (base_lin_vel[0:2] - cmd_cpu[0:2]).abs()
            base_ang_vel_z_abs_error = abs(float(base_ang_vel[2].item() - cmd_cpu[2].item()))

            record = {
                "sample_index": step_idx - args.warmup_steps,
                "command": [float(args.cmd_vx), float(args.cmd_vy), float(args.cmd_wz)],
                "base_lin_vel": [float(x) for x in base_lin_vel.tolist()],
                "base_ang_vel": [float(x) for x in base_ang_vel.tolist()],
                "base_lin_vel_xy_abs_error": [float(x) for x in base_lin_vel_xy_abs_error.tolist()],
                "base_ang_vel_z_abs_error": base_ang_vel_z_abs_error,
                "joint_pos_rel": [float(x) for x in joint_pos_rel.tolist()],
                "joint_vel_rel": [float(x) for x in joint_vel_rel.tolist()],
                "last_action": [float(x) for x in last_action.tolist()],
                "target_pos_rel": [float(x) for x in target_pos_rel.tolist()],
                "abs_error": [float(x) for x in abs_error.tolist()],
                "mean_abs_error": float(abs_error.mean().item()),
                "max_abs_error": float(abs_error.max().item()),
                "mean_base_lin_vel_xy_abs_error": float(base_lin_vel_xy_abs_error.mean().item()),
            }
            step_records.append(record)

    with trace_path.open("w", encoding="utf-8") as f:
        for record in step_records:
            f.write(json.dumps(record) + "\n")

    joint_names = list(base_env.scene["robot"].joint_names)
    error_tensor = torch.tensor([r["abs_error"] for r in step_records], dtype=torch.float32)
    mean_per_joint = error_tensor.mean(dim=0)
    max_per_joint = error_tensor.max(dim=0).values
    lin_vel_error_tensor = torch.tensor(
        [r["base_lin_vel_xy_abs_error"] for r in step_records], dtype=torch.float32
    )
    ang_vel_z_error_tensor = torch.tensor(
        [r["base_ang_vel_z_abs_error"] for r in step_records], dtype=torch.float32
    )
    observed_vx_tensor = torch.tensor([r["base_lin_vel"][0] for r in step_records], dtype=torch.float32)
    observed_vy_tensor = torch.tensor([r["base_lin_vel"][1] for r in step_records], dtype=torch.float32)
    observed_wz_tensor = torch.tensor([r["base_ang_vel"][2] for r in step_records], dtype=torch.float32)

    summary = {
        "checkpoint": checkpoint_path,
        "policy_source": policy_label,
        "output_dir": str(output_dir),
        "physics_dt": float(base_env.physics_dt),
        "step_dt": float(base_env.step_dt),
        "config": {
            "stiffness": args.stiffness,
            "damping": args.damping,
            "solver_velocity_iters": args.solver_velocity_iters,
            "enable_external_forces_every_iteration": args.enable_external_forces_every_iteration,
            "action_scale": args.action_scale,
            "warmup_steps": args.warmup_steps,
            "sample_steps": args.sample_steps,
            "command": [args.cmd_vx, args.cmd_vy, args.cmd_wz],
            "device": args.device,
            "seed": args.seed,
        },
        "overall_mean_abs_error": float(error_tensor.mean().item()),
        "overall_max_abs_error": float(error_tensor.max().item()),
        "velocity_tracking": {
            "command": [args.cmd_vx, args.cmd_vy, args.cmd_wz],
            "mean_observed_vx": float(observed_vx_tensor.mean().item()),
            "mean_observed_vy": float(observed_vy_tensor.mean().item()),
            "mean_observed_wz": float(observed_wz_tensor.mean().item()),
            "mean_abs_error_vx": float(lin_vel_error_tensor[:, 0].mean().item()),
            "mean_abs_error_vy": float(lin_vel_error_tensor[:, 1].mean().item()),
            "mean_abs_error_wz": float(ang_vel_z_error_tensor.mean().item()),
            "max_abs_error_vx": float(lin_vel_error_tensor[:, 0].max().item()),
            "max_abs_error_vy": float(lin_vel_error_tensor[:, 1].max().item()),
            "max_abs_error_wz": float(ang_vel_z_error_tensor.max().item()),
        },
        "per_joint": [
            {"name": name, "mean": float(mean), "max": float(maximum)}
            for name, mean, maximum in zip(
                joint_names, mean_per_joint.tolist(), max_per_joint.tolist()
            )
        ],
        "files": {
            "trace_jsonl": str(trace_path),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    base_env.close()
    return summary


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        summary = run_probe(args)
        print(json.dumps(summary, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
