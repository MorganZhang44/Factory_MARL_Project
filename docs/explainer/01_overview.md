# 01 · Project Overview and Directory Layout

## Problem Statement

The project trains two Unitree Go2 robots to cooperatively contain and capture a moving intruder inside a cluttered warehouse-like environment. The core challenge is coordination: one robot should pressure the target at short range while the other moves into a better enclosing position.

## Why MAPPO

The research stack uses **MAPPO** (Multi-Agent Proximal Policy Optimization):

- PPO-based and stable for multi-agent policy training
- shared actor parameters for both robots
- agent identity kept in the observation so the shared policy can specialize behavior
- centralized training with decentralized execution

## Current Repository Layout

```text
Factory_MARL_Project/
├── core/                          # runtime orchestration and dashboard
├── simulation/                    # legacy Isaac Lab simulation line
├── perception/                    # perception HTTP service
├── navdp/                         # navigation planning HTTP service
├── locomotion/                    # locomotion adapter HTTP service
├── marl/
│   ├── marl_service.py            # runtime MARL service
│   ├── v13_final.pt               # runtime checkpoint
│   └── research/                  # training, evaluation, and analysis tree
│       ├── marl/
│       │   ├── envs/pursuit_env.py
│       │   ├── policies/actor.py
│       │   ├── policies/critic.py
│       │   ├── trainers/mappo_trainer.py
│       │   ├── rewards/pursuit_reward.py
│       │   ├── buffers/rollout_buffer.py
│       │   └── utils/
│       ├── scripts/
│       ├── configs/
│       └── checkpoints/
├── sim2real/                      # real-robot subproject
└── docs/
```

## Dependency View for MARL Training

```text
marl/research/scripts/train_mappo.py
  -> marl/research/marl/trainers/mappo_trainer.py
      -> marl/research/marl/envs/pursuit_env.py
          -> marl/research/marl/rewards/pursuit_reward.py
          -> marl/research/marl/utils/map_utils.py
          -> marl/research/marl/utils/astar.py
      -> marl/research/marl/policies/actor.py
      -> marl/research/marl/policies/critic.py
      -> marl/research/marl/buffers/rollout_buffer.py
      -> marl/research/marl/utils/normalizer.py
```

## Runtime vs Research Split

- `marl/marl_service.py` and `marl/v13_final.pt` belong to the online runtime stack.
- `marl/research/` contains training scripts, evaluation tools, analysis utilities, and the full research code tree.

That separation is intentional: runtime stays small, while research remains reproducible and self-contained.
