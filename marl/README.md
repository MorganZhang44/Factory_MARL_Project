# MARL Module

`marl` is the standalone multi-agent decision module used by the runtime
system. The directory is intentionally split into three parts:

1. the online service entrypoint,
2. the default runtime checkpoint used by the online service,
3. a separate `research/` tree for training and offline experiments.

## Runtime Role

The runtime module takes:

- world-frame position/velocity for `agent_1`
- world-frame position/velocity for `agent_2`
- world-frame position/velocity for `intruder_1`

and returns:

- one world-frame `subgoal` per robot,
- role assignment metadata,
- fallback/debug status used by `core` and the dashboard.

At runtime, `marl_service.py` builds the current observation, assigns roles,
loads the default `v13` checkpoint when available, and falls back to geometric
closing logic when policy inference is disabled or unavailable.

## Directory Layout

```text
marl/
├── marl_service.py
├── README.md
├── environment.yml
├── Dockerfile
├── v13_final.pt              # runtime checkpoint only
└── research/                 # separate research/training tree
```

## Runtime Boundary

The online integration boundary is:

- `marl/marl_service.py`
- `scripts/launch_marl.sh`

The runtime service uses the in-tree default release at:

```text
marl/research
```

The runtime service loads the default checkpoint from:

```text
marl/v13_final.pt
```

In other words:

- `research/` provides the policy and utility code imported by `marl_service.py`,
- `v13_final.pt` is the runtime model file.

## Research Tree

All training-oriented code has been moved under:

```text
marl/research
```

That tree contains:

- training entrypoints,
- evaluation scripts,
- MAPPO environment code,
- policy/critic definitions,
- rollout buffer,
- rewards,
- utility modules.

This keeps the runtime surface small while preserving the full `v13` research
stack in one place.

## Launch

Create the environment:

```bash
conda env create -f marl/environment.yml
```

Run the service:

```bash
./scripts/launch_marl.sh
```

By default the service listens on `http://127.0.0.1:8892`.

## Endpoints

- `GET /health`
- `POST /act`

## Checkpoints

Default runtime checkpoint:

```text
marl/v13_final.pt
```

Backward-compatible fallback checkpoint:

```text
marl/checkpoints/mappo_latest.pt
```

If no checkpoint is present, or if the runtime environment cannot import
`torch`, the service stays runnable and falls back to a simple intercept offset
policy. This keeps the module boundary testable before trained weights are
fully wired in.
