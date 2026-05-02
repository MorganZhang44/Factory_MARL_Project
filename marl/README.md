# MARL Module

`marl` is the standalone multi-agent decision module.

## Role

The runtime module takes:

- world-frame position/velocity for `agent_1`
- world-frame position/velocity for `agent_2`
- world-frame position/velocity for `intruder_1`

and returns:

- one world-frame `subgoal` per robot

Internally, the current MAPPO policy consumes a `13D` observation per robot and
predicts a world-frame relative offset. `marl_service.py` converts that offset
into the absolute world-frame `subgoal` returned to `core`.

## Runtime Boundary

This module does **not** expose the training stack to the rest of the project.

Training-oriented files such as:

- `marl/trainers/`
- `marl/buffers/`
- `marl/rewards/`
- `marl/envs/`

remain internal research code.

The system integration boundary is:

- `marl/marl_service.py`
- `scripts/launch_marl.sh`

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

Default checkpoint path:

```text
results/checkpoints/final.pt
```

Backward-compatible fallback path:

```text
marl/checkpoints/mappo_latest.pt
```

If no checkpoint is present, or if the runtime environment cannot import
`torch`, the service stays runnable and falls back to a simple intercept offset
policy. This keeps the module boundary testable before trained weights are
fully wired in.
