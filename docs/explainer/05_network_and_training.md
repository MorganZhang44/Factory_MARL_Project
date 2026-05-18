# 05 · Network Structure and MAPPO Training

**Primary files:**

- `marl/research/marl/policies/actor.py`
- `marl/research/marl/policies/critic.py`
- `marl/research/marl/trainers/mappo_trainer.py`
- `marl/research/marl/buffers/rollout_buffer.py`

## Actor

The actor is the policy network used at inference time. It maps the 21-dimensional observation to a 2-dimensional action representing a subgoal offset.

The current structure is a compact MLP:

- Linear -> Tanh
- Linear -> Tanh
- linear mean head for action output
- learned log standard deviation for stochastic training

At inference time, the mean action is typically used.

## Critic

The critic estimates state value during training. It is used to compute advantages and stabilize PPO updates, but it is not required for deployment demos or runtime service inference.

## Observation Normalization

`marl/research/marl/utils/normalizer.py` maintains running observation statistics. This matters because position, velocity, and LiDAR features live on different numeric scales. The saved normalization statistics must stay aligned with the checkpoint used for inference.

## MAPPO Loop

The training cycle is:

1. collect rollouts
2. compute GAE advantages
3. run multiple PPO update epochs over minibatches

This is standard MAPPO structure, but the quality of the learned behavior depends heavily on the environment and reward terms described in the previous sections.

## Main Hyperparameters

The training configuration in `marl/research/configs/mappo_config.yaml` controls:

- actor and critic learning rates
- `gamma`
- `gae_lambda`
- PPO clip epsilon
- rollout length
- minibatch size
- epoch count per update
- total training steps

## Checkpoints

Research checkpoints live under `marl/research/checkpoints/`. The runtime checkpoint kept at `marl/v13_final.pt` is the compact deployment artifact, while the research tree may contain intermediate or comparison checkpoints used during training and evaluation.
