"""
mappo_trainer.py
MAPPO (Multi-Agent PPO) trainer.

Design choices:
  - Parameter sharing: one actor, one critic for all agents
  - CTDE: centralized critic sees global state (concat of all obs)
  - Observation normalization with RunningMeanStd
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..envs.pursuit_env import PursuitEnv
from ..policies.actor import Actor
from ..policies.critic import Critic
from ..buffers.rollout_buffer import RolloutBuffer
from ..utils.normalizer import RunningMeanStd


class MAPPOTrainer:
    def __init__(self, cfg: dict, device: str = "auto"):
        self.cfg = cfg
        env_cfg   = cfg["env"]
        mappo_cfg = cfg["mappo"]

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Dimensions
        self.n_agents    = env_cfg["n_agents"]
        self.obs_dim     = int(env_cfg.get("obs_dim", 13))   # 13D: state(12) + agent_id(1)
        self.action_dim  = 2
        self.global_dim  = self.n_agents * self.obs_dim       # 26D for critic

        # Hyperparameters
        self.lr_actor        = mappo_cfg["lr_actor"]
        self.lr_critic       = mappo_cfg["lr_critic"]
        self.gamma           = mappo_cfg["gamma"]
        self.gae_lambda      = mappo_cfg["gae_lambda"]
        self.clip_eps        = mappo_cfg["clip_epsilon"]
        self.n_epochs        = mappo_cfg["n_epochs"]
        self.rollout_steps   = mappo_cfg["rollout_steps"]
        self.mini_batch_sz   = mappo_cfg["mini_batch_size"]
        self.entropy_coeff   = mappo_cfg["entropy_coeff"]
        self.vf_coeff        = mappo_cfg["value_loss_coeff"]
        self.max_grad_norm   = mappo_cfg["max_grad_norm"]
        self.total_timesteps = mappo_cfg["total_timesteps"]
        self.save_interval   = mappo_cfg["save_interval"]

        # Environment
        self.env = PursuitEnv(cfg, render_mode=None)

        # Networks (parameter sharing: one actor/critic for all agents)
        self.actor  = Actor(self.obs_dim, self.action_dim, hidden_dim=64,
                            map_half=env_cfg["map_half"]).to(self.device)
        self.critic = Critic(self.global_dim, hidden_dim=128).to(self.device)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor,  eps=1e-5)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.rollout_steps,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=str(self.device),
        )

        # Observation normalizer
        self.obs_norm = RunningMeanStd(shape=(self.obs_dim,))

        # Logging
        self.total_steps   = 0
        self.episode_count = 0
        self.ep_rewards: list = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, save_dir: str = "results/checkpoints"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        obs, _ = self.env.reset()
        ep_reward = 0.0
        done      = False

        print(f"[MAPPO] Device: {self.device}")
        print(f"[MAPPO] Total timesteps: {self.total_timesteps:,}")
        print(f"[MAPPO] Rollout steps: {self.rollout_steps} | Mini-batch: {self.mini_batch_sz}")

        while self.total_steps < self.total_timesteps:
            self.buffer.reset()

            # ── Rollout collection ──────────────────────────────────────
            while not self.buffer.full:
                self.obs_norm.update(obs)
                obs_n = self.obs_norm.normalize(obs)  # (n_agents, obs_dim)

                obs_t = torch.FloatTensor(obs_n).to(self.device)
                with torch.no_grad():
                    actions, log_probs = self.actor.get_action(obs_t)
                    gs = obs_t.reshape(1, -1)                          # (1, global_dim)
                    values = self.critic(gs).squeeze().expand(self.n_agents)  # (n_agents,)

                act_np  = actions.cpu().numpy()
                logp_np = log_probs.cpu().numpy()
                val_np  = values.cpu().numpy()

                next_obs, rewards, terminated, truncated, info = self.env.step(act_np)
                done = terminated or truncated

                self.buffer.add(obs_n, act_np, rewards, val_np, logp_np, float(done))
                self.total_steps += self.n_agents
                ep_reward += float(rewards.mean())

                if done:
                    self.episode_count += 1
                    self.ep_rewards.append(ep_reward)
                    if self.episode_count % 10 == 0:
                        mean_r = float(np.mean(self.ep_rewards[-10:]))
                        print(
                            f"  Step {self.total_steps:>9,} | Ep {self.episode_count:>4} | "
                            f"MeanR {mean_r:+.2f} | "
                            f"Captured: {info['captured']} | "
                            f"MinDist: {info['min_dist']:.2f}m"
                        )
                    obs, _ = self.env.reset()
                    ep_reward = 0.0
                else:
                    obs = next_obs

            # ── GAE computation ─────────────────────────────────────────
            obs_n = self.obs_norm.normalize(obs)
            obs_t = torch.FloatTensor(obs_n).to(self.device)
            with torch.no_grad():
                gs = obs_t.reshape(1, -1)
                last_val = self.critic(gs).squeeze().expand(self.n_agents).cpu().numpy()
            self.buffer.compute_returns_and_advantages(last_val, done)

            # ── PPO update ───────────────────────────────────────────────
            metrics = self._update()

            # ── Checkpoint ──────────────────────────────────────────────
            if self.total_steps % self.save_interval < self.rollout_steps * self.n_agents:
                ckpt = save_path / f"step_{self.total_steps}.pt"
                self._save(ckpt)

        self._save(save_path / "final.pt")
        print("[MAPPO] Training complete.")

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(self) -> Dict[str, float]:
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_logp_b, adv_b, ret_b, gs_b in \
                    self.buffer.get_mini_batches(self.mini_batch_sz):

                # Actor loss (PPO clip)
                new_logp, entropy = self.actor.evaluate(obs_b, act_b)
                ratio  = torch.exp(new_logp - old_logp_b)
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coeff * entropy.mean()

                # Critic loss (MSE)
                v_pred      = self.critic(gs_b).squeeze(-1)
                critic_loss = self.vf_coeff * nn.functional.mse_loss(v_pred, ret_b)

                # Actor update
                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # Critic update
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, path: Path):
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "obs_norm_mean":  self.obs_norm.mean,
            "obs_norm_var":   self.obs_norm.var,
            "obs_norm_count": self.obs_norm.count,
            "total_steps":    self.total_steps,
        }, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.obs_norm.mean  = ckpt["obs_norm_mean"]
        self.obs_norm.var   = ckpt["obs_norm_var"]
        self.obs_norm.count = ckpt["obs_norm_count"]
        self.total_steps    = ckpt.get("total_steps", 0)
        print(f"  [ckpt] loaded ← {path}")
