"""
mappo_trainer.py
MAPPO (Multi-Agent PPO) trainer for the pursuit task.

Supports two modes via cfg:
  - Vanilla shared-actor MAPPO (use_intruder_policy=False, use_roles=False)
  - Role-aware MAPPO + co-trained intruder policy
    (env.use_roles=True + mappo.use_intruder_policy=True)

Design choices:
  - Parameter sharing: one actor, one critic for all *agents* (not the intruder)
  - CTDE: centralized critic over concat of all agent obs
  - Role flag is part of the obs → the same shared actor handles both PURSUER and ENCIRCLER
  - Intruder is a separate actor + critic (no parameter sharing with agents)
  - Intruder hybrid: episodes alternate between scripted (random walk) and RL-policy
    intruder, mixed by `intruder_hybrid_prob_*`. Hybrid prevents GAE corruption
    by keeping each episode purely scripted OR purely RL.
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
from ..policies.intruder_actor import IntruderActor, IntruderCritic
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
        self.use_roles   = bool(env_cfg.get("use_roles", False))
        default_obs_dim  = 22 if self.use_roles else 21
        self.obs_dim     = int(env_cfg.get("obs_dim", default_obs_dim))
        self.action_dim  = 2
        self.global_dim  = self.n_agents * self.obs_dim

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

        # Intruder co-training settings
        self.use_intruder_policy = bool(mappo_cfg.get("use_intruder_policy", False))
        self.intruder_obs_dim    = int(mappo_cfg.get("intruder_obs_dim", 20))
        self.intruder_lr_actor   = float(mappo_cfg.get("intruder_lr_actor",  3.0e-4))
        self.intruder_lr_critic  = float(mappo_cfg.get("intruder_lr_critic", 3.0e-4))
        self.hybrid_p_start = float(mappo_cfg.get("intruder_hybrid_prob_start", 0.5))
        self.hybrid_p_end   = float(mappo_cfg.get("intruder_hybrid_prob_end",   0.1))
        self.hybrid_decay_frac = float(mappo_cfg.get("intruder_hybrid_decay_frac", 0.5))
        self.intruder_max_offset = float(env_cfg.get("intruder_max_offset", 2.0))

        # Environment
        self.env = PursuitEnv(cfg, render_mode=None)

        # Networks (parameter sharing for agents)
        # IMPORTANT: actor's action_limit must match env's per-step `max_offset` so the
        # output range isn't wasted (was 10 with env-clip to 3 → most range unused).
        agent_action_limit = float(env_cfg.get("agent_max_offset", 3.0))
        self.actor  = Actor(self.obs_dim, self.action_dim, hidden_dim=64,
                            map_half=agent_action_limit).to(self.device)
        self.critic = Critic(self.global_dim, hidden_dim=128).to(self.device)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor,  eps=1e-5)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        # Rollout buffer for agents
        self.buffer = RolloutBuffer(
            n_steps=self.rollout_steps,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=str(self.device),
        )
        self.obs_norm = RunningMeanStd(shape=(self.obs_dim,))

        # Intruder networks + buffer + normalizer
        if self.use_intruder_policy:
            self.intruder_actor  = IntruderActor(
                obs_dim=self.intruder_obs_dim, action_dim=self.action_dim,
                hidden_dim=64, max_offset=self.intruder_max_offset,
            ).to(self.device)
            self.intruder_critic = IntruderCritic(
                obs_dim=self.intruder_obs_dim, hidden_dim=128,
            ).to(self.device)
            self.opt_intruder_actor  = optim.Adam(
                self.intruder_actor.parameters(),  lr=self.intruder_lr_actor,  eps=1e-5
            )
            self.opt_intruder_critic = optim.Adam(
                self.intruder_critic.parameters(), lr=self.intruder_lr_critic, eps=1e-5
            )
            self.intruder_buffer = RolloutBuffer(
                n_steps=self.rollout_steps,
                n_agents=1,
                obs_dim=self.intruder_obs_dim,
                action_dim=self.action_dim,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=str(self.device),
            )
            self.intruder_obs_norm = RunningMeanStd(shape=(self.intruder_obs_dim,))

        # Logging
        self.total_steps   = 0
        self.episode_count = 0
        self.ep_rewards: list = []
        self.np_random = np.random.RandomState(seed=None)
        self.current_episode_uses_intruder_rl = False

    # ------------------------------------------------------------------
    # Hybrid intruder schedule
    # ------------------------------------------------------------------

    def _hybrid_prob_script(self) -> float:
        """Probability of using SCRIPTED intruder (= 1 - p_RL)."""
        decay_target = self.total_timesteps * self.hybrid_decay_frac
        if self.total_steps >= decay_target or decay_target <= 0:
            return self.hybrid_p_end
        frac = self.total_steps / decay_target
        return self.hybrid_p_start + (self.hybrid_p_end - self.hybrid_p_start) * frac

    def _decide_intruder_mode(self):
        if not self.use_intruder_policy:
            self.current_episode_uses_intruder_rl = False
            return
        p_script = self._hybrid_prob_script()
        self.current_episode_uses_intruder_rl = (self.np_random.random() >= p_script)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, save_dir: str = "results/checkpoints"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        obs, _ = self.env.reset()
        intruder_obs = self.env.get_intruder_obs() if self.use_intruder_policy else None
        self._decide_intruder_mode()
        ep_reward = 0.0
        done      = False

        print(f"[MAPPO] Device: {self.device}")
        print(f"[MAPPO] Total timesteps: {self.total_timesteps:,}")
        print(f"[MAPPO] Rollout steps: {self.rollout_steps} | Mini-batch: {self.mini_batch_sz}")
        print(f"[MAPPO] Use roles: {self.use_roles}  obs_dim={self.obs_dim}")
        print(f"[MAPPO] Intruder policy: {self.use_intruder_policy}  "
              f"hybrid_p_start={self.hybrid_p_start} → end={self.hybrid_p_end}")

        while self.total_steps < self.total_timesteps:
            self.buffer.reset()  # main buffer reset every cycle; intruder buffer accumulates

            # ── Rollout collection ──────────────────────────────────────
            while not self.buffer.full:
                # ---- Agent action ----
                self.obs_norm.update(obs)
                obs_n = self.obs_norm.normalize(obs)
                obs_t = torch.FloatTensor(obs_n).to(self.device)
                with torch.no_grad():
                    actions, log_probs = self.actor.get_action(obs_t)
                    gs = obs_t.reshape(1, -1)
                    values = self.critic(gs).squeeze().expand(self.n_agents)
                act_np  = actions.cpu().numpy()
                logp_np = log_probs.cpu().numpy()
                val_np  = values.cpu().numpy()

                # ---- Intruder action (only when this episode is RL) ----
                intruder_action_for_env = None
                int_obs_n = int_action_np = None
                int_logp = int_value = 0.0
                if self.use_intruder_policy and self.current_episode_uses_intruder_rl:
                    self.intruder_obs_norm.update(intruder_obs)
                    int_obs_n = self.intruder_obs_norm.normalize(intruder_obs)
                    int_obs_t = torch.FloatTensor(int_obs_n).to(self.device)
                    with torch.no_grad():
                        int_action_t, int_logp_t = self.intruder_actor.get_action(int_obs_t)
                        int_value_t = self.intruder_critic(int_obs_t).squeeze()
                    int_action_np = int_action_t.cpu().numpy()
                    int_logp = float(int_logp_t.cpu().numpy())
                    int_value = float(int_value_t.cpu().numpy())
                    intruder_action_for_env = int_action_np

                # ---- Env step ----
                next_obs, rewards, terminated, truncated, info = self.env.step(
                    act_np, intruder_action=intruder_action_for_env
                )
                done = terminated or truncated

                # ---- Buffer add ----
                self.buffer.add(obs_n, act_np, rewards, val_np, logp_np, float(done))
                self.total_steps += self.n_agents
                ep_reward += float(rewards.mean())

                if (self.use_intruder_policy
                        and self.current_episode_uses_intruder_rl
                        and not self.intruder_buffer.full
                        and int_obs_n is not None):
                    self.intruder_buffer.add(
                        obs=int_obs_n[np.newaxis, :].astype(np.float32),
                        action=int_action_np[np.newaxis, :].astype(np.float32),
                        reward=np.array([info["intruder_reward"]], dtype=np.float32),
                        value=np.array([int_value], dtype=np.float32),
                        log_prob=np.array([int_logp], dtype=np.float32),
                        done=float(done),
                    )

                # ---- Episode end ----
                if done:
                    self.episode_count += 1
                    self.ep_rewards.append(ep_reward)
                    if self.episode_count % 10 == 0:
                        mean_r = float(np.mean(self.ep_rewards[-10:]))
                        p_script = self._hybrid_prob_script()
                        flag = "RL" if self.current_episode_uses_intruder_rl else "scripted"
                        print(
                            f"  Step {self.total_steps:>9,} | Ep {self.episode_count:>4} | "
                            f"MeanR {mean_r:+.2f} | Captured: {info['captured']} | "
                            f"MinDist: {info['min_dist']:.2f}m | "
                            f"Intruder: {flag} (p_script={p_script:.2f})"
                        )
                    obs, _ = self.env.reset()
                    if self.use_intruder_policy:
                        intruder_obs = self.env.get_intruder_obs()
                    self._decide_intruder_mode()
                    ep_reward = 0.0
                else:
                    obs = next_obs
                    if self.use_intruder_policy:
                        intruder_obs = info["intruder_obs"]

            # ── Main GAE + update ──────────────────────────────────────
            obs_n_last = self.obs_norm.normalize(obs)
            obs_t_last = torch.FloatTensor(obs_n_last).to(self.device)
            with torch.no_grad():
                gs_last = obs_t_last.reshape(1, -1)
                last_val = self.critic(gs_last).squeeze().expand(self.n_agents).cpu().numpy()
            self.buffer.compute_returns_and_advantages(last_val, done)
            metrics = self._update()

            # ── Intruder update if its buffer is full ───────────────────
            if self.use_intruder_policy and self.intruder_buffer.full:
                int_obs_n_last = self.intruder_obs_norm.normalize(intruder_obs)
                int_obs_t_last = torch.FloatTensor(int_obs_n_last).to(self.device)
                with torch.no_grad():
                    last_int_val_t = self.intruder_critic(int_obs_t_last).squeeze()
                last_int_val_np = last_int_val_t.cpu().numpy().reshape(-1)
                if last_int_val_np.size != 1:
                    last_int_val_np = last_int_val_np[:1]
                self.intruder_buffer.compute_returns_and_advantages(last_int_val_np, done)
                int_metrics = self._update_intruder()
                self.intruder_buffer.reset()
                print(f"  [intruder] actor_loss={int_metrics['actor_loss']:+.4f} "
                      f"critic_loss={int_metrics['critic_loss']:.4f} "
                      f"entropy={int_metrics['entropy']:.3f}")

            # ── Checkpoint ──────────────────────────────────────────────
            if self.total_steps % self.save_interval < self.rollout_steps * self.n_agents:
                ckpt = save_path / f"step_{self.total_steps}.pt"
                self._save(ckpt)

        self._save(save_path / "final.pt")
        print("[MAPPO] Training complete.")

    # ------------------------------------------------------------------
    # PPO update — agents (shared actor + centralized critic)
    # ------------------------------------------------------------------

    def _update(self) -> Dict[str, float]:
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.n_epochs):
            for obs_b, act_b, old_logp_b, adv_b, ret_b, gs_b in \
                    self.buffer.get_mini_batches(self.mini_batch_sz):
                new_logp, entropy = self.actor.evaluate(obs_b, act_b)
                ratio  = torch.exp(new_logp - old_logp_b)
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coeff * entropy.mean()

                v_pred      = self.critic(gs_b).squeeze(-1)
                critic_loss = self.vf_coeff * nn.functional.mse_loss(v_pred, ret_b)

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

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
    # PPO update — intruder
    # ------------------------------------------------------------------

    def _update_intruder(self) -> Dict[str, float]:
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.n_epochs):
            for obs_b, act_b, old_logp_b, adv_b, ret_b, gs_b in \
                    self.intruder_buffer.get_mini_batches(self.mini_batch_sz):
                # For intruder n=1, gs == obs (the buffer reshapes accordingly).
                new_logp, entropy = self.intruder_actor.evaluate(obs_b, act_b)
                ratio  = torch.exp(new_logp - old_logp_b)
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coeff * entropy.mean()

                v_pred      = self.intruder_critic(gs_b).squeeze(-1)
                critic_loss = self.vf_coeff * nn.functional.mse_loss(v_pred, ret_b)

                self.opt_intruder_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.intruder_actor.parameters(), self.max_grad_norm)
                self.opt_intruder_actor.step()

                self.opt_intruder_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.intruder_critic.parameters(), self.max_grad_norm)
                self.opt_intruder_critic.step()

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
        ckpt = {
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "obs_norm_mean":  self.obs_norm.mean,
            "obs_norm_var":   self.obs_norm.var,
            "obs_norm_count": self.obs_norm.count,
            "total_steps":    self.total_steps,
            "obs_dim":        self.obs_dim,
            "use_roles":      self.use_roles,
        }
        if self.use_intruder_policy:
            ckpt.update({
                "intruder_actor":          self.intruder_actor.state_dict(),
                "intruder_critic":         self.intruder_critic.state_dict(),
                "intruder_obs_norm_mean":  self.intruder_obs_norm.mean,
                "intruder_obs_norm_var":   self.intruder_obs_norm.var,
                "intruder_obs_norm_count": self.intruder_obs_norm.count,
                "intruder_obs_dim":        self.intruder_obs_dim,
                "intruder_max_offset":     self.intruder_max_offset,
            })
        torch.save(ckpt, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.obs_norm.mean  = ckpt["obs_norm_mean"]
        self.obs_norm.var   = ckpt["obs_norm_var"]
        self.obs_norm.count = ckpt["obs_norm_count"]
        self.total_steps    = ckpt.get("total_steps", 0)
        if self.use_intruder_policy and "intruder_actor" in ckpt:
            self.intruder_actor.load_state_dict(ckpt["intruder_actor"])
            self.intruder_critic.load_state_dict(ckpt["intruder_critic"])
            self.intruder_obs_norm.mean  = ckpt["intruder_obs_norm_mean"]
            self.intruder_obs_norm.var   = ckpt["intruder_obs_norm_var"]
            self.intruder_obs_norm.count = ckpt["intruder_obs_norm_count"]
        print(f"  [ckpt] loaded ← {path}")
