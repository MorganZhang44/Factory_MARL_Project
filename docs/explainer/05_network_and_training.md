# 05 · 神经网络结构与 MAPPO 训练

**对应文件：** `marl/policies/actor.py`  
**对应文件：** `marl/policies/critic.py`  
**对应文件：** `marl/trainers/mappo_trainer.py`  
**对应文件：** `marl/buffers/rollout_buffer.py`

---

## Actor（策略网络）

**文件：** `marl/policies/actor.py`

Actor 是 AI 的"大脑"——给定观测，输出动作（Subgoal 偏移量）。

```python
# marl/policies/actor.py

class Actor(nn.Module):
    def __init__(self, obs_dim=21, action_dim=2, hidden_dim=64, map_half=10.0):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),   # 21 → 64
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh() # 64 → 64
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)  # 64 → 2
        self.log_std   = nn.Parameter(torch.zeros(action_dim))  # 可学习的标准差

    def get_action(self, obs, deterministic=False):
        feat   = self.net(obs)
        mean   = torch.tanh(self.mean_head(feat)) * self.map_half  # 输出 ∈ [-10, +10]
        std    = self.log_std.exp()
        dist   = Normal(mean, std)
        action = dist.mean if deterministic else dist.rsample()    # 推理用 mean，训练时采样
        return action.clamp(-map_half, map_half), log_prob
```

**网络结构：**
```
输入 (21维)
  → Linear(21→64) + Tanh
  → Linear(64→64) + Tanh
  → mean_head: Linear(64→2) + Tanh × map_half  →  Subgoal [dx, dy] ∈ [-10, +10]
```

在实际部署中，输出的 `[dx, dy]` 会被进一步限制在 **3 米以内**（`pursuit_env.py` 的 `step()` 里处理）。

---

## Critic（价值网络）

**文件：** `marl/policies/critic.py`

Critic 在**训练时**估计当前状态的价值（"这个局面大概能拿多少分"）。
**推理/Demo 时不使用 Critic**，只用 Actor。

```python
# marl/policies/critic.py
class Critic(nn.Module):
    # 结构类似 Actor，但输出 1 个标量（状态价值）
    # 输入可以是全局状态（两只狗的 obs 拼接），而不是单只狗的局部 obs
```

---

## 观测归一化：`marl/utils/normalizer.py`

```python
# 训练时在线统计观测的均值和方差，做 Running Normalization
obs_normalized = (obs - mean) / sqrt(var + 1e-8)
```

**为什么需要归一化？**
神经网络对输入数值的范围非常敏感。位置可能是 `[-10, 10]`，速度是 `[-1.5, 1.5]`，LiDAR 是 `[0, 1]`，量纲差异很大。归一化后统一到 `[-1, 1]` 附近，训练更稳定。

均值和方差随 `final.pt` 一起保存，推理时必须加载同一套统计数据，否则网络输入分布会错位。

---

## MAPPO 训练主循环

**文件：** `marl/trainers/mappo_trainer.py`  
**入口：** `scripts/train_mappo.py`

```
训练大循环（总计 300 万步）：
│
├── Phase 1：收集 Rollout（2048 步）
│   对每一步：
│     obs → Actor → action → env.step() → reward, next_obs
│     将 (obs, action, reward, done, value) 存入 rollout_buffer
│
├── Phase 2：计算优势函数 GAE
│   GAE(γ=0.99, λ=0.95) 估计每步动作的"相对好坏"
│
└── Phase 3：PPO 更新（重复 10 个 Epoch）
    将 2048 步数据分成 mini-batch（256 条）
    对每个 mini-batch：
      - 计算新旧策略比值 ratio
      - PPO Clip：防止更新幅度过大（clip_epsilon = 0.2）
      - 更新 Actor 和 Critic 权重
```

---

## 关键超参数

```yaml
# configs/mappo_config.yaml
mappo:
  lr_actor:       3.0e-4   # Actor 学习率
  lr_critic:      3.0e-4   # Critic 学习率
  gamma:          0.99     # 折扣因子（重视长期收益）
  gae_lambda:     0.95     # GAE 平滑系数
  clip_epsilon:   0.2      # PPO Clip 阈值（防止策略突变）
  n_epochs:       10       # 每批数据重复训练 10 次
  rollout_steps:  2048     # 每次收集 2048 步才更新
  mini_batch_size: 256     # 每个 mini-batch 大小
  total_timesteps: 3_000_000  # 总训练步数
```

---

## 检查点与模型权重

```
results/checkpoints/
├── step_100000.pt    ← 每 10 万步保存一次
├── step_200000.pt
├── ...
└── final.pt          ← 训练完成后的最终权重（推理时用这个）
```

`final.pt` 包含：
- `actor`：Actor 网络权重
- `critic`：Critic 网络权重
- `obs_norm_mean/var/count`：归一化统计数据（推理时必须加载）
