# 04 · 奖励函数设计

**对应文件：** `marl/rewards/pursuit_reward.py`  
**对应配置：** `configs/mappo_config.yaml → reward:`

这是整个项目**最核心的设计部分**，也是迭代次数最多的模块。

---

## 最终奖励公式

```python
# marl/rewards/pursuit_reward.py  →  def compute():

total = r_progress      # 靠近奖励（密集）
      + r_pin           # 黏人奖励（持续）
      + r_capture       # 合围成功奖励（稀疏）
      + r_time_bonus    # 时间效率奖励
      + r_proximity     # 碰撞惩罚
      + r_step          # 步数惩罚
```

---

## 各奖励项详解

### 1. `r_progress`：密集进度奖励

```python
progress   = mean(dists_before - dists_after)   # 每步平均靠近了多少
r_progress = w_distance * progress * 10.0       # w_distance = 1.0
```

每步只要两只狗的平均距离缩短了，就加分。这是最基础的"追人"激励。

---

### 2. `r_pin`：黏人奖励（持续奖励）

```python
r_pin = 0.0
for d in dists_after:
    if d <= 2.0:           # 距离 Intruder 2 米以内
        r_pin += w_pin     # w_pin = 2.0，每只狗每步 +2
```

**设计动机**：让 AI 学会"先冲上去贴住目标，拿持续分，等队友包抄"。
没有这个奖励，AI 倾向于两只狗都保持距离，观望等待完美时机。

---

### 3. `r_capture`：合围成功奖励

```python
if d1 <= 1.5 and d2 <= 1.5:         # 两狗都在 1.5m 内
    cos_theta = dot(v1, v2) / (d1 * d2)
    if cos_theta <= 0.0:             # 夹角 ≥ 90°
        angle_quality = (1.0 - cos_theta) / 2.0   # 0.5（90°) ~ 1.0（180°）
        r_capture = w_capture * angle_quality      # w_capture = 200.0
        # 最高得分：完美对角线 180° 包围 → +200 分
        # 最低得分：恰好 90° 夹角    → +100 分
```

**夹角加成**：越接近 180 度的对角线包围，分数越高，激励 AI 追求更优质的合围阵型。

---

### 4. `r_time_bonus`：时间效率奖励

```python
# 只在 r_capture 触发时同步计算
time_fraction = (max_steps - step_count) / max_steps   # 剩余步数比例
r_time_bonus  = w_time * time_fraction                 # w_time = 100.0
```

| 抓捕时刻 | time_fraction | r_time_bonus |
|----------|--------------|--------------|
| 第 1 步  | 499/500 ≈ 1.0 | **+100 分** |
| 第 100 步 | 400/500 = 0.8 | +80 分 |
| 第 250 步 | 250/500 = 0.5 | +50 分 |
| 第 499 步 | 1/500 ≈ 0   | ≈ 0 分 |

**设计动机**：解决"AI 磨蹭绕圈等最佳阵型"的问题。
每磨蹭 10 步，不只扣 1 分（`r_step × 10`），还损失 2 分时间奖金，总代价 **3 分**，有效逼迫 AI 速战速决。

---

### 5. `r_proximity`：重叠惩罚

```python
d_agents = dist(agent_0, agent_1)
if d_agents < sep_threshold:     # sep_threshold = 1.5m
    r_proximity = w_proximity * (sep_threshold - d_agents)  # w_proximity = -0.5
```

防止两只狗堆在一起，保持最小间距，便于从两个方向包抄。

---

### 6. `r_step`：步数惩罚

```python
r_step = w_step   # w_step = -0.1，每步固定扣 0.1 分
```

---

## 奖励权重配置

```yaml
# configs/mappo_config.yaml
reward:
  w_distance:           1.0     # r_progress 权重
  w_capture:          200.0     # r_capture 权重（最大奖励）
  w_step:              -0.1     # 步数惩罚
  w_proximity:         -0.5     # 重叠惩罚
  separation_threshold: 1.5     # 重叠判定阈值（米）
  w_pin:               2.0      # 黏人奖励权重
  w_time:            100.0      # 时间效率奖励权重
```

---

## 奖励函数迭代历史

| 版本 | 主要变化 | 问题/结果 |
|------|---------|-----------|
| v1 | 单狗触碰即 Capture，per-step 角度惩罚 | AI "Reward Hacking"：原地不动等最佳角度 |
| v2 | 改为双人合围规则 + 加入 `r_pin` | AI 学会"主动贴身"，不再原地等待 |
| v3 | 加入 LiDAR 观测（`obs_dim: 13→21`） | AI 不再进入死胡同 |
| v4（当前）| 加入 `r_time_bonus` | AI 速战速决，MeanR 从 +550 提升至 +850 |
