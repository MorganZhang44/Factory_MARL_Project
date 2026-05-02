# 03 · 观测空间与 LiDAR

**对应文件：** `marl/envs/pursuit_env.py → _get_obs()`  
**对应文件：** `marl/utils/map_utils.py → ray_cast()`

---

## 什么是观测空间（Observation Space）？

神经网络在每个决策时刻，只能"看到"我们放进 `obs` 向量里的信息。
设计观测空间，就是在决定："给 AI 提供哪些感知线索？"

---

## 当前观测向量：21 维

```python
# marl/envs/pursuit_env.py  →  def _get_obs():

obs[i] = np.concatenate([
    [float(i)],                          # 1:  agent_id（0=主攻, 1=包抄）
    self.agent_pos[i] / pos_scale,       # 2-3: 自身位置 [x, y]（归一化到 [-1,1]）
    self.agent_vel[i] / vel_scale,       # 4-5: 自身速度 [vx, vy]
    self.agent_pos[j] / pos_scale,       # 6-7: 队友位置 [x, y]
    self.agent_vel[j] / vel_scale,       # 8-9: 队友速度 [vx, vy]
    self.target_pos   / pos_scale,       # 10-11: Intruder 位置 [x, y]
    self.target_vel   / vel_scale,       # 12-13: Intruder 速度 [vx, vy]
    lidar,                               # 14-21: 8方向 LiDAR 距离
])
```

| 编号 | 内容 | 维度 |
|------|------|------|
| 1 | agent_id（0 或 1，用于角色分工） | 1 |
| 2-3 | 自身位置 | 2 |
| 4-5 | 自身速度 | 2 |
| 6-7 | 队友位置 | 2 |
| 8-9 | 队友速度 | 2 |
| 10-11 | Intruder 位置 | 2 |
| 12-13 | Intruder 速度 | 2 |
| **14-21** | **LiDAR 8 方向射线距离** | **8** |

---

## 为什么需要 LiDAR？

**加 LiDAR 之前**，神经网络对地图一无所知。它只知道"我在哪，目标在哪"，但完全不知道哪里有墙。

结果：神经网络会把 Subgoal 设在死胡同里，Agent 被 A* 乖乖带进墙角卡死，追捕失败。

**加 LiDAR 之后**，神经网络能感知到：
- 前方 0.3m 有墙 → 不能往前走
- 左侧 5m 是开路 → 可以绕左边包抄

---

## LiDAR 射线实现

```python
# marl/utils/map_utils.py  →  def ray_cast():

def ray_cast(self, ox, oy, angle, max_range=8.0, step=0.2):
    """
    从 (ox, oy) 出发，沿 angle 方向发射一条光线。
    每次前进 0.2m，直到碰到障碍物或达到最大距离 8m。
    返回归一化距离 [0, 1]：0 = 紧贴着墙，1 = 前方全是空地。
    """
    x, y, dist = ox, oy, 0.0
    while dist < max_range:
        x += cos(angle) * step
        y += sin(angle) * step
        dist += step
        if self.is_collision(x, y):
            return dist / max_range   # 有墙，返回归一化距离
    return 1.0                        # 全程无障碍，返回 1.0
```

8 条射线的方向（0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°）：

```python
# marl/envs/pursuit_env.py  →  _get_obs() 内

lidar_angles = [i * np.pi / 4 for i in range(8)]  # 8 个方向
lidar = np.array([
    self.obs_map.ray_cast(px, py, a, max_range=8.0)
    for a in lidar_angles
])
```

---

## Sim-to-Real 一致性

在仿真中，LiDAR 是用 `ray_cast()` 函数模拟的几何光线投射。
在真实 Go 2 上，LiDAR/深度相机也会返回各方向的距离值。

**数据格式完全一致**，神经网络在仿真里学到的"墙近了要绕开"这个本能，可以零代价迁移到真实机器人上。

---

## 观测空间的配置

```yaml
# configs/mappo_config.yaml
env:
  obs_dim: 21   # 13 维状态 + 8 维 LiDAR
```

> **注意**：`obs_dim` 必须与 `_get_obs()` 实际生成的向量长度完全一致，否则神经网络加载权重时会报 shape mismatch 错误。
