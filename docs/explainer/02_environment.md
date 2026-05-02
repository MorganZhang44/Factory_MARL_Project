# 02 · 仿真环境与物理逻辑

**对应文件：** `marl/envs/pursuit_env.py`

---

## 环境的作用

`PursuitEnv` 是整个仿真的"物理引擎"，负责：
1. 管理 Agent 和 Intruder 的位置、速度
2. 每步执行 Agent 的动作（移动）
3. 驱动 Intruder 做随机游走
4. 计算这一步的奖励
5. 判断是否结束（Episode Termination）

---

## 关键函数：`step(actions)`

```python
# marl/envs/pursuit_env.py  →  def step(self, actions):

def step(self, actions):
    # 1. 将 AI 输出的 Subgoal 交给 A* 移动 Agent
    for i in range(self.n_agents):
        raw = actions[i]
        mag = np.linalg.norm(raw)
        if mag > 3.0:          # ← 限制最大 3 米，保证响应灵敏
            raw = (raw / mag) * 3.0
        sg_world = self.agent_pos[i] + raw
        self._subgoals[i] = np.clip(sg_world, ...)   # 不能出地图

    for i in range(self.n_agents):
        path = astar(self.obs_map, agent_pos[i], subgoal[i])  # A* 规划
        self.agent_pos[i], self.agent_vel[i] = move_along_path(path)

    # 2. 驱动 Intruder 随机游走
    self.target_pos, self.target_vel = self._step_intruder()

    # 3. 计算奖励
    rewards = self.reward_fn.compute(...)

    # 4. 判断结束条件（双人合围）
    captured = check_encirclement(...)
    return obs, rewards, terminated, truncated, info
```

---

## Intruder 的随机游走逻辑

```python
# marl/envs/pursuit_env.py  →  def _step_intruder():

def _step_intruder(self):
    noise = np.random.normal(0, 0.3, 2)    # 每步加随机扰动
    vel = self.target_vel + noise
    vel = normalize(vel) * self.intruder_spd  # 保持恒定速度 1.0 m/s

    new_pos = self.target_pos + vel * self.dt
    if collision(new_pos):   # 撞墙就反弹
        vel = -vel
        new_pos = self.target_pos + vel * self.dt
    return new_pos, vel
```

Intruder 速度恒定为 `1.0 m/s`，方向带噪声，撞墙反弹。训练时，Agent 的速度上限是 `1.5 m/s`，保证追得上。

---

## 终止条件：双人合围

这是我们设计的最关键规则——**单只狗碰到 Intruder 不算成功！**

```python
# marl/envs/pursuit_env.py  →  step() 第 4 步

captured = False
d1, d2 = dist(agent_0, target), dist(agent_1, target)

if d1 <= 1.5 and d2 <= 1.5:          # 两狗都在 1.5m 范围内
    v1 = agent_0_pos - target_pos
    v2 = agent_1_pos - target_pos
    cos_theta = dot(v1, v2) / (d1 * d2)
    if cos_theta <= 0.0:              # 夹角 ≥ 90°，两狗在目标两侧
        captured = True               # 才算真正 Capture！
```

这条规则从环境层面**强制要求协作**，AI 不能靠单兵作战完成任务。

---

## 地图工具：`marl/utils/map_utils.py`

```python
# 仓库地图的所有障碍物都在这里定义
PERIMETER_WALLS = [
    RectObstacle(cx=-0.85, cy=3.45, w=9.52, h=0.16),  # 后墙
    RectObstacle(cx=-5.50, cy=-0.65, w=0.16, h=8.16), # 左墙
    ...
]
INTERIOR_WALLS = [
    RectObstacle(cx=-2.30, cy=-0.85, w=0.425, h=1.275), # 左中柱
    RectObstacle(cx=0.40, cy=-0.95, w=0.425, h=1.375),  # 中心柱
    RectObstacle(cx=2.95, cy=-0.65, w=0.45, h=1.40),    # 右中柱
]
```

`ObstacleMap` 把这些几何体转换成一个 40×40 的二进制栅格（Grid），供 A* 路径规划使用。

---

## A* 路径规划：`marl/utils/astar.py`

```python
# 用法示意
path = astar(
    obstacle_map = env.obs_map,
    start_world  = (agent_x, agent_y),   # 起点（世界坐标）
    goal_world   = (subgoal_x, subgoal_y) # 终点（MARL 给的 Subgoal）
)
# 返回一串中间路径点，Agent 沿着这些点移动，自动绕开障碍物
```

A* 是传统算法，不需要学习，对地图有完整的"上帝视角"。它负责"怎么走到目标"，MARL 负责"战术上去哪"。
