# Core 模块说明

## 1. 模块任务

`core/` 是整个项目的中心调度层与可视化层，负责：

- 订阅 Simulation 的 ROS2 数据；
- 维护一份系统状态镜像；
- 调用 `perception`、`marl`、`navdp`、`locomotion`；
- 把各模块结果组合成控制输出；
- 驱动 dashboard 展示全链状态。

它是整套系统的中枢。

---

## 2. 主要入口

入口脚本：

- [scripts/launch_core_dashboard.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_core_dashboard.sh)

核心文件：

- [core/ros2/factory_core/factory_core/control_node.py](/home/yyz/projects/Factory_MARL_Project/core/ros2/factory_core/factory_core/control_node.py)
- [core/ros2/factory_core/factory_core/visualization_node.py](/home/yyz/projects/Factory_MARL_Project/core/ros2/factory_core/factory_core/visualization_node.py)
- [core/ros2/factory_core/factory_core/state_mirror.py](/home/yyz/projects/Factory_MARL_Project/core/ros2/factory_core/factory_core/state_mirror.py)

---

## 3. 模块功能

### 3.1 状态订阅与镜像

订阅来自 Simulation 的：

- pose
- camera
- depth
- semantic
- IMU
- LiDAR
- locomotion observation
- aggregate state

并把这些数据统一镜像到 `state_mirror` 中。

### 3.2 外部模块调用

`Core` 负责调用：

- `perception`
- `marl`
- `navdp`
- `locomotion`

并把它们的结果统一管理。

### 3.3 控制链组合

典型控制链为：

```text
Simulation state -> Core -> MARL subgoal -> NavDP path -> Locomotion command -> Simulation
```

### 3.4 近距离接管 / fallback

当前 `Core` 自己负责近距离收口逻辑：

- `2m` 内：朝 intruder 前进并转向；
- `0.5m` 内：停止前进，只面朝 intruder；
- 如果 `NavDP` path 首尾距离 `< 0.2m`：
  - 也会进入只转头不前进状态。

### 3.5 Dashboard

提供浏览器可视化页面，展示：

- WorldState
- Robot
- Perception
- MARL
- NavDP
- Locomotion

---

## 4. 输入

### 4.1 来自 Simulation 的输入

通过 ROS2 获取：

- 机器人与 intruder 的 pose
- 机器人图像与深度图
- semantic segmentation
- IMU
- LiDAR
- locomotion observation

### 4.2 来自外部 HTTP 服务的输入

通过 HTTP 获取：

- `Perception` 的感知结果
- `MARL` 的子目标和角色
- `NavDP` 的 path
- `Locomotion` 的低层动作与速度结果

---

## 5. 输出

### 5.1 对 Simulation 的输出

通过 ROS2 发布：

- locomotion motion command

### 5.2 对浏览器的输出

通过 WebSocket / HTTP 状态接口输出镜像状态，用于 dashboard。

---

## 6. 当前关键逻辑

### 6.1 `body_velocity_command`

`Core` 会把规划结果转成给 locomotion 的：

- `vx`
- `vy`
- `wz`

在 fallback 情况下，`wz` 可直接打满到 `±1.0`。

### 6.2 Core fallback

当前 `Core` 的 fallback 已成为整链最后收口的主导层。

### 6.3 可观测性

`Core` 还负责把：

- planning output
- locomotion output
- marl output
- perception output

都镜像给 dashboard，方便排障。

---

## 7. 推荐运行环境

- `core`

---

## 8. 一句话概括

`core/` 是这套系统的中枢控制层，负责接收仿真数据、调度外部模块、输出控制命令，并把全链状态暴露给 dashboard。
