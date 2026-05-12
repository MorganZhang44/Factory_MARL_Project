# Locomotion 模块说明

## 1. 模块任务

`locomotion/` 是低层运动策略模块，负责把：

- 路径信息
- 速度命令
- 机器人 observation

转换成：

- 低层 12D joint action
- world velocity 输出

供仿真中 Go2 机器人执行。

---

## 2. 主要入口

入口脚本：

- [scripts/launch_locomotion.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_locomotion.sh)

服务文件：

- [locomotion/locomotion_service.py](/home/yyz/projects/Factory_MARL_Project/locomotion/locomotion_service.py)

---

## 3. 模块功能

### 3.1 低层动作推理

根据 observation 和目标命令输出：

- 12D 关节动作

### 3.2 速度跟踪

当前输入命令的核心是：

- `body_velocity_command = [vx, vy, wz]`

模块会把这组命令写入 locomotion policy observation。

### 3.3 调试 / fallback 输出

同时保留：

- world velocity
- action scale

便于 `Core` 与 dashboard 调试。

---

## 4. 输入

来自 `Core` 的 HTTP 请求，通常包含：

- `robot_id`
- `robot_state`
- `path`
- `body_velocity_command`
- `locomotion_observation`
- `simulation_state`

---

## 5. 输出

通过 HTTP 返回：

- `velocity`
- `action`
- `action_scale`
- 其他控制辅助字段

这些输出会被 `Core` 进一步发送回 `Simulation`。

---

## 6. 当前系统里的作用

典型位置是：

```text
Core -> Locomotion -> Simulation
```

也就是：

- `Core` 决定高层意图；
- `Locomotion` 负责把高层意图变成低层 joint action。

---

## 7. 推荐运行环境

- `locomotion`

---

## 8. 一句话概括

`locomotion/` 是控制闭环的最后一层策略服务，负责把规划与速度命令变成 Go2 可执行的低层动作。
