# NavDP 模块说明

## 1. 模块任务

`navdp/` 是路径规划模块的服务适配层，负责把：

- 当前机器人状态
- 当前子目标

转换成一条世界系 waypoint 路径，供 `Core -> Locomotion` 使用。

---

## 2. 主要入口

入口脚本：

- [scripts/launch_navdp.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_navdp.sh)

服务文件：

- [navdp/navdp_service.py](/home/yyz/projects/Factory_MARL_Project/navdp/navdp_service.py)

---

## 3. 模块功能

### 3.1 路径规划服务

对外提供：

- `POST /plan`

### 3.2 规划模式

当前支持：

- straight-line fallback
- real NavDP planner（当环境与依赖满足时）

### 3.3 日志输出

现在每次规划都会把 waypoint 打印到终端，包括：

- robot
- planner
- count
- start
- goal
- first
- last
- 全量 waypoints

这对联调很有帮助。

---

## 4. 输入

来自 `Core` 的 HTTP 请求，通常包含：

- `robot_id`
- `robot_state.position`
- `robot_state.velocity`
- `subgoal`
- `robot_yaw`
- `local_goal`
- 可能还包含传感器数据（真实 planner 情况下）

---

## 5. 输出

通过 HTTP 返回：

- `planner`
- `waypoints`
- 在真实 planner 情况下可能有 `local_waypoints`

这些结果随后由 `Core` 镜像到 dashboard。

---

## 6. 当前在系统中的角色

典型链路为：

```text
MARL subgoal / chase subgoal -> NavDP -> waypoints -> Locomotion
```

但当前系统也允许 `Core` 在近距离或 path 失效时绕过 NavDP 的推进逻辑，进入 heading-only 或近距离接管。

---

## 7. 推荐运行环境

- `navdp`

---

## 8. 一句话概括

`navdp/` 是子目标到路径的桥梁层，它把上层目标转成 waypoint 路径，并在终端提供清晰的 path 可观测性。
