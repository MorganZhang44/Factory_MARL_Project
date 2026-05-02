# Perception Environment Compat

这条目录是隔离出来的兼容仿真线。

目标：

- 原样保留 perception 原始项目里的环境结构
- 不改现有 `simulation/standalone/validate_slam_scene.py`
- 只做当前工程里狗和人相关的命名、初始化、运动控制、ROS2 topic 适配

入口：

- 运行脚本：`simulation/standalone/run_perception_env_compat.py`
- 启动命令：`./scripts/launch_simulation_perception_env.sh`

当前适配规则：

- `go2_dog_1 -> agent_1`
- `go2_dog_2 -> agent_2`
- `suspect -> intruder_1`
- CCTV 名称保持不变

说明：

- 这条线是并行方案，不替换现有 simulation 主线
- 当前隔离目录只包含 cloned `environment/`
- 当前兼容运行器默认不启用原始项目里的自主巡逻
- 狗和人会在启动后被重新压回当前项目旧版基线：
  - `agent_1`: `(-2.0, -2.0, 0.42)`, yaw `0`
  - `agent_2`: `(-2.0, 1.6, 0.42)`, yaw `0`
  - `intruder_1`: `(2.0, -0.5, 1.34)`, yaw `180`
- intruder 默认固定在基线位置和朝向，只接受我们后续明确接入的控制，不走自主巡逻
- 后续如果要继续适配，优先在这个目录和它的独立启动脚本里改
