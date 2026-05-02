# Environment Rewrite

这是一条全新重写的 simulation 环境线。

目标：

- 迁移 `test/environment_updated` 的环境层结构
- 不修改现有 `simulation/standalone/validate_slam_scene.py`
- 使用独立的 cfg / bridge / static geometry / local scene asset
- 继续适配当前项目的 ROS2 topic 契约

入口：

- 运行脚本：`simulation/standalone/run_environment_rewrite.py`
- 启动命令：`./scripts/launch_simulation_rewrite.sh`

当前约束：

- 内部实体命名保持 `go2_dog_1` / `go2_dog_2` / `suspect`
- 对外 ROS2 topic 仍映射成 `agent_1` / `agent_2` / `intruder_1`
- 默认不启用自主巡逻
- intruder 默认固定在旧版稳定基线位置和朝向

这条线和旧版 USDA runtime 并行存在，后续迁移应优先在这里继续做。
