# MARL 模块说明

## 1. 模块任务

`marl/` 是多智能体决策模块，负责根据两只狗和 intruder 的状态，输出每只狗的世界系子目标。

它的核心职责是：

- 进行双狗协同追捕决策；
- 输出子目标；
- 提供角色分配（pursuer / encircler）；
- 提供策略状态给 dashboard 与 trace。

---

## 2. 主要入口

入口脚本：

- [scripts/launch_marl.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_marl.sh)

服务文件：

- [marl/marl_service.py](/home/yyz/projects/Factory_MARL_Project/marl/marl_service.py)

当前项目中同时保留了多个 release：

- `v9`
- `v13`
- `v13_1`

当前主链默认资源基线是 `v13` 系列。

---

## 3. 模块功能

### 3.1 角色分配

为两只狗输出：

- `pursuer`
- `encircler`

### 3.2 子目标规划

根据状态输出每只狗的：

- world-frame `subgoal`

### 3.3 地图 lidar 复刻

当前 service 内部复刻了 map-based lidar 特征，用于对齐 release observation 契约。

### 3.4 内部 fallback

当前 `marl_service` 已重新保留内部 fallback 能力：

- 能切到 closing fallback；
- 能输出 `fallback_active`、`decision_mode`、`lock_only`、`look_at` 等状态。

但当前实际控制链最后怎么收口，更多由 `Core` 决定。

### 3.5 Trace 记录

当前 `marl` 还会记录推理 trace，便于离线分析：

- 输入状态；
- roles；
- observation；
- normalized observation；
- policy action；
- chosen action；
- 输出 subgoal；
- spawn snapshot。

---

## 4. 输入

来自 `Core` 的 HTTP 请求，当前主链最关键的输入是：

- `agent_1` 的 world-frame position / velocity / yaw
- `agent_2` 的 world-frame position / velocity / yaw
- `intruder` 的 world-frame position / velocity

Service 内部会进一步构造 observation。

---

## 5. 输出

通过 HTTP 返回：

- 每只狗的 `subgoal`
- `offset`
- `role`
- `role_name`
- `decision_mode`
- `fallback_active`
- `lock_only`
- `look_at`

这些结果会被 `Core` 镜像到 dashboard。

---

## 6. 当前运行口径

当前这版 service 已不再是最早的简化 13D 接法，而是更贴近 release 契约的 runtime。

同时，项目中还保留了原始 release demo，用于：

- 研究策略行为；
- 对照 fallback 逻辑；
- 做独立可视化检查。

---

## 7. 推荐运行环境

- `marl`

---

## 8. 一句话概括

`marl/` 是多智能体策略模块，负责双狗协同子目标规划和角色分配，并提供较强的可观测性与离线分析能力。
