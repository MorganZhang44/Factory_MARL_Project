# Factory MARL 项目

这个仓库是一个基于 Isaac Sim / Isaac Lab 的模块化多智能体拦截系统。

当前已经打通的主链路是：

```text
Simulation -> Core -> NavDP -> Locomotion -> Simulation
```

可视化属于 Core，本身只读取 Core 的状态镜像，不直接进入控制闭环。

这个项目当前的组织原则是：

* 每个模块保持自己的运行边界
* 日常开发优先使用本机环境
* Docker 主要用于给队友分享稳定环境

在当前阶段，本机开发仍然是主工作流。

---

## 模块说明

当前正在使用的模块有：

* `simulation/` - Isaac Sim / Isaac Lab 场景运行模块
* `core/` - 总通信层与 dashboard
* `navdp/` - 路径规划适配层与真实 NavDP 接入
* `locomotion/` - 低层运动适配层与 policy
* `ros2/` - ROS2 工具、launch 资产和工作区相关内容

文档主要放在：

* `docs/architecture/`

建议优先阅读：

* [docs/architecture/key.md](docs/architecture/key.md)
* [docs/architecture/12_runtime_environments.md](docs/architecture/12_runtime_environments.md)
* [docs/architecture/13_stable_motion_baseline.md](docs/architecture/13_stable_motion_baseline.md)
* [docs/architecture/14_docker_and_github_management_plan.md](docs/architecture/14_docker_and_github_management_plan.md)

---

## 运行时归属

每个模块都只能在自己的环境里运行。

当前归属如下：

* `simulation` -> `isaaclab51`
* `core` + dashboard -> `core`
* `navdp` -> `navdp`
* `locomotion` -> `locomotion`
* `ros2/` 当前保留为工具和 launch 资产，不作为单独部署的运行服务

这是架构约束，不只是使用习惯。

---

## 本机快速启动

### 1. Simulation

运行环境：`isaaclab51`

```bash
./scripts/launch_simulation.sh --keep-open
```

现在默认会使用：

```text
--device cuda:0
```

如果你需要强制使用 CPU：

```bash
./scripts/launch_simulation.sh --device cpu
```

### 2. NavDP

运行环境：`navdp`

```bash
./scripts/launch_navdp.sh
```

如果你想给队友一个更稳的基线，可以强制让 NavDP 用 CPU：

```bash
NAVDP_DEVICE=cpu ./scripts/launch_navdp.sh
```

如果是 Docker 共享环境，当前更推荐使用：

```text
planner=auto + device=cpu
```

这样在真实 NavDP planner 无法满足运行条件时，会自动回退到直线路径基线，而不会直接把共享环境跑死。

### 3. Locomotion

运行环境：`locomotion`

```bash
./scripts/launch_locomotion.sh
```

### 4. Core + Dashboard

运行环境：`core`

```bash
./scripts/launch_core_dashboard.sh
```

Dashboard 地址：

```text
http://localhost:8770
```

Core 状态接口：

```text
http://localhost:8765
```

---

## 推荐启动顺序

建议分 4 个终端启动：

### 终端 1

```bash
./scripts/launch_simulation.sh --keep-open
```

### 终端 2

```bash
./scripts/launch_navdp.sh
```

### 终端 3

```bash
./scripts/launch_locomotion.sh
```

### 终端 4

```bash
./scripts/launch_core_dashboard.sh
```

然后浏览器打开：

```text
http://localhost:8770
```

---

## Docker 用法

这个仓库里的 Docker 主要用于给队友分享可复现环境。

当前仍然建议你自己优先使用本机环境开发。

### 当前已经准备的 Docker 范围

目前仓库已经补了这些模块的 Docker 支持：

* `core`
* `navdp`
* `locomotion`

另外还提供了一个 Simulation 的 headless 初始容器脚手架：

* `simulation/Dockerfile.headless`

这个 Simulation Docker 目前应当视为“可继续完善的基线版本”，而不是最终稳定的 Isaac Sim 容器方案。

### Compose

主 Compose 文件：

```text
compose.yaml
```

目前的 Compose 设计为 Linux 下使用 host networking，方便保持当前 ROS2 通信方式简单直接。

启动共享服务容器：

```bash
docker compose up --build core navdp locomotion
```

或者直接：

```bash
docker compose up --build
```

这会启动：

* Core
* NavDP
* Locomotion

默认不会启动 Simulation。

其中 `core` 容器会使用独立的容器内 ROS2 工作区卷，不直接复用宿主机的
`ros2/workspace` 构建产物，这样可以避免本机旧的 `colcon build/install`
状态与容器环境冲突。

### 可选的 Simulation Compose Profile

仓库里保留了一个实验性的 Simulation profile：

```bash
docker compose --profile simulation up --build simulation
```

这个 profile 主要用于 headless 基线验证和队友环境共享尝试，不是当前最推荐的本机开发路径。

注意：Isaac Sim 官方容器需要接受 NVIDIA Isaac Sim 的附加许可。仓库里的
Simulation Compose 配置已经默认传入：

```text
ACCEPT_EULA=Y
OMNI_KIT_ACCEPT_EULA=YES
```

---

## 环境分享策略

当前建议的工作方式是：

* 先在本机把环境和功能调通
* 当环境达到“值得分享”的稳定节点时，再同步更新 Docker

这意味着 Docker 不需要在每次很小的本地依赖实验之后都立即重建。

比较适合更新 Docker 的时机：

* 某个模块环境已经基本稳定
* 一条新链路已经跑通
* 队友准备开始接这个模块
* 准备合并稳定版本

---

## 队友使用建议

### 方式 A：适合日常开发

直接使用本机模块环境，按脚本启动。

这种方式最适合：

* 改代码
* 调试
* 改场景
* 调性能

### 方式 B：适合环境分享

使用 Docker 跑：

* `core`
* `navdp`
* `locomotion`

`simulation` 先建议仍然走本机，除非 headless 容器方案已经在对应机器上验证通过。

所以当前最实用的混合方式是：

* Simulation 在宿主机本机运行
* 其他服务模块走 Docker

---

## 当前重要端口

```text
Core state API:         8765
Dashboard frontend:     8770
NavDP adapter:          8889
Locomotion adapter:     8890
```

---

## 当前关键入口文件

Simulation：

* `simulation/standalone/validate_slam_scene.py`

Core：

* `core/ros2/factory_core/factory_core/control_node.py`
* `core/ros2/factory_core/factory_core/visualization_node.py`

NavDP：

* `navdp/navdp_service.py`

Locomotion：

* `locomotion/locomotion_service.py`

---

## 常见问题与排查

下面这些问题都是真正在当前项目里遇到过的，队友第一次拉起环境时很可能会踩到。

### 1. `docker: permission denied while trying to connect to the Docker daemon socket`

说明当前 shell 还没有拿到 Docker daemon 权限。

可以用下面任一方式处理：

```bash
newgrp docker
```

或者退出当前登录会话后重新登录。

如果只是临时执行一条命令，也可以用：

```bash
sg docker -c 'docker compose up --build'
```

---

### 2. `could not select device driver "" with capabilities: [[gpu]]`

说明 Docker 已安装，但 NVIDIA 容器运行时没有配好。

需要安装并配置 `nvidia-container-toolkit`，然后重启 Docker。

验证方式：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

如果这条命令不能正常打印 GPU 信息，Simulation Docker 也通常起不来。

---

### 3. Isaac Sim 容器提示需要接受 EULA

如果日志里出现：

```text
Please accept the EULA above by setting the ACCEPT_EULA environment variable.
```

说明容器里没有传入 Isaac Sim 所需的许可环境变量。

当前仓库里的 Compose 配置已经默认传入：

```text
ACCEPT_EULA=Y
OMNI_KIT_ACCEPT_EULA=YES
```

如果你自己手动运行容器，需要保持这两个变量都存在。

---

### 4. `docker-compose-plugin` 找不到

在部分 Ubuntu 22.04 环境里，系统仓库提供的是：

* `docker.io`
* `docker-compose-v2`

而不是 `docker-compose-plugin`。

如果 `apt install docker-compose-plugin` 报错，可以改成：

```bash
sudo apt install -y docker.io docker-compose-v2
```

---

### 5. Core Docker 一直重启，日志里有 `File exists` / `ament_index` / `colcon`

这是 ROS2 工作区构建产物冲突。

之前的问题是：

* 容器直接使用了宿主机 bind mount 进来的 `ros2/workspace`
* 里面已经有旧的 `build/install/log`
* `colcon build --symlink-install` 在容器里重建时与旧状态冲突

当前仓库已经把 `core` 容器改成使用独立命名卷工作区，并支持启动前自动清理：

```text
CORE_WORKSPACE=/docker-workspaces/core
CORE_CLEAN_WORKSPACE=1
```

如果你自己改 Compose 或启动脚本，不要再让容器直接复用宿主机旧的 `build/install/log`。

---

### 6. Simulation Docker 能发 topic，但 Core 收不到数据

这个问题之前不是 Docker 网络问题，而是 `core_control_node` 自己被控制循环卡住了。

具体现象：

* `ros2 topic list` 能看到 Simulation 的 topic
* `simulation` 自己能收到自己发出的 `/factory/simulation/state`
* 但 Core 的 `/api/state` 里还是全 `seen: false`

根因是：

* `core_control_node` 原来使用单线程 executor
* `50 Hz` 的控制循环里还在同步调用 NavDP / Locomotion HTTP 接口
* 结果把订阅回调饿住了

当前代码已经修复为：

* `MultiThreadedExecutor`
* 状态订阅与控制定时器拆到不同 callback group

如果后面又出现“能看到 topic 但 Core 不更新状态”的情况，优先回头检查这里。

---

### 7. Simulation Docker 日志里出现 `No module named 'isaaclab'`

说明容器里只有 Isaac Sim，没有把项目当前依赖的 Isaac Lab Python 包补进去。

当前仓库里的 `simulation/Dockerfile.headless` 已经安装：

```text
isaaclab==2.3.2.post1
```

并补了对应的 `PYTHONPATH`。

如果你更换了 Isaac Sim / Isaac Lab 版本，这里需要一起同步。

---

### 8. Simulation Docker 日志里说 `Could not import system rclpy`

如果后面马上出现：

```text
Attempting to load internal rclpy for ROS Distro: ...
rclpy loaded
```

这通常不是致命错误。

意思是：

* 系统 Python 环境里没有独立安装 `rclpy`
* 但 Isaac Sim 内置的 ROS2 bridge 已经接管成功

只有在后续真的没有 ROS2 topic 发出来时，这条日志才需要继续深挖。

---

### 9. Simulation Docker 日志里有大量 headless 图形警告

例如：

* `Invalid sync scope for buffer resource 'shared swapchain buffer'`
* `_createExtendCursor: No windowing`

这类告警在 headless Isaac Sim 容器里比较常见。

只要同时还能看到类似：

```text
app ready
Scene setup complete. Running validation steps...
Isaac Sim ROS2 publisher active under /factory/simulation
```

一般就不是阻塞性错误。

---

### 10. Docker 里的 NavDP 最稳默认配置是什么

如果只是给队友分享环境，当前更稳的设置是：

```text
NAVDP_PLANNER=auto
NAVDP_DEVICE=cpu
```

原因是：

* 当真实 planner 在当前机器条件下跑不起来时
* 它会回退到 `straight_line_v1`
* 不会直接把整套共享环境带崩

这是“先让队友跑起来”的更稳默认值。

---

### 11. Simulation Docker 首次 build 很慢

这是正常现象。

原因主要是：

* `nvcr.io/nvidia/isaac-sim:5.1.0` 基础镜像很大
* 首次拉取和构建耗时明显高于普通 Python 服务镜像

如果是第一次构建，请预留足够时间，不要因为下载慢就误判为构建卡死。

---

### 12. 如何快速判断整条 Docker 链是否正常

先启动：

```bash
docker compose up --build
docker compose --profile simulation up --build simulation
```

然后检查：

```bash
http://127.0.0.1:8765/health
http://127.0.0.1:8770/health
http://127.0.0.1:8889/health
http://127.0.0.1:8890/health
```

再检查：

```bash
http://127.0.0.1:8765/api/state
```

如果 `aggregate_state_seen=true`，而且机器人下的 `pose / camera / observation / planning / locomotion`
都已经变成 `seen: true`，说明当前端到端链路已经真正通了。

---

## 给队友的提醒

1. 不要在别的模块环境里启动当前模块。
2. Dashboard 只从 Core 读状态，不应插入控制闭环。
3. Simulation 是世界状态和动作应用的唯一真实来源。
4. 当前稳定路线跟随基线记录在：

   [docs/architecture/13_stable_motion_baseline.md](docs/architecture/13_stable_motion_baseline.md)

5. 当前 Docker 与 GitHub 管理策略记录在：

   [docs/architecture/14_docker_and_github_management_plan.md](docs/architecture/14_docker_and_github_management_plan.md)

---

## 当前状态

当前项目状态可以概括为：

* 本机端到端链路是主要支持路径
* `core / navdp / locomotion` 已经补了 Docker 分享基础设施
* Simulation Docker 目前是分阶段建设中的 headless baseline，应视为实验性方案
