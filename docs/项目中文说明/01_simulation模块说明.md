# Simulation 模块说明

## 1. 模块任务

`simulation/` 模块负责：

- 使用 Isaac Sim / Isaac Lab 搭建并运行工厂场景；
- 生成机器人、入侵者、相机、LiDAR 等仿真数据；
- 通过 ROS2 向 `Core` 发布仿真世界状态与传感器数据；
- 接收 `Core` 下发的 locomotion 控制命令，并将其作用到仿真中的机器人。

这是整个项目的“物理世界”和“数据源”。

---

## 2. 当前主要运行线

当前仓库里存在多条运行线：

- `legacy`
- `rewrite`
- `rebuild`

当前推荐、最稳定、最适合全链联调的是：

- `legacy`

入口脚本：

- [scripts/launch_simulation.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_simulation.sh)

主要入口文件：

- [simulation/standalone/validate_slam_scene.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/validate_slam_scene.py)

---

## 3. 模块功能

### 3.1 场景加载

- 加载 `slam_scene.usda`
- 构建地面、墙体、障碍物等静态环境

### 3.2 Actor 管理

- 生成两只 Go2 机器人：
  - `agent_1`
  - `agent_2`
- 生成一个 humanoid intruder：
  - `intruder_1`

### 3.3 传感器挂载

为每只狗提供：

- 前视 RGB camera
- depth camera
- semantic segmentation camera
- IMU
- RayCaster LiDAR

此外场景内还挂有多台固定 CCTV 相机。

### 3.4 ROS2 发布

向 `Core` 发布：

- 机器人 pose
- intruder pose
- RGB / depth / semantic 图像
- IMU
- LiDAR scan
- LiDAR point cloud
- locomotion observation
- aggregate state

### 3.5 控制执行

接收 `Core` 发布的 locomotion motion command，并在仿真中驱动机器人。

### 3.6 录像

当前支持两类视频同时录制：

- monitor-wall 合成视频
- top-down 俯视视频

---

## 4. 输入

### 4.1 来自启动命令的输入

常见参数包括：

- `--runtime legacy`
- `--keep-open`
- `--move-intruder`
- `--record-video`
- `--video-seconds`
- `--video-output`
- `--topdown-video-output`
- `--disable-ros2`

### 4.2 来自 Core 的输入

通过 ROS2 接收：

- `locomotion/motion_command`

其中包含：

- 机器人平面速度命令
- 低层关节 action（如果存在）

---

## 5. 输出

### 5.1 对 Core 的输出

通过 ROS2 输出：

- `/factory/simulation/state`
- `/factory/simulation/<robot>/pose`
- `/factory/simulation/<robot>/camera/image_raw`
- `/factory/simulation/<robot>/camera/depth`
- `/factory/simulation/<robot>/camera/semantic_segmentation`
- `/factory/simulation/<robot>/imu`
- `/factory/simulation/<robot>/lidar/scan`
- `/factory/simulation/<robot>/lidar/points`
- `/factory/simulation/<robot>/locomotion/observation`
- `/factory/simulation/cctv/...`

### 5.2 文件输出

录像时输出：

- monitor 视频
- top-down 视频

---

## 6. 当前行为约定

### 6.1 intruder

当前 `legacy` 下：

- 默认静止；
- 加 `--move-intruder` 时才沿固定路线移动。

### 6.2 无 locomotion 指令时

当前不会再每一拍强制回到默认站姿，而是保持当前关节 target。

### 6.3 推荐运行环境

- `isaaclab51`

---

## 7. 模块与其他模块的关系

它与其他模块的关系是：

```text
Simulation -> Core
Simulation <- Core
```

其中：

- `Simulation -> Core` 负责发布状态和传感器数据；
- `Core -> Simulation` 负责下发控制命令。

---

## 8. 一句话概括

`simulation/` 是整个项目的仿真物理世界与传感器数据源，同时也是控制命令最终落地执行的地方。
