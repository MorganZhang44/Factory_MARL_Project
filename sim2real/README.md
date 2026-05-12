# Sim2Real

`sim2real/` 现在是一条独立的真机接入线，目标是先把 Unitree Go2 的真实状态、传感器和相机稳定接进现有项目风格的 `core + dashboard`，再逐步往控制和更复杂模块扩展。

这套目录现在按“**只拿走 `sim2real/` 这一整个文件夹**”来组织：

- 启动脚本在 `sim2real/scripts/`
- ROS2 包在 `sim2real/ros2/`
- Unitree ROS2 消息副本在 `sim2real/unitree_ros2/`
- Unitree Python SDK 副本在 `sim2real/unitree_sdk2_python/`

也就是说，当前运行不再依赖仓库根目录下其他业务模块（`simulation/`、`marl/`、`navdp/`、`locomotion/`、`perception/`）。

## 当前结构

- `ros2/factory_core_sim2real`
  - `sim2real` 专用的 core 副本
  - 负责订阅真机 topic、汇总状态、提供 `/api/state` 和 websocket
- `ros2/factory_bringup_sim2real`
  - `sim2real` 专用 launch
- `unitree_ros2`
  - 官方 ROS2 消息/示例仓库的本地副本
- `unitree_sdk2_python`
  - 官方 Python SDK 的本地副本
- `scripts/go2_forward_back_test.py`
  - 最小真机运动测试脚本

## 当前已接入的数据

### 第一批

- `/sportmodestate`
- `/utlidar/robot_pose`
- `/utlidar/robot_odom`

### 第二批

- `/utlidar/imu`
- `/utlidar/cloud`

### 相机

相机没有直接走 `/frontvideostream` 解码，而是走官方推荐的 `VideoClient`：

- `camera_worker.py` 用 `unitree_sdk2py.go2.video.video_client.VideoClient`
- 周期性拉取 JPEG 帧
- 写到本地缓存：
  - `/tmp/factory_sim2real/front_camera.jpg`
- `core_control_node` 定时读取这张缓存图，喂给 dashboard

这样做的原因是：

- `VideoClient` 单独跑是稳定的
- 把 `VideoClient` 和 `rclpy` 硬塞进同一个 ROS2 节点时，`ChannelFactoryInitialize` 容易打架
- 分成独立 worker 后更稳，也更好排障

## 当前页面能力

`sim2real` dashboard 现在已经能显示：

- 机器人位姿
- `sportmodestate` 状态
- IMU 状态
- LiDAR 点云俯视图
  - 按高度着色
- 前相机画面

默认地址：

- `http://127.0.0.1:8770/`

Core 状态 API：

- `http://127.0.0.1:8765/`

## 环境准备

推荐直接用独立环境：

```bash
cd sim2real
./scripts/rebuild_env.sh
conda activate sim2real
```

`sim2real/environment.yml` 提供 conda 基础环境；
`sim2real/requirements.txt` 提供 pip 侧依赖。

如果你已经在 `sim2real` 环境里，只需要：

```bash
cd sim2real
pip install -r requirements.txt
```

## 启动 dashboard

```bash
conda activate sim2real
cd sim2real
./scripts/launch_dashboard.sh
```

这个脚本会自动做这些事：

1. 激活 `sim2real` conda 环境
2. 配置 CycloneDDS
3. source 本目录内的 `unitree_ros2/cyclonedds_ws/install/setup.bash`
4. build `ros2`
5. launch `factory_bringup_sim2real`

默认网卡：

- `eno1`

如需覆盖：

```bash
SIM2REAL_NET_IFACE=enp6s0 ./scripts/launch_dashboard.sh
```

## 构建 Unitree ROS2 消息工作区

如果项目内的 `unitree_ros2` 需要重新编：

```bash
conda activate sim2real
cd sim2real
./scripts/build_unitree_ros2.sh
```

## 真机运动测试

当前提供了一个非常小的运动脚本：

- `scripts/go2_forward_back_test.py`

它会执行：

1. `StandUp()`
2. `BalanceStand()`
3. 前进 `0.1 m/s` 持续 `1 秒`
4. 后退 `0.1 m/s` 持续 `1 秒`
5. `StopMove()`
6. `Damp()`

运行：

```bash
conda activate sim2real
cd sim2real
python scripts/go2_forward_back_test.py --iface eno1
```

如果狗已经站好，可以跳过站立步骤：

```bash
python scripts/go2_forward_back_test.py --iface eno1 --skip-standup
```

## 关键文件

- `ros2/factory_core_sim2real/factory_core_sim2real/control_node.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/state_mirror.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/visualization_node.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/camera_worker.py`
- `ros2/factory_bringup_sim2real/launch/core_dashboard.launch.py`
- `scripts/launch_dashboard.sh`

## 当前已知事实

- 点云当前默认读取：
  - `/utlidar/cloud`
- 雷达 frame：
  - `utlidar_lidar`
- 前相机当前通过 SDK 成功拉到：
  - `1920 x 1080`
  - `jpeg`
- dashboard 当前应能看到：
  - `pose=True`
  - `status=True`
  - `camera=True`
  - `imu=True`
  - `lidar_points=True`

## 后续自然方向

- 把 `/wirelesscontroller` 接进页面做人工接管观测
- 给相机和点云加录制/快照功能
- 把 `SportClient` 控制进一步封成 `sim2real` 内部 bridge
- 再决定是否把现有更高层模块接进真机链
