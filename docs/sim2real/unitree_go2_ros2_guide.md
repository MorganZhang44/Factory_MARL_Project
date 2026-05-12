# Unitree Go2 + Ubuntu + ROS2 控制完整流程

> 基于宇树官方仓库（`unitree_ros2` / `unitree_sdk2` / `unitree_sdk2_python`）整理。
> 少数项标注 `[待验证]` 的为社区单源信息，使用前请二次核对。
> 整理日期：2026-05-10

---

## 0. 总体架构

Go2 的上层运动算法（行走、转弯、跳跃等）是封装好的 "sport service"，跑在狗内部的板子上。用户机（Ubuntu PC）通过 **有线网口 + CycloneDDS**（不是 ROS 默认的 FastDDS）与狗通信。

两条通信路径：

| 路径 | 用途 | 哪些型号能用 |
|---|---|---|
| 有线 + CycloneDDS（DDS topic） | 高层运控、读传感器、低层关节力矩 | **EDU** 全开放；**Air/Pro** 只能读和高层运控 |
| WiFi + WebRTC | App 控制、远程视频流 | 全部，但功能受限 |

**研究/科研一律走有线 DDS**。下面所有步骤默认你的 Go2 是 **EDU**，或者只做高层控制不需要写 `/lowcmd`。

---

## 1. 软硬件版本矩阵

| 维度 | 推荐 | 说明 |
|---|---|---|
| Go2 型号 | EDU（科研推荐） | Air/Pro 不能写 `/lowcmd`，只能高层 |
| Ubuntu | **22.04 LTS** | 20.04 也行，对应 ROS2 Foxy |
| ROS2 distro | **Humble**（22.04）/ Foxy（20.04） | 24.04+Jazzy 官方未列，社区能跑但要折腾 [待验证] |
| CycloneDDS | **必须 0.10.2** | Humble 下 apt 自带版本可用；Foxy 下必须源码编译 |
| GCC / CMake | gcc 9.4+, CMake ≥ 3.10 | |
| Python | ≥ 3.8 | |

**铁律**：不要用老的 `unitree_legged_sdk`，那是 Go1 的，Go2 跑不起来。

---

## 2. 安装依赖

假设 Ubuntu 22.04 + ROS2 Humble，**先装好 ROS2 Humble desktop full**（这步走 ROS 官方文档，省略）。

```bash
# C++ SDK 编译依赖
sudo apt install cmake g++ build-essential libyaml-cpp-dev \
    libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev

# ROS2 + CycloneDDS 桥
sudo apt install ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl libyaml-cpp-dev

# 调试工具
sudo apt install ros-humble-rviz2 ros-humble-plotjuggler-ros \
    ros-humble-rqt-graph
```

**Foxy 用户** 额外需要源码编译 CycloneDDS 0.10.2（Humble 跳过这一段）：

```bash
mkdir -p ~/unitree_ros2/cyclonedds_ws/src && cd ~/unitree_ros2/cyclonedds_ws/src
git clone https://github.com/ros2/rmw_cyclonedds -b foxy
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd .. && colcon build --packages-select cyclonedds
# 编译前要在 .bashrc 里临时注释掉 source /opt/ros/foxy/setup.bash
```

**Python SDK**（高层运动用 Python 写很方便）：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python
pip3 install -e .
# 自动装 cyclonedds==0.10.2, numpy, opencv-python
```

**克隆 ROS2 包仓库**：

```bash
cd ~ && git clone https://github.com/unitreerobotics/unitree_ros2
```

---

## 3. 网络连接

| 项 | 值 |
|---|---|
| 物理接线 | 网线接 Go2 **背部朝后** 的 RJ45 网口（不是 USB-C） |
| Go2 后端口 IP | `192.168.123.18` |
| Go2 内部 eth0 | `192.168.123.161` |
| **PC 网口 IP** | **`192.168.123.99`**（官方示例） |
| 子网掩码 | `255.255.255.0` |
| 子网段 | `192.168.123.0/24` |

**配置 PC 网口**（GUI 或命令行二选一）：

```bash
# 命令行示例（替换 enp3s0 成你的网口名）
sudo ip addr add 192.168.123.99/24 dev enp3s0
sudo ip link set enp3s0 up

# 验证
ping 192.168.123.18           # 通即说明物理层 OK
ssh unitree@192.168.123.18    # 默认密码 123 [待验证：依固件版本]
```

**查自己 PC 网口名**：`ip a` 找有线那一行（典型 `enp3s0` / `enp2s0` / `eno1`）。

**配置 CycloneDDS 环境** —— 这是最关键的一步。在 `~/unitree_ros2/setup.sh` 写：

```bash
#!/bin/bash
source /opt/ros/humble/setup.bash
source $HOME/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
    <NetworkInterface name="enp3s0" priority="default" multicast="default" />
</Interfaces></General></Domain></CycloneDDS>'
```

**注意**：`name="enp3s0"` 必须替换成你 PC 的实际网口名，写错了 `ros2 topic list` 看不到狗的任何话题。

**改完 IP 必须重启 PC**（官方 README 强调，常见症状：不重启就拿不到 topic）。

---

## 4. 编译 ROS2 包

```bash
source ~/unitree_ros2/setup.sh
cd ~/unitree_ros2 && colcon build           # 编译消息包 unitree_go / unitree_api
cd ~/unitree_ros2/example && colcon build   # 编译示例
```

每次开新终端都得 `source ~/unitree_ros2/setup.sh`，建议直接写进 `.bashrc`。

---

## 5. 核心 Topic 一览

狗一上电、PC 网络配好后，跑：

```bash
ros2 topic list
```

应当看到至少这些（前缀可能是 `/` 或 `/lf/`）：

| Topic | Msg type | 含义 |
|---|---|---|
| `/sportmodestate` | `unitree_go/msg/SportModeState` | 高层状态：位置、速度、步态、IMU、足端力 |
| `/lowstate` | `unitree_go/msg/LowState` | 20 个关节的 q/dq/tau/温度，BMS 电量 |
| `/lowcmd` | `unitree_go/msg/LowCmd` | 低层关节命令（**仅 EDU 能写**） |
| `/wirelesscontroller` | `unitree_go/msg/WirelessController` | 手柄摇杆 lx/ly/rx/ry + 按键 |
| `/utlidar/cloud` | `sensor_msgs/PointCloud2` | LiDAR 点云，frame `utlidar_lidar` |
| `/utlidar/imu` | `sensor_msgs/Imu` | LiDAR 自带 IMU [待验证] |
| `/api/sport/request` | `unitree_api/msg/Request` | 高层运动命令通道（用 SportClient 包装） |

`SportModeState.mode` 枚举：`0=idle, 1=balanceStand, 3=locomotion, 5=lieDown, 7=damping, 8=recoveryStand, 10=sit`

`SportModeState.gait_type`：`0=idle, 1=trot, 2=run, 3=climb stair, 4=forwardDownStair, 9=adjust`

---

## 6. 让狗动起来 —— 高层运动 API

最简单的就是 Python 调 `SportClient`。先 **把狗吊起来或者放在软垫上**，再跑下面这段。

```python
# go2_walk_demo.py
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 第一参数 = DDS domain id（默认 0），第二参数 = 网卡名
ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else "")

sc = SportClient()
sc.SetTimeout(10.0)
sc.Init()

sc.StandUp();         time.sleep(2)
sc.BalanceStand();    time.sleep(1)

# 前进 0.3 m/s 持续 3 秒
sc.Move(0.3, 0.0, 0.0); time.sleep(3); sc.StopMove(); time.sleep(1)

# 原地左转 0.5 rad/s 持续 2 秒
sc.Move(0.0, 0.0, 0.5); time.sleep(2); sc.StopMove(); time.sleep(1)

# 横向左移
sc.Move(0.0, 0.2, 0.0); time.sleep(2); sc.StopMove()

sc.StandDown();       time.sleep(2)
sc.Damp()             # 卸力，必做
```

跑：`python3 go2_walk_demo.py enp3s0`

`Move(vx, vy, vyaw)` 单位：m/s, m/s, rad/s。

**SportClient 全部方法**（来自 `sport_client.hpp`）：

| 类别 | 方法 |
|---|---|
| 站立/卸力 | `StandUp`, `StandDown`, `BalanceStand`, `RecoveryStand`, `Damp`, `StopMove` |
| 行走 | `Move(vx,vy,vyaw)`, `SpeedLevel(0/1/2)`, `Euler(roll,pitch,yaw)` |
| 步态切换 | `TrotRun`, `StaticWalk`, `EconomicGait`, `ClassicWalk(bool)`, `WalkUpright(bool)`, `CrossStep(bool)` |
| 表演动作 | `Hello`, `Stretch`, `Sit`, `RiseSit`, `Scrape`, `Heart`, `Content`, `Dance1`, `Dance2` |
| 高难度 | `FrontFlip`, `FrontJump`, `BackFlip`, `LeftFlip`, `HandStand(bool)` |
| 模式 | `FreeWalk`, `FreeBound(bool)`, `FreeJump(bool)`, `FreeAvoid(bool)`, `SwitchAvoidMode` |
| 杂项 | `AutoRecoverSet/Get(bool)`, `SwitchJoystick(bool)` |

**Air 型号** 不支持 `FreeAvoid / CrossStep / FreeBound / FreeJump / HandStand`。

**C++ 版本** 几乎一致：

```cpp
#include <unitree/robot/go2/sport/sport_client.hpp>
using namespace unitree::robot::go2;

SportClient sc;
sc.SetTimeout(10.0f);
sc.Init();
sc.StandUp();
sc.Move(0.3f, 0.0f, 0.0f);
```

也可直接跑官方 example：

```bash
./install/unitree_ros2_example/bin/sport_mode_ctrl
# ⚠ 这个例程会自动让狗向前走 1m，跑前留好至少 2m 净空
```

---

## 7. 摄像头数据

Go2 前置 RGB 相机：**1280×720 @ 15Hz，水平 100°**。

**关键点**：官方 SDK 默认 **没有** 把相机以 ROS topic 形式发出来，要走 `VideoClient` 拉 JPEG 帧：

```python
import sys, cv2, numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else "")
vc = VideoClient(); vc.SetTimeout(3.0); vc.Init()

while True:
    code, data = vc.GetImageSample()
    if code != 0:
        continue
    img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("go2", img)
    if cv2.waitKey(1) == 27:
        break
```

如果你 **一定要 ROS topic** 形式的相机，两条路：

1. 自己写 ROS2 node，把上面的 JPEG 转 `sensor_msgs/Image` 发布。
2. 用第三方 [`abizovnuralem/go2_ros2_sdk`](https://github.com/abizovnuralem/go2_ros2_sdk)，它把视频流封装成了 `/camera/image_raw`。

---

## 8. LiDAR 数据

Go2 自带的是 **4D LiDAR L1**。它默认就发到 ROS2：

```bash
ros2 topic list | grep utlidar
ros2 topic hz   /utlidar/cloud         # 应该 ~10Hz
ros2 topic info /utlidar/cloud         # type: sensor_msgs/msg/PointCloud2

# 可视化
ros2 run rviz2 rviz2
# 在 rviz：Fixed Frame 改成 utlidar_lidar → Add → By topic → /utlidar/cloud
```

LiDAR 坐标原点在底部安装面中心，+X 方向与出线方向相反。

**注意**：独立 LiDAR L1 用户需要装 `unilidar_sdk`，但 Go2 自带的不用 —— 直接订 `/utlidar/cloud` 就够了。

---

## 9. 调试命令大全

```bash
# 拓扑/频率/带宽
ros2 topic list
ros2 topic info /sportmodestate
ros2 topic hz   /lowstate
ros2 topic bw   /utlidar/cloud
ros2 node list
ros2 node info /your_node

# 看消息内容
ros2 topic echo /sportmodestate
ros2 topic echo --no-arr /lowstate            # 数组截断好读
ros2 topic echo --no-arr /utlidar/cloud
ros2 topic echo /wirelesscontroller

# 录包/回放
ros2 bag record /sportmodestate /lowstate /utlidar/cloud -o run01
ros2 bag info run01
ros2 bag play run01

# 可视化
ros2 run rviz2 rviz2
ros2 run rqt_graph rqt_graph
ros2 run plotjuggler plotjuggler              # 拖 topic 进去画曲线

# 现成 example 二进制（编译 example 后产生）
./install/unitree_ros2_example/bin/read_motion_state
./install/unitree_ros2_example/bin/read_low_state
./install/unitree_ros2_example/bin/read_wireless_controller
./install/unitree_ros2_example/bin/sport_mode_ctrl     # ⚠ 会动
./install/unitree_ros2_example/bin/low_level_ctrl      # ⚠ 仅 EDU 直接驱关节
```

---

## 10. 安全 / 常见坑

1. **第一次跑代码必须把狗吊起来或放软垫上**。先做 IMU + 腿部校准（蓝灯闪 → 绿灯 = 完成），再用绳挂起来验 `StandUp`，确认正常再放地面。
2. **紧急停止**：手柄 `L2(长按) + B(单击)` → Damping mode（电机断力，狗会瘫倒）。`L2 + X` → 恢复站立。
3. **`/lowcmd` 是 EDU 专属**。Air/Pro 即便代码不报错也不会响应；强行写还可能触发服务异常。
4. **CycloneDDS 版本必须严格 0.10.2**。版本错了 topic 列表为空但不报错。
5. **网卡名写错** → 同上，topic 列表空。先 `ip a` 确认。
6. **改完 IP 必须重启 PC**（README 原话）。
7. **高层 vs 低层切换**：发了 `/lowcmd` 后高层 sport service 会被禁用；要恢复用 `RobotStateClient.ServiceSwitch("sport_mode", true)`，或重启狗 [待验证]。
8. **`sport_mode_ctrl` 例程默认让狗向前走 1m**，启动前清场。
9. **老固件有 RCE 漏洞**（CVE-2026-27509/27510），上手前先 App 升级到最新固件，且不要把狗暴露公网。
10. **WiFi 路径** 只能跑高层运控（WebRTC），不能写 `/lowcmd`，研究用一律走有线。

---

## 11. 推荐学习顺序

1. PC 配 IP → `ping 192.168.123.18` 通
2. `ros2 topic list` 看到 `/sportmodestate` 等
3. 跑 `read_motion_state` 例程读状态，先 **不动** 狗
4. 把狗吊起来 → Python 跑 `StandUp` + `StandDown` + `Damp`
5. 验证完放地面 → `Move(0.2, 0, 0)` 试前进
6. `rviz2` 看 `/utlidar/cloud`
7. `VideoClient` 拉相机
8. 有需要再碰 `/lowcmd`（EDU only）

---

## 来源

- [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) — ROS2 包主仓
- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) — C++ SDK
- [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — Python SDK
- [sport_client.hpp](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/sport/sport_client.hpp) — 完整 API 签名
- [go2_sport_client.cpp](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_sport_client.cpp) — C++ 例程
- [go2_sport_client.py](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/high_level/go2_sport_client.py) — Python 例程
- [Quadruped.de Go2 manual](https://www.docs.quadruped.de/projects/go2/html/controller.html) — 手柄/Air vs EDU 差异
- [abizovnuralem/go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk) — 第三方 ROS2 桥（相机/WebRTC）
- [Mybotshop Go2 network setting](https://forum.mybotshop.de/t/unitree-go2-network-setting/1153) — 网络配置实测
