# Perception 模块说明

## 1. 模块任务

`perception/` 是感知服务模块，负责根据 `Core` 提供的打包输入，输出：

- 机器人状态估计；
- intruder 位置估计；
- 感知侧融合结果。

它不直接启动 Isaac Sim，也不自己接 ROS2 传感器。

---

## 2. 主要入口

入口脚本：

- [scripts/launch_perception.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_perception.sh)

服务文件：

- `perception/perception_service.py`

---

## 3. 模块功能

### 3.1 感知推理

接收 `Core` 打包后的请求，估计：

- 机器人位置 / 姿态 / 速度等状态；
- intruder 位置；
- 感知置信信息与融合结果。

### 3.2 离线回放支持

项目还支持对 perception request 做离线回放和分析。

---

## 4. 输入

来自 `Core` 的 HTTP 请求，一般包含：

- 图像
- LiDAR
- 机器人状态
- 仿真状态

`Core` 负责镜像、打包、节流与调用。

---

## 5. 输出

通过 HTTP 返回：

- 感知估计结果
- intruder 估计
- 相关辅助字段

并由 `Core` 镜像到 dashboard。

---

## 6. 运行边界

推荐数据流：

```text
Simulation -> Core -> Perception
```

也就是说：

- `Simulation` 只发原始数据；
- `Core` 负责组织输入；
- `Perception` 只做估计。

---

## 7. 推荐运行环境

- `perception`

---

## 8. 一句话概括

`perception/` 是一个纯 HTTP 感知服务，吃 `Core` 打包后的数据，专注做状态估计与 intruder 感知输出。
