# 06 · 系统分层架构与 Sim-to-Real

**对应文件：** `scripts/interactive_demo.py`（部署接口参考）  
**对应文件：** `scripts/train_mappo.py`（训练入口）

---

## 系统分层架构

我们的系统分为三层，每层职责完全分离：

```
┌──────────────────────────────────────────────────┐
│  Layer 1：MARL 决策层（神经网络）                  │
│                                                    │
│  输入：21 维观测向量（位置+速度+LiDAR）             │
│  输出：Subgoal [dx, dy]（下一步目标点偏移量）       │
│                                                    │
│  文件：marl/policies/actor.py                      │
│  权重：results/checkpoints/final.pt               │
└─────────────────────┬────────────────────────────┘
                       │  Subgoal 坐标
┌─────────────────────▼────────────────────────────┐
│  Layer 2：Navigation 执行层                        │
│                                                    │
│  仿真中：A* 路径规划（marl/utils/astar.py）         │
│  真实中：ROS 2 Nav2（ros2_ws/ 目录）               │
│                                                    │
│  负责：从当前位置规划避障路径到 Subgoal             │
└─────────────────────┬────────────────────────────┘
                       │  速度/位置指令
┌─────────────────────▼────────────────────────────┐
│  Layer 3：Unitree Go 2 实体                        │
│                                                    │
│  Camera：视觉感知                                  │
│  LiDAR：局部避障（底层安全保障）                   │
│  电机控制器：执行运动                              │
└──────────────────────────────────────────────────┘
```

---

## 为什么这样分层？

**关注点分离（Separation of Concerns）**：

| 层次 | 解决的问题 | 使用的技术 |
|------|-----------|-----------|
| MARL 决策层 | "战术上该去哪" | 强化学习、神经网络 |
| Navigation 层 | "物理上怎么走到那" | A* / ROS Nav2 |
| 机器人底层 | "安全执行，不撞人" | Camera + LiDAR + PID |

MARL 只需要输出一个坐标，完全不需要知道路上有什么障碍物，也不需要控制电机。这使得整个系统模块化，每层可以独立升级替换。

---

## Sim-to-Real 可靠性分析

### 为什么可以直接迁移？

**1. MARL 层的输入/输出格式不依赖仿真**

神经网络的输入（21 维向量）全部是抽象的数值：
- 位置/速度：归一化的浮点数，仿真和真实 Go 2 格式相同
- LiDAR 射线：仿真用 `ray_cast()` 计算，真实用 Go 2 雷达读取——**都是一组距离值，格式完全一致**

**2. Navigation 层独立替换**

仿真中用 A*，真实部署时换成 ROS 2 Nav2，MARL 层代码**零改动**。

**3. 无图像依赖**

我们没有把 RGB 图像作为 MARL 的输入（图像的 Sim-to-Real Gap 极大，仿真光影和现实不同）。选择 LiDAR 射线数据，避开了这个核心难题。

---

## 换地图是否需要重训？

| 场景 | 是否需要重训 | 原因 |
|------|------------|------|
| 相同规模的仓库（不同障碍物布局） | ❌ 不需要 / ⚠️ Fine-tune 50 万步 | LiDAR 射线提供了足够的局部感知 |
| 地图尺寸变化（比如 20m→40m） | ✅ 需要重训 | 坐标归一化的 `map_half` 参数改变 |
| Agent 数量从 2 变成 3 | ✅ 需要重训 | 观测向量和网络结构都要改 |

**Fine-tune 流程（换地图时）：**
```
1. 更新 marl/utils/map_utils.py 中的障碍物定义
2. 检查 configs/mappo_config.yaml 中的 map_half 是否需要调整
3. 加载现有 final.pt 作为初始权重，继续训练 50 万步
4. 总时间约 10 分钟（vs 从头训练的 25 分钟）
```

---

## 如何运行各个脚本

```bash
# 训练（300 万步，约 25 分钟）
python3 scripts/train_mappo.py

# 可视化 Demo（鼠标控制 Intruder，观察实时追捕）
python3 scripts/interactive_demo.py

# 策略路线分析（生成 6 个随机场景的 Subgoal 图）
python3 scripts/analyze_subgoals.py

# 调试：移动目标测试
python3 scripts/debug_policy.py

# 调试：固定目标测试（检查确定性）
python3 scripts/debug_policy.py --fixed --steps 20

# 批量评估（运行 100 局取平均抓捕率）
python3 scripts/eval_mappo.py
```
