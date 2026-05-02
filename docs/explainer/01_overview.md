# 01 · 项目总览与文件夹结构

## 问题定义

用两只 **Unitree Go 2 机器狗**，在有障碍物的仓库环境里，协作围捕一个随机游走的移动目标（Intruder）。
核心挑战：两只狗需要学会**分工合作**，一只贴身压制，一只绕后包抄，形成包围圈。

## 算法选型：MAPPO

我们使用 **MAPPO（Multi-Agent Proximal Policy Optimization）**。

- 基于 PPO，专为多智能体场景设计
- **参数共享（Parameter Sharing）**：两只狗共用一套神经网络，但输入中包含 `agent_id`（0 或 1），让模型自动区分角色
- **集中式训练，分散式执行（CTDE）**：训练时利用全局信息优化，执行时每只狗只用自己的局部观测

## 文件夹结构

```
Factory_MARL_Project/
│
├── marl/                        ← ⭐ 我们写的 MARL 算法
│   ├── envs/pursuit_env.py      ← 仿真环境（物理引擎）
│   ├── policies/actor.py        ← 策略网络（Actor）
│   ├── policies/critic.py       ← 价值网络（Critic，训练专用）
│   ├── trainers/mappo_trainer.py← MAPPO 训练主循环
│   ├── rewards/pursuit_reward.py← 奖励函数（最核心的设计）
│   ├── buffers/rollout_buffer.py← 经验缓冲区
│   └── utils/
│       ├── map_utils.py         ← 地图工具 + LiDAR 射线投射
│       ├── astar.py             ← A* 路径规划
│       └── normalizer.py        ← 观测归一化器
│
├── scripts/                     ← ⭐ 运行入口
│   ├── train_mappo.py           ← 训练入口
│   ├── interactive_demo.py      ← 可视化 Demo（鼠标控制 intruder）
│   ├── analyze_subgoals.py      ← 分析脚本（生成路线图）
│   ├── debug_policy.py          ← 调试脚本
│   └── eval_mappo.py            ← 批量评估
│
├── configs/
│   └── mappo_config.yaml        ← 所有超参数（唯一的配置文件）
│
├── results/checkpoints/         ← 训练产出的模型权重
│   └── final.pt                 ← 最终使用的权重文件
│
├── simulation/                  ← Isaac Lab 3D 仿真（队友负责）
└── docs/explainer/              ← 本文档目录
```

## 各文件的依赖关系

```
train_mappo.py
  └── mappo_trainer.py
        ├── pursuit_env.py     （环境：每步移动、奖励、终止）
        │     ├── map_utils.py （地图、碰撞检测、LiDAR 射线）
        │     ├── astar.py     （A* 路径规划）
        │     └── pursuit_reward.py （奖励计算）
        ├── actor.py           （策略网络：obs → action）
        ├── critic.py          （价值网络：obs → value）
        ├── rollout_buffer.py  （缓冲区：存 experience）
        └── normalizer.py      （归一化：稳定训练输入）
```

> 训练完成后，推理/Demo 时只用到：`actor.py` + `normalizer.py` + `pursuit_env.py`
