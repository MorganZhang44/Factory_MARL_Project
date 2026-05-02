# Perception

当前项目里的 `perception/` 是一个独立 HTTP 服务模块。

它不直接启动 Isaac Sim，也不直接读取 scene。它吃的是 `core` 打包后的请求，
对外提供：

- `GET /health`
- `POST /estimate`

入口文件：

- `perception/perception_service.py`

## 当前保留的内容

这一版仓库里保留的是当前主链真正会用到的部分：

- `perception_service.py`
  - 当前项目的 HTTP 适配层
- `perception/perception/`
  - 感知算法内核
- `environment/static_scene_geometry.py`
  - 感知内核会用到的静态几何辅助
- `environment/types.py`
  - 当前运行时使用的数据结构
- `environment.yml`
  - `perception` 独立环境
- `Dockerfile`
  - `perception` Docker 镜像

之前从 `newtest` / `newtest2` 迁过来的环境侧运行脚手架、可视化脚手架、独立
pose server、视频转码等不再保留，因为当前项目主链没有用到它们。

## 本机启动

```bash
./scripts/launch_perception.sh
```

默认环境：

```text
perception
```

## Docker 启动

```bash
docker compose up --build perception
```

## 离线重放

可以用录下来的 perception request 直接离线回放：

```bash
python scripts/replay_perception_record.py output/perception_records --device cpu
```

## 运行边界

当前推荐的数据流是：

```text
simulation -> core -> perception
```

也就是说：

- `simulation` 发布原始传感器和状态
- `core` 负责镜像、打包、节流和 HTTP 调用
- `perception` 只负责估计输出
