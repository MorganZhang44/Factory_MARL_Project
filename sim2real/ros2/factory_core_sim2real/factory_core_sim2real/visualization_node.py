from __future__ import annotations

import re
from typing import Any

import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from rclpy.node import Node


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sim2Real Robot Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background: #f5f7fb;
      color: #172233;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7fb; }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #ffffff;
      border-bottom: 1px solid #dbe2ee;
    }
    h1 { margin: 0; font-size: 18px; }
    main {
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(340px, 420px);
      gap: 16px;
    }
    .panel {
      background: #ffffff;
      border: 1px solid #dbe2ee;
      border-radius: 10px;
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 15px;
      font-weight: 650;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      border: 1px solid #dce5f2;
      background: #f8fbff;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
    }
    .ok { background: #1f9b5c; }
    .wait { background: #9aa7bc; }
    .stale { background: #d39a1a; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    canvas {
      width: 100%;
      display: block;
      background: #fbfcff;
      border: 1px solid #e3e9f2;
      border-radius: 8px;
    }
    .image-box {
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border: 1px solid #e3e9f2;
      border-radius: 8px;
      background: #eef2f8;
      display: grid;
      place-items: center;
    }
    .image-box img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .metric {
      border: 1px solid #e3e9f3;
      border-radius: 8px;
      padding: 10px;
      background: #fbfcff;
    }
    .metric .label {
      font-size: 11px;
      color: #5e6a80;
      margin-bottom: 4px;
    }
    .metric .value {
      font-size: 14px;
      font-weight: 650;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 6px 4px;
      border-bottom: 1px solid #edf1f7;
      vertical-align: top;
    }
    th { color: #5e6a80; width: 120px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      max-height: 520px;
      overflow: auto;
    }
    .note {
      font-size: 12px;
      color: #5e6a80;
      line-height: 1.45;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Sim2Real Robot Dashboard</h1>
    <div id="connection" class="note">Connecting to core websocket</div>
  </header>
  <main>
    <section class="panel">
      <div class="title">Robot State</div>
      <div class="chips" id="status-chips"></div>
      <div class="grid" id="metric-grid"></div>
      <div>
        <div class="note" style="margin-bottom:8px;">Front Camera</div>
        <div id="camera-box" class="image-box"><span class="note">Waiting for camera image</span></div>
      </div>
      <div>
        <div class="note" style="margin-bottom:8px;">LiDAR top-down point cloud</div>
        <canvas id="lidar-canvas" width="420" height="260"></canvas>
      </div>
      <table>
        <tbody id="detail-table"></tbody>
      </table>
    </section>
    <section class="panel">
      <div class="title">Raw Snapshot</div>
      <div class="note">The raw mirrored snapshot is kept here so additional real-robot fields can be added later.</div>
      <pre id="raw-json">{}</pre>
    </section>
  </main>
<script>
const CORE_WS_URL = "__CORE_WS_URL__";

function dotClass(fresh, seen) {
  if (fresh) return "ok";
  if (seen) return "stale";
  return "wait";
}

function fmtNum(v, digits = 3) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "--";
  return v.toFixed(digits);
}

function fmtVec(v, digits = 3) {
  if (!Array.isArray(v)) return "--";
  return "[" + v.map(x => fmtNum(x, digits)).join(", ") + "]";
}

function setMetricGrid(metrics) {
  const grid = document.getElementById("metric-grid");
  grid.innerHTML = metrics.map(({label, value}) => `
    <div class="metric">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");
}

function setDetailTable(rows) {
  const table = document.getElementById("detail-table");
  table.innerHTML = rows.map(([k, v]) => `
    <tr><th>${k}</th><td>${v}</td></tr>
  `).join("");
}

function drawArrow(ctx, cx, cy, angle, length, color) {
  const tx = cx + Math.cos(angle) * length;
  const ty = cy - Math.sin(angle) * length;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  const head = 7;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - Math.cos(angle - Math.PI / 6) * head, ty + Math.sin(angle - Math.PI / 6) * head);
  ctx.lineTo(tx - Math.cos(angle + Math.PI / 6) * head, ty + Math.sin(angle + Math.PI / 6) * head);
  ctx.closePath();
  ctx.fill();
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function heightColor(z, zMin, zMax, fresh) {
  const span = Math.max(1e-6, zMax - zMin);
  const t = clamp01((z - zMin) / span);
  const stops = [
    [61, 110, 245],   // low: blue
    [36, 163, 95],    // mid-low: green
    [215, 154, 26],   // mid-high: amber
    [203, 74, 63],    // high: red
  ];
  const scaled = t * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const localT = scaled - idx;
  const c0 = stops[idx];
  const c1 = stops[idx + 1];
  const r = Math.round(lerp(c0[0], c1[0], localT));
  const g = Math.round(lerp(c0[1], c1[1], localT));
  const b = Math.round(lerp(c0[2], c1[2], localT));
  const a = fresh ? 0.82 : 0.55;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function drawLidar(canvas, lidarPoints = {}) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.42;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dfe6f0";
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * frac, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath(); ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius); ctx.stroke();

  const points = Array.isArray(lidarPoints.points_xyz) ? lidarPoints.points_xyz : [];
  const maxPointRange = points.reduce((acc, point) => Math.max(acc, Math.hypot(point[0] || 0, point[1] || 0)), 0);
  const range = Math.max(1, Math.min(maxPointRange * 1.2 || 8, 12));
  const zValues = points.map(point => Number(point[2]) || 0);
  const zMin = zValues.length ? Math.min(...zValues) : 0;
  const zMax = zValues.length ? Math.max(...zValues) : 1;
  for (const point of points) {
    const px = cx + (point[0] / range) * radius;
    const py = cy - (point[1] / range) * radius;
    ctx.fillStyle = heightColor(Number(point[2]) || 0, zMin, zMax, !!lidarPoints.fresh);
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  drawArrow(ctx, cx, cy, 0, 12, "#1b6fd1");
  ctx.fillStyle = "#4d5b71";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`${points.length} pts`, 8, 16);
  ctx.fillText(`z ${zMin.toFixed(2)}-${zMax.toFixed(2)} m`, 8, height - 10);
  ctx.fillText(`${range.toFixed(1)} m`, width - 56, height - 10);

  const legendX = width - 18;
  const legendY = 26;
  const legendH = height - 52;
  const grad = ctx.createLinearGradient(0, legendY + legendH, 0, legendY);
  grad.addColorStop(0.0, "rgba(61,110,245,0.95)");
  grad.addColorStop(0.35, "rgba(36,163,95,0.95)");
  grad.addColorStop(0.7, "rgba(215,154,26,0.95)");
  grad.addColorStop(1.0, "rgba(203,74,63,0.95)");
  ctx.fillStyle = grad;
  ctx.fillRect(legendX, legendY, 8, legendH);
  ctx.strokeStyle = "#cfd7e5";
  ctx.strokeRect(legendX, legendY, 8, legendH);
  ctx.fillStyle = "#4d5b71";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(zMax.toFixed(2), legendX - 38, legendY + 4);
  ctx.fillText(zMin.toFixed(2), legendX - 38, legendY + legendH);
}

function renderState(state) {
  const robot = (state.robots && state.robots.agent_1) || {};
  const pose = robot.pose || {};
  const status = robot.status || {};
  const camera = robot.camera || {};
  const imu = robot.imu || {};
  const lidarPoints = robot.lidar_points || {};

  document.getElementById("status-chips").innerHTML = [
    ["pose", pose.fresh, pose.seen],
    ["sport", status.fresh, status.seen],
    ["camera", camera.fresh, camera.seen],
    ["imu", imu.fresh, imu.seen],
    ["lidar", lidarPoints.fresh, lidarPoints.seen],
  ].map(([label, fresh, seen]) => `
    <div class="chip"><span class="dot ${dotClass(fresh, seen)}"></span>${label}</div>
  `).join("");

  const sportMode = status.mode;
  const velocity = status.velocity || ((state.aggregate_state || {}).robot_velocities || {}).agent_1;
  setMetricGrid([
    { label: "Mode", value: sportMode ?? "--" },
    { label: "Pose XY", value: fmtVec((pose.position || []).slice(0, 2)) },
    { label: "Yaw", value: fmtNum(pose.yaw, 3) },
    { label: "Velocity", value: fmtVec(velocity) },
    { label: "Yaw Speed", value: fmtNum(status.yaw_speed, 3) },
    { label: "Body Height", value: fmtNum(status.body_height, 3) },
    { label: "Camera", value: camera.width ? `${camera.width}×${camera.height}` : "--" },
    { label: "Lidar Points", value: lidarPoints.point_count ?? "--" },
  ]);

  setDetailTable([
    ["Pose frame", pose.frame_id || "--"],
    ["Pose position", fmtVec(pose.position)],
    ["Pose orientation", fmtVec(pose.orientation)],
    ["Sport position", fmtVec(status.position)],
    ["Sport velocity", fmtVec(status.velocity)],
    ["Gait type", status.gait_type ?? "--"],
    ["Progress", fmtNum(status.progress, 3)],
    ["Foot raise", fmtNum(status.foot_raise_height, 3)],
    ["Gyroscope", fmtVec(imu.angular_velocity)],
    ["Acceleration", fmtVec(imu.linear_acceleration)],
    ["IMU orientation", fmtVec(imu.orientation)],
    ["Camera frame", camera.frame_id || "--"],
    ["Camera encoding", camera.encoding || "--"],
    ["Lidar frame", lidarPoints.frame_id || "--"],
    ["Lidar point count", lidarPoints.point_count ?? "--"],
  ]);

  const cameraBox = document.getElementById("camera-box");
  if (camera.image) {
    cameraBox.innerHTML = `<img src="${camera.image}" alt="front camera" />`;
  } else {
    cameraBox.innerHTML = `<span class="note">Waiting for camera image</span>`;
  }

  drawLidar(document.getElementById("lidar-canvas"), lidarPoints);

  document.getElementById("raw-json").textContent = JSON.stringify({
    aggregate_state: state.aggregate_state,
    robot: robot,
  }, null, 2);
}

function connect() {
  const status = document.getElementById("connection");
  const ws = new WebSocket(CORE_WS_URL);
  ws.onopen = () => { status.textContent = "Connected to core websocket"; };
  ws.onclose = () => {
    status.textContent = "Connection lost, retrying in 1s";
    setTimeout(connect, 1000);
  };
  ws.onerror = () => { status.textContent = "WebSocket error"; };
  ws.onmessage = (event) => {
    try {
      renderState(JSON.parse(event.data));
    } catch (err) {
      status.textContent = "Failed to parse state update";
      console.error(err);
    }
  };
}

connect();
</script>
</body>
</html>
"""


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("factory_core_sim2real_visualization")
        self.declare_parameter("web_host", "0.0.0.0")
        self.declare_parameter("web_port", 8770)
        self.declare_parameter("core_ws_url", "ws://127.0.0.1:8765/ws")
        self.web_host = str(self.get_parameter("web_host").value)
        self.web_port = int(self.get_parameter("web_port").value)
        self.core_ws_url = str(self.get_parameter("core_ws_url").value)
        self._server: uvicorn.Server | None = None

        app = FastAPI(title="Factory Sim2Real Visualization")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(self._render_page())

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "owner": "visualization_sim2real"}

        config = uvicorn.Config(app, host=self.web_host, port=self.web_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self.get_logger().info(
            f"Sim2real dashboard listening on http://{self.web_host}:{self.web_port}; "
            f"reading Core state from {self.core_ws_url}"
        )
        self._server.run()

    def _render_page(self) -> str:
        page = HTML_PAGE.replace("__CORE_WS_URL__", self.core_ws_url)
        return re.sub(r"__[^_]+__", "", page)

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
