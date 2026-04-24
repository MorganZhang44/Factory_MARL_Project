from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from rclpy.node import Node


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "simulation" / "assets" / "scenes" / "slam_scene.usda"


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Factory Core Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6fa;
      color: #192334;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6fa; color: #192334; }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid #d8e0ec;
      background: #ffffff;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.45; }
    .app {
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      border-right: 1px solid #d8e0ec;
      background: #f8faff;
      padding: 14px;
      display: grid;
      align-content: start;
      gap: 10px;
    }
    .nav-group {
      display: grid;
      gap: 8px;
    }
    .nav-button {
      border: 1px solid #d7e0ee;
      background: #ffffff;
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      font-size: 13px;
      color: #192334;
      cursor: pointer;
    }
    .nav-button.active {
      background: #e9f1ff;
      border-color: #a9c0f3;
    }
    .nav-note {
      font-size: 12px;
      color: #5d6980;
      line-height: 1.45;
    }
    main {
      padding: 14px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .page { display: none !important; gap: 14px; }
    .page.active { display: grid !important; }
    .two-col { grid-template-columns: minmax(420px, 1.35fr) minmax(320px, 1fr); }
    .full-grid { grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
    .section {
      background: #ffffff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      padding: 12px;
    }
    .section h2 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }
    .subheading {
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 650;
      color: #314056;
    }
    .stack { display: grid; gap: 10px; }
    .grid { display: grid; gap: 12px; }
    .summary-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .summary-card, .robot-card, .module-card {
      border: 1px solid #e2e8f1;
      border-radius: 8px;
      padding: 10px;
      background: #fbfcff;
    }
    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 650;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #dbe4f0;
      background: #f7faff;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: #304055;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
    }
    .ok { background: #24a35f; }
    .stale { background: #d39a1a; }
    .waiting { background: #99a5ba; }
    canvas {
      width: 100%;
      display: block;
      background: #fbfcff;
      border: 1px solid #e3e9f2;
      border-radius: 6px;
    }
    .image-box {
      width: 100%;
      aspect-ratio: 4 / 3;
      overflow: hidden;
      border: 1px solid #e3e9f2;
      border-radius: 6px;
      background: #eef2f8;
      display: grid;
      place-items: center;
    }
    .image-box img { width: 100%; height: 100%; object-fit: cover; image-rendering: pixelated; }
    .robot-layout {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
      gap: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 6px 7px;
      border-bottom: 1px solid #ebeff5;
      text-align: left;
      vertical-align: top;
    }
    th { color: #59657a; font-weight: 650; }
    .metric-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid #e2e8f1;
      border-radius: 8px;
      padding: 8px;
      background: #fbfcff;
    }
    .metric .label { font-size: 11px; color: #637189; margin-bottom: 4px; }
    .metric .value { font-size: 14px; font-weight: 650; }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
      gap: 12px;
    }
    .small-note { font-size: 12px; color: #5f6c82; line-height: 1.45; }
    .raw {
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 420px;
      overflow: auto;
    }
    @media (max-width: 1120px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid #d8e0ec; }
      .two-col, .split, .robot-layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Factory Core Dashboard</h1>
    <div class="mono" id="connection">connecting to core</div>
  </header>
  <div class="app">
    <aside>
      <div class="nav-group">
        <button class="nav-button active" data-page="worldstate">WorldState</button>
        <button class="nav-button" data-page="robot">Robot</button>
        <button class="nav-button" data-page="navdp">NavDP</button>
        <button class="nav-button" data-page="locomotion">Locomotion</button>
      </div>
      <div class="nav-note">
        Dashboard reads mirrored state from Core WebSocket only. It does not sit in the control loop.
      </div>
      <div class="section">
        <div class="subheading">Quick Status</div>
        <div class="stack" id="module-summary"></div>
      </div>
    </aside>
    <main>
      <section class="page two-col active" id="page-worldstate">
        <div class="section stack">
          <h2>WorldState</h2>
          <canvas id="world-map" width="1024" height="700"></canvas>
        </div>
        <div class="stack">
          <div class="section">
            <h2>Module State</h2>
            <div class="grid summary-grid" id="world-module-grid"></div>
          </div>
          <div class="section">
            <h2>Active Entities</h2>
            <div class="stack" id="world-entity-list"></div>
          </div>
          <div class="section">
            <h2>Flow Summary</h2>
            <div class="small-note" id="world-summary"></div>
          </div>
        </div>
      </section>

      <section class="page stack" id="page-robot">
        <div class="section">
          <h2>Robot</h2>
          <div class="small-note">Per-robot sensor view, motion state, joint telemetry, and observed pose from locomotion observation.</div>
        </div>
        <div class="grid full-grid" id="robot-grid"></div>
      </section>

      <section class="page stack" id="page-navdp">
        <div class="section">
          <h2>NavDP</h2>
          <div class="small-note">Camera input, planning connection state, planning frequency, and current route overlay.</div>
        </div>
        <div class="grid full-grid" id="navdp-grid"></div>
      </section>

      <section class="page stack" id="page-locomotion">
        <div class="section">
          <h2>Locomotion</h2>
          <div class="small-note">Command connection state, update frequency, action summary, and commanded pose preview.</div>
        </div>
        <div class="grid full-grid" id="locomotion-grid"></div>
      </section>
    </main>
  </div>
<script>
const CORE_WS_URL = "__CORE_WS_URL__";
const SCENE_MAP = __SCENE_MAP__;
const jointOrder = [
  "FL_hip", "FR_hip", "RL_hip", "RR_hip",
  "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
  "FL_calf", "FR_calf", "RL_calf", "RR_calf"
];

const connection = document.getElementById("connection");
const moduleSummary = document.getElementById("module-summary");
const worldModuleGrid = document.getElementById("world-module-grid");
const worldEntityList = document.getElementById("world-entity-list");
const worldSummary = document.getElementById("world-summary");
const worldCanvas = document.getElementById("world-map");
const robotGrid = document.getElementById("robot-grid");
const navdpGrid = document.getElementById("navdp-grid");
const locomotionGrid = document.getElementById("locomotion-grid");

let latestData = null;

function fmtNum(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "-";
}

function fmtHz(value) {
  const num = Number(value);
  return Number.isFinite(num) ? `${num.toFixed(1)} Hz` : "-";
}

function fmtVec(value, digits = 2) {
  if (!Array.isArray(value)) return "-";
  return `[${value.slice(0, 3).map(v => fmtNum(v, digits)).join(", ")}]`;
}

function normStatus(item) {
  if (!item || !item.seen) return "waiting";
  return item.fresh ? "ok" : "stale";
}

function dot(status) {
  return `<span class="status-dot ${status}"></span>${status}`;
}

function yawFromPose(pose) {
  return Number.isFinite(Number(pose && pose.yaw)) ? Number(pose.yaw) : 0;
}

function buildWorldBounds(data) {
  const xs = [];
  const ys = [];
  for (const rect of SCENE_MAP.rects || []) {
    xs.push(rect.center[0] - rect.scale[0] * 0.6, rect.center[0] + rect.scale[0] * 0.6);
    ys.push(rect.center[1] - rect.scale[1] * 0.6, rect.center[1] + rect.scale[1] * 0.6);
  }
  for (const robot of Object.values(data.robots || {})) {
    const pos = robot.pose && robot.pose.position;
    if (Array.isArray(pos)) {
      xs.push(pos[0]);
      ys.push(pos[1]);
    }
    const waypoints = (robot.planning && robot.planning.waypoints) || [];
    for (const point of waypoints) {
      xs.push(point[0]);
      ys.push(point[1]);
    }
  }
  for (const intruder of Object.values(data.intruders || {})) {
    const pos = intruder.pose && intruder.pose.position;
    if (Array.isArray(pos)) {
      xs.push(pos[0]);
      ys.push(pos[1]);
    }
  }
  if (!xs.length || !ys.length) return { xmin: -7, xmax: 7, ymin: -6, ymax: 5 };
  return {
    xmin: Math.min(...xs) - 1.0,
    xmax: Math.max(...xs) + 1.0,
    ymin: Math.min(...ys) - 1.0,
    ymax: Math.max(...ys) + 1.0,
  };
}

function makeProjector(canvas, bounds) {
  const pad = 26;
  const width = canvas.width - pad * 2;
  const height = canvas.height - pad * 2;
  const sx = width / Math.max(1e-6, bounds.xmax - bounds.xmin);
  const sy = height / Math.max(1e-6, bounds.ymax - bounds.ymin);
  const scale = Math.min(sx, sy);
  const ox = pad + (width - (bounds.xmax - bounds.xmin) * scale) * 0.5;
  const oy = pad + (height - (bounds.ymax - bounds.ymin) * scale) * 0.5;
  return {
    x(value) { return ox + (value - bounds.xmin) * scale; },
    y(value) { return canvas.height - (oy + (value - bounds.ymin) * scale); },
    scale,
  };
}

function drawArrow(ctx, x, y, yaw, size, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-yaw);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size * 0.75, -size * 0.55);
  ctx.lineTo(-size * 0.45, 0);
  ctx.lineTo(-size * 0.75, size * 0.55);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawSceneMap(canvas, data, focusRobotId = null) {
  const ctx = canvas.getContext("2d");
  const bounds = buildWorldBounds(data);
  const proj = makeProjector(canvas, bounds);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const stepMeters = 1;
  ctx.strokeStyle = "#e2e8f1";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(bounds.xmin); x <= bounds.xmax; x += stepMeters) {
    ctx.beginPath();
    ctx.moveTo(proj.x(x), proj.y(bounds.ymin));
    ctx.lineTo(proj.x(x), proj.y(bounds.ymax));
    ctx.stroke();
  }
  for (let y = Math.ceil(bounds.ymin); y <= bounds.ymax; y += stepMeters) {
    ctx.beginPath();
    ctx.moveTo(proj.x(bounds.xmin), proj.y(y));
    ctx.lineTo(proj.x(bounds.xmax), proj.y(y));
    ctx.stroke();
  }

  for (const rect of SCENE_MAP.rects || []) {
    const color = rect.kind === "floor" ? "#d8d8ce" : (rect.kind === "wall" ? "#2d3138" : "#aa654f");
    ctx.fillStyle = color;
    const w = rect.scale[0] * proj.scale;
    const h = rect.scale[1] * proj.scale;
    const x = proj.x(rect.center[0]) - w * 0.5;
    const y = proj.y(rect.center[1]) - h * 0.5;
    ctx.fillRect(x, y, w, h);
  }

  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const lidar = robot.lidar || {};
    const pose = robot.pose || {};
    if (Array.isArray(pose.position) && Array.isArray(lidar.points)) {
      ctx.fillStyle = robotId === focusRobotId ? "rgba(48, 126, 235, 0.18)" : "rgba(36, 163, 95, 0.14)";
      const rx = pose.position[0];
      const ry = pose.position[1];
      const yaw = yawFromPose(pose);
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      for (const point of lidar.points) {
        const px = rx + point[0] * cosYaw - point[1] * sinYaw;
        const py = ry + point[0] * sinYaw + point[1] * cosYaw;
        ctx.beginPath();
        ctx.arc(proj.x(px), proj.y(py), robotId === focusRobotId ? 2 : 1.3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const planning = robot.planning || {};
    const waypoints = Array.isArray(planning.waypoints) ? planning.waypoints : [];
    if (waypoints.length > 1) {
      ctx.strokeStyle = robotId === focusRobotId ? "#6a3bc2" : "#7f8aa3";
      ctx.lineWidth = robotId === focusRobotId ? 3.5 : 2.5;
      ctx.beginPath();
      waypoints.forEach((point, index) => {
        if (index === 0) ctx.moveTo(proj.x(point[0]), proj.y(point[1]));
        else ctx.lineTo(proj.x(point[0]), proj.y(point[1]));
      });
      ctx.stroke();
    }
    if (Array.isArray(planning.subgoal)) {
      ctx.strokeStyle = "#e09b15";
      ctx.lineWidth = 2.5;
      const gx = proj.x(planning.subgoal[0]);
      const gy = proj.y(planning.subgoal[1]);
      ctx.beginPath();
      ctx.arc(gx, gy, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(gx - 12, gy);
      ctx.lineTo(gx + 12, gy);
      ctx.moveTo(gx, gy - 12);
      ctx.lineTo(gx, gy + 12);
      ctx.stroke();
    }
  }

  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const pose = robot.pose || {};
    if (!Array.isArray(pose.position)) continue;
    const x = proj.x(pose.position[0]);
    const y = proj.y(pose.position[1]);
    ctx.fillStyle = robotId === focusRobotId ? "#1b6fd1" : "#2a8d62";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
    drawArrow(ctx, x, y, yawFromPose(pose), 14, "#103e73");
    ctx.fillStyle = "#1b2430";
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText(robotId, x + 10, y - 10);
  }

  for (const [intruderId, intruder] of Object.entries(data.intruders || {})) {
    const pose = intruder.pose || {};
    if (!Array.isArray(pose.position)) continue;
    const x = proj.x(pose.position[0]);
    const y = proj.y(pose.position[1]);
    ctx.strokeStyle = "#c94b42";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x - 8, y - 8);
    ctx.lineTo(x + 8, y + 8);
    ctx.moveTo(x + 8, y - 8);
    ctx.lineTo(x - 8, y + 8);
    ctx.stroke();
    ctx.fillStyle = "#1b2430";
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText(intruderId, x + 12, y - 10);
  }

  ctx.fillStyle = "#4d5b71";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("x [m]", canvas.width - 52, canvas.height - 14);
  ctx.save();
  ctx.translate(18, 48);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("y [m]", 0, 0);
  ctx.restore();
}

function drawLidar(canvas, lidar) {
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
  const points = Array.isArray(lidar.points) ? lidar.points : [];
  const maxPointRange = points.reduce((acc, point) => Math.max(acc, Math.hypot(point[0] || 0, point[1] || 0)), 0);
  const range = Math.max(1, Math.min(Number(lidar.range_max) || 12, maxPointRange * 1.2 || 8));
  ctx.fillStyle = lidar.fresh ? "rgba(36, 163, 95, 0.75)" : "rgba(153, 165, 186, 0.6)";
  for (const point of points) {
    const px = cx + (point[0] / range) * radius;
    const py = cy - (point[1] / range) * radius;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  drawArrow(ctx, cx, cy, 0, 10, "#1b6fd1");
  ctx.fillStyle = "#4d5b71";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`${points.length} pts`, 8, 16);
  ctx.fillText(`${range.toFixed(1)} m`, width - 52, height - 10);
}

function dogPoseFromJoints(jointRel, actionScale = 1.0) {
  const values = Array.isArray(jointRel) ? jointRel : [];
  const index = Object.fromEntries(jointOrder.map((name, i) => [name, values[i] || 0]));
  function leg(nameHip, nameThigh, nameCalf, side, front) {
    const hipY = side * (0.22 + (index[nameHip] || 0) * 0.12 * actionScale);
    const hipX = front * 0.42;
    const upper = 0.43;
    const lower = 0.44;
    const thigh = 0.85 + (index[nameThigh] || 0) * actionScale;
    const calf = -1.35 + (index[nameCalf] || 0) * actionScale;
    const kneeX = hipX + Math.cos(thigh) * upper;
    const kneeZ = -0.02 - Math.sin(thigh) * upper;
    const footX = kneeX + Math.cos(thigh + calf) * lower;
    const footZ = kneeZ - Math.sin(thigh + calf) * lower;
    return {
      hip: [hipX, hipY, 0.0],
      knee: [kneeX, hipY, kneeZ],
      foot: [footX, hipY, footZ],
    };
  }
  return {
    body: {
      length: 0.95,
      width: 0.36,
      height: 0.28,
    },
    legs: [
      leg("FL_hip", "FL_thigh", "FL_calf", 1, 1),
      leg("FR_hip", "FR_thigh", "FR_calf", -1, 1),
      leg("RL_hip", "RL_thigh", "RL_calf", 1, -1),
      leg("RR_hip", "RR_thigh", "RR_calf", -1, -1),
    ],
  };
}

function projectPoint3d(point) {
  const x = point[0];
  const y = point[1];
  const z = point[2];
  return [x + y * 0.45, -z - y * 0.18];
}

function drawDogPose(canvas, jointRel, accent, actionScale = 1.0) {
  const ctx = canvas.getContext("2d");
  const pose = dogPoseFromJoints(jointRel, actionScale);
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dfe6f0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(22, height - 28);
  ctx.lineTo(width - 18, height - 28);
  ctx.stroke();

  const scale = Math.min(width / 2.5, height / 1.9);
  const cx = width * 0.5;
  const cy = height * 0.62;
  function tp(point) {
    const [px, py] = projectPoint3d(point);
    return [cx + px * scale * 0.42, cy + py * scale * 0.42];
  }

  const bodyCorners = [
    tp([ pose.body.length * 0.5,  pose.body.width * 0.5, pose.body.height]),
    tp([ pose.body.length * 0.5, -pose.body.width * 0.5, pose.body.height]),
    tp([-pose.body.length * 0.5, -pose.body.width * 0.5, pose.body.height]),
    tp([-pose.body.length * 0.5,  pose.body.width * 0.5, pose.body.height]),
  ];
  ctx.fillStyle = "#b7cadf";
  ctx.beginPath();
  bodyCorners.forEach((point, index) => index === 0 ? ctx.moveTo(point[0], point[1]) : ctx.lineTo(point[0], point[1]));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#6f8095";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.strokeStyle = accent;
  ctx.lineWidth = 4;
  for (const leg of pose.legs) {
    const hip = tp(leg.hip);
    const knee = tp(leg.knee);
    const foot = tp(leg.foot);
    ctx.beginPath();
    ctx.moveTo(hip[0], hip[1]);
    ctx.lineTo(knee[0], knee[1]);
    ctx.lineTo(foot[0], foot[1]);
    ctx.stroke();
    ctx.fillStyle = accent;
    for (const point of [hip, knee, foot]) {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = "#4d5b71";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("observed / commanded posture", 10, 18);
}

function renderModuleSummary(data) {
  const robots = Object.values(data.robots || {});
  const simulationOk = robots.some(robot => robot.pose && robot.pose.fresh) || Object.values(data.intruders || {}).some(entry => entry.pose && entry.pose.fresh);
  const navdpOk = robots.some(robot => robot.planning && robot.planning.seen);
  const locomotionOk = robots.some(robot => robot.locomotion && robot.locomotion.seen);
  const robotObsOk = robots.some(robot => robot.observation && robot.observation.seen);
  const items = [
    ["core", "ok", "state mirror serving dashboard"],
    ["simulation", simulationOk ? "ok" : "waiting", `aggregate ${data.aggregate_state_seen ? "seen" : "missing"}`],
    ["robot telemetry", robotObsOk ? "ok" : "waiting", "from locomotion observation topic"],
    ["navdp", navdpOk ? "ok" : "waiting", "planning results mirrored in core"],
    ["locomotion", locomotionOk ? "ok" : "waiting", "command outputs mirrored in core"],
  ];
  moduleSummary.innerHTML = items.map(([name, status, note]) => `
    <div class="summary-card">
      <div class="panel-title"><span>${name}</span><span>${dot(status)}</span></div>
      <div class="small-note">${note}</div>
    </div>`).join("");
}

function renderWorldState(data) {
  drawSceneMap(worldCanvas, data, null);
  const robots = Object.entries(data.robots || {});
  const intruders = Object.entries(data.intruders || {});
  const modules = [
    ["Simulation", data.aggregate_state, `aggregate ${fmtNum(data.aggregate_state.age_sec)} s`],
    ["Core State Mirror", { seen: true, fresh: true, hz: 1 / 0.1 }, `${data.topic_prefix}`],
    ["NavDP", robots.find(([, robot]) => robot.planning && robot.planning.seen)?.[1]?.planning || {}, `${robots.filter(([, robot]) => robot.planning && robot.planning.fresh).length}/${robots.length} fresh`],
    ["Locomotion", robots.find(([, robot]) => robot.locomotion && robot.locomotion.seen)?.[1]?.locomotion || {}, `${robots.filter(([, robot]) => robot.locomotion && robot.locomotion.fresh).length}/${robots.length} fresh`],
  ];
  worldModuleGrid.innerHTML = modules.map(([name, item, note]) => `
    <div class="summary-card">
      <div class="panel-title"><span>${name}</span><span>${dot(normStatus(item))}</span></div>
      <div class="chip-row">
        <span class="chip">age ${fmtNum(item.age_sec)} s</span>
        <span class="chip">freq ${fmtHz(item.hz)}</span>
      </div>
      <div class="small-note">${note}</div>
    </div>`).join("");

  worldEntityList.innerHTML = "";
  for (const [robotId, robot] of robots) {
    const pose = robot.pose || {};
    const planning = robot.planning || {};
    worldEntityList.insertAdjacentHTML("beforeend", `
      <div class="summary-card">
        <div class="panel-title"><span>${robotId}</span><span>${dot(normStatus(pose))}</span></div>
        <div class="chip-row">
          <span class="chip">pose ${fmtVec(pose.position)}</span>
          <span class="chip">yaw ${fmtNum(pose.yaw)} rad</span>
          <span class="chip">path ${Array.isArray(planning.waypoints) ? planning.waypoints.length : 0}</span>
        </div>
      </div>`);
  }
  for (const [intruderId, intruder] of intruders) {
    const pose = intruder.pose || {};
    worldEntityList.insertAdjacentHTML("beforeend", `
      <div class="summary-card">
        <div class="panel-title"><span>${intruderId}</span><span>${dot(normStatus(pose))}</span></div>
        <div class="chip-row"><span class="chip">pose ${fmtVec(pose.position)}</span></div>
      </div>`);
  }

  const freshPaths = robots.filter(([, robot]) => robot.planning && robot.planning.fresh).length;
  const freshLocomotion = robots.filter(([, robot]) => robot.locomotion && robot.locomotion.fresh).length;
  worldSummary.textContent =
    `Simulation publishes into Core on ${data.topic_prefix}. ` +
    `${freshPaths}/${robots.length} robots have fresh NavDP routes and ` +
    `${freshLocomotion}/${robots.length} robots have fresh locomotion outputs. ` +
    `Map view overlays scene geometry, robot headings, intruder position, path waypoints, and LiDAR points.`;
}

function jointRows(observation, locomotion) {
  const names = Array.isArray(observation.joint_names) ? observation.joint_names : jointOrder;
  const pos = Array.isArray(observation.joint_position_rel) ? observation.joint_position_rel : [];
  const vel = Array.isArray(observation.joint_velocity_rel) ? observation.joint_velocity_rel : [];
  const action = Array.isArray(locomotion.action) ? locomotion.action : [];
  return names.map((name, index) => `
    <tr>
      <td>${name}</td>
      <td>${fmtNum(pos[index], 3)}</td>
      <td>${fmtNum(vel[index], 3)}</td>
      <td>${fmtNum(action[index], 3)}</td>
    </tr>`).join("");
}

function renderRobotPage(data) {
  robotGrid.innerHTML = "";
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const pose = robot.pose || {};
    const camera = robot.camera || {};
    const lidar = robot.lidar || {};
    const observation = robot.observation || {};
    const locomotion = robot.locomotion || {};
    const cameraHtml = camera.image ? `<img src="${camera.image}" alt="${robotId} camera" />` : `<span class="mono">waiting for camera</span>`;
    const card = document.createElement("div");
    card.className = "robot-card";
    card.innerHTML = `
      <div class="panel-title"><span>${robotId}</span><span>${dot(normStatus(observation))}</span></div>
      <div class="chip-row">
        <span class="chip">pose ${fmtVec(pose.position)}</span>
        <span class="chip">speed ${fmtNum(observation.planar_speed)} m/s</span>
        <span class="chip">camera ${fmtHz(camera.hz)}</span>
        <span class="chip">lidar ${fmtHz(lidar.hz)}</span>
      </div>
      <div class="robot-layout">
        <div class="stack">
          <div>
            <div class="subheading">Camera</div>
            <div class="image-box">${cameraHtml}</div>
          </div>
          <div>
            <div class="subheading">LiDAR</div>
            <canvas width="360" height="220" data-role="lidar"></canvas>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="subheading">State</div>
            <div class="metric-list">
              <div class="metric"><div class="label">Yaw</div><div class="value">${fmtNum(pose.yaw)} rad</div></div>
              <div class="metric"><div class="label">Base Lin Vel</div><div class="value mono">${fmtVec(observation.base_linear_velocity)}</div></div>
              <div class="metric"><div class="label">Base Ang Vel</div><div class="value mono">${fmtVec(observation.base_angular_velocity)}</div></div>
              <div class="metric"><div class="label">Command Slot</div><div class="value mono">${fmtVec(observation.command_slot)}</div></div>
            </div>
          </div>
          <div>
            <div class="subheading">Observed Pose</div>
            <canvas width="360" height="220" data-role="pose"></canvas>
          </div>
        </div>
      </div>
      <div style="margin-top:10px">
        <div class="subheading">Joint Telemetry</div>
        <table>
          <thead><tr><th>Joint</th><th>Pos Rel</th><th>Vel Rel</th><th>Action</th></tr></thead>
          <tbody>${jointRows(observation, locomotion)}</tbody>
        </table>
      </div>`;
    robotGrid.appendChild(card);
    drawLidar(card.querySelector('[data-role="lidar"]'), lidar);
    drawDogPose(card.querySelector('[data-role="pose"]'), observation.joint_position_rel, "#1b6fd1", 1.0);
  }
}

function renderNavdpPage(data) {
  navdpGrid.innerHTML = "";
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const planning = robot.planning || {};
    const camera = robot.camera || {};
    const pose = robot.pose || {};
    const card = document.createElement("div");
    card.className = "module-card";
    card.innerHTML = `
      <div class="panel-title"><span>${robotId}</span><span>${dot(normStatus(planning))}</span></div>
      <div class="chip-row">
        <span class="chip">planner ${planning.planner || "-"}</span>
        <span class="chip">planning ${fmtHz(planning.hz)}</span>
        <span class="chip">camera ${fmtHz(camera.hz)}</span>
        <span class="chip">age ${fmtNum(planning.age_sec)} s</span>
      </div>
      <div class="split">
        <div class="stack">
          <div>
            <div class="subheading">Camera Input</div>
            <div class="image-box">${camera.image ? `<img src="${camera.image}" alt="${robotId} camera" />` : `<span class="mono">waiting for camera</span>`}</div>
          </div>
          <div>
            <div class="subheading">Planning Summary</div>
            <div class="metric-list">
              <div class="metric"><div class="label">Subgoal</div><div class="value mono">${fmtVec(planning.subgoal)}</div></div>
              <div class="metric"><div class="label">Local Goal</div><div class="value mono">${fmtVec(planning.local_goal)}</div></div>
              <div class="metric"><div class="label">Waypoints</div><div class="value">${Array.isArray(planning.waypoints) ? planning.waypoints.length : 0}</div></div>
              <div class="metric"><div class="label">Robot Yaw</div><div class="value">${fmtNum(pose.yaw)} rad</div></div>
            </div>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="subheading">Path Overlay</div>
            <canvas width="360" height="360" data-role="path-map"></canvas>
          </div>
          <div class="small-note mono">
            first ${Array.isArray(planning.waypoints) && planning.waypoints.length ? fmtVec(planning.waypoints[0]) : "-"}<br />
            last ${Array.isArray(planning.waypoints) && planning.waypoints.length ? fmtVec(planning.waypoints[planning.waypoints.length - 1]) : "-"}<br />
            local_first ${Array.isArray(planning.local_waypoints) && planning.local_waypoints.length ? fmtVec(planning.local_waypoints[0]) : "-"}<br />
            local_last ${Array.isArray(planning.local_waypoints) && planning.local_waypoints.length ? fmtVec(planning.local_waypoints[planning.local_waypoints.length - 1]) : "-"}
          </div>
        </div>
      </div>`;
    navdpGrid.appendChild(card);
    drawSceneMap(card.querySelector('[data-role="path-map"]'), data, robotId);
  }
}

function renderLocomotionPage(data) {
  locomotionGrid.innerHTML = "";
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const observation = robot.observation || {};
    const locomotion = robot.locomotion || {};
    const action = Array.isArray(locomotion.action) ? locomotion.action : [];
    const actionScale = Number(locomotion.action_scale) || 0.25;
    const maxAbs = action.reduce((acc, value) => Math.max(acc, Math.abs(Number(value) || 0)), 0);
    const card = document.createElement("div");
    card.className = "module-card";
    card.innerHTML = `
      <div class="panel-title"><span>${robotId}</span><span>${dot(normStatus(locomotion))}</span></div>
      <div class="chip-row">
        <span class="chip">controller ${locomotion.controller || "-"}</span>
        <span class="chip">freq ${fmtHz(locomotion.hz)}</span>
        <span class="chip">age ${fmtNum(locomotion.age_sec)} s</span>
        <span class="chip">action scale ${fmtNum(actionScale, 3)}</span>
      </div>
      <div class="split">
        <div class="stack">
          <div class="metric-list">
            <div class="metric"><div class="label">Body Velocity Command</div><div class="value mono">${fmtVec(locomotion.body_velocity_command)}</div></div>
            <div class="metric"><div class="label">World Velocity</div><div class="value mono">${fmtVec(locomotion.velocity)}</div></div>
            <div class="metric"><div class="label">Target</div><div class="value mono">${fmtVec(locomotion.target)}</div></div>
            <div class="metric"><div class="label">max |action|</div><div class="value">${fmtNum(maxAbs, 3)}</div></div>
          </div>
          <div>
            <div class="subheading">Commanded Pose</div>
            <canvas width="360" height="220" data-role="cmd-pose"></canvas>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="subheading">Observed vs Commanded</div>
            <table>
              <thead><tr><th>Joint</th><th>Observed</th><th>Command</th></tr></thead>
              <tbody>${jointOrder.map((name, index) => `
                <tr>
                  <td>${name}</td>
                  <td>${fmtNum(Array.isArray(observation.joint_position_rel) ? observation.joint_position_rel[index] : null, 3)}</td>
                  <td>${fmtNum(action[index], 3)}</td>
                </tr>`).join("")}</tbody>
            </table>
          </div>
        </div>
      </div>`;
    locomotionGrid.appendChild(card);
    drawDogPose(card.querySelector('[data-role="cmd-pose"]'), action, "#c05a2d", actionScale * 2.5);
  }
}

function renderAll(data) {
  latestData = data;
  renderModuleSummary(data);
  renderWorldState(data);
  renderRobotPage(data);
  renderNavdpPage(data);
  renderLocomotionPage(data);
}

function connect() {
  const ws = new WebSocket(CORE_WS_URL);
  ws.onopen = () => { connection.textContent = `connected: ${CORE_WS_URL}`; };
  ws.onmessage = event => renderAll(JSON.parse(event.data));
  ws.onclose = () => {
    connection.textContent = `reconnecting: ${CORE_WS_URL}`;
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}

document.querySelectorAll(".nav-button").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach(item => item.classList.toggle("active", item === button));
    document.querySelectorAll(".page").forEach(page => page.classList.remove("active"));
    document.getElementById(`page-${button.dataset.page}`).classList.add("active");
    if (latestData) renderAll(latestData);
  });
});

connect();
</script>
</body>
</html>
"""


class CoreVisualizationNode(Node):
    """Dashboard frontend only.

    This node does not subscribe to ROS2 simulation topics. The browser receives
    state from the Core control layer's WebSocket state API.
    """

    def __init__(self) -> None:
        super().__init__("factory_core_visualization")

        self.declare_parameter("web_host", "0.0.0.0")
        self.declare_parameter("web_port", 8770)
        self.declare_parameter("core_ws_url", "ws://127.0.0.1:8765/ws")
        self.declare_parameter("scene_map_path", str(DEFAULT_SCENE_PATH))

        self.web_host = str(self.get_parameter("web_host").value)
        self.web_port = int(self.get_parameter("web_port").value)
        self.core_ws_url = str(self.get_parameter("core_ws_url").value)
        scene_map_path = Path(str(self.get_parameter("scene_map_path").value))
        self.scene_map = _load_scene_map(scene_map_path)

        self._server = None
        self._server_thread = threading.Thread(target=self._run_web_server, daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            f"Dashboard frontend listening on http://{self.web_host}:{self.web_port}; "
            f"reading Core state from {self.core_ws_url}"
        )

    def _run_web_server(self) -> None:
        app = create_app(self.core_ws_url, self.scene_map)
        config = uvicorn.Config(app, host=self.web_host, port=self.web_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.run()

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        return super().destroy_node()


def create_app(core_ws_url: str, scene_map: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Factory Core Dashboard Frontend")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html = HTML_PAGE.replace("__CORE_WS_URL__", core_ws_url)
        return html.replace("__SCENE_MAP__", json.dumps(scene_map))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "owner": "visualization", "core_ws_url": core_ws_url}

    return app


def _load_scene_map(scene_path: Path) -> dict[str, Any]:
    if not scene_path.exists():
        return {"source": str(scene_path), "rects": []}
    content = scene_path.read_text(encoding="utf-8")
    rects: list[dict[str, Any]] = []
    cube_pattern = re.compile(
        r'def Cube "([^"]+)".*?float3 xformOp:scale = \(([^)]+)\).*?double3 xformOp:translate = \(([^)]+)\)',
        re.DOTALL,
    )
    for name, scale_text, translate_text in cube_pattern.findall(content):
        scale = [float(value.strip()) for value in scale_text.split(",")]
        translate = [float(value.strip()) for value in translate_text.split(",")]
        lowered = name.lower()
        if "floor" in lowered:
            kind = "floor"
        elif "wall" in lowered:
            kind = "wall"
        elif "obstacle" in lowered:
            kind = "obstacle"
        else:
            continue
        rects.append({"name": name, "kind": kind, "scale": scale[:2], "center": translate[:2]})
    return {"source": str(scene_path), "rects": rects}


def main() -> None:
    rclpy.init()
    node = CoreVisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
