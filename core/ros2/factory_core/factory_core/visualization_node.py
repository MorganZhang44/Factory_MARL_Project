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
    .worldstate-layout { grid-template-columns: minmax(420px, 1.35fr) minmax(320px, 1fr); }
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
    .camera-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
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
    @media (max-width: 1480px) {
      .worldstate-layout { grid-template-columns: 1fr; }
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
        <button class="nav-button" data-page="perception">Perception</button>
        <button class="nav-button" data-page="marl">MARL</button>
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
      <section class="page two-col worldstate-layout active" id="page-worldstate">
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
            <h2>CCTV</h2>
            <div class="grid camera-grid" id="world-cctv-grid"></div>
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

      <section class="page stack" id="page-perception">
        <div class="section">
          <h2>Perception</h2>
          <div class="small-note">Core-to-perception connection state, localization output, intruder estimate, and sensor coverage summary.</div>
        </div>
        <div class="grid full-grid" id="perception-grid"></div>
      </section>

      <section class="page stack" id="page-marl">
        <div class="section">
          <h2>MARL</h2>
          <div class="small-note">Core-mirrored MARL runtime state, current observation summary, action semantics, and per-robot decision subgoals. Distinct symbols make robot pose, intruder pose, and MARL target easier to read at a glance.</div>
        </div>
        <div class="grid full-grid" id="marl-grid"></div>
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
const worldCctvGrid = document.getElementById("world-cctv-grid");
const worldSummary = document.getElementById("world-summary");
const worldCanvas = document.getElementById("world-map");
const robotGrid = document.getElementById("robot-grid");
const perceptionGrid = document.getElementById("perception-grid");
const marlGrid = document.getElementById("marl-grid");
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

function drawDiamondMarker(ctx, x, y, size, color, fill = true) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x + size, y);
  ctx.lineTo(x, y + size);
  ctx.lineTo(x - size, y);
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
  ctx.restore();
}

function drawTargetMarker(ctx, x, y, size, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(x, y, size * 0.75, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y + size);
  ctx.stroke();
  ctx.restore();
}

function drawArrowLine(ctx, x1, y1, x2, y2, color, dashed = false) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1.0e-6) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  if (dashed) ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  const ux = dx / len;
  const uy = dy / len;
  const head = 10;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - ux * head - uy * head * 0.55, y2 - uy * head + ux * head * 0.55);
  ctx.lineTo(x2 - ux * head + uy * head * 0.55, y2 - uy * head - ux * head * 0.55);
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
    const lidarPoints = robot.lidar_points || {};
    const pose = robot.pose || {};
    const points = Array.isArray(lidar.points) ? lidar.points : (
      Array.isArray(lidarPoints.points_xyz) ? lidarPoints.points_xyz : []
    );
    if (Array.isArray(pose.position) && points.length > 0) {
      ctx.fillStyle = robotId === focusRobotId ? "rgba(48, 126, 235, 0.18)" : "rgba(36, 163, 95, 0.14)";
      const rx = pose.position[0];
      const ry = pose.position[1];
      const yaw = yawFromPose(pose);
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      for (const point of points) {
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

function drawMarlMap(canvas, data, focusRobotId = null) {
  drawSceneMap(canvas, data, focusRobotId);
  const ctx = canvas.getContext("2d");
  const bounds = buildWorldBounds(data);
  const proj = makeProjector(canvas, bounds);
  const marl = data.marl || {};
  const subgoals = marl.subgoals || {};
  const intruderEntry = Object.entries(data.intruders || {})[0];
  const intruderId = intruderEntry ? intruderEntry[0] : "intruder";
  const intruderPose = intruderEntry && intruderEntry[1] ? intruderEntry[1].pose : null;
  const intruderPos = intruderPose && Array.isArray(intruderPose.position) ? intruderPose.position : null;

  if (intruderPos) {
    const ix = proj.x(intruderPos[0]);
    const iy = proj.y(intruderPos[1]);
    drawDiamondMarker(ctx, ix, iy, 10, "#cb4a3f", true);
    ctx.fillStyle = "#1b2430";
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText(intruderId, ix + 12, iy - 10);
  }

  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const pose = robot.pose || {};
    const robotPos = Array.isArray(pose.position) ? pose.position : null;
    const subgoal = subgoals[robotId];
    const subgoalPos = subgoal && Array.isArray(subgoal.subgoal) ? subgoal.subgoal : null;
    if (!robotPos || !subgoalPos) continue;
    const rx = proj.x(robotPos[0]);
    const ry = proj.y(robotPos[1]);
    const gx = proj.x(subgoalPos[0]);
    const gy = proj.y(subgoalPos[1]);
    const accent = robotId === focusRobotId ? "#7a42d1" : "#d39a1a";
    drawArrowLine(ctx, rx, ry, gx, gy, accent, true);
    drawTargetMarker(ctx, gx, gy, robotId === focusRobotId ? 13 : 11, accent);
    if (intruderPos) {
      drawArrowLine(ctx, gx, gy, proj.x(intruderPos[0]), proj.y(intruderPos[1]), "#cb4a3f", false);
    }
  }

  const legendX = 16;
  const legendY = 16;
  ctx.fillStyle = "rgba(255,255,255,0.94)";
  ctx.fillRect(legendX, legendY, 230, 92);
  ctx.strokeStyle = "#d8e0ec";
  ctx.lineWidth = 1;
  ctx.strokeRect(legendX, legendY, 230, 92);
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillStyle = "#233044";
  ctx.fillText("MARL legend", legendX + 10, legendY + 16);
  ctx.fillStyle = "#2a8d62";
  ctx.beginPath();
  ctx.arc(legendX + 18, legendY + 34, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#233044";
  ctx.fillText("robot pose", legendX + 34, legendY + 38);
  drawDiamondMarker(ctx, legendX + 18, legendY + 54, 6, "#cb4a3f", true);
  ctx.fillStyle = "#233044";
  ctx.fillText("intruder pose", legendX + 34, legendY + 58);
  drawTargetMarker(ctx, legendX + 18, legendY + 74, 7, "#d39a1a");
  ctx.fillStyle = "#233044";
  ctx.fillText("MARL subgoal", legendX + 34, legendY + 78);
  drawArrowLine(ctx, legendX + 138, legendY + 34, legendX + 168, legendY + 34, "#7a42d1", true);
  ctx.fillStyle = "#233044";
  ctx.fillText("robot -> subgoal", legendX + 174, legendY + 38);
}

function drawLidar(canvas, lidar, lidarPoints = {}) {
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
  const points = Array.isArray(lidar.points) ? lidar.points : (
    Array.isArray(lidarPoints.points_xyz) ? lidarPoints.points_xyz : []
  );
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
  const cctv = Object.values(data.cctv_cameras || {});
  const perception = data.perception || {};
  const marl = data.marl || {};
  const simulationOk = robots.some(robot => robot.pose && robot.pose.fresh) || Object.values(data.intruders || {}).some(entry => entry.pose && entry.pose.fresh);
  const perceptionOk = perception && perception.seen;
  const marlOk = marl && marl.seen;
  const navdpOk = robots.some(robot => robot.planning && robot.planning.seen);
  const locomotionOk = robots.some(robot => robot.locomotion && robot.locomotion.seen);
  const robotObsOk = robots.some(robot => robot.observation && robot.observation.seen);
  const cctvOk = cctv.some(camera => camera && camera.seen);
  const items = [
    ["core", "ok", "state mirror serving dashboard"],
    ["simulation", simulationOk ? "ok" : "waiting", `aggregate ${data.aggregate_state_seen ? "seen" : "missing"}`],
    ["perception", perceptionOk ? normStatus(perception) : "waiting", perceptionOk ? `actual ${fmtHz(perception.hz)}` : "awaiting perception results"],
    ["marl", marlOk ? normStatus(marl) : "waiting", marlOk ? `${marl.policy_mode || "-"} / ${marl.action_semantics || "-"}` : "awaiting mirrored marl output"],
    ["cctv", cctvOk ? "ok" : "waiting", `${cctv.filter(camera => camera && camera.fresh).length}/${cctv.length} fresh`],
    ["robot telemetry", robotObsOk ? "ok" : "waiting", "from locomotion observation topic"],
    ["navdp", navdpOk ? "ok" : "waiting", "actual planner result frequency"],
    ["locomotion", locomotionOk ? "ok" : "waiting", "actual locomotion result frequency"],
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
  const perception = data.perception || {};
  const marl = data.marl || {};
  const modules = [
    ["Simulation", data.aggregate_state, `aggregate ${fmtNum(data.aggregate_state.age_sec)} s`],
    ["Core State Mirror", { seen: true, fresh: true, hz: 1 / 0.1 }, `${data.topic_prefix}`],
    ["Perception", perception, perception.seen ? `intruder ${perception.intruder_estimate && perception.intruder_estimate.detected ? "detected" : "pending"}` : "awaiting core-perception updates"],
    ["MARL", marl, marl.seen ? `${marl.policy_mode || "-"} / ${Object.keys((marl.subgoals || {})).length} subgoals` : "awaiting core-marl updates"],
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

  const cctv = Object.entries(data.cctv_cameras || {});
  const cctvSemantics = data.cctv_semantics || {};
  worldCctvGrid.innerHTML = "";
  for (const [cameraId, camera] of cctv) {
    const semantic = cctvSemantics[cameraId] || {};
    const image = camera.image ? `<img src="${camera.image}" alt="${cameraId} CCTV" />` : `<span class="mono">waiting for ${cameraId}</span>`;
    worldCctvGrid.insertAdjacentHTML("beforeend", `
      <div class="summary-card">
        <div class="panel-title"><span>${cameraId}</span><span>${dot(normStatus(camera))}</span></div>
        <div class="chip-row">
          <span class="chip">freq ${fmtHz(camera.hz)}</span>
          <span class="chip">seg ${fmtHz(semantic.hz)}</span>
          <span class="chip">${camera.width || "-"}x${camera.height || "-"}</span>
        </div>
        <div class="image-box">${image}</div>
      </div>`);
  }

  const freshPaths = robots.filter(([, robot]) => robot.planning && robot.planning.fresh).length;
  const freshLocomotion = robots.filter(([, robot]) => robot.locomotion && robot.locomotion.fresh).length;
  const freshCctv = cctv.filter(([, camera]) => camera && camera.fresh).length;
  const marlNote = data.marl && data.marl.seen
    ? `MARL is mirrored at ${fmtHz(data.marl.hz)} in ${data.marl.policy_mode || "-"} mode. `
    : "MARL has not published a mirrored decision preview yet. ";
  const perceptionNote = data.perception && data.perception.seen
    ? `Perception is updating at ${fmtHz(data.perception.hz)} and ${
        data.perception.intruder_estimate && data.perception.intruder_estimate.detected ? "has" : "does not yet have"
      } an intruder estimate. `
    : "Perception has not published a mirrored estimate yet. ";
  worldSummary.textContent =
    `Simulation publishes into Core on ${data.topic_prefix}. ` +
    marlNote +
    perceptionNote +
    `${freshPaths}/${robots.length} robots have fresh NavDP routes and ` +
    `${freshLocomotion}/${robots.length} robots have fresh locomotion outputs. ` +
    `${freshCctv}/${cctv.length} CCTV feeds are fresh. ` +
    `Map view overlays scene geometry, robot headings, intruder position, path waypoints, and LiDAR points.`;
}

function jointRows(observation, locomotion) {
  const names = Array.isArray(observation.joint_names) ? observation.joint_names : jointOrder;
  const pos = Array.isArray(observation.joint_position_rel) ? observation.joint_position_rel : [];
  const vel = Array.isArray(observation.joint_velocity_rel) ? observation.joint_velocity_rel : [];
  const action = Array.isArray(observation.last_action) ? observation.last_action : [];
  const actionScale = Number(locomotion.action_scale) || 0.25;
  return names.map((name, index) => `
    <tr>
      <td>${name}</td>
      <td>${fmtNum(pos[index], 3)}</td>
      <td>${fmtNum(vel[index], 3)}</td>
      <td>${fmtNum((Number(action[index]) || 0) * actionScale, 3)}</td>
    </tr>`).join("");
}

function renderRobotPage(data) {
  robotGrid.innerHTML = "";
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const pose = robot.pose || {};
    const camera = robot.camera || {};
    const depth = robot.depth || {};
    const semantic = robot.semantic || {};
    const imu = robot.imu || {};
    const lidar = robot.lidar || {};
    const lidarPoints = robot.lidar_points || {};
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
        <span class="chip">depth ${fmtHz(depth.hz)}</span>
        <span class="chip">seg ${fmtHz(semantic.hz)}</span>
        <span class="chip">imu ${fmtHz(imu.hz)}</span>
        <span class="chip">lidar ${fmtHz(lidar.hz)}</span>
        <span class="chip">cloud ${fmtHz(lidarPoints.hz)} / ${lidarPoints.point_count || 0} pts</span>
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
          <thead><tr><th>Joint</th><th>Observed Pos Rel</th><th>Observed Vel Rel</th><th>Target Pos Rel</th></tr></thead>
          <tbody>${jointRows(observation, locomotion)}</tbody>
        </table>
      </div>`;
    robotGrid.appendChild(card);
    drawLidar(card.querySelector('[data-role="lidar"]'), lidar, lidarPoints);
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

function computeXyError(estPos, gtPos) {
  if (!Array.isArray(estPos) || !Array.isArray(gtPos) || estPos.length < 2 || gtPos.length < 2) return null;
  const dx = estPos[0] - gtPos[0];
  const dy = estPos[1] - gtPos[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function drawPerceptionComparisonMap(canvas, data) {
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
    ctx.beginPath(); ctx.moveTo(proj.x(x), proj.y(bounds.ymin)); ctx.lineTo(proj.x(x), proj.y(bounds.ymax)); ctx.stroke();
  }
  for (let y = Math.ceil(bounds.ymin); y <= bounds.ymax; y += stepMeters) {
    ctx.beginPath(); ctx.moveTo(proj.x(bounds.xmin), proj.y(y)); ctx.lineTo(proj.x(bounds.xmax), proj.y(y)); ctx.stroke();
  }
  for (const rect of SCENE_MAP.rects || []) {
    const color = rect.kind === "floor" ? "#d8d8ce" : (rect.kind === "wall" ? "#2d3138" : "#aa654f");
    ctx.fillStyle = color;
    const w = rect.scale[0] * proj.scale;
    const h = rect.scale[1] * proj.scale;
    ctx.fillRect(proj.x(rect.center[0]) - w * 0.5, proj.y(rect.center[1]) - h * 0.5, w, h);
  }

  const perception = data.perception || {};

  // Draw ground truth intruder (red X)
  for (const [intruderId, intruder] of Object.entries(data.intruders || {})) {
    const pose = intruder.pose || {};
    if (!Array.isArray(pose.position)) continue;
    const x = proj.x(pose.position[0]);
    const y = proj.y(pose.position[1]);
    ctx.strokeStyle = "#c94b42";
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x - 8, y - 8); ctx.lineTo(x + 8, y + 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + 8, y - 8); ctx.lineTo(x - 8, y + 8); ctx.stroke();
    ctx.fillStyle = "#c94b42";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("GT " + intruderId, x + 12, y - 10);
  }

  // Draw perception intruder estimate (orange diamond)
  const intruderEst = perception.intruder_estimate || {};
  if (intruderEst.detected && Array.isArray(intruderEst.position_world)) {
    const ex = proj.x(intruderEst.position_world[0]);
    const ey = proj.y(intruderEst.position_world[1]);
    ctx.fillStyle = "#e09b15";
    ctx.beginPath();
    ctx.moveTo(ex, ey - 10); ctx.lineTo(ex + 8, ey); ctx.lineTo(ex, ey + 10); ctx.lineTo(ex - 8, ey);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#e09b15";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("EST intruder", ex + 12, ey + 4);

    // Draw error line between GT and estimate
    for (const intruder of Object.values(data.intruders || {})) {
      const pose = intruder.pose || {};
      if (!Array.isArray(pose.position)) continue;
      const gx = proj.x(pose.position[0]);
      const gy = proj.y(pose.position[1]);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#e09b15";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Draw ground truth dogs (filled circle) and estimated dogs (ring)
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const pose = robot.pose || {};
    const estimate = (perception.dogs || {})[robotId] || {};
    const color = robotId === "agent_1" ? "#2a8d62" : "#1b6fd1";
    // GT
    if (Array.isArray(pose.position)) {
      const gx = proj.x(pose.position[0]);
      const gy = proj.y(pose.position[1]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(gx, gy, 8, 0, Math.PI * 2); ctx.fill();
      drawArrow(ctx, gx, gy, yawFromPose(pose), 14, color);
      ctx.fillStyle = "#1b2430";
      ctx.font = "12px ui-monospace, monospace";
      ctx.fillText("GT " + robotId, gx + 10, gy - 12);
    }
    // Estimate
    if (estimate.detected && Array.isArray(estimate.position_world)) {
      const ex = proj.x(estimate.position_world[0]);
      const ey = proj.y(estimate.position_world[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(ex, ey, 10, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("EST", ex + 14, ey + 4);
      // Error line
      if (Array.isArray(pose.position)) {
        const gx = proj.x(pose.position[0]);
        const gy = proj.y(pose.position[1]);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // Legend
  ctx.fillStyle = "#4d5b71";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("● GT (filled)    ○ EST (ring)    ◆ EST intruder    ✕ GT intruder", 10, canvas.height - 10);
}

function renderPerceptionPage(data) {
  perceptionGrid.innerHTML = "";
  const perception = data.perception || {};
  const intruder = perception.intruder_estimate || {};

  // --- Comparison Map ---
  const mapCard = document.createElement("div");
  mapCard.className = "module-card";
  mapCard.innerHTML = `
    <div class="panel-title"><span>Perception vs Ground Truth Map</span><span>${dot(normStatus(perception))}</span></div>
    <div class="small-note" style="margin-bottom:8px">Filled markers = ground truth from simulation. Rings/diamonds = perception estimates. Dashed lines show error.</div>
    <canvas id="perception-comparison-map" width="1024" height="600"></canvas>`;
  perceptionGrid.appendChild(mapCard);
  drawPerceptionComparisonMap(mapCard.querySelector("#perception-comparison-map"), data);

  // --- Intruder Estimate vs GT ---
  const gtIntruder = Object.values(data.intruders || {})[0] || {};
  const gtIntruderPose = gtIntruder.pose || {};
  const intruderXyError = computeXyError(intruder.position_world, gtIntruderPose.position);

  const intruderCard = document.createElement("div");
  intruderCard.className = "module-card";
  intruderCard.innerHTML = `
    <div class="panel-title"><span>Intruder: Estimate vs Ground Truth</span><span>${dot(intruder.detected ? "ok" : "waiting")}</span></div>
    <div class="chip-row">
      <span class="chip">freq ${fmtHz(perception.hz)}</span>
      <span class="chip">pipeline ${perception.pipeline || "-"}</span>
      <span class="chip">step ${perception.step || "-"}</span>
    </div>
    <table>
      <thead><tr><th>Field</th><th>Perception Estimate</th><th>Ground Truth (Sim)</th><th>Error</th></tr></thead>
      <tbody>
        <tr>
          <td><b>Position XY</b></td>
          <td class="mono">${fmtVec(intruder.position_world)}</td>
          <td class="mono">${fmtVec(gtIntruderPose.position)}</td>
          <td style="font-weight:650;color:${intruderXyError !== null && intruderXyError < 0.5 ? '#24a35f' : '#c94b42'}">${intruderXyError !== null ? intruderXyError.toFixed(3) + " m" : "-"}</td>
        </tr>
        <tr>
          <td>Detected</td>
          <td>${intruder.detected ? "✅ yes" : "❌ no"}</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Confidence</td>
          <td>${fmtNum(intruder.confidence, 3)}</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Camera Detections</td>
          <td>${intruder.camera_detections || 0}</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>LiDAR Detections</td>
          <td>${intruder.lidar_detections || 0}</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>`;
  perceptionGrid.appendChild(intruderCard);

  // --- Per-Dog Estimate vs GT ---
  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const estimate = (perception.dogs || {})[robotId] || {};
    const gtPose = robot.pose || {};
    const camera = robot.camera || {};
    const lidar = robot.lidar || {};
    const lidarPoints = robot.lidar_points || {};
    const dogXyError = estimate.xy_error_m;
    const estPos = estimate.position_world;
    const gtPos = gtPose.position;

    const observation = robot.observation || {};
    const gtVel = observation.base_linear_velocity;
    const gtSpeed = observation.planar_speed;

    const card = document.createElement("div");
    card.className = "module-card";
    card.innerHTML = `
      <div class="panel-title"><span>${robotId}: Estimate vs Ground Truth</span><span>${dot(estimate.detected ? (estimate.localized ? "ok" : "stale") : "waiting")}</span></div>
      <div class="chip-row">
        <span class="chip">localized ${estimate.localized ? "✅" : "❌"}</span>
        <span class="chip" style="font-weight:650;color:${dogXyError != null && dogXyError < 0.3 ? '#24a35f' : '#c94b42'}">xy error ${fmtNum(dogXyError, 3)} m</span>
        <span class="chip">scan score ${fmtNum(estimate.scan_match_score, 3)}</span>
        <span class="chip">scan inliers ${estimate.scan_inliers || 0}</span>
      </div>
      <table>
        <thead><tr><th>Field</th><th>Perception Estimate</th><th>Ground Truth (Sim)</th></tr></thead>
        <tbody>
          <tr>
            <td><b>Position</b></td>
            <td class="mono">${fmtVec(estPos)}</td>
            <td class="mono">${fmtVec(gtPos)}</td>
          </tr>
          <tr>
            <td><b>Velocity</b></td>
            <td class="mono">${fmtVec(estimate.velocity_world)}</td>
            <td class="mono">${fmtVec(gtVel)} <span style="color:#5d6980">(body)</span></td>
          </tr>
          <tr>
            <td><b>Planar Speed</b></td>
            <td class="mono">${fmtNum(Array.isArray(estimate.velocity_world) ? Math.hypot(estimate.velocity_world[0] || 0, estimate.velocity_world[1] || 0) : null, 3)} m/s</td>
            <td class="mono">${fmtNum(gtSpeed, 3)} m/s</td>
          </tr>
          <tr>
            <td><b>Yaw</b></td>
            <td>${fmtNum(estimate.yaw_rad)} rad</td>
            <td>${fmtNum(gtPose.yaw)} rad</td>
          </tr>
          <tr>
            <td>Prediction Only</td>
            <td>${estimate.used_prediction_only ? "⚠️ yes" : "no"}</td>
            <td>-</td>
          </tr>
        </tbody>
      </table>
      <div class="split" style="margin-top:10px">
        <div class="stack">
          <div>
            <div class="subheading">Camera Input</div>
            <div class="image-box">${camera.image ? `<img src="${camera.image}" alt="${robotId} camera" />` : `<span class="mono">waiting for camera</span>`}</div>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="subheading">LiDAR Evidence</div>
            <canvas width="360" height="220" data-role="perception-lidar"></canvas>
          </div>
        </div>
      </div>`;
    perceptionGrid.appendChild(card);
    drawLidar(card.querySelector('[data-role="perception-lidar"]'), lidar, lidarPoints);
  }
}

function renderMarlPage(data) {
  marlGrid.innerHTML = "";
  const marl = data.marl || {};
  const subgoals = marl.subgoals || {};
  const input = marl.input || {};

  const statusCard = document.createElement("div");
  statusCard.className = "module-card";
  statusCard.innerHTML = `
    <div class="panel-title"><span>MARL Runtime</span><span>${dot(normStatus(marl))}</span></div>
    <div class="chip-row">
      <span class="chip">mode ${marl.policy_mode || "-"}</span>
      <span class="chip">freq ${fmtHz(marl.hz)}</span>
      <span class="chip">age ${fmtNum(marl.age_sec)} s</span>
      <span class="chip">obs ${marl.obs_dim || 13}D</span>
    </div>
    <div class="metric-list">
      <div class="metric"><div class="label">Action Semantics</div><div class="value mono">${marl.action_semantics || "-"}</div></div>
      <div class="metric"><div class="label">Checkpoint Loaded</div><div class="value">${marl.checkpoint_loaded ? "yes" : "no"}</div></div>
      <div class="metric"><div class="label">Checkpoint</div><div class="value mono">${marl.checkpoint || "-"}</div></div>
      <div class="metric"><div class="label">Load Error</div><div class="value mono">${marl.load_error || "-"}</div></div>
    </div>`;
  marlGrid.appendChild(statusCard);

  const mapCard = document.createElement("div");
  mapCard.className = "module-card";
  mapCard.innerHTML = `
    <div class="panel-title"><span>Decision Overlay</span><span>${dot(normStatus(marl))}</span></div>
    <div class="small-note" style="margin-bottom:8px">Dashed lines show robot to MARL subgoal. Red arrows show subgoal to intruder. This is a mirrored preview only.</div>
    <canvas width="1024" height="560" data-role="marl-map"></canvas>`;
  marlGrid.appendChild(mapCard);
  drawMarlMap(mapCard.querySelector('[data-role="marl-map"]'), {
    ...data,
    robots: Object.fromEntries(Object.entries(data.robots || {}).map(([robotId, robot]) => [
      robotId,
      { ...robot, planning: { ...(robot.planning || {}), subgoal: (subgoals[robotId] || {}).subgoal || (robot.planning || {}).subgoal } },
    ])),
  }, null);

  for (const [robotId, robot] of Object.entries(data.robots || {})) {
    const robotInput = (input.robots || {})[robotId] || {};
    const subgoal = subgoals[robotId] || {};
    const pose = robot.pose || {};
    const observation = robot.observation || {};
    const card = document.createElement("div");
    card.className = "module-card";
    card.innerHTML = `
      <div class="panel-title"><span>${robotId}</span><span>${dot(subgoal.subgoal ? "ok" : "waiting")}</span></div>
      <div class="chip-row">
        <span class="chip">mode ${subgoal.mode || "-"}</span>
        <span class="chip">priority ${subgoal.priority || "-"}</span>
        <span class="chip">pose ${fmtVec(pose.position)}</span>
      </div>
      <div class="split">
        <div class="stack">
          <div>
            <div class="subheading">Decision Input</div>
            <div class="metric-list">
              <div class="metric"><div class="label">Robot Position</div><div class="value mono">${fmtVec(robotInput.position)}</div></div>
              <div class="metric"><div class="label">Robot Velocity</div><div class="value mono">${fmtVec(robotInput.velocity)}</div></div>
              <div class="metric"><div class="label">Observation Speed</div><div class="value">${fmtNum(observation.planar_speed, 3)} m/s</div></div>
              <div class="metric"><div class="label">Intruder Position</div><div class="value mono">${fmtVec((input.intruder || {}).position)}</div></div>
            </div>
          </div>
          <div>
            <div class="subheading">Decision Output</div>
            <div class="metric-list">
              <div class="metric"><div class="label">Subgoal</div><div class="value mono">${fmtVec(subgoal.subgoal)}</div></div>
              <div class="metric"><div class="label">Offset</div><div class="value mono">${fmtVec(subgoal.offset)}</div></div>
              <div class="metric"><div class="label">Action Mode</div><div class="value">${subgoal.mode || "-"}</div></div>
              <div class="metric"><div class="label">Priority</div><div class="value">${subgoal.priority || "-"}</div></div>
            </div>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="subheading">Local Spatial View</div>
            <canvas width="360" height="360" data-role="marl-map-${robotId}"></canvas>
          </div>
          <div class="small-note mono">
            input_robot ${fmtVec(robotInput.position)}<br />
            input_vel ${fmtVec(robotInput.velocity)}<br />
            intruder ${fmtVec((input.intruder || {}).position)}<br />
            output_subgoal ${fmtVec(subgoal.subgoal)}<br />
            output_offset ${fmtVec(subgoal.offset)}
          </div>
        </div>
      </div>`;
    marlGrid.appendChild(card);
    drawMarlMap(card.querySelector(`[data-role="marl-map-${robotId}"]`), {
      ...data,
      robots: Object.fromEntries(Object.entries(data.robots || {}).map(([rid, item]) => [
        rid,
        { ...item, planning: { ...(item.planning || {}), subgoal: (subgoals[rid] || {}).subgoal || (item.planning || {}).subgoal } },
      ])),
    }, robotId);
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
            <div class="subheading">Observed vs Last Applied Target Relative Joint Pose</div>
            <table>
              <thead><tr><th>Joint</th><th>Observed Pos Rel</th><th>Observed Vel Rel</th><th>Target Pos Rel</th></tr></thead>
              <tbody>${jointOrder.map((name, index) => `
                <tr>
                  <td>${name}</td>
                  <td>${fmtNum(Array.isArray(observation.joint_position_rel) ? observation.joint_position_rel[index] : null, 3)}</td>
                  <td>${fmtNum(Array.isArray(observation.joint_velocity_rel) ? observation.joint_velocity_rel[index] : null, 3)}</td>
                  <td>${fmtNum((Number(Array.isArray(observation.last_action) ? observation.last_action[index] : null) || 0) * actionScale, 3)}</td>
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
  renderPerceptionPage(data);
  renderMarlPage(data);
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
