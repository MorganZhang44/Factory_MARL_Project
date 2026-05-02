"""
interactive_demo.py
═══════════════════════════════════════════════════════
交互式 Demo: 你用鼠标控制入侵者(红色★)，
训练好的 MAPPO agent(蓝色●) 会追捕你！

Controls:
  Mouse  ── 移动入侵者
  R      ── 重新开始
  ESC    ── 退出

Run:
  python3 scripts/interactive_demo.py
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import math
import time
from pathlib import Path
from collections import deque

import numpy as np
import pygame
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.envs.pursuit_env import PursuitEnv
from marl.utils.map_utils import (
    ALL_RECTS, ALL_CIRCLES, PERIMETER_WALLS, INTERIOR_WALLS, BOXES, PILLARS
)
from marl.utils.astar import astar
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd


# ── Config ────────────────────────────────────────────────────────────
CFG_PATH  = "configs/mappo_config.yaml"
CKPT_PATH = "results/checkpoints/final.pt"

SCREEN_W, SCREEN_H = 820, 760
MAP_PX     = 700          # map area in pixels (square)
VIEW_HALF  = 7.0          # viewport bounds (zoomed into the inner room)
MAP_OFFSET = (60, 30)     # top-left corner of the map on screen

FPS = 20

# ── Colors (dark theme) ───────────────────────────────────────────────
BG          = (15,  20,  40)
MAP_BG      = (22,  28,  55)
GRID_COLOR  = (40,  50,  80)
WALL_COLOR  = (50,  60,  90)
OBS_COLOR   = (60,  75, 110)

DOG1_COLOR  = ( 74, 144, 217)   # bright blue
DOG2_COLOR  = (  0, 180, 216)   # teal
SUSPECT_COL = (231,  76,  60)   # red
CAPTURE_COL = (255, 215,   0)   # gold flash

PATH1_COL   = (100, 170, 240, 160)
PATH2_COL   = (80,  200, 230, 160)

TEXT_COL    = (210, 220, 240)
DIM_COL     = (100, 120, 170)
WARN_COL    = (255, 160,  50)
GREEN_COL   = ( 80, 200, 120)


# ── Coordinate helpers ────────────────────────────────────────────────

class CoordMapper:
    def __init__(self, view_half: float, px: int, offset: tuple):
        self.view_half = view_half
        self.px       = px
        self.offset   = offset
        self.scale    = px / (2 * view_half)   # pixels per metre

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int((x + self.view_half) * self.scale) + self.offset[0]
        sy = int((-y + self.view_half) * self.scale) + self.offset[1]
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        x = (sx - self.offset[0]) / self.scale - self.view_half
        y = -((sy - self.offset[1]) / self.scale - self.view_half)
        return x, y

    def m_to_px(self, metres: float) -> int:
        return max(1, int(metres * self.scale))


# ── Pre-render obstacle surface ───────────────────────────────────────

def build_map_surface(cm: CoordMapper) -> pygame.Surface:
    surf = pygame.Surface((MAP_PX, MAP_PX))
    surf.fill(MAP_BG)

    # Grid lines (draw them specifically across the VIEW bounds)
    for i in range(-int(cm.view_half), int(cm.view_half) + 1):
        sx, _  = cm.world_to_screen(i, 0)
        _, sy  = cm.world_to_screen(0, i)
        ox, oy = cm.offset
        if 0 <= sx - ox <= MAP_PX:
            pygame.draw.line(surf, GRID_COLOR, (sx - ox, 0), (sx - ox, MAP_PX), 1)
        if 0 <= sy - oy <= MAP_PX:
            pygame.draw.line(surf, GRID_COLOR, (0, sy - oy), (MAP_PX, sy - oy), 1)

    # Rectangular obstacles
    for obs in ALL_RECTS:
        is_perim = obs in PERIMETER_WALLS
        col = WALL_COLOR if is_perim else OBS_COLOR
        x0, y0 = cm.world_to_screen(obs.cx - obs.w / 2, obs.cy + obs.h / 2)
        w_px    = cm.m_to_px(obs.w)
        h_px    = cm.m_to_px(obs.h)
        ox, oy  = cm.offset
        pygame.draw.rect(surf, col, (x0 - ox, y0 - oy, w_px + 1, h_px + 1))

    # Circular obstacles (pillars)
    for obs in ALL_CIRCLES:
        cx, cy = cm.world_to_screen(obs.cx, obs.cy)
        r_px   = cm.m_to_px(obs.r)
        ox, oy = cm.offset
        pygame.draw.circle(surf, OBS_COLOR, (cx - ox, cy - oy), r_px + 1)

    return surf


# ── Drawing helpers ───────────────────────────────────────────────────

def draw_dashed_path(screen, points, color, cm: CoordMapper, dash=8, gap=5):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        p1 = cm.world_to_screen(*points[i])
        p2 = cm.world_to_screen(*points[i + 1])
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1:
            continue
        steps = max(1, int(dist / (dash + gap)))
        for s in range(steps):
            t0 = s * (dash + gap) / dist
            t1 = min(1.0, t0 + dash / dist)
            sx0 = int(p1[0] + dx * t0)
            sy0 = int(p1[1] + dy * t0)
            sx1 = int(p1[0] + dx * t1)
            sy1 = int(p1[1] + dy * t1)
            pygame.draw.line(screen, color, (sx0, sy0), (sx1, sy1), 2)


def draw_agent(screen, pos, color, radius_m, cm: CoordMapper, label: str, vel=None):
    cx, cy = cm.world_to_screen(*pos)
    r      = cm.m_to_px(radius_m)

    # Glow ring
    glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*color, 40), (r * 2, r * 2), r * 2)
    screen.blit(glow_surf, (cx - r * 2, cy - r * 2))

    # Body
    pygame.draw.circle(screen, color, (cx, cy), r)
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, 2)

    # Velocity arrow
    if vel is not None:
        spd = np.linalg.norm(vel)
        if spd > 0.1:
            arrow_len = cm.m_to_px(spd * 0.5)
            vx, vy = vel[0] / spd, vel[1] / spd
            ex = int(cx + vx * arrow_len)
            ey = int(cy - vy * arrow_len)
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (ex, ey), 2)

    # Label
    font_sm = pygame.font.SysFont("Arial", 11, bold=True)
    lbl = font_sm.render(label, True, (255, 255, 255))
    screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def draw_suspect(screen, pos, radius_m, cm: CoordMapper, trail: deque):
    cx, cy = cm.world_to_screen(*pos)
    r      = cm.m_to_px(radius_m)

    # Trail
    for i, tp in enumerate(trail):
        alpha = int(180 * i / max(len(trail), 1))
        trail_r = max(2, r - i // 3)
        s = pygame.Surface((trail_r * 2, trail_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*SUSPECT_COL, alpha), (trail_r, trail_r), trail_r)
        tx, ty = cm.world_to_screen(*tp)
        screen.blit(s, (tx - trail_r, ty - trail_r))

    # Glow
    glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*SUSPECT_COL, 50), (r * 2, r * 2), r * 2)
    screen.blit(glow, (cx - r * 2, cy - r * 2))

    # Body (star shape approximated by two overlapping circles + cross)
    pygame.draw.circle(screen, SUSPECT_COL, (cx, cy), r)
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, 2)
    font_sm = pygame.font.SysFont("Arial", 11, bold=True)
    lbl = font_sm.render("YOU", True, (255, 255, 255))
    screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def draw_capture_radius(screen, pos, radius_m, cm, color=(255, 215, 0)):
    cx, cy = cm.world_to_screen(*pos)
    r      = cm.m_to_px(radius_m)
    surf   = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*color, 50), (r + 2, r + 2), r)
    pygame.draw.circle(surf, (*color, 120), (r + 2, r + 2), r, 2)
    screen.blit(surf, (cx - r - 2, cy - r - 2))


def draw_hud(screen, font, font_big, step, elapsed, n_captured, dist_min,
             captured_flash, escaped_flash, map_px, map_off):
    W = SCREEN_W
    ox, oy = map_off
    panel_x = ox + map_px + 12

    # Title
    title = font_big.render("MAPPO Demo", True, TEXT_COL)
    screen.blit(title, (panel_x, oy + 5))

    # Instructions
    inst_lines = [
        ("Mouse", "Move suspect"),
        ("R",     "Restart"),
        ("ESC",   "Quit"),
    ]
    y = oy + 45
    for key, desc in inst_lines:
        k_surf = font.render(f"{key}:", True, WARN_COL)
        d_surf = font.render(desc, True, DIM_COL)
        screen.blit(k_surf, (panel_x, y))
        screen.blit(d_surf, (panel_x + 55, y))
        y += 20

    pygame.draw.line(screen, (60, 80, 120), (panel_x, y + 2), (W - 10, y + 2), 1)
    y += 14

    # Stats
    stats = [
        ("Steps",    f"{step}"),
        ("Time",     f"{elapsed:.1f}s"),
        ("Min Dist", f"{dist_min:.2f}m"),
        ("Captures", f"{n_captured}"),
    ]
    for label, val in stats:
        l_surf = font.render(f"{label}:", True, DIM_COL)
        v_surf = font.render(val, True, TEXT_COL)
        screen.blit(l_surf, (panel_x, y))
        screen.blit(v_surf, (panel_x + 80, y))
        y += 22

    pygame.draw.line(screen, (60, 80, 120), (panel_x, y + 2), (W - 10, y + 2), 1)
    y += 14

    # Legend
    legend = [
        (DOG1_COLOR,  "Dog 1 (agent)"),
        (DOG2_COLOR,  "Dog 2 (agent)"),
        (SUSPECT_COL, "Suspect (you)"),
    ]
    for col, label in legend:
        pygame.draw.circle(screen, col, (panel_x + 8, y + 8), 7)
        l_surf = font.render(label, True, TEXT_COL)
        screen.blit(l_surf, (panel_x + 20, y + 1))
        y += 22

    # Capture flash
    if captured_flash > 0:
        flash = font_big.render("CAPTURED!", True, CAPTURE_COL)
        fx = (W - flash.get_width()) // 2
        fy = oy + map_px // 2 - 30
        # background box
        bg = pygame.Surface((flash.get_width() + 30, flash.get_height() + 16), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        screen.blit(bg, (fx - 15, fy - 8))
        screen.blit(flash, (fx, fy))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("MAPPO Interactive Demo  –  Move your mouse to control the suspect!")
    clock  = pygame.time.Clock()

    font     = pygame.font.SysFont("Arial", 13)
    font_big = pygame.font.SysFont("Arial", 22, bold=True)

    # ── Load config & model ──────────────────────────────────────────
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    MAP_HALF     = float(env_cfg["map_half"])
    AGENT_RADIUS = float(env_cfg["agent_radius"])
    AGENT_SPD    = float(env_cfg["agent_max_speed"])
    CAP_RADIUS   = float(env_cfg["capture_radius"])
    DT           = float(env_cfg["dt"])
    MAX_STEPS    = int(env_cfg["max_steps"])

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    obs_norm = RunningMeanStd(shape=(int(cfg["env"].get("obs_dim", 13)),))
    obs_norm.mean  = ckpt["obs_norm_mean"]
    obs_norm.var   = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]
    actor = Actor(int(cfg["env"].get("obs_dim", 13)), 2, 64, map_half=MAP_HALF)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # ── Env for A* map & agent movement ─────────────────────────────
    env = PursuitEnv(cfg, render_mode=None)
    cm  = CoordMapper(VIEW_HALF, MAP_PX, MAP_OFFSET)

    # Pre-render map
    map_surf = build_map_surface(cm)

    # ── Episode state ────────────────────────────────────────────────
    def reset_episode():
        env.reset()
        return {
            "step":       0,
            "start_time": time.time(),
            "caps":       0,
            "cap_flash":  0,
            "target_vel": np.zeros(2),
            "trail":      deque(maxlen=25),
            "dist_min":   99.9,
        }

    state = reset_episode()

    # ── Main loop ────────────────────────────────────────────────────
    running = True
    pygame.mouse.set_visible(False)   # hide system cursor (we draw our own)

    while running:
        dt_real = clock.tick(FPS) / 1000.0   # actual frame time

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = reset_episode()

        # ── Mouse → world (capped velocity) ──────────────────────────
        mx, my = pygame.mouse.get_pos()
        
        # Convert mouse screen position to world coordinates
        target_mouse_x, target_mouse_y = cm.screen_to_world(mx, my)

        # Calculate exact vector towards the mouse from CURRENT intruder position
        desired_pos = np.array([target_mouse_x, target_mouse_y])
        current_pos = env.target_pos.copy()
        
        raw_diff = desired_pos - current_pos
        dist_to_mouse = float(np.linalg.norm(raw_diff))
        
        # Max speed limit for human intruder
        INTRUDER_MAX_SPD = AGENT_SPD
        TRAIN_SPD        = float(env_cfg.get("intruder_spd", 1.0))  # match training

        if dist_to_mouse > 1e-4:
            # Mouse is moving — follow it at capped speed
            max_step    = INTRUDER_MAX_SPD * dt_real
            actual_step = min(dist_to_mouse, max_step)
            new_pos     = current_pos + (raw_diff / dist_to_mouse) * actual_step
            raw_vel     = (new_pos - current_pos) / max(dt_real, 1e-4)
            # Normalise to TRAIN_SPD so the network always sees an in-distribution
            # velocity magnitude (it was trained with intruder_spd = 1.0 m/s)
            vel_mag = float(np.linalg.norm(raw_vel))
            tvel    = (raw_vel / vel_mag * TRAIN_SPD) if vel_mag > 0.05 else env.target_vel
            state["_last_tvel"] = tvel.copy()
        else:
            # Mouse stationary: Target velocity is exactly zero.
            # (Note: This is pure visualization logic. Any symmetry breaking
            # should happen in the MARL environment or policy, not here.)
            tvel = np.zeros(2)
            new_pos = current_pos    # intruder doesn't actually move

        # Clamp into free space
        if env.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
            new_pos = current_pos

        env.target_pos = new_pos
        env.target_vel = tvel
        state["trail"].append((new_pos[0], new_pos[1]))





        # ── MARL policy ──────────────────────────────────────────────
        obs = env._get_obs()   # uses updated target_pos/vel
        obs_n = obs_norm.normalize(obs)
        with torch.no_grad():
            act, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
        act_np = act.numpy()

        # ── Move agents ───────────────────────────────────────────────
        for i in range(env.n_agents):
            raw = np.array(act_np[i], dtype=np.float64)
            mag = float(np.linalg.norm(raw))
            if mag > 3.0:                       # cap at 3 m (same as training)
                raw = (raw / mag) * 3.0
            sg = env.agent_pos[i] + raw
            sg = np.clip(sg, -MAP_HALF + 0.3, MAP_HALF - 0.3)
            env._subgoals[i] = sg
            path = astar(env.obs_map, tuple(env.agent_pos[i].tolist()), tuple(sg.tolist()))
            env._paths[i] = path
            env.agent_pos[i], env.agent_vel[i] = env._move_along_path(
                env.agent_pos[i], path, AGENT_SPD
            )



        state["step"] += 1

        # ── Capture check (strict dual-agent encirclement) ────────────
        # Requires: both agents within CAP_RADIUS AND angle between them
        # (relative to intruder) >= 90°  (cos_theta <= 0)
        dists = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
        state["dist_min"] = min(state["dist_min"], float(dists.min()))
        d1, d2 = float(dists[0]), float(dists[1])
        captured = False
        if d1 <= CAP_RADIUS and d2 <= CAP_RADIUS:
            v1 = env.agent_pos[0] - env.target_pos
            v2 = env.agent_pos[1] - env.target_pos
            if d1 > 0.05 and d2 > 0.05:
                cos_theta = float(np.dot(v1, v2) / (d1 * d2))
                if cos_theta <= 0.0:   # angle >= 90°
                    captured = True

        if captured:
            caps_kept = state.get("caps", 0) + 1
            state = reset_episode()
            state["caps"]      = caps_kept
            state["cap_flash"] = FPS * 2


        if state["cap_flash"] > 0:
            state["cap_flash"] -= 1

        # ── Draw ──────────────────────────────────────────────────────
        screen.fill(BG)

        # Map background + obstacles
        screen.blit(map_surf, MAP_OFFSET)

        # A* paths (dashed)
        for i, (path, col) in enumerate([(env._paths[0], PATH1_COL), (env._paths[1], PATH2_COL)]):
            if path:
                draw_dashed_path(screen, path, col[:3], cm)

        # Subgoals (small circles)
        for i, col in enumerate([DOG1_COLOR, DOG2_COLOR]):
            sg = env._subgoals[i]
            if sg is not None:
                cx, cy = cm.world_to_screen(*sg)
                pygame.draw.circle(screen, col, (cx, cy), 6)
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 2)
                # Draw a thin line from agent to subgoal
                ax, ay = cm.world_to_screen(*env.agent_pos[i])
                pygame.draw.line(screen, col, (ax, ay), (cx, cy), 1)

        # Capture radius circle around each agent
        for i in range(env.n_agents):
            draw_capture_radius(screen, env.agent_pos[i], CAP_RADIUS, cm)

        # Suspect trail + body
        draw_suspect(screen, env.target_pos, 0.35, cm, state["trail"])

        # Agents
        draw_agent(screen, env.agent_pos[0], DOG1_COLOR, AGENT_RADIUS, cm, "A1", env.agent_vel[0])
        draw_agent(screen, env.agent_pos[1], DOG2_COLOR, AGENT_RADIUS, cm, "A2", env.agent_vel[1])

        # HUD
        elapsed = time.time() - state["start_time"]
        draw_hud(
            screen, font, font_big,
            step=state["step"],
            elapsed=elapsed,
            n_captured=state["caps"],
            dist_min=state["dist_min"],
            captured_flash=state["cap_flash"],
            escaped_flash=0,
            map_px=MAP_PX,
            map_off=MAP_OFFSET,
        )

        # Map border
        ox, oy = MAP_OFFSET
        pygame.draw.rect(screen, (80, 100, 160), (ox - 1, oy - 1, MAP_PX + 2, MAP_PX + 2), 2)

        # Custom mouse cursor (crosshair)
        pygame.draw.line(screen, SUSPECT_COL, (mx - 12, my), (mx + 12, my), 2)
        pygame.draw.line(screen, SUSPECT_COL, (mx, my - 12), (mx, my + 12), 2)
        pygame.draw.circle(screen, SUSPECT_COL, (mx, my), 5, 2)

        # Axis labels
        for label in [-6, -3, 0, 3, 6]:
            lx, ly = cm.world_to_screen(label, -cm.view_half)
            surf = font.render(str(label), True, DIM_COL)
            if 0 <= lx - ox <= MAP_PX:
                screen.blit(surf, (lx - surf.get_width() // 2, ly + 6))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
