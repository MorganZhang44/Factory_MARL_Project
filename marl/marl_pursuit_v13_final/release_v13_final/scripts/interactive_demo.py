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

import argparse
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
DEFAULT_CFG_PATH  = "configs/mappo_role_intruder_300k.yaml"
DEFAULT_CKPT_PATH = "checkpoints/v13_final.pt"

SCREEN_W, SCREEN_H = 1280, 800
MAP_PX     = 720          # map area in pixels (square)
VIEW_HALF  = 7.0          # viewport bounds (zoomed into the inner room)
MAP_OFFSET = (40, 40)     # top-left corner of the map on screen
DEBUG_PANEL_X = MAP_OFFSET[0] + MAP_PX + 16   # right panel start
DEBUG_PANEL_W = SCREEN_W - DEBUG_PANEL_X - 16

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

def find_reachable_near(env, ideal_pos, target_pos, max_radial_relax: float = 0.5):
    """If ideal_pos is in a wall, sweep angle around target_pos to find nearest
    free cell at the same distance. Tries small angle perturbations first, then
    larger, then also shrinks the radial distance."""
    if not env.obs_map.is_collision(float(ideal_pos[0]), float(ideal_pos[1])):
        return ideal_pos
    base_v = ideal_pos - target_pos
    base_dist = float(np.linalg.norm(base_v))
    if base_dist < 0.1:
        return ideal_pos
    base_angle = float(np.arctan2(base_v[1], base_v[0]))
    for d_factor in (1.0, 0.85, 0.7, 0.55):
        for d_deg in (5, -5, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60, 80, -80):
            ang = base_angle + np.radians(d_deg)
            r = base_dist * d_factor
            cand = target_pos + r * np.array([np.cos(ang), np.sin(ang)])
            if not env.obs_map.is_collision(float(cand[0]), float(cand[1])):
                return cand
    return ideal_pos


def compute_closing_actions(
    env,
    agent_pos: np.ndarray,
    target_pos: np.ndarray,
    capture_radius: float,
    agent_yaw: np.ndarray = None,
    face_thresh: float = 0.5,
):
    """Deterministic fallback. Returns (actions, lock_flags) where
    `actions` is the (2,2) per-agent offset for the env's regular motion,
    and `lock_flags` is a length-2 boolean: True means the agent should
    SKIP forward translation this frame and ONLY rotate yaw toward the
    intruder. The demo's main loop honours `lock_flags` to give us a
    rotate-in-place phase that the env's heading-locked dynamics
    otherwise can't express.

    Slot geometry:
      pursuer (closer): radius `target_dist` along its current angle
      encircler (farther): opposite side at same radius
    Lock-in conditions per agent:
      ① already inside (or just outside) the capture range, AND
      ② within `align_radius` of its ideal slot OR not facing the target.
    """
    target_dist  = capture_radius * 0.7   # 1.05 m for capture_radius=1.5
    align_radius = 0.45                   # within 0.45m of ideal slot
    cap_safe     = capture_radius - 0.1   # 1.40m (safe interior of cap zone)
    overshoot    = target_dist - 0.3      # 0.75m → past slot, near target
    face_lock_th = 0.7                    # cos_face above this = "fully facing"

    d = np.linalg.norm(agent_pos - target_pos, axis=1)
    pursuer_idx   = int(np.argmin(d))
    encircler_idx = 1 - pursuer_idx
    actions     = np.zeros((2, 2), dtype=np.float64)
    lock_flags  = [False, False]

    v_p = agent_pos[pursuer_idx] - target_pos
    n_p = float(np.linalg.norm(v_p))
    v_e = agent_pos[encircler_idx] - target_pos
    n_e = float(np.linalg.norm(v_e))

    # Encirclement angle (between v_p and v_e). cos<=0 means angle>=90°.
    if n_p > 0.05 and n_e > 0.05:
        cos_pe = float(np.dot(v_p, v_e) / (n_p * n_e))
    else:
        cos_pe = 1.0
    encircled = cos_pe <= 0.0

    # Pursuer ideal slot: keep its angle, radius = target_dist
    if n_p > 0.1:
        ideal_p = target_pos + (v_p / n_p) * target_dist
    else:
        ideal_p = agent_pos[pursuer_idx].copy()
    ideal_p = find_reachable_near(env, ideal_p, target_pos)

    # Encircler ideal slot logic:
    #   • If already on opposite side (cos_pe<=0): aim at the EXACT antipode.
    #   • If on same side as pursuer:               aim at a perpendicular
    #     waypoint first so encircler swings tangentially around the target
    #     instead of trying to go through the pursuer.
    # Encircler ideal — combined staged-detour + orbit-step:
    #   • cos_pe ≤ -0.5  (≥ ~120° apart) → final slot at antipode,
    #     radius = target_dist; lockable.
    #   • cos_pe ≤  0.3  (≥ ~72° apart)  → antipode at wider radius (2.0 m)
    #     so encircler doesn't lock; not lockable.
    #   • cos_pe >  0.3  (same side)     → BLEND of perpendicular tangent
    #     and "anti-pursuer" direction, at radius cap_radius+0.3 = 1.8 m.
    #     The blend pushes the encircler around the target in the chosen
    #     swing direction while also nudging it AWAY from the pursuer's
    #     current angle — this lets it reach the antipode without trying
    #     to walk through the pursuer. Not lockable.
    encircler_on_detour = False
    if n_p > 0.1:
        u_p = v_p / n_p
        if cos_pe <= -0.5:
            ideal_e = target_pos - u_p * target_dist
        elif cos_pe <= 0.3:
            ideal_e = target_pos - u_p * 2.0
            encircler_on_detour = True
        else:
            if n_e > 0.05:
                cross = float(v_p[0]*v_e[1] - v_p[1]*v_e[0])
                if abs(cross) < 0.05:
                    sign = 1.0
                else:
                    sign = 1.0 if cross >= 0.0 else -1.0
            else:
                sign = 1.0
            perp = np.array([-u_p[1], u_p[0]]) * sign
            ideal_e = target_pos + perp * (capture_radius + 0.3)  # 1.8 m
            encircler_on_detour = True
    else:
        ideal_e = target_pos + np.array([target_dist, 0.0])
    ideal_e = find_reachable_near(env, ideal_e, target_pos)

    ideals = {pursuer_idx: ideal_p, encircler_idx: ideal_e}
    lockable = {pursuer_idx: True, encircler_idx: not encircler_on_detour}

    # Pre-compute facing cosines for both agents.
    cf = [1.0, 1.0]
    if agent_yaw is not None:
        for i in range(2):
            to_t = target_pos - agent_pos[i]
            nt = float(np.linalg.norm(to_t))
            if nt > 1e-3:
                head = np.array([np.cos(float(agent_yaw[i])),
                                 np.sin(float(agent_yaw[i]))])
                cf[i] = float(np.dot(head, to_t / nt))

    for i in range(2):
        gap_to_slot = float(np.linalg.norm(agent_pos[i] - ideals[i]))
        d_to_target = d[i]

        slot_close   = gap_to_slot <= align_radius
        in_safe_zone = d_to_target <= cap_safe        # safely inside cap zone
        too_close    = d_to_target <= overshoot       # past slot, near target
        not_facing   = cf[i] < face_lock_th

        # Lock fires when EITHER:
        #   ① agent reached its (lockable) slot AND is inside cap zone
        #   ② agent is in safe zone AND encirclement geometry is good
        #   ③ agent overshot AND geometry is encircled (lock + face)
        # Do NOT lock on "too_close alone" — if geometry is wrong (cos>0)
        # the agent must keep moving so the OTHER agent's detour can complete.
        lock_now = (
            (slot_close and in_safe_zone and lockable[i])
            or (in_safe_zone and encircled)
            or (too_close and encircled)
        )
        if lock_now:
            lock_flags[i] = True
            actions[i] = np.zeros(2)
        else:
            actions[i] = ideals[i] - agent_pos[i]

    # Cap magnitude (matches env's max_offset)
    for i in range(2):
        mag = float(np.linalg.norm(actions[i]))
        if mag > 3.0:
            actions[i] = actions[i] / mag * 3.0
    return actions, lock_flags


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


def _kv(screen, font, x, y, key, val, key_col=DIM_COL, val_col=TEXT_COL, key_w=110):
    k = font.render(key, True, key_col)
    v = font.render(val, True, val_col)
    screen.blit(k, (x, y))
    screen.blit(v, (x + key_w, y))


def _section(screen, font_bold, x, y, title, w):
    pygame.draw.line(screen, (60, 80, 120), (x, y), (x + w, y), 1)
    s = font_bold.render(title, True, TEXT_COL)
    screen.blit(s, (x, y + 4))
    return y + 24


def draw_lidar(screen, x, y, lidar_vals, label, color, font_small):
    """Draw 8 lidar rays as a small radial diagram with normalised lengths."""
    cx, cy, R = x + 30, y + 30, 28
    pygame.draw.circle(screen, (40, 50, 80), (cx, cy), R, 1)
    for i, v in enumerate(lidar_vals):
        ang = i * np.pi / 4
        ex = int(cx + R * float(v) * np.cos(ang))
        ey = int(cy - R * float(v) * np.sin(ang))
        col = (int(255 * (1 - v)), int(180 * v), 60) if v < 0.7 else (60, 200, 120)
        pygame.draw.line(screen, col, (cx, cy), (ex, ey), 2)
        pygame.draw.circle(screen, col, (ex, ey), 2)
    pygame.draw.circle(screen, color, (cx, cy), 3)
    s = font_small.render(label, True, color)
    screen.blit(s, (x + 65, y + 10))
    # Min lidar value (closest wall)
    min_v = float(min(lidar_vals))
    s2 = font_small.render(f"min ray = {min_v:.2f}", True, DIM_COL)
    screen.blit(s2, (x + 65, y + 28))


def draw_debug_panel(screen, font, font_bold, font_big, font_small, dbg):
    """All-in-one debug panel. `dbg` is a dict with everything we need."""
    px = DEBUG_PANEL_X
    pw = DEBUG_PANEL_W
    y = MAP_OFFSET[1]

    # ─── Title ───
    title = font_big.render("MAPPO Debug", True, TEXT_COL)
    screen.blit(title, (px, y))
    y += 28

    # Mode badge
    mode_col = (231, 76, 60) if dbg["fallback_active"] else (80, 200, 120)
    mode_lbl = "FALLBACK (geom)" if dbg["fallback_active"] else "RL POLICY"
    badge = font_small.render(f"[{mode_lbl}]", True, mode_col)
    screen.blit(badge, (px, y))
    y += 18

    # Controls hint
    for k, v in [("Mouse", "intruder"), ("R", "reset"), ("ESC", "quit")]:
        _kv(screen, font_small, px, y, f"{k}:", v, key_col=WARN_COL, key_w=42)
        y += 14

    # ─── Episode ───
    y = _section(screen, font_bold, px, y + 6, "EPISODE", pw)
    _kv(screen, font, px, y, "Step:",     str(dbg["step"]));         y += 18
    _kv(screen, font, px, y, "Time:",     f"{dbg['elapsed']:.1f}s"); y += 18
    _kv(screen, font, px, y, "Captures:", str(dbg["caps"]));         y += 18
    _kv(screen, font, px, y, "Min dist:", f"{dbg['dist_min']:.2f}m"); y += 18

    # ─── Capture conditions ───
    y = _section(screen, font_bold, px, y + 6, "CAPTURE CONDITION", pw)
    cap_col = (80, 200, 120) if dbg["captured"] else TEXT_COL
    _kv(screen, font, px, y, "captured?", "YES" if dbg["captured"] else "NO",
        val_col=cap_col); y += 18
    g = (80, 200, 120); r = (231, 76, 60)
    _kv(screen, font, px, y, "d1 ≤ 1.5m:", f"{dbg['d1']:.2f}",
        val_col=g if dbg["d1"] <= 1.5 else r); y += 18
    _kv(screen, font, px, y, "d2 ≤ 1.5m:", f"{dbg['d2']:.2f}",
        val_col=g if dbg["d2"] <= 1.5 else r); y += 18
    _kv(screen, font, px, y, "cos θ ≤ 0:", f"{dbg['cos_theta']:+.2f}",
        val_col=g if dbg["cos_theta"] <= 0 else r); y += 18
    _kv(screen, font, px, y, "angle θ:", f"{np.degrees(np.arccos(np.clip(dbg['cos_theta'], -1, 1))):.0f}°"); y += 18
    _kv(screen, font, px, y, "facing OK:", "YES" if dbg.get("facing_ok", False) else "NO",
        val_col=g if dbg.get("facing_ok", False) else r); y += 18
    hold      = int(dbg.get("capture_hold", 0))
    hold_tgt  = int(dbg.get("capture_hold_target", 20))
    hold_sec  = hold * 0.1   # dt=0.1
    tgt_sec   = hold_tgt * 0.1
    hold_col  = g if hold >= hold_tgt else (WARN_COL if hold > 0 else r)
    _kv(screen, font, px, y, "hold:", f"{hold_sec:.1f}/{tgt_sec:.1f}s ({hold}/{hold_tgt})",
        val_col=hold_col); y += 18

    # ─── Fallback state ───
    y = _section(screen, font_bold, px, y + 6, "FALLBACK CONTROLLER", pw)
    _kv(screen, font, px, y, "close frames:", str(dbg["close_frames"])); y += 18
    _kv(screen, font, px, y, "max d (both):", f"{dbg['max_d']:.2f}m"); y += 18
    _kv(screen, font, px, y, "trigger:",
        ("≤ 2m, 1f"   if dbg['max_d'] <= 2.0 else
         "≤ 3m, 5f"   if dbg['max_d'] <= 3.0 else
         "≤ 4m, 10f"  if dbg['max_d'] <= 4.0 else
         "≤ 6m, 20f"  if dbg['max_d'] <= 6.0 else
         "out of range")); y += 18
    lf = dbg.get("lock_flags", [False, False])
    _kv(screen, font, px, y, "lock A1/A2:",
        ("ROT" if lf[0] else "MOVE") + " / " + ("ROT" if lf[1] else "MOVE"),
        val_col=(WARN_COL if any(lf) else DIM_COL)); y += 18

    # ─── Per-agent state ───
    role_col = [(231, 76, 60), (0, 180, 216)]   # PURSUER, ENCIRCLER
    role_lbl = ["P (pursuer)", "E (encircler)"]
    for i in range(2):
        col = role_col[int(dbg['roles'][i])] if dbg["use_roles"] else (DOG1_COLOR if i == 0 else DOG2_COLOR)
        lbl = role_lbl[int(dbg['roles'][i])] if dbg["use_roles"] else f"Agent {i}"
        y = _section(screen, font_bold, px, y + 6, f"AGENT {i} — {lbl}", pw)
        pygame.draw.circle(screen, col, (px + pw - 12, y - 16), 6)
        _kv(screen, font, px, y, "pos:",  f"({dbg['agent_pos'][i][0]:+.2f}, {dbg['agent_pos'][i][1]:+.2f})"); y += 18
        v = dbg['agent_vel'][i]; spd = float(np.linalg.norm(v))
        _kv(screen, font, px, y, "vel:",  f"({v[0]:+.2f}, {v[1]:+.2f})  |v|={spd:.2f}"); y += 18
        _kv(screen, font, px, y, "d to YOU:", f"{dbg['d_target'][i]:.2f}m"); y += 18
        a = dbg['actions'][i]; amag = float(np.linalg.norm(a))
        _kv(screen, font, px, y, "action:", f"({a[0]:+.2f}, {a[1]:+.2f})  |a|={amag:.2f}"); y += 18
        _kv(screen, font, px, y, "subgoal:", f"({dbg['subgoals'][i][0]:+.2f}, {dbg['subgoals'][i][1]:+.2f})"); y += 18

    # ─── Intruder ───
    y = _section(screen, font_bold, px, y + 6, "INTRUDER", pw)
    _kv(screen, font, px, y, "pos:", f"({dbg['target_pos'][0]:+.2f}, {dbg['target_pos'][1]:+.2f})"); y += 18
    tv = dbg['target_vel']; tvm = float(np.linalg.norm(tv))
    _kv(screen, font, px, y, "vel:", f"({tv[0]:+.2f}, {tv[1]:+.2f})  |v|={tvm:.2f}"); y += 18
    _kv(screen, font, px, y, "d agents:", f"{float(np.linalg.norm(dbg['agent_pos'][0] - dbg['agent_pos'][1])):.2f}m"); y += 18

    # ─── Lidar mini-displays ───
    y = _section(screen, font_bold, px, y + 6, "LIDAR (8 rays)", pw)
    draw_lidar(screen, px,         y, dbg['lidar'][0],
               role_lbl[int(dbg['roles'][0])] if dbg["use_roles"] else "A1",
               role_col[int(dbg['roles'][0])] if dbg["use_roles"] else DOG1_COLOR, font_small)
    draw_lidar(screen, px + 200,   y, dbg['lidar'][1],
               role_lbl[int(dbg['roles'][1])] if dbg["use_roles"] else "A2",
               role_col[int(dbg['roles'][1])] if dbg["use_roles"] else DOG2_COLOR, font_small)
    y += 70

    # ─── CAPTURED flash overlay (centred on map) ───
    if dbg["cap_flash"] > 0:
        flash = font_big.render("CAPTURED!", True, CAPTURE_COL)
        fx = MAP_OFFSET[0] + (MAP_PX - flash.get_width()) // 2
        fy = MAP_OFFSET[1] + MAP_PX // 2 - 30
        bg = pygame.Surface((flash.get_width() + 30, flash.get_height() + 16), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        screen.blit(bg, (fx - 15, fy - 8))
        screen.blit(flash, (fx, fy))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=DEFAULT_CFG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("MAPPO Interactive Demo  –  Move your mouse to control the suspect!")
    clock  = pygame.time.Clock()

    font_small = pygame.font.SysFont("Consolas", 11)
    font       = pygame.font.SysFont("Consolas", 13)
    font_bold  = pygame.font.SysFont("Consolas", 13, bold=True)
    font_big   = pygame.font.SysFont("Arial",    22, bold=True)

    # ── Load config & model ──────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    MAP_HALF     = float(env_cfg["map_half"])
    AGENT_RADIUS = float(env_cfg["agent_radius"])
    AGENT_SPD    = float(env_cfg["agent_max_speed"])
    CAP_RADIUS   = float(env_cfg["capture_radius"])
    DT           = float(env_cfg["dt"])
    MAX_STEPS    = int(env_cfg["max_steps"])
    USE_ROLES    = bool(env_cfg.get("use_roles", False))
    OBS_DIM      = int(env_cfg.get("obs_dim", 22 if USE_ROLES else 21))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Prefer obs_dim from checkpoint if present (handles old vs new ckpts)
    OBS_DIM = int(ckpt.get("obs_dim", OBS_DIM))
    obs_norm = RunningMeanStd(shape=(OBS_DIM,))
    obs_norm.mean  = ckpt["obs_norm_mean"]
    obs_norm.var   = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]
    actor = Actor(OBS_DIM, 2, 64, map_half=MAP_HALF)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"[demo] loaded ckpt={args.checkpoint} obs_dim={OBS_DIM} use_roles={USE_ROLES}")

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
            "stuck_frames": 0,
            "min_d_seen":   99.9,
            "close_frames": 0,
            "capture_hold": 0,
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

        # ── Mouse → world (capped + speed-limited to training distribution) ──
        mx, my = pygame.mouse.get_pos()
        target_mouse_x, target_mouse_y = cm.screen_to_world(mx, my)
        desired_pos = np.array([target_mouse_x, target_mouse_y])
        current_pos = env.target_pos.copy()
        raw_diff = desired_pos - current_pos
        dist_to_mouse = float(np.linalg.norm(raw_diff))

        # Use the SAME speed the policy saw during training (key fix: was
        # reading wrong cfg key `intruder_spd` -> default 1.0; correct key is
        # `intruder_speed` and training was 1.2 m/s).
        TRAIN_SPD = float(env_cfg.get("intruder_speed", env_cfg.get("intruder_spd", 1.2)))

        if dist_to_mouse > 1e-4:
            # Cap distance the intruder can move this frame at training speed
            max_step    = TRAIN_SPD * dt_real
            actual_step = min(dist_to_mouse, max_step)
            new_pos     = current_pos + (raw_diff / dist_to_mouse) * actual_step
            raw_vel     = (new_pos - current_pos) / max(dt_real, 1e-4)
            vel_mag = float(np.linalg.norm(raw_vel))
            tvel_target = (raw_vel / vel_mag * TRAIN_SPD) if vel_mag > 0.05 else np.zeros(2)
        else:
            tvel_target = np.zeros(2)
            new_pos = current_pos

        if env.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
            new_pos = current_pos

        # ── EMA smoothing for intruder velocity (kills high-frequency mouse jitter) ──
        # Without this, target_vel direction can flip every frame as the user wiggles
        # the mouse — out of training distribution and a major cause of agent jitter.
        prev_tvel = state.get("_last_tvel", np.zeros(2))
        alpha = 0.25
        tvel = (1.0 - alpha) * prev_tvel + alpha * tvel_target
        state["_last_tvel"] = tvel

        env.target_pos = new_pos
        env.target_vel = tvel
        state["trail"].append((new_pos[0], new_pos[1]))

        # ── Re-evaluate roles with the SAME hysteresis as training ──
        # Without this, role flag in obs is frozen since reset → policy sees
        # stale role context that no longer matches geometry.
        if USE_ROLES and env.n_agents == 2:
            d_now = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
            env._maybe_switch_roles(d_now)





        # ── MARL policy ──────────────────────────────────────────────
        obs = env._get_obs()   # uses updated target_pos/vel
        obs_n = obs_norm.normalize(obs)
        with torch.no_grad():
            act, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
        act_np = act.numpy()

        # ── Closing-maneuver fallback ────────────────────────────────
        # Aggressive activation: as soon as agents are in striking range, count
        # the consecutive frames. Fire the deterministic controller based on
        # how close they already are.
        d_to_target = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
        max_d = float(d_to_target.max())
        min_d = float(d_to_target.min())

        if max_d <= 6.0:
            state["close_frames"] = state.get("close_frames", 0) + 1
        else:
            state["close_frames"] = 0

        # Tiered activation under v12+ dynamics (yaw + 2 s capture lock):
        #   max_d ≤ 2.0 : already adjacent  → fire immediately
        #   max_d ≤ 3.0 : close             → fire after 5 frames (0.25 s)
        #   max_d ≤ 4.0 : engaging          → fire after 10 frames (0.5 s)
        #   max_d ≤ 6.0 : approaching       → fire after 20 frames (1.0 s)
        # Looser triggers because the closing-maneuver fallback has the
        # rotate-in-place lock and orbit detour the RL policy can't learn.
        cf = state["close_frames"]
        fallback_active = (
            (max_d <= 2.0 and cf >= 1)
            or (max_d <= 3.0 and cf >= 5)
            or (max_d <= 4.0 and cf >= 10)
            or (max_d <= 6.0 and cf >= 20)
        )

        lock_flags = [False, False]
        if fallback_active:
            act_np, lock_flags = compute_closing_actions(
                env,
                env.agent_pos,
                env.target_pos,
                CAP_RADIUS,
                agent_yaw=env.agent_yaw,
                face_thresh=env.face_capture_thresh,
            )

        # ── Move agents ───────────────────────────────────────────────
        for i in range(env.n_agents):
            if lock_flags[i]:
                # LOCK mode: skip translation, rotate yaw toward target.
                to_t = env.target_pos - env.agent_pos[i]
                nt = float(np.linalg.norm(to_t))
                if nt > 1e-3:
                    desired = float(np.arctan2(to_t[1], to_t[0]))
                    err = (desired - float(env.agent_yaw[i]) + np.pi) % (2.0*np.pi) - np.pi
                    max_step = env.agent_max_omega * env.dt
                    err = float(np.clip(err, -max_step, max_step))
                    env.agent_yaw[i] = (float(env.agent_yaw[i]) + err + np.pi) % (2.0*np.pi) - np.pi
                env.agent_vel[i] = np.zeros(2)
                env._subgoals[i] = env.agent_pos[i].copy()
                env._paths[i] = []
                continue

            raw = np.array(act_np[i], dtype=np.float64)
            mag = float(np.linalg.norm(raw))
            if mag > 3.0:                       # cap at 3 m (same as training)
                raw = (raw / mag) * 3.0
            sg = env.agent_pos[i] + raw
            sg = np.clip(sg, -MAP_HALF + 0.3, MAP_HALF - 0.3)
            env._subgoals[i] = sg
            path = astar(env.obs_map, tuple(env.agent_pos[i].tolist()), tuple(sg.tolist()))
            env._paths[i] = path
            # Hard agent-agent collision + yaw rate limit
            other_pos = env.agent_pos[1 - i].copy() if env.n_agents == 2 else None
            (env.agent_pos[i],
             env.agent_vel[i],
             env.agent_yaw[i]) = env._move_along_path(
                env.agent_pos[i], path, AGENT_SPD,
                other_pos=other_pos,
                current_yaw=float(env.agent_yaw[i]),
                max_omega=env.agent_max_omega,
            )



        state["step"] += 1

        # ── Capture check (geometry + hold timer) ──────────────────────
        # Geometry: both within CAP_RADIUS, encirclement angle ≥ 90°,
        # both agents' heading pointing at the intruder.
        # Capture finalises only when geometry has held for ≥ capture_hold_steps.
        dists = np.linalg.norm(env.agent_pos - env.target_pos, axis=1)
        state["dist_min"] = min(state["dist_min"], float(dists.min()))
        d1, d2 = float(dists[0]), float(dists[1])
        v1 = env.agent_pos[0] - env.target_pos
        v2 = env.agent_pos[1] - env.target_pos
        if d1 > 0.05 and d2 > 0.05:
            cos_theta = float(np.dot(v1, v2) / (d1 * d2))
        else:
            cos_theta = 1.0
        facing_ok = True
        for ai in range(env.n_agents):
            head = np.array([np.cos(env.agent_yaw[ai]), np.sin(env.agent_yaw[ai])])
            to_t = env.target_pos - env.agent_pos[ai]
            nt = float(np.linalg.norm(to_t))
            if nt > 0.05 and float(np.dot(head, to_t / nt)) < env.face_capture_thresh:
                facing_ok = False
                break
        capture_geom = (d1 <= CAP_RADIUS and d2 <= CAP_RADIUS
                        and cos_theta <= 0.0 and facing_ok)
        # Update hold counter
        if capture_geom:
            state["capture_hold"] = state.get("capture_hold", 0) + 1
        else:
            state["capture_hold"] = 0
        captured = state["capture_hold"] >= env.capture_hold_steps

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

        # Agents — color by ROLE so role-switching is visible
        # PURSUER=tomato (close-in chaser), ENCIRCLER=teal (flanker)
        ROLE_COLORS = [(231, 76, 60), (0, 180, 216)]   # PURSUER, ENCIRCLER
        if USE_ROLES:
            c0 = ROLE_COLORS[int(env.roles[0])]
            c1 = ROLE_COLORS[int(env.roles[1])]
            label0 = "P" if env.roles[0] == 0 else "E"
            label1 = "P" if env.roles[1] == 0 else "E"
        else:
            c0, c1 = DOG1_COLOR, DOG2_COLOR
            label0, label1 = "A1", "A2"
        draw_agent(screen, env.agent_pos[0], c0, AGENT_RADIUS, cm, label0, env.agent_vel[0])
        draw_agent(screen, env.agent_pos[1], c1, AGENT_RADIUS, cm, label1, env.agent_vel[1])

        # ── Build debug bundle and draw debug panel ───────────────────
        elapsed = time.time() - state["start_time"]
        # Compute lidar (8 rays) for each agent — for the panel only
        LIDAR_ANGLES = [k * np.pi / 4 for k in range(8)]
        lidar_per_agent = []
        for i in range(env.n_agents):
            px_i, py_i = float(env.agent_pos[i][0]), float(env.agent_pos[i][1])
            lidar_per_agent.append([env.obs_map.ray_cast(px_i, py_i, a, max_range=8.0)
                                    for a in LIDAR_ANGLES])
        dbg = {
            "step":       state["step"],
            "elapsed":    elapsed,
            "caps":       state["caps"],
            "dist_min":   state["dist_min"],
            "cap_flash":  state["cap_flash"],
            "captured":   captured,
            "d1": d1, "d2": d2,
            "cos_theta":  cos_theta,
            "facing_ok":  facing_ok,
            "capture_hold":        state.get("capture_hold", 0),
            "capture_hold_target": env.capture_hold_steps,
            "fallback_active": fallback_active,
            "close_frames":    state.get("close_frames", 0),
            "lock_flags":      list(lock_flags),
            "max_d":           max_d,
            "use_roles":       USE_ROLES,
            "roles":           env.roles.copy() if USE_ROLES else np.zeros(env.n_agents, dtype=np.int64),
            "agent_pos":       env.agent_pos.copy(),
            "agent_vel":       env.agent_vel.copy(),
            "actions":         act_np.copy(),
            "subgoals":        env._subgoals.copy(),
            "d_target":        dists,
            "target_pos":      env.target_pos.copy(),
            "target_vel":      env.target_vel.copy(),
            "lidar":           lidar_per_agent,
        }
        draw_debug_panel(screen, font, font_bold, font_big, font_small, dbg)

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
