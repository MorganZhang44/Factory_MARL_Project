# Copyright (c) 2026, Multi-Agent Surveillance Project
# Real-time visualization dashboard for surveillance system.

from __future__ import annotations

import torch
import numpy as np
from typing import Optional
import os
import time

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from perception.fusion import FusionResult


class SurveillanceVisualizer:
    """Premium real-time visualization of the surveillance system.

    Creates a professional dashboard with:
    - Multi-camera surveillance wall
    - High-fidelity top-down tactical map
    - Real-time error and confidence telemetry
    - Dark mode aesthetics with HSL-tailored colors
    """

    def __init__(
        self,
        output_dir: str = "output/viz",
        scene_size: float = 20.0,
        save_frames: bool = True,
        display: bool = False,
        map_bounds: dict[str, float] | None = None,
        camera_markers: list[tuple[float, float, str]] | None = None,
    ):
        self.output_dir = output_dir
        self.scene_size = scene_size
        self.save_frames = save_frames
        self.display = display
        self.map_bounds = map_bounds
        self.camera_markers = camera_markers or []
        self.frame_count = 0
        self._error_history: list[float] = []
        self._conf_history: list[float] = []
        
        # Premium Color Palette (HSL based)
        self.colors = {
            "bg": "#0f172a",          # Slate 900
            "panel": "#1e293b",       # Slate 800
            "text": "#f8fafc",        # Slate 50
            "text_dim": "#94a3b8",    # Slate 400
            "suspect": "#ef4444",     # Red 500
            "gt": "#22c55e",          # Green 500
            "dog": "#3b82f6",         # Blue 500
            "cam": "#eab308",         # Yellow 500
            "grid": "#334155",        # Slate 700
            "accent": "#8b5cf6",      # Violet 500
        }

        if save_frames:
            os.makedirs(output_dir, exist_ok=True)

    def render_frame(
        self,
        fusion_result: FusionResult,
        camera_images: dict[str, np.ndarray] | None = None,
        dog_positions: dict[str, torch.Tensor] | None = None,
        static_objects: list[tuple[float, float, str]] | None = None,
    ) -> np.ndarray | None:
        if not HAS_MPL:
            self._log_text(fusion_result)
            return None

        self._error_history.append(fusion_result.error_meters if fusion_result.error_meters != float("inf") else 0.0)
        self._conf_history.append(fusion_result.confidence)

        # Create figure with dark background
        plt.rcParams['text.color'] = self.colors["text"]
        plt.rcParams['axes.labelcolor'] = self.colors["text"]
        plt.rcParams['xtick.color'] = self.colors["text_dim"]
        plt.rcParams['ytick.color'] = self.colors["text_dim"]
        
        fig = plt.figure(figsize=(20, 10), facecolor=self.colors["bg"])
        gs = gridspec.GridSpec(2, 4, figure=fig)

        # ---- Panel 1: Tactical Map (Large) ----
        ax_map = fig.add_subplot(gs[0:2, 0:2])
        self._draw_topdown_map(ax_map, fusion_result, dog_positions, static_objects)

        # ---- Panel 2: Camera Feed Wall ----
        ax_cams = fig.add_subplot(gs[0, 2:4])
        if camera_images:
            self._draw_camera_montage(ax_cams, camera_images, fusion_result)
        else:
            self._draw_empty_panel(ax_cams, "CAMERA FEEDS OFFLINE")

        # ---- Panel 3: Error Telemetry ----
        ax_error = fig.add_subplot(gs[1, 2])
        self._draw_error_plot(ax_error)

        # ---- Panel 4: Confidence & Status ----
        ax_status = fig.add_subplot(gs[1, 3])
        self._draw_status_panel(ax_status, fusion_result)

        plt.tight_layout(pad=3.0)

        # Save frame
        if self.save_frames:
            filepath = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.png")
            fig.savefig(filepath, dpi=100, bbox_inches="tight", facecolor=self.colors["bg"])

        # Convert to RGB array
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        # Display
        if self.display and HAS_CV2:
            cv2.imshow("Surveillance Dashboard [PRO]", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        self.frame_count += 1
        return buf

    def _draw_topdown_map(self, ax, result, dog_pos, static_obj):
        ax.set_facecolor(self.colors["panel"])

        if self.map_bounds is not None:
            x_min = self.map_bounds["x_min"]
            x_max = self.map_bounds["x_max"]
            y_min = self.map_bounds["y_min"]
            y_max = self.map_bounds["y_max"]
        else:
            half = self.scene_size / 2
            x_min = -half
            x_max = half
            y_min = -half
            y_max = half

        padding = 1.0
        
        # Grid lines
        ax.grid(True, color=self.colors["grid"], linestyle='--', alpha=0.5)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect("equal")
        
        # Scene Boundary
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=3, edgecolor=self.colors["grid"], facecolor="none", zorder=1)
        ax.add_patch(rect)

        # Static Obstacles
        if static_obj:
            for x, y, label in static_obj:
                color = "#475569" if label == "pillar" else "#78350f"
                ax.add_patch(Circle((x, y), 0.4, color=color, alpha=0.8, zorder=2))

        # Camera Vision Cones (simplified)
        for cx, cy, lbl in self.camera_markers:
            ax.plot(cx, cy, "D", color=self.colors["cam"], markersize=10, zorder=10)
            ax.annotate(lbl.upper(), (cx, cy), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, color=self.colors["cam"],
                        fontweight='bold', ha='center')
            # Add simple FOV indicator
            # self._draw_fov(ax, cx, cy, ang, 60, 5.0)

        # Dogs
        if dog_pos:
            for name, pos in dog_pos.items():
                px, py = pos[0].item(), pos[1].item()
                ax.plot(px, py, "o", color=self.colors["dog"], markersize=12, label=name, zorder=8)
                ax.annotate(name.upper(), (px, py), textcoords="offset points",
                           xytext=(0, 10), fontsize=9, color=self.colors["dog"], 
                           fontweight='bold', ha='center')

        # Ground Truth
        if result.ground_truth is not None:
            gt = result.ground_truth
            ax.plot(gt[0].item(), gt[1].item(), "x", color=self.colors["gt"],
                    markersize=14, markeredgewidth=4, label="Target (GT)", zorder=9)

        # Estimated Position
        if result.detected and result.position_world is not None:
            est = result.position_world
            # Draw uncertainty circle
            uncert = max(0.2, (1.1 - result.confidence) * 1.5)
            circle = Circle((est[0].item(), est[1].item()), uncert, 
                            color=self.colors["suspect"], alpha=0.2, zorder=4)
            ax.add_patch(circle)
            ax.plot(est[0].item(), est[1].item(), "o", color=self.colors["suspect"],
                    markersize=10, markeredgewidth=2, markerfacecolor="none", label="Estimated", zorder=10)

            # Error Vector
            if result.ground_truth is not None:
                gt = result.ground_truth
                ax.plot([gt[0].item(), est[0].item()], [gt[1].item(), est[1].item()],
                        "-", color="#fbbf24", linewidth=2, alpha=0.8, zorder=5)

        ax.set_title("TACTICAL SURVEILLANCE MAP", fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc="upper left", fontsize=8, facecolor=self.colors["panel"], framealpha=0.9)

    def _draw_camera_montage(self, ax, images, result: FusionResult):
        feeds = list(images.values())
        names = list(images.keys())
        
        cols = 3 if len(feeds) > 4 else 2 if len(feeds) > 1 else 1
        rows = max(1, int(np.ceil(len(feeds) / cols)))
        h, w = feeds[0].shape[:2]
        montage = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Map camera names to detections for easy lookup
        cam_to_det = {det.camera_name: det for det in result.camera_detections}

        total_slots = rows * cols
        for i in range(min(total_slots, len(feeds))):
            r, c = divmod(i, cols)
            cam_name = names[i]
            img = feeds[i][:, :, :3].copy()
            
            # Draw bounding box if detected
            if cam_name in cam_to_det:
                det = cam_to_det[cam_name]
                if det.detected and det.bbox_2d:
                    x1, y1, x2, y2 = det.bbox_2d
                    # Draw rectangle in red
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 50, 50), 3)
                    # Add label
                    cv2.putText(img, f"SUSPECT {int(det.confidence*100)}%", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

            img_resized = cv2.resize(img, (w, h))
            montage[r * h:(r + 1) * h, c * w:(c + 1) * w] = img_resized
            
            # Add camera label
            cv2.putText(montage, cam_name.upper(), (c*w + 10, r*h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ax.imshow(montage)
        ax.set_title("LIVE MULTI-FEED MONITOR", fontsize=12, fontweight='bold')
        ax.axis("off")

    def _draw_error_plot(self, ax):
        ax.set_facecolor(self.colors["panel"])
        ax.plot(self._error_history, color=self.colors["suspect"], linewidth=2)
        ax.fill_between(range(len(self._error_history)), self._error_history, 
                        color=self.colors["suspect"], alpha=0.1)
        ax.set_title("LOCALIZATION ERROR (m)", fontsize=10, fontweight='bold')
        ax.set_ylim(0, 5.0)
        ax.grid(True, color=self.colors["grid"], alpha=0.3)

    def _draw_status_panel(self, ax, result):
        ax.set_facecolor(self.colors["panel"])
        ax.axis("off")
        
        # Detection Status
        status_color = self.colors["gt"] if result.detected else self.colors["suspect"]
        status_text = "ACQUIRED" if result.detected else "LOST"
        
        ax.text(0.1, 0.85, "TARGET STATUS:", fontsize=10, color=self.colors["text_dim"])
        ax.text(0.1, 0.75, status_text, fontsize=24, color=status_color, fontweight='bold')
        
        # Confidence Bar
        ax.text(0.1, 0.55, "CONFIDENCE:", fontsize=10, color=self.colors["text_dim"])
        rect_bg = Rectangle((0.1, 0.45), 0.8, 0.08, color=self.colors["grid"], alpha=0.5)
        rect_fg = Rectangle((0.1, 0.45), 0.8 * result.confidence, 0.08, color=self.colors["accent"])
        ax.add_patch(rect_bg)
        ax.add_patch(rect_fg)
        ax.text(0.9, 0.45, f"{int(result.confidence*100)}%", fontsize=10, ha='right', va='bottom')
        
        # Sensor Count
        ax.text(0.1, 0.25, f"ACTIVE CAMS:   {result.num_camera_detections}", fontsize=11)
        ax.text(0.1, 0.15, f"ACTIVE LIDAR:  {result.num_lidar_detections}", fontsize=11)
        
        ax.set_title("SYSTEM TELEMETRY", fontsize=12, fontweight='bold')

    def _draw_empty_panel(self, ax, message):
        ax.set_facecolor(self.colors["panel"])
        ax.text(0.5, 0.5, message, ha='center', va='center', color=self.colors["suspect"], fontweight='bold')
        ax.axis("off")

    def _draw_fov(self, ax, x, y, angle, fov, dist):
        # Draw a wedge representing FOV
        from matplotlib.patches import Wedge
        wedge = Wedge((x, y), dist, angle - fov/2, angle + fov/2, 
                      color=self.colors["cam"], alpha=0.1, zorder=1)
        ax.add_patch(wedge)

    def _log_text(self, result: FusionResult):
        # Already implemented, keeping for fallback
        pos_str = f"({result.position_world[0]:.2f}, {result.position_world[1]:.2f})" if result.position_world is not None else "LOST"
        print(f"[STEP {result.step:04d}] Status: {'ACQUIRED' if result.detected else 'SEARCHING'} | Est: {pos_str} | Err: {result.error_meters:.3f}m")

    def close(self):
        """Clean up visualization resources."""
        if self.display and HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
