"""Lightweight writer for dog LiDAR point-cloud overview videos."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from perception.scan_matching import StaticMap2D


class DogPointCloudVideoWriter:
    """Render top-down LiDAR point-cloud videos for both dogs."""

    def __init__(
        self,
        output_path: str,
        map_bounds: dict[str, float],
        static_map: StaticMap2D | None = None,
        fps: float = 12.0,
        panel_size: tuple[int, int] = (720, 540),
    ):
        self.output_path = Path(output_path)
        self.map_bounds = map_bounds
        self.static_points = None if static_map is None else np.asarray(static_map.points_xy, dtype=np.float32)
        self.fps = float(fps)
        self.panel_size = panel_size
        self.writer: cv2.VideoWriter | None = None
        self._est_history: dict[str, list[tuple[float, float]]] = {"go2_dog_1": [], "go2_dog_2": []}
        self._gt_history: dict[str, list[tuple[float, float]]] = {"go2_dog_1": [], "go2_dog_2": []}

    def add_frame(
        self,
        step: int,
        dog_states: dict[str, dict],
        dog_lidar_data: dict[str, dict],
        gt_dog_data: dict[str, dict],
    ):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.panel_size[0] * 2, self.panel_size[1]),
            )

        panels = []
        for dog_name in ("go2_dog_1", "go2_dog_2"):
            lidar_name = "dog1_lidar" if dog_name.endswith("1") else "dog2_lidar"
            panels.append(
                self._render_panel(
                    dog_name=dog_name,
                    step=step,
                    state=dog_states.get(dog_name),
                    lidar=dog_lidar_data.get(lidar_name),
                    gt=gt_dog_data.get(dog_name),
                )
            )

        frame = np.hstack(panels)
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _render_panel(
        self,
        dog_name: str,
        step: int,
        state: dict | None,
        lidar: dict | None,
        gt: dict | None,
    ) -> np.ndarray:
        width, height = self.panel_size
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        self._draw_border(canvas)
        self._draw_static_map(canvas)
        self._draw_lidar_hits(canvas, lidar)

        if gt is not None:
            gt_xy = (float(gt["pos"][0]), float(gt["pos"][1]))
            self._gt_history[dog_name].append(gt_xy)
            self._draw_history(canvas, self._gt_history[dog_name], (40, 160, 40))
            self._draw_pose(canvas, gt_xy, float(gt["yaw"]), (40, 160, 40), "GT")

        if state is not None:
            est_xy = (float(state["pos"][0]), float(state["pos"][1]))
            self._est_history[dog_name].append(est_xy)
            self._draw_history(canvas, self._est_history[dog_name], (30, 60, 220))
            self._draw_pose(canvas, est_xy, float(state["euler"][2]), (30, 60, 220), "EST")

        cv2.putText(canvas, dog_name.upper(), (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 120, 10), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"step {step}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (90, 90, 90), 2, cv2.LINE_AA)

        if state is not None:
            cv2.putText(
                canvas,
                f"score {state['scan_match_score']:.3f} | inliers {state['scan_inliers']} | lidar {'Y' if state['lidar_corrected'] else 'N'}",
                (18, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (50, 50, 50),
                2,
                cv2.LINE_AA,
            )

        if state is not None and gt is not None:
            xy_error = math.hypot(float(state["pos"][0] - gt["pos"][0]), float(state["pos"][1] - gt["pos"][1]))
            cv2.putText(
                canvas,
                f"xy error {xy_error:.3f} m",
                (18, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (50, 50, 50),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def _draw_border(self, canvas: np.ndarray):
        pad = 22
        cv2.rectangle(canvas, (pad, pad), (canvas.shape[1] - pad, canvas.shape[0] - pad), (150, 150, 150), 2)

    def _draw_static_map(self, canvas: np.ndarray):
        if self.static_points is None:
            return
        for point in self.static_points[::2]:
            px, py = self._world_to_pixel(float(point[0]), float(point[1]), canvas.shape[1], canvas.shape[0])
            cv2.circle(canvas, (px, py), 1, (205, 205, 205), -1, lineType=cv2.LINE_AA)

    def _draw_lidar_hits(self, canvas: np.ndarray, lidar: dict | None):
        if lidar is None:
            return
        hit_points = lidar["hit_points"]
        if isinstance(hit_points, torch.Tensor):
            hits = hit_points.detach().cpu().numpy()
        else:
            hits = np.asarray(hit_points)
        if hits.ndim == 3:
            hits = hits[0]
        if hits.size == 0:
            return
        valid = np.isfinite(hits).all(axis=1)
        hits = hits[valid]
        for point in hits:
            px, py = self._world_to_pixel(float(point[0]), float(point[1]), canvas.shape[1], canvas.shape[0])
            cv2.circle(canvas, (px, py), 2, (0, 170, 255), -1, lineType=cv2.LINE_AA)

    def _draw_history(self, canvas: np.ndarray, history: list[tuple[float, float]], color: tuple[int, int, int]):
        if len(history) < 2:
            return
        recent = history[-80:]
        points = [
            self._world_to_pixel(x, y, canvas.shape[1], canvas.shape[0])
            for x, y in recent
        ]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(canvas, start, end, color, 2, lineType=cv2.LINE_AA)

    def _draw_pose(
        self,
        canvas: np.ndarray,
        xy: tuple[float, float],
        yaw: float,
        color: tuple[int, int, int],
        label: str,
    ):
        px, py = self._world_to_pixel(xy[0], xy[1], canvas.shape[1], canvas.shape[0])
        cv2.circle(canvas, (px, py), 7, color, -1, lineType=cv2.LINE_AA)
        arrow_len = 24
        end = (
            int(round(px + arrow_len * math.cos(yaw))),
            int(round(py - arrow_len * math.sin(yaw))),
        )
        cv2.arrowedLine(canvas, (px, py), end, color, 2, tipLength=0.25)
        cv2.putText(canvas, label, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _world_to_pixel(self, x: float, y: float, width: int, height: int) -> tuple[int, int]:
        x_min = self.map_bounds["x_min"]
        x_max = self.map_bounds["x_max"]
        y_min = self.map_bounds["y_min"]
        y_max = self.map_bounds["y_max"]
        pad = 28
        px = pad + (x - x_min) / max(x_max - x_min, 1.0e-6) * (width - 2 * pad)
        py = pad + (y_max - y) / max(y_max - y_min, 1.0e-6) * (height - 2 * pad)
        return int(round(px)), int(round(py))
