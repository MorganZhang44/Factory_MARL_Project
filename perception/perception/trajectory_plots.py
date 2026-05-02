"""Trajectory plotting helpers for localization outputs."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import numpy as np


def plot_dog_trajectory_comparison(
    dog_log_path: str,
    save_path: str,
    scene_size: float = 20.0,
):
    """Plot predicted vs ground-truth dog trajectories from the dog localization log."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping dog trajectory plot.")
        return

    if not os.path.exists(dog_log_path):
        print(f"  [WARN] Dog localization log not found: {dog_log_path}")
        return

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(dog_log_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_rows[row["dog"]].append(row)

    if not grouped_rows:
        print("  [WARN] No dog trajectory data to plot.")
        return

    dog_names = sorted(grouped_rows.keys())
    num_rows = len(dog_names)
    fig, axes = plt.subplots(num_rows, 2, figsize=(16, 6 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])

    half = scene_size / 2.0
    for row_idx, dog_name in enumerate(dog_names):
        rows = grouped_rows[dog_name]
        steps = [int(row["step"]) for row in rows]
        est_x = [float(row["est_x"]) for row in rows]
        est_y = [float(row["est_y"]) for row in rows]
        gt_x = [float(row["gt_x"]) for row in rows]
        gt_y = [float(row["gt_y"]) for row in rows]
        xy_error = [float(row["xy_error_m"]) for row in rows]
        yaw_error = [float(row["yaw_error_deg"]) for row in rows]
        lidar_corrected_idx = [idx for idx, row in enumerate(rows) if int(row["lidar_corrected"]) == 1]

        ax_traj = axes[row_idx, 0]
        ax_err = axes[row_idx, 1]

        ax_traj.set_facecolor("#f8fafc")
        ax_traj.plot(gt_x, gt_y, "-", color="#16a34a", linewidth=2.5, label="Ground Truth")
        ax_traj.plot(est_x, est_y, "--", color="#dc2626", linewidth=2.2, label="Estimated")
        ax_traj.scatter(gt_x[0], gt_y[0], color="#15803d", s=90, marker="o", label="GT Start", zorder=5)
        ax_traj.scatter(gt_x[-1], gt_y[-1], color="#15803d", s=90, marker="s", label="GT End", zorder=5)
        ax_traj.scatter(est_x[0], est_y[0], color="#b91c1c", s=75, marker="o", label="Est Start", zorder=5)
        ax_traj.scatter(est_x[-1], est_y[-1], color="#b91c1c", s=75, marker="s", label="Est End", zorder=5)

        if lidar_corrected_idx:
            ax_traj.scatter(
                [est_x[idx] for idx in lidar_corrected_idx],
                [est_y[idx] for idx in lidar_corrected_idx],
                color="#2563eb",
                s=45,
                marker="D",
                label="LiDAR Corrected",
                zorder=6,
            )

        for idx in range(0, len(rows), max(1, len(rows) // 12 or 1)):
            ax_traj.plot(
                [gt_x[idx], est_x[idx]],
                [gt_y[idx], est_y[idx]],
                color="#f59e0b",
                linewidth=0.8,
                alpha=0.45,
            )

        rect = plt.Rectangle(
            (-half, -half),
            scene_size,
            scene_size,
            linewidth=2,
            edgecolor="#94a3b8",
            facecolor="none",
        )
        ax_traj.add_patch(rect)
        ax_traj.set_xlim(-half - 1.0, half + 1.0)
        ax_traj.set_ylim(-half - 1.0, half + 1.0)
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.25)
        ax_traj.set_title(
            f"{dog_name}: Estimated vs Ground Truth Trajectory\n"
            f"Mean XY Error = {np.mean(xy_error):.3f} m",
            fontsize=12,
            fontweight="bold",
        )
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.legend(loc="upper left", fontsize=9, framealpha=0.95)

        ax_err.set_facecolor("#f8fafc")
        ax_err.plot(steps, xy_error, color="#dc2626", linewidth=2.0, label="XY Error (m)")
        ax_err.fill_between(steps, 0.0, xy_error, color="#fecaca", alpha=0.45)
        ax_err.axhline(np.mean(xy_error), color="#991b1b", linestyle="--", linewidth=1.2)
        ax_err.set_xlabel("Step")
        ax_err.set_ylabel("XY Error (m)", color="#991b1b")
        ax_err.tick_params(axis="y", labelcolor="#991b1b")
        ax_err.grid(True, alpha=0.25)

        ax_yaw = ax_err.twinx()
        ax_yaw.plot(steps, yaw_error, color="#2563eb", linewidth=1.8, label="Yaw Error (deg)")
        ax_yaw.set_ylabel("Yaw Error (deg)", color="#1d4ed8")
        ax_yaw.tick_params(axis="y", labelcolor="#1d4ed8")

        if lidar_corrected_idx:
            lidar_steps = [steps[idx] for idx in lidar_corrected_idx]
            lidar_xy = [xy_error[idx] for idx in lidar_corrected_idx]
            ax_err.scatter(lidar_steps, lidar_xy, color="#2563eb", s=28, zorder=5)

        ax_err.set_title(
            f"{dog_name}: Localization Error Over Time\n"
            f"Mean Yaw Error = {np.mean(yaw_error):.2f} deg",
            fontsize=12,
            fontweight="bold",
        )

        handles_left, labels_left = ax_err.get_legend_handles_labels()
        handles_right, labels_right = ax_yaw.get_legend_handles_labels()
        ax_err.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=9)

    fig.suptitle("Dog Self-Localization Trajectory Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [INFO] Dog trajectory comparison plot saved to: {save_path}", flush=True)

