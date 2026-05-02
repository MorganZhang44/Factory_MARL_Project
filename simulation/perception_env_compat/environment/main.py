# Copyright (c) 2026, Multi-Agent Surveillance Project
# Main entry point: Isaac Lab standalone script for multi-agent surveillance
# with CV-based suspect localization.
#
# Usage:
#   conda activate env_isaaclab
#   python main.py --num_steps 500
#   python main.py --headless --num_steps 1000

from __future__ import annotations
print("[DEBUG] main.py started.", flush=True)

import argparse
import os
import sys
import time
import math
import torch
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Isaac Lab / Sim launch — MUST happen before any other Isaac imports
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Surveillance: CV-based Suspect Localization in Isaac Lab"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps.")
    parser.add_argument("--step_dt", type=float, default=1.0 / 60.0, help="Simulation timestep (s).")
    parser.add_argument("--no_move", action="store_true", help="Disable entity movement.")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization (faster).")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--pose_server", action="store_true", help="Publish live dog/intruder poses over HTTP.")
    parser.add_argument("--pose_host", type=str, default="127.0.0.1", help="Pose HTTP server host.")
    parser.add_argument("--pose_port", type=int, default=8765, help="Pose HTTP server port.")
    parser.add_argument(
        "--pose_print_interval",
        type=int,
        default=0,
        help="Print live dog/intruder pose API snapshots every N steps (0 disables).",
    )
    return parser.parse_args()


args = parse_args()

# Launch Isaac Sim App
print("[DEBUG] Initializing AppLauncher...", flush=True)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app
print("[DEBUG] App launched successfully.", flush=True)

# ---------------------------------------------------------------------------
# Now safe to import Isaac Lab modules and project modules
# ---------------------------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

# Add project root to path. This file lives in environment/, while perception/
# remains a sibling package at the repository root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("[DEBUG] Importing scene_cfg...", flush=True)
from environment.config.scene_cfg import SurveillanceSceneCfg
print("[DEBUG] Importing sensor_cfg...", flush=True)
from environment.config.sensor_cfg import get_intrinsics_from_camera_cfg, SURVEILLANCE_CAM_CFG, DOG_CAM_CFG
print("[DEBUG] Importing scene_manager...", flush=True)
from environment.scene.surveillance_scene import SurveillanceSceneManager
print("[DEBUG] Importing detectors...", flush=True)
from perception.camera_detector import CameraDetector
from perception.lidar_detector import LidarDetector
print("[DEBUG] Importing localizer...", flush=True)
from perception.dog_localizer import DogLocalizer
from perception.scan_matching import build_static_map_from_scene_cfg
print("[DEBUG] Importing fusion...", flush=True)
from perception.fusion import SensorFusion, FusionResult
print("[DEBUG] Importing visualizer...", flush=True)
from perception.visualization import SurveillanceVisualizer
from perception.trajectory_plots import plot_dog_trajectory_comparison
from perception.dog_pointcloud_video import DogPointCloudVideoWriter
from environment.localization_mesh import create_static_localization_mesh
from perception.pose_server import PoseHttpServer, PoseStateStore
from environment.static_scene_geometry import (
    MAP_BOUNDS,
    SCENE_SIZE,
    get_camera_markers,
    get_surveillance_camera_names,
    get_static_object_positions,
    get_visualization_obstacles,
)

SURVEILLANCE_CAMERA_NAMES = get_surveillance_camera_names()


def setup_scene() -> tuple[InteractiveScene, SimulationContext]:
    """Create the simulation context and build the scene.

    Returns:
        (scene, sim_context) tuple.
    """
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=args.step_dt,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    sim = SimulationContext(sim_cfg)

    # Set main camera view (for the GUI viewport)
    center_x = MAP_BOUNDS["center_x"]
    center_y = MAP_BOUNDS["center_y"]
    span = max(MAP_BOUNDS["width"], MAP_BOUNDS["height"])
    sim.set_camera_view(
        eye=[center_x, center_y - span * 1.4, span * 0.95],
        target=[center_x, center_y, 0.0],
    )

    mesh_info = create_static_localization_mesh()
    print(
        f"[DEBUG] Localization mesh ready: {mesh_info['mesh_path']} "
        f"({mesh_info['num_vertices']} verts, {mesh_info['num_triangles']} tris)",
        flush=True,
    )

    # Create scene
    scene_cfg = SurveillanceSceneCfg()
    scene = InteractiveScene(scene_cfg)

    return scene, sim


def get_entity(scene: InteractiveScene, name: str):
    """Helper to get an entity by name from any collection."""
    if name in scene.articulations.keys():
        return scene.articulations[name]
    if name in scene.sensors.keys():
        return scene.sensors[name]
    if hasattr(scene, "rigid_objects") and name in scene.rigid_objects.keys():
        return scene.rigid_objects[name]
    return None


def get_scene_keys(scene: InteractiveScene) -> list[str]:
    """Helper to get all entity keys from the scene."""
    keys = list(scene.articulations.keys())
    keys += list(scene.sensors.keys())
    if hasattr(scene, "rigid_objects"):
        keys += list(scene.rigid_objects.keys())
    return keys


def setup_perception(scene: InteractiveScene) -> tuple[CameraDetector, LidarDetector, SensorFusion]:
    """Set up the perception pipeline.

    Args:
        scene: The interactive scene.

    Returns:
        (camera_detector, lidar_detector, sensor_fusion) tuple.
    """
    # Build camera intrinsics dict
    # Surveillance cams use SURVEILLANCE_CAM_CFG params
    surv_intrinsics = get_intrinsics_from_camera_cfg(SURVEILLANCE_CAM_CFG)
    dog_intrinsics = get_intrinsics_from_camera_cfg(DOG_CAM_CFG)

    camera_intrinsics = {}
    camera_conventions = {}
    for cam_name in [k for k in get_scene_keys(scene) if "cam" in k.lower()]:
        is_dog = "dog" in cam_name
        camera_intrinsics[cam_name] = dog_intrinsics if is_dog else surv_intrinsics
        camera_conventions[cam_name] = "world"

    camera_detector = CameraDetector(
        camera_intrinsics=camera_intrinsics,
        camera_conventions=camera_conventions,
        suspect_class_name="suspect",
        min_pixel_threshold=10,
        max_depth_threshold=30.0,
    )

    static_positions = get_static_object_positions()

    lidar_detector = LidarDetector(
        height_min=0.3,
        height_max=2.0,
        cluster_distance=1.5,
        min_cluster_points=3,
        static_object_positions=static_positions,
        static_object_radius=1.0,
    )

    sensor_fusion = SensorFusion(
        camera_weight=1.1,
        lidar_weight=0.55,
        temporal_alpha=0.7,
        outlier_threshold=3.0,
        history_size=5000,
        camera_gate_distance=1.8,
        lidar_gate_distance=2.8,
    )

    return camera_detector, lidar_detector, sensor_fusion


def collect_dog_imu_data(scene: InteractiveScene, sim_time: float, motion_hints: dict | None = None) -> dict:
    """Collect IMU data from both dogs.

    Returns:
        {dog_name: {"ang_vel_b": tensor, "lin_acc_b": tensor, "projected_gravity_b": tensor, "timestamp": float}}
    """
    from isaaclab.utils.math import quat_apply_inverse

    dog_names = ["go2_dog_1", "go2_dog_2"]
    imu_data = {}
    
    for dog_name in dog_names:
        try:
            # Try to get IMU data. In our scene_cfg, we configured it.
            # Names might be different in the scene dictionary.
            imu_name = f"{dog_name}_imu"
            if imu_name in get_scene_keys(scene):
                imu = get_entity(scene, imu_name)
                imu_data[dog_name] = {
                    "ang_vel_b": imu.data.ang_vel_b[0],
                    "lin_acc_b": imu.data.lin_acc_b[0],
                    "projected_gravity_b": imu.data.projected_gravity_b[0],
                    "timestamp": sim_time,
                }
            else:
                # Fallback only to keep the loop alive if the IMU sensor handle is missing.
                dog = get_entity(scene, dog_name)
                if dog is None:
                    continue
                base_quat_w = dog.data.root_quat_w[0]
                body_lin_acc_w = dog.data.body_lin_acc_w[0, 0]
                body_ang_vel_w = dog.data.body_ang_vel_w[0, 0]
                gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=base_quat_w.device)
                imu_data[dog_name] = {
                    "ang_vel_b": quat_apply_inverse(base_quat_w.unsqueeze(0), body_ang_vel_w.unsqueeze(0)).squeeze(0),
                    "lin_acc_b": quat_apply_inverse(base_quat_w.unsqueeze(0), body_lin_acc_w.unsqueeze(0)).squeeze(0),
                    "projected_gravity_b": quat_apply_inverse(base_quat_w.unsqueeze(0), gravity_w.unsqueeze(0)).squeeze(0),
                    "timestamp": sim_time,
                }
            if motion_hints and dog_name in motion_hints and dog_name in imu_data:
                hint = motion_hints[dog_name]
                device = imu_data[dog_name]["ang_vel_b"].device
                imu_data[dog_name]["odom_vel_w"] = torch.tensor(
                    hint.get("linear_velocity", (0.0, 0.0, 0.0)),
                    dtype=torch.float32,
                    device=device,
                )
                imu_data[dog_name]["odom_ang_vel_w"] = torch.tensor(
                    hint.get("angular_velocity", (0.0, 0.0, 0.0)),
                    dtype=torch.float32,
                    device=device,
                )
        except (KeyError, AttributeError):
            continue
    return imu_data


def collect_camera_data(scene: InteractiveScene) -> tuple[dict, dict]:
    """Collect camera data (images + poses) from all cameras in the scene.

    Returns:
        (camera_data, camera_poses) where:
        - camera_data: {cam_name: {"semantic_segmentation": tensor, "depth": tensor, "info": dict}}
        - camera_poses: {cam_name: (pos_tensor, quat_tensor)}
    """
    camera_names = [k for k in get_scene_keys(scene) if "cam" in k.lower()]
    camera_data = {}
    camera_poses = {}

    for cam_name in camera_names:
        try:
            cam = scene[cam_name]
            data = cam.data

            if data.output is None:
                continue

            # Get image data from output dict
            semantic = data.output.get("semantic_segmentation")
            depth = data.output.get("depth")
            rgb = data.output.get("rgb")

            if semantic is None:
                continue

            # CCTV check: if it's a fixed CCTV, we ignore depth to simulate monocular
            is_cctv = "cam_" in cam_name
            if is_cctv:
                depth = None

            # Squeeze batch dimension if present
            if semantic.dim() == 4:
                semantic = semantic[0, :, :, 0]
            elif semantic.dim() == 3:
                semantic = semantic[:, :, 0] if semantic.shape[-1] == 1 else semantic[0]

            if depth is not None:
                if depth.dim() == 4:
                    depth = depth[0, :, :, 0]
                elif depth.dim() == 3:
                    depth = depth[:, :, 0] if depth.shape[-1] == 1 else depth[0]

            if rgb is not None:
                if rgb.dim() == 4:
                    rgb = rgb[0, :, :, :3]
                elif rgb.dim() == 3:
                    rgb = rgb[:, :, :3]
            
            # Get semantic info
            info = {}
            if data.info is not None and len(data.info) > 0:
                env_info = data.info[0] if isinstance(data.info, list) else data.info
                if isinstance(env_info, dict):
                    info = env_info.get("semantic_segmentation", {})

            camera_data[cam_name] = {
                "semantic_segmentation": semantic,
                "depth": depth,
                "rgb": rgb,
                "info": info,
                "intrinsic_matrix": data.intrinsic_matrices[0] if data.intrinsic_matrices is not None else None,
            }

            pos = data.pos_w[0] if data.pos_w is not None else torch.zeros(3, device=semantic.device)
            quat = data.quat_w_ros[0] if data.quat_w_ros is not None else torch.tensor([1.0, 0.0, 0.0, 0.0], device=semantic.device)

            camera_poses[cam_name] = (pos, quat)

        except (KeyError, AttributeError, RuntimeError) as e:
            print(f"  [WARN] Could not read camera '{cam_name}': {e}")
            continue

    return camera_data, camera_poses


def collect_lidar_data(scene: InteractiveScene) -> dict:
    """Collect LiDAR data from both dogs.

    Returns:
        {lidar_name: {"hit_points": tensor, "pos_w": tensor, "quat_w": tensor}}
    """
    lidar_names = ["dog1_lidar", "dog2_lidar"]
    lidar_data = {}

    for lidar_name in lidar_names:
        try:
            lidar = scene[lidar_name]
            data = lidar.data

            if data.ray_hits_w is None:
                continue

            lidar_data[lidar_name] = {
                "hit_points": data.ray_hits_w,   # (B, N, 3) world-frame hit positions
                "pos_w": data.pos_w,              # (B, 3) sensor position
                "quat_w": data.quat_w,            # (B, 4) sensor orientation
            }
        except (KeyError, AttributeError, RuntimeError) as e:
            print(f"  [WARN] Could not read LiDAR '{lidar_name}': {e}")
            continue

    return lidar_data


def collect_camera_rgb(scene: InteractiveScene) -> dict[str, np.ndarray]:
    """Collect RGB images from all cameras (surveillance + mobile) for visualization.

    Returns:
        {cam_name: np.ndarray (H, W, 3) uint8}
    """
    camera_names = [k for k in get_scene_keys(scene) if "cam" in k.lower()]
    images = {}

    for cam_name in camera_names:
        try:
            cam = get_entity(scene, cam_name)
            rgb = cam.data.output.get("rgb")
            if rgb is not None:
                if rgb.dim() == 4:
                    rgb = rgb[0]  # (B, H, W, C) -> (H, W, C)
                # Ensure uint8 and remove alpha
                rgb_np = rgb[:, :, :3].cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255.0).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                images[cam_name] = rgb_np
        except (KeyError, AttributeError):
            continue

    return images


def angular_error_deg(est_yaw: float, gt_yaw: float) -> float:
    """Smallest absolute yaw error in degrees."""
    delta = math.atan2(math.sin(est_yaw - gt_yaw), math.cos(est_yaw - gt_yaw))
    return abs(math.degrees(delta))


def _json_scalar(value) -> float | int | bool | None:
    """Convert tensor/numpy/Python scalar to a JSON-safe scalar."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return value


def _json_vector(value) -> list[float] | None:
    """Convert tensors, numpy arrays, tuples, or lists to a JSON-safe float list."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            scalar = _json_scalar(item)
            result.append(float(scalar) if scalar is not None else None)
        return result
    scalar = _json_scalar(value)
    return [float(scalar)] if scalar is not None else None


def _euler_payload(euler_rad) -> dict[str, list[float] | None]:
    values = _json_vector(euler_rad)
    degrees = [math.degrees(v) if v is not None else None for v in values] if values is not None else None
    return {"rad": values, "deg": degrees}


def _heading_pose_from_velocity(velocity) -> tuple[list[float] | None, dict[str, list[float] | None] | None, bool]:
    """Infer yaw-only orientation from planar velocity for target estimates."""
    vel = _json_vector(velocity)
    if vel is None or len(vel) < 2 or vel[0] is None or vel[1] is None:
        return None, None, False
    speed_xy = math.hypot(float(vel[0]), float(vel[1]))
    if speed_xy < 1.0e-3:
        return None, None, False
    yaw = math.atan2(float(vel[1]), float(vel[0]))
    quat = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
    return quat, _euler_payload([0.0, 0.0, yaw]), True


def _dog_pose_payload(
    dog_name: str,
    step: int,
    sim_time: float,
    est_state: dict,
    gt_pos,
    gt_quat,
    gt_vel,
    gt_euler_rad,
    xy_error: float,
    yaw_error_deg: float,
) -> dict:
    """Build the public JSON payload for one robot dog."""
    return {
        "name": dog_name,
        "step": int(step),
        "time": float(sim_time),
        "estimate": {
            "pos": _json_vector(est_state.get("pos")),
            "vel": _json_vector(est_state.get("vel")),
            "quat": _json_vector(est_state.get("quat")),
            "euler": _euler_payload(est_state.get("euler")),
            "speed": _json_scalar(est_state.get("speed")),
            "localized": bool(est_state.get("localized", False)),
            "lidar_corrected": bool(est_state.get("lidar_corrected", False)),
            "only_imu_prediction": bool(est_state.get("only_imu_prediction", False)),
            "scan_match_score": _json_scalar(est_state.get("scan_match_score")),
            "scan_inliers": _json_scalar(est_state.get("scan_inliers")),
        },
        "ground_truth": {
            "pos": _json_vector(gt_pos),
            "vel": _json_vector(gt_vel),
            "quat": _json_vector(gt_quat),
            "euler": _euler_payload(gt_euler_rad),
            "speed": _json_scalar(torch.norm(gt_vel).item()),
        },
        "errors": {
            "xy_m": float(xy_error),
            "yaw_deg": float(yaw_error_deg),
        },
    }


def _intruder_pose_payload(
    step: int,
    sim_time: float,
    fusion_result: FusionResult,
    suspect_asset,
) -> dict:
    """Build the public JSON payload for the intruder/suspect."""
    from isaaclab.utils.math import euler_xyz_from_quat

    gt_pos = suspect_asset.data.root_pos_w[0]
    gt_quat = suspect_asset.data.root_quat_w[0]
    gt_vel = suspect_asset.data.root_lin_vel_w[0]
    gt_ang_vel = suspect_asset.data.root_ang_vel_w[0]
    gt_roll, gt_pitch, gt_yaw = euler_xyz_from_quat(gt_quat.unsqueeze(0))
    gt_euler = [gt_roll[0].item(), gt_pitch[0].item(), gt_yaw[0].item()]
    est_vel = fusion_result.velocity_world
    est_quat, est_euler, heading_valid = _heading_pose_from_velocity(est_vel)

    return {
        "name": "intruder",
        "step": int(step),
        "time": float(sim_time),
        "estimate": {
            "pos": _json_vector(fusion_result.position_world),
            "vel": _json_vector(est_vel),
            "quat": est_quat,
            "euler": est_euler,
            "speed": _json_scalar(torch.norm(est_vel).item()) if est_vel is not None else None,
            "heading_from_velocity": heading_valid,
            "detected": bool(fusion_result.detected),
            "confidence": float(fusion_result.confidence),
            "num_camera_detections": int(fusion_result.num_camera_detections),
            "num_lidar_detections": int(fusion_result.num_lidar_detections),
        },
        "ground_truth": {
            "pos": _json_vector(gt_pos),
            "vel": _json_vector(gt_vel),
            "angular_vel": _json_vector(gt_ang_vel),
            "quat": _json_vector(gt_quat),
            "euler": _euler_payload(gt_euler),
            "speed": _json_scalar(torch.norm(gt_vel).item()),
        },
        "errors": {
            "xy_m": _json_scalar(fusion_result.error_meters),
        },
    }


# ===========================================================================
# Main simulation loop
# ===========================================================================

def main():
    print("=" * 70)
    print("  Multi-Agent Surveillance: CV-Based Suspect Localization")
    print("=" * 70)
    move_entities = not args.no_move
    save_viz = not args.no_viz

    print(f"  Steps: {args.num_steps} | dt: {args.step_dt:.4f}s | Headless: {args.headless}")
    print(f"  Movement: {move_entities} | Save viz: {save_viz}")
    print(f"  Pose server: {'on' if args.pose_server else 'off'}")
    print("=" * 70, flush=True)

    # Setup
    print("\n[1/4] Setting up scene...", flush=True)
    scene, sim = setup_scene()

    print("[2/4] Setting up perception pipeline...", flush=True)
    camera_detector, lidar_detector, sensor_fusion = setup_perception(scene)

    print("[3/4] Initializing scene manager...", flush=True)
    # Reset the simulator so articulated assets and sensors are populated.
    # The InteractiveScene instance we constructed in setup_scene() remains the
    # authoritative scene handle in current Isaac Lab versions.
    sim.reset()
    if hasattr(scene, "reset"):
        scene.reset()
    print(f"  [DEBUG] Scene attributes: {dir(scene)}")
    scene_manager = SurveillanceSceneManager(scene, sim_dt=args.step_dt)

    pose_store = PoseStateStore()
    pose_server: PoseHttpServer | None = None
    if args.pose_server:
        pose_server = PoseHttpServer(pose_store, host=args.pose_host, port=args.pose_port)
        pose_server.start()
        print(f"  [INFO] Pose HTTP server running at {pose_server.url}", flush=True)
        print("  [INFO] Endpoints: /health, /poses, /poses/dogs, /poses/intruder", flush=True)

    if move_entities:
        # Align the simulated world with the patrol-route start state before
        # initializing the onboard localization filters.
        scene_manager.step_movement(0)
        sim.step()
        scene.update(args.step_dt)
        if not args.headless:
            simulation_app.update()

    print("[3.5/4] Initializing dog localizers...", flush=True)
    sensor_interval = max(1, int(round((1.0 / 12.0) / args.step_dt)))
    dog_lidar_interval = max(1, int(round(0.066 / args.step_dt)))
    static_map = build_static_map_from_scene_cfg()
    print(
        f"  [DEBUG] Detection interval={sensor_interval} steps | "
        f"Dog LiDAR interval={dog_lidar_interval} steps | "
        f"Static map points={static_map.metadata['num_points']}",
        flush=True,
    )
    dog_localizers = {
            name: DogLocalizer(
                name,
                args.step_dt,
                initial_pose=(get_entity(scene, name).data.root_pos_w[0], get_entity(scene, name).data.root_quat_w[0]),
                lidar_mount_height=0.35,
                static_map=static_map,
                lidar_update_steps=dog_lidar_interval,
                imu_gravity_gain=0.08,
                zupt_speed_threshold=0.05,
                max_planar_speed_mps=1.60,
                min_stable_match_score=0.62,
                max_lidar_speed_mps=1.80,
            )
        for name in ["go2_dog_1", "go2_dog_2"]
    }

    print("[4/4] Initializing visualizer...", flush=True)
    viz = SurveillanceVisualizer(
        output_dir=os.path.join(args.output_dir, "viz"),
        scene_size=SCENE_SIZE,
        save_frames=save_viz,
        display=not args.headless,
        map_bounds=MAP_BOUNDS,
        camera_markers=get_camera_markers(),
    )

    static_viz_objects = get_visualization_obstacles()

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, "localization_log.csv"), "w")
    log_file.write("step,time,detected,est_x,est_y,est_z,gt_x,gt_y,gt_z,error_m,confidence,n_cams,n_lidars\n")

    debug_intruder_file = None
    debug_steps_env = os.environ.get("DEBUG_INTRUDER_STEPS", "")
    debug_step_range: tuple[int, int] | None = None
    if debug_steps_env:
        try:
            start_s, end_s = debug_steps_env.split(":", 1)
            debug_step_range = (int(start_s), int(end_s))
            debug_intruder_file = open(os.path.join(args.output_dir, "intruder_detection_debug.csv"), "w")
            debug_intruder_file.write(
                "step,time,sensor,type,detected,x,y,z,gt_x,gt_y,gt_z,error_m,confidence,pixels_or_points,extra\n"
            )
        except ValueError:
            print(f"  [WARN] Ignoring malformed DEBUG_INTRUDER_STEPS={debug_steps_env!r}")

    dog_log_file = open(os.path.join(args.output_dir, "dog_localization_log.csv"), "w")
    dog_log_file.write(
        "step,time,dog,localized,lidar_corrected,scan_score,scan_inliers,only_imu,"
        "est_x,est_y,est_z,gt_x,gt_y,gt_z,xy_error_m,"
        "est_roll,est_pitch,est_yaw,gt_roll,gt_pitch,gt_yaw,yaw_error_deg,"
        "est_vx,est_vy,est_vz,gt_vx,gt_vy,gt_vz,speed_est,speed_gt\n"
    )
    dog_pointcloud_writer = DogPointCloudVideoWriter(
        output_path=os.path.join(args.output_dir, "dog_pointcloud_video.avi"),
        map_bounds=MAP_BOUNDS,
        static_map=static_map,
        fps=max(1.0, round(1.0 / (dog_lidar_interval * args.step_dt), 1)),
    )

    # -----------------------------------------------------------------------
    # Warm-up: run a few steps to let cameras and rendering pipeline initialize
    # -----------------------------------------------------------------------
    print("\n--- Warming up cameras & rendering... ", end="", flush=True)
    for _ in range(20):
        sim.step()
        scene.update(args.step_dt)
        if not args.headless:
            simulation_app.update()
    print("done.", flush=True)

    # -----------------------------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------------------------
    # Main simulation loop
    print("\n--- Starting simulation loop ---", flush=True)
    start_time = time.time()
    latest_dog_states: dict[str, dict] = {}
    
    for step in range(args.num_steps):
        if step == 0:
            print(f"  [DEBUG] Scene keys: {get_scene_keys(scene)}")
        # Check if user closed the window
        if not simulation_app.is_running():
            print("\n  [INFO] Window closed by user. Stopping.", flush=True)
            break

        sim_time = step * args.step_dt
        pose_store.update(step=step, sim_time=sim_time)
        printed_pose_snapshot = False

        # Move entities
        if move_entities and step > 0:
            scene_manager.step_movement(step)

        # Step simulation
        sim.step()
        scene.update(args.step_dt)

        # Pump the Omniverse event loop (critical for GUI to stay responsive)
        # simulation_app.update() handles rendering + UI events + timeline
        if not args.headless:
            simulation_app.update()

        # --- Dog Localization Monitoring (Run every step for accuracy) ---
        from isaaclab.utils.math import euler_xyz_from_quat
        dog_data = collect_dog_imu_data(scene, sim_time, scene_manager.get_dog_motion_hints())
        dog_lidar_data = collect_lidar_data(scene) if step % dog_lidar_interval == 0 else {}
        
        if step == 0 and not dog_data:
            print("  [WARN] No dog IMU data collected! Check sensor names.")
        for dog_name, dog_imu in dog_data.items():
            try:
                localizer = dog_localizers[dog_name]
                lidar_name = "dog1_lidar" if "1" in dog_name else "dog2_lidar"
                dog_lidar = dog_lidar_data.get(lidar_name)
                
                est_state = localizer.update(dog_imu, dog_lidar)
                latest_dog_states[dog_name] = est_state
                
                # Ground truth for comparison
                dog_asset = get_entity(scene, dog_name)
                gt_dog_pos = dog_asset.data.root_pos_w[0]
                gt_dog_quat = dog_asset.data.root_quat_w[0]
                gt_dog_vel_w = dog_asset.data.root_lin_vel_w[0]
                gt_dog_speed = torch.norm(gt_dog_vel_w).item()
                
                gt_roll, gt_pitch, gt_yaw = euler_xyz_from_quat(gt_dog_quat.unsqueeze(0))
                est_roll, est_pitch, est_yaw = est_state["euler"]
                xy_error = torch.norm(est_state["pos"][:2] - gt_dog_pos[:2]).item()
                yaw_error = angular_error_deg(est_yaw, gt_yaw[0].item())
                pose_store.update_dog(
                    dog_name,
                    _dog_pose_payload(
                        dog_name=dog_name,
                        step=step,
                        sim_time=sim_time,
                        est_state=est_state,
                        gt_pos=gt_dog_pos,
                        gt_quat=gt_dog_quat,
                        gt_vel=gt_dog_vel_w,
                        gt_euler_rad=[gt_roll[0].item(), gt_pitch[0].item(), gt_yaw[0].item()],
                        xy_error=xy_error,
                        yaw_error_deg=yaw_error,
                    ),
                )
                
                dog_log_file.write(
                    f"{step},{sim_time:.4f},{dog_name},"
                    f"{int(est_state['localized'])},{int(est_state['lidar_corrected'])},"
                    f"{est_state['scan_match_score']:.4f},{est_state['scan_inliers']},{int(est_state['only_imu_prediction'])},"
                    f"{est_state['pos'][0]:.3f},{est_state['pos'][1]:.3f},{est_state['pos'][2]:.3f},"
                    f"{gt_dog_pos[0]:.3f},{gt_dog_pos[1]:.3f},{gt_dog_pos[2]:.3f},"
                    f"{xy_error:.3f},"
                    f"{est_roll:.3f},{est_pitch:.3f},{est_yaw:.3f},"
                    f"{gt_roll[0].item():.3f},{gt_pitch[0].item():.3f},{gt_yaw[0].item():.3f},"
                    f"{yaw_error:.3f},"
                    f"{est_state['vel'][0]:.3f},{est_state['vel'][1]:.3f},{est_state['vel'][2]:.3f},"
                    f"{gt_dog_vel_w[0]:.3f},{gt_dog_vel_w[1]:.3f},{gt_dog_vel_w[2]:.3f},"
                    f"{est_state['speed']:.3f},{gt_dog_speed:.3f}\n"
                )
                if step % 10 == 0:
                    dog_log_file.flush()
            except Exception as e:
                if step % 100 == 0:
                    print(f"  [WARN] Dog localization failed: {e}")

        if dog_lidar_data:
            gt_dog_data = {}
            for dog_name in ["go2_dog_1", "go2_dog_2"]:
                dog_asset = get_entity(scene, dog_name)
                if dog_asset is None:
                    continue
                gt_roll, gt_pitch, gt_yaw = euler_xyz_from_quat(dog_asset.data.root_quat_w[0].unsqueeze(0))
                gt_dog_data[dog_name] = {
                    "pos": dog_asset.data.root_pos_w[0].detach().cpu(),
                    "yaw": gt_yaw[0].item(),
                }
                if dog_name not in latest_dog_states:
                    latest_dog_states[dog_name] = dog_localizers[dog_name].get_estimate()
            dog_pointcloud_writer.add_frame(step, latest_dog_states, dog_lidar_data, gt_dog_data)

        if step % 50 == 0 and latest_dog_states:
            for dog_name in ["go2_dog_1", "go2_dog_2"]:
                state = latest_dog_states.get(dog_name)
                if not state:
                    continue
                print(
                    f"    [DOG] {dog_name}: "
                    f"xy=({state['pos'][0]:.2f}, {state['pos'][1]:.2f}) "
                    f"yaw={math.degrees(state['euler'][2]):.1f}deg "
                    f"v={state['speed']:.2f}m/s "
                    f"lidar={'Y' if state['lidar_corrected'] else 'N'} "
                    f"score={state['scan_match_score']:.3f} "
                    f"inliers={state['scan_inliers']}",
                    flush=True,
                )

        # Collect sensor data periodically
        if step % sensor_interval == 0:
            # Camera data
            camera_data, camera_poses = collect_camera_data(scene)
            cam_detections = camera_detector.detect_all_cameras(camera_data, camera_poses)

            # CV2 Recording RGB feeds to MP4
            if True:
                frames = []
                feed_order = [*SURVEILLANCE_CAMERA_NAMES, "dog1_cam", "dog2_cam"]
                cam_to_det = {det.camera_name: det for det in cam_detections}
                rendered_frames = {}
                prototype_frame = None

                for c in feed_order:
                    if c in camera_data and camera_data[c]["rgb"] is not None:
                        # Convert RGBA/RGB tensor to numpy BGR for OpenCV
                        im_rgb = camera_data[c]["rgb"].cpu().numpy()
                        if im_rgb.dtype != np.uint8:
                            im_rgb = (im_rgb * 255.0).astype(np.uint8) if im_rgb.max() <= 1.0 else im_rgb.astype(np.uint8)
                        
                        im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Draw bounding box if detected
                        if c in cam_to_det:
                            det = cam_to_det[c]
                            if det.detected and det.bbox_2d:
                                x1, y1, x2, y2 = det.bbox_2d
                                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(
                                    im,
                                    f"DET: {int(det.confidence*100)}%",
                                    (x1, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    (0, 0, 255),
                                    2,
                                )
                        
                        # Add text label
                        cv2.putText(im, c.upper(), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        rendered_frames[c] = im
                        if prototype_frame is None:
                            prototype_frame = im

                if prototype_frame is not None:
                    for c in feed_order:
                        if c not in rendered_frames:
                            placeholder = np.zeros_like(prototype_frame)
                            cv2.putText(placeholder, c.upper(), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            rendered_frames[c] = placeholder
                        frames.append(rendered_frames[c])

                    tile_size = (480, 360)
                    resized_frames = []
                    for frame in frames:
                        interpolation = cv2.INTER_AREA if frame.shape[1] >= tile_size[0] else cv2.INTER_LINEAR
                        resized_frames.append(cv2.resize(frame, tile_size, interpolation=interpolation))
                    frames = resized_frames
                    cols = 3 if len(frames) > 4 else 2 if len(frames) > 1 else 1
                    rows = math.ceil(len(frames) / cols)
                    while len(frames) < rows * cols:
                        frames.append(np.zeros_like(frames[0]))
                    row_images = []
                    for row_idx in range(rows):
                        start = row_idx * cols
                        row_images.append(np.hstack(frames[start:start + cols]))
                    monitor_wall = np.vstack(row_images)
                    
                    if not hasattr(simulation_app, "_video_writer"):
                        # MJPG is extremely compatible for AVI
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        simulation_app._video_writer = cv2.VideoWriter(
                            'output/surveillance_video.avi',
                            fourcc,
                            12.0,
                            (tile_size[0] * cols, tile_size[1] * rows),
                        )
                        print("[INFO] Started recording surveillance video to output/surveillance_video.avi")
                    simulation_app._video_writer.write(monitor_wall)

            # Monitor all active detections
            for det in cam_detections:
                if det.detected and "dog" in det.camera_name:
                    # Log dog detections as they are more 'exclusive'
                    gt_pos = scene_manager.get_suspect_ground_truth()
                    err = torch.norm(det.position_world - gt_pos).item()
                    if step % 20 == 0:
                        print(f"    [INFO] Mobile Det ({det.camera_name}): err={err:.3f}m", flush=True)
            
            # Debug: print camera info on first collection or every 50 steps
            if step == 0 or (step % 50 == 0 and step <= 500):
                print(f"\n  [DEBUG] Cameras with data: {list(camera_data.keys())}", flush=True)
                if step == 0:
                    for cname, cdata in camera_data.items():
                        seg = cdata["semantic_segmentation"]
                        dep = cdata["depth"]
                        dep_shape = dep.shape if dep is not None else "N/A"
                        info = cdata.get("info", {})
                        unique_ids = torch.unique(seg).tolist() if seg is not None else []
                        id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
                        print(f"    {cname}: seg shape={seg.shape}, depth shape={dep_shape}, "
                              f"unique_ids={unique_ids[:10]}, labels={id_to_labels}", flush=True)
                gt_pos = scene_manager.get_suspect_ground_truth()
                for det in cam_detections:
                    if det.detected:
                        cam_p, cam_q = camera_poses[det.camera_name]
                        print(f"    {det.camera_name}: detected pos={det.position_world.cpu().tolist()}, "
                              f"GT={gt_pos.cpu().tolist()}, pixels={det.num_pixels}, depth={det.mean_depth:.2f}m, "
                              f"cam_pos={cam_p.cpu().tolist()}, cam_quat={cam_q.cpu().tolist()}", flush=True)
                    else:
                        print(f"    {det.camera_name}: not detected (pixels={det.num_pixels})", flush=True)

            # LiDAR data
            lidar_data = dog_lidar_data if dog_lidar_data else collect_lidar_data(scene)
            lidar_detections = lidar_detector.detect_all_lidars(lidar_data)

            # Ground truth
            gt_pos = scene_manager.get_suspect_ground_truth()

            if debug_intruder_file is not None and debug_step_range is not None:
                debug_start, debug_end = debug_step_range
                if debug_start <= step <= debug_end:
                    for det in cam_detections:
                        if det.detected and det.position_world is not None:
                            err = torch.norm(det.position_world[:2] - gt_pos[:2]).item()
                            pos = det.position_world
                        else:
                            err = float("inf")
                            pos = torch.tensor([float("nan"), float("nan"), float("nan")], device=gt_pos.device)
                        extra = (
                            f"bbox={det.bbox_2d};occ={getattr(det, 'occlusion_score', 0.0):.3f};"
                            f"vis={getattr(det, 'visibility_score', 0.0):.3f};depth={int(getattr(det, 'used_depth', False))}"
                        )
                        debug_intruder_file.write(
                            f"{step},{sim_time:.4f},{det.camera_name},camera,{int(det.detected)},"
                            f"{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},"
                            f"{gt_pos[0]:.3f},{gt_pos[1]:.3f},{gt_pos[2]:.3f},"
                            f"{err:.4f},{det.confidence:.4f},{det.num_pixels},{extra}\n"
                        )
                    for det in lidar_detections:
                        if det.detected and det.position_world is not None:
                            err = torch.norm(det.position_world[:2] - gt_pos[:2]).item()
                            pos = det.position_world
                        else:
                            err = float("inf")
                            pos = torch.tensor([float("nan"), float("nan"), float("nan")], device=gt_pos.device)
                        extra = f"cluster_std={getattr(det, 'cluster_std', 0.0):.3f}"
                        debug_intruder_file.write(
                            f"{step},{sim_time:.4f},{det.lidar_name},lidar,{int(det.detected)},"
                            f"{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},"
                            f"{gt_pos[0]:.3f},{gt_pos[1]:.3f},{gt_pos[2]:.3f},"
                            f"{err:.4f},{det.confidence:.4f},{det.num_points},{extra}\n"
                        )

            # Fusion
            fusion_result = sensor_fusion.fuse(
                camera_detections=cam_detections,
                lidar_detections=lidar_detections,
                ground_truth=gt_pos,
                timestamp=sim_time,
                step=step,
            )
            suspect_asset = get_entity(scene, "suspect")
            if suspect_asset is not None:
                pose_store.update_intruder(_intruder_pose_payload(step, sim_time, fusion_result, suspect_asset))

            if args.pose_print_interval > 0 and step % args.pose_print_interval == 0 and not printed_pose_snapshot:
                snapshot = pose_store.snapshot()
                print(
                    "[POSE_API] "
                    f"step={snapshot.get('step')} "
                    f"dogs={snapshot.get('dogs')} "
                    f"intruder={snapshot.get('intruder')}",
                    flush=True,
                )
                printed_pose_snapshot = True

            # Track trajectories
            scene_manager.record_ground_truth()
            scene_manager.record_estimation(fusion_result.position_world)

            # Log
            est = fusion_result.position_world
            est_str = f"{est[0]:.3f},{est[1]:.3f},{est[2]:.3f}" if est is not None else "N/A,N/A,N/A"
            gt_str = f"{gt_pos[0]:.3f},{gt_pos[1]:.3f},{gt_pos[2]:.3f}"
            log_file.write(
                f"{step},{sim_time:.4f},{fusion_result.detected},"
                f"{est_str},{gt_str},"
                f"{fusion_result.error_meters:.4f},{fusion_result.confidence:.4f},"
                f"{fusion_result.num_camera_detections},{fusion_result.num_lidar_detections}\n"
            )

            # Print progress
            if step % 50 == 0:
                dog_positions = scene_manager.get_dog_positions()
                print(
                    f"  Step {step:4d}/{args.num_steps} | "
                    f"t={sim_time:6.2f}s | "
                    f"Detected={fusion_result.detected} | "
                    f"Err={fusion_result.error_meters:6.3f}m | "
                    f"Conf={fusion_result.confidence:.2f} | "
                    f"Cams={fusion_result.num_camera_detections} "
                    f"LiDARs={fusion_result.num_lidar_detections}",
                    flush=True,
                )


            # Visualization (every 20 steps to reduce overhead)
            if step % 20 == 0 and save_viz:
                dog_positions = scene_manager.get_dog_positions()
                camera_rgb = collect_camera_rgb(scene)
                viz.render_frame(
                    fusion_result=fusion_result,
                    camera_images=camera_rgb,
                    dog_positions=dog_positions,
                    static_objects=static_viz_objects,
                )

    # -----------------------------------------------------------------------
    # Cleanup and Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    
    if hasattr(simulation_app, "_video_writer"):
        simulation_app._video_writer.release()
        print("[INFO] Simulation video closed and saved successfully.")
    
    log_file.flush()
    log_file.close()
    if debug_intruder_file is not None:
        debug_intruder_file.flush()
        debug_intruder_file.close()
    dog_log_file.flush()
    dog_log_file.close()
    dog_pointcloud_writer.close()

    # Summary statistics
    history = sensor_fusion.get_history()
    errors = [r.error_meters for r in history if r.detected and r.error_meters != float("inf")]
    detected_count = sum(1 for r in history if r.detected)

    print("\n" + "=" * 70)
    print("  Simulation Complete")
    print("=" * 70)
    print(f"  Total steps:       {args.num_steps}")
    print(f"  Elapsed time:      {elapsed:.1f}s ({args.num_steps / elapsed:.1f} steps/s)")
    print(f"  Detection rate:    {detected_count}/{len(history)} ({100 * detected_count / max(1, len(history)):.1f}%)")
    if errors:
        print(f"  Mean error:        {np.mean(errors):.3f} m")
        print(f"  Median error:      {np.median(errors):.3f} m")
        print(f"  Max error:         {np.max(errors):.3f} m")
        print(f"  Min error:         {np.min(errors):.3f} m")
    print(f"  Log saved to:      {os.path.join(args.output_dir, 'localization_log.csv')}")
    if save_viz:
        print(f"  Viz frames saved:  {os.path.join(args.output_dir, 'viz/')}")
    
    # -----------------------------------------------------------------------
    # Generate trajectory comparison plot
    # -----------------------------------------------------------------------
    trajectories = scene_manager.get_trajectories()
    traj_path = os.path.join(args.output_dir, "trajectory_comparison.png")
    _plot_trajectory_comparison(trajectories, traj_path, scene_size=20.0)
    print(f"  Trajectory plot:   {traj_path}")

    dog_traj_path = os.path.join(args.output_dir, "dog_trajectory_comparison.png")
    plot_dog_trajectory_comparison(
        dog_log_path=os.path.join(args.output_dir, "dog_localization_log.csv"),
        save_path=dog_traj_path,
        scene_size=20.0,
    )
    print(f"  Dog traj plot:    {dog_traj_path}")
    print(f"  Dog pointcloud:   {os.path.join(args.output_dir, 'dog_pointcloud_video.avi')}")
    print("=" * 70, flush=True)

    # Shutdown
    viz.close()
    if pose_server is not None:
        pose_server.stop()
    simulation_app.close()


def _plot_trajectory_comparison(trajectories: dict, save_path: str, scene_size: float = 20.0):
    """Plot ground truth vs estimated trajectory and save to file.

    Args:
        trajectories: dict with 'ground_truth' and 'estimated' lists.
        save_path: Path to save the plot image.
        scene_size: Scene size for axis limits.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("  [WARN] matplotlib not available, skipping trajectory plot.")
        return

    gt = trajectories["ground_truth"]
    est = trajectories["estimated"]

    if not gt:
        print("  [WARN] No trajectory data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ---- Panel 1: Top-down trajectory map ----
    ax1 = axes[0]
    half = scene_size / 2

    # Ground truth trajectory (green)
    gt_x = [p[0] for p in gt]
    gt_y = [p[1] for p in gt]
    ax1.plot(gt_x, gt_y, '-', color='limegreen', linewidth=2.5, label='Ground Truth', zorder=3)
    ax1.plot(gt_x[0], gt_y[0], 'o', color='green', markersize=12, label='Start (GT)', zorder=5)
    ax1.plot(gt_x[-1], gt_y[-1], 's', color='green', markersize=12, label='End (GT)', zorder=5)

    # Estimated trajectory (red, only valid points)
    est_x = [p[0] for p in est if p[0] is not None]
    est_y = [p[1] for p in est if p[1] is not None]
    if est_x:
        ax1.plot(est_x, est_y, '--', color='red', linewidth=2.0, alpha=0.8, label='CV Estimated', zorder=4)
        ax1.scatter(est_x, est_y, c='red', s=15, alpha=0.5, zorder=4)

    # Draw error lines between matched GT and estimated points
    for i, (g, e) in enumerate(zip(gt, est)):
        if e[0] is not None and i % 5 == 0:  # every 5th point
            ax1.plot([g[0], e[0]], [g[1], e[1]], '-', color='orange', linewidth=0.5, alpha=0.4)

    # Scene boundary
    rect = plt.Rectangle((-half, -half), scene_size, scene_size,
                          linewidth=2, edgecolor='gray', facecolor='#1a1a2e', alpha=0.3)
    ax1.add_patch(rect)

    # Camera positions
    cam_positions = [
        (9.0, 9.0, "NE"), (-9.0, 9.0, "NW"),
        (9.0, -9.0, "SE"), (-9.0, -9.0, "SW"),
    ]
    for cx, cy, label in cam_positions:
        ax1.plot(cx, cy, 'D', color='#ff6b6b', markersize=10, zorder=6)
        ax1.annotate(f'Cam {label}', (cx, cy), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color='#ff6b6b', fontweight='bold')

    ax1.set_xlim(-half - 1, half + 1)
    ax1.set_ylim(-half - 1, half + 1)
    ax1.set_aspect('equal')
    ax1.set_title('Trajectory Comparison: Ground Truth vs CV Estimation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)', fontsize=11)
    ax1.set_ylabel('Y (meters)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Error over time ----
    ax2 = axes[1]
    # Compute per-point errors
    point_errors = []
    for g, e in zip(gt, est):
        if e[0] is not None:
            dx = g[0] - e[0]
            dy = g[1] - e[1]
            point_errors.append(math.sqrt(dx*dx + dy*dy))
        else:
            point_errors.append(float('nan'))

    valid_idx = [i for i, e in enumerate(point_errors) if not math.isnan(e)]
    valid_errors = [point_errors[i] for i in valid_idx]

    if valid_errors:
        ax2.plot(valid_idx, valid_errors, '-', color='crimson', linewidth=1.5, alpha=0.8)
        ax2.fill_between(valid_idx, 0, valid_errors, color='crimson', alpha=0.15)
        mean_err = np.mean(valid_errors)
        ax2.axhline(y=mean_err, color='orange', linestyle='--', linewidth=1.5,
                     label=f'Mean: {mean_err:.2f}m')
        ax2.legend(fontsize=10, framealpha=0.9)

    ax2.set_title('Localization Error Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Error (meters)', fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [INFO] Trajectory comparison plot saved to: {save_path}", flush=True)



print("[DEBUG] Calling main()...", flush=True)
if __name__ == "__main__":
    main()
