#!/usr/bin/env python3
"""Isaac Gym move-state demo.

This keeps the source table/box scene, adds a pallet 5 m diagonally from the
table, preloads two stacked cubes, and moves the robot root to the four side
stances while arms/hands hold a locally generated carry posture.
"""

from __future__ import annotations

import math
import sys
import argparse
from pathlib import Path

try:
    from isaacgym import gymapi  # type: ignore
    from isaacgym import gymtorch  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on local Isaac Gym install
    raise SystemExit(
        "Isaac Gym Python package is not importable. Activate the Isaac Gym environment first."
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("PyTorch is required for move_test1.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
ASSET_ROOT = MOVE_ROOT / "assets"
ROBOT_ASSET_FILE = "integrated/tianyi_xhand_move.urdf"
IK_TRAJECTORY_FILE = MOVE_ROOT / "logs" / "last_ik_joint_trajectory.pt"

sys.path.insert(0, str(REPO_ROOT))
from move.planning import (  # noqa: E402
    BOX_POSE,
    BOX_SIZE,
    PALLET_THICKNESS,
    TABLE_POSE,
    TABLE_SIZE,
    BoxPlacement,
    Pose,
    build_move_scene_plan,
    sample_root_trajectory,
)


COMPUTE_DEVICE_ID = 0
GRAPHICS_DEVICE_ID = 0
BOX_MASS = 0.35
PRESET_BOX_MASS = 0.40
MOBILE_DOF_NAMES = {
    "first_leg_pitch_joint",
    "second_leg_pitch_joint",
    "waist_pitch_joint",
    "body_yaw_joint",
}


def _make_transform(x: float, y: float, z: float, yaw: float = 0.0) -> "gymapi.Transform":
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(x), float(y), float(z))
    half = yaw * 0.5
    pose.r = gymapi.Quat(0.0, 0.0, math.sin(half), math.cos(half))
    return pose


def _set_color(gym, env, actor, color) -> None:
    gym.set_rigid_body_color(
        env,
        actor,
        0,
        gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(float(color[0]), float(color[1]), float(color[2])),
    )


def _set_shape_friction(
    gym,
    env,
    actor,
    friction: float,
    rolling_friction: float = 0.02,
    torsion_friction: float = 0.02,
) -> None:
    props = gym.get_actor_rigid_shape_properties(env, actor)
    for prop in props:
        prop.friction = float(friction)
        prop.rolling_friction = float(rolling_friction)
        prop.torsion_friction = float(torsion_friction)
        prop.restitution = 0.0
    gym.set_actor_rigid_shape_properties(env, actor, props)


def create_sim(gym):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False
    return gym.create_sim(
        COMPUTE_DEVICE_ID,
        GRAPHICS_DEVICE_ID,
        gymapi.SIM_PHYSX,
        sim_params,
    )


def load_robot_asset(gym, sim):
    options = gymapi.AssetOptions()
    options.fix_base_link = False
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    options.replace_cylinder_with_capsule = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    options.vhacd_enabled = True
    options.vhacd_params.resolution = 200000
    options.vhacd_params.max_convex_hulls = 32
    options.vhacd_params.max_num_vertices_per_ch = 64
    return gym.load_asset(sim, str(ASSET_ROOT), ROBOT_ASSET_FILE, options)


def _load_hold_targets(dof_names: list[str], lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    if not IK_TRAJECTORY_FILE.exists():
        return _generate_hold_targets(dof_names, lower, upper)

    payload = torch.load(IK_TRAJECTORY_FILE, map_location="cpu")
    saved_names = list(payload.get("dof_names", []))
    targets = payload.get("filtered_targets")
    if targets is None:
        targets = payload["targets"]
    saved_hold = targets.detach().cpu().float()[-1]
    if saved_names:
        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        hold_values = []
        unexpected_missing = []
        for name in dof_names:
            if name in saved_index:
                hold_values.append(saved_hold[saved_index[name]])
            elif name in MOBILE_DOF_NAMES:
                hold_values.append(torch.tensor(0.0, dtype=torch.float32))
            else:
                unexpected_missing.append(name)
        if unexpected_missing:
            raise RuntimeError(f"Recorded move trajectory is missing DOFs: {unexpected_missing}")
        hold = torch.stack(hold_values).float()
    else:
        hold = saved_hold
        if hold.numel() != len(dof_names):
            raise RuntimeError("Recorded move trajectory has no dof_names and a mismatched DOF count")
    return torch.maximum(torch.minimum(hold, upper), lower)


def _generate_hold_targets(dof_names: list[str], lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    from move.grab_test import run as build_grab_plan

    _report, plan = build_grab_plan(place_mode="move")
    saved_names = list(plan["move_dof_names"])
    saved_q = plan["move_lift_target"].detach().cpu().float()
    saved_index = {name: idx for idx, name in enumerate(saved_names)}
    hold_values = []
    unexpected_missing = []
    for name, lo, hi in zip(dof_names, lower, upper):
        if name in saved_index:
            hold_values.append(saved_q[saved_index[name]])
        elif name in MOBILE_DOF_NAMES:
            hold_values.append(torch.tensor(0.0, dtype=torch.float32).clamp(lo, hi))
        else:
            unexpected_missing.append(name)
    if unexpected_missing:
        raise RuntimeError(f"Generated move plan is missing DOFs: {unexpected_missing}")
    hold = torch.stack(hold_values).float()
    return torch.maximum(torch.minimum(hold, upper), lower)


def _set_actor_root_pose(gym, sim, root_states: torch.Tensor, actor_indices: torch.Tensor, actor_index: int, pose: Pose) -> None:
    root_state = root_states[actor_index]
    root_state[0] = pose.x
    root_state[1] = pose.y
    root_state[2] = pose.z
    half = pose.yaw * 0.5
    root_state[3] = 0.0
    root_state[4] = 0.0
    root_state[5] = math.sin(half)
    root_state[6] = math.cos(half)
    root_state[7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(actor_indices),
        int(actor_indices.numel()),
    )


def _box_world_pose(stack_center: tuple[float, float, float], placement: BoxPlacement) -> tuple[float, float, float]:
    local_center = placement.center_local
    return (
        stack_center[0] - 0.5 + local_center[0],
        stack_center[1] - 0.5 + local_center[1],
        stack_center[2] + local_center[2],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Isaac Gym move-state stacking scene.")
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs forever.")
    args = parser.parse_args()

    scene_plan = build_move_scene_plan()
    stack = scene_plan.stack_scene

    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    robot_asset = load_robot_asset(gym, sim)

    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True

    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, stack.pallet_size[0], stack.pallet_size[1], PALLET_THICKNESS, fixed_opts)

    box_opts = gymapi.AssetOptions()
    box_opts.density = BOX_MASS / (BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2])
    source_box_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], box_opts)

    preset_assets = []
    for placement in stack.preset_boxes:
        sx, sy, sz = placement.size
        opts = gymapi.AssetOptions()
        opts.density = PRESET_BOX_MASS / (sx * sy * sz)
        preset_assets.append(gym.create_box(sim, sx, sy, sz, opts))

    target_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], fixed_opts)

    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(6.0, 6.0, 2.0), 1)
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), "tianyi_xhand_move", 0, 1)
    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    source_box = gym.create_actor(env, source_box_asset, _make_transform(*BOX_POSE), "source_box", 0, 0)
    pallet_z = stack.pallet_surface_z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(
        env,
        pallet_asset,
        _make_transform(stack.pallet_center[0], stack.pallet_center[1], pallet_z),
        "pallet_surface",
        0,
        0,
    )

    preset_actors = []
    for asset, placement in zip(preset_assets, stack.preset_boxes):
        preset_actors.append(
            gym.create_actor(
                env,
                asset,
                _make_transform(*_box_world_pose(stack.pallet_center, placement)),
                placement.name,
                0,
                0,
            )
        )

    target = gym.create_actor(
        env,
        target_asset,
        _make_transform(*_box_world_pose(stack.pallet_center, stack.next_box)),
        "box_3_target_ghost",
        0,
        0,
    )

    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, source_box, (0.9, 0.05, 0.02))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    for actor in preset_actors:
        _set_color(gym, env, actor, (0.15, 0.25, 0.85))
    _set_color(gym, env, target, (0.95, 0.75, 0.10))

    _set_shape_friction(gym, env, robot, 2.0)
    _set_shape_friction(gym, env, table, 1.2)
    _set_shape_friction(gym, env, source_box, 3.0)
    _set_shape_friction(gym, env, pallet, 1.4)
    for actor in preset_actors:
        _set_shape_friction(gym, env, actor, 1.8)

    dof_names = gym.get_asset_dof_names(robot_asset)
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        if name.startswith("xhand_"):
            props["stiffness"][i] = 120.0
            props["damping"][i] = 12.0
            props["effort"][i] = max(float(props["effort"][i]), 20.0)
        else:
            props["stiffness"][i] = 360.0
            props["damping"][i] = 36.0
            props["effort"][i] = max(float(props["effort"][i]), 90.0)
    gym.set_actor_dof_properties(env, robot, props)

    lower = torch.tensor(props["lower"], dtype=torch.float32)
    upper = torch.tensor(props["upper"], dtype=torch.float32)
    hold_targets = _load_hold_targets(list(dof_names), lower, upper)
    dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    dof_state["pos"] = hold_targets.numpy()
    dof_state["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot, hold_targets.numpy())

    gym.prepare_sim(sim)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)
    robot_actor_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
    robot_actor_indices = torch.tensor([robot_actor_index], dtype=torch.int32)

    trajectory_frames = []
    for route in scene_plan.routes:
        for start, goal in zip(route.waypoints, route.waypoints[1:]):
            segment = sample_root_trajectory(start.pose, goal.pose)
            trajectory_frames.extend((route.side, goal.label, pose) for pose in segment)
        final_pose = route.waypoints[-1].pose
        trajectory_frames.extend((route.side, route.waypoints[-1].label, final_pose) for _ in range(30))

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(stack.pallet_center[0] + 1.8, stack.pallet_center[1] - 2.0, 1.5),
            gymapi.Vec3(stack.pallet_center[0], stack.pallet_center[1], 0.55),
        )

    print("Move-state scene:")
    print(f"  pallet_center={stack.pallet_center}")
    print(f"  third_box_target_min={stack.next_box.min_corner} size={stack.next_box.size}")
    for stance in scene_plan.stances:
        pose = stance.pose
        tmp = stance.tmp_pose
        face = stance.face_center
        print(f"  stance {stance.side}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} yaw={math.degrees(pose.yaw):.1f}deg")
        print(f"    tmp: x={tmp.x:.3f} y={tmp.y:.3f} z={tmp.z:.3f} yaw={math.degrees(tmp.yaw):.1f}deg")
        print(f"    placement_side_center=({face[0]:.3f},{face[1]:.3f},{face[2]:.3f})")
    print("  route keypoints:")
    for route in scene_plan.routes:
        print(f"    route {route.side}:")
        for waypoint in route.waypoints:
            pose = waypoint.pose
            print(
                f"      {waypoint.label}: x={pose.x:.3f} y={pose.y:.3f} "
                f"z={pose.z:.3f} yaw={math.degrees(pose.yaw):.1f}deg"
            )

    frame = 0
    try:
        while viewer is None or not gym.query_viewer_has_closed(viewer):
            side, waypoint_label, pose = trajectory_frames[frame % len(trajectory_frames)]
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(gym, sim, root_states, robot_actor_indices, robot_actor_index, pose)
            gym.set_actor_dof_position_targets(env, robot, hold_targets.numpy())

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            if frame % 120 == 0:
                print(
                    f"frame={frame} side={side} "
                    f"target={waypoint_label} "
                    f"root=({pose.x:.2f},{pose.y:.2f},{pose.z:.2f}) yaw={math.degrees(pose.yaw):.1f}deg"
                )
            frame += 1
            if args.max_frames > 0 and frame >= args.max_frames:
                break
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
