#!/usr/bin/env python3
"""Move test 2: run the test1 route with actual lift-joint IK."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    from isaacgym import gymapi  # type: ignore
    from isaacgym import gymtorch  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on local Isaac Gym install
    raise SystemExit("Isaac Gym Python package is not importable. Activate the gym environment first.") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("PyTorch is required for move_test2.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
ASSET_ROOT = MOVE_ROOT / "assets"
ROBOT_ASSET_FILE = "integrated/tianyi_xhand_move.urdf"
sys.path.insert(0, str(REPO_ROOT))

from move_pre.mobile_ik import MoveLiftIK, target_box_center_z  # noqa: E402
from move_pre.planning import (  # noqa: E402
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
from move_pre.tasks.move_test1 import (  # noqa: E402
    BOX_MASS,
    PRESET_BOX_MASS,
    _load_hold_targets,
    _make_transform,
    _set_color,
    _set_shape_friction,
    create_sim,
    load_robot_asset,
)


def _box_world_pose(stack_center: tuple[float, float, float], placement: BoxPlacement) -> tuple[float, float, float]:
    local_center = placement.center_local
    return (
        stack_center[0] - 0.5 + local_center[0],
        stack_center[1] - 0.5 + local_center[1],
        stack_center[2] + local_center[2],
    )


def _set_actor_root_pose(gym, sim, root_states: torch.Tensor, actor_index: int, pose: Pose) -> None:
    root_state = root_states[actor_index]
    root_state[0] = pose.x
    root_state[1] = pose.y
    root_state[2] = 0.0
    half = pose.yaw * 0.5
    root_state[3] = 0.0
    root_state[4] = 0.0
    root_state[5] = math.sin(half)
    root_state[6] = math.cos(half)
    root_state[7:13] = 0.0
    actor_indices = torch.tensor([actor_index], dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(actor_indices),
        int(actor_indices.numel()),
    )


def _blend_targets(hold_targets: torch.Tensor, lift_targets: torch.Tensor, alpha: float) -> torch.Tensor:
    return hold_targets * (1.0 - alpha) + lift_targets * alpha


def _lift_alpha(label: str) -> float:
    if "target" in label or "retreat" in label or "tmp" in label:
        return 1.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test1 route with lift-joint IK instead of root_z lifting.")
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs forever.")
    args = parser.parse_args()

    scene_plan = build_move_scene_plan()
    stack = scene_plan.stack_scene
    target_support_z = stack.pallet_surface_z + stack.next_box.min_corner[2]
    desired_box_center_z = target_box_center_z(target_support_z, BOX_SIZE[2])

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
    target_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], fixed_opts)
    box_opts = gymapi.AssetOptions()
    box_opts.density = BOX_MASS / (BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2])
    source_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], box_opts)

    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(6.5, 6.0, 2.0), 1)
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), "robot_move_test2", 0, 1)
    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    source = gym.create_actor(env, source_asset, _make_transform(*BOX_POSE), "source_box", 0, 0)
    pallet = gym.create_actor(
        env,
        pallet_asset,
        _make_transform(stack.pallet_center[0], stack.pallet_center[1], stack.pallet_surface_z - PALLET_THICKNESS * 0.5),
        "pallet",
        0,
        0,
    )
    preset_actors = []
    for placement in stack.preset_boxes:
        sx, sy, sz = placement.size
        opts = gymapi.AssetOptions()
        opts.density = PRESET_BOX_MASS / (sx * sy * sz)
        asset = gym.create_box(sim, sx, sy, sz, opts)
        preset_actors.append(gym.create_actor(env, asset, _make_transform(*_box_world_pose(stack.pallet_center, placement)), placement.name, 0, 0))
    target = gym.create_actor(
        env,
        target_asset,
        _make_transform(*_box_world_pose(stack.pallet_center, stack.next_box)),
        "target_ghost",
        0,
        0,
    )

    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, source, (0.9, 0.05, 0.02))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    for actor in preset_actors:
        _set_color(gym, env, actor, (0.15, 0.25, 0.85))
    _set_color(gym, env, target, (0.95, 0.75, 0.10))
    for actor in (robot, table, source, pallet, *preset_actors):
        _set_shape_friction(gym, env, actor, 2.0)

    dof_names = gym.get_asset_dof_names(robot_asset)
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        props["stiffness"][i] = 120.0 if name.startswith("xhand_") else 360.0
        props["damping"][i] = 12.0 if name.startswith("xhand_") else 36.0
        props["effort"][i] = max(float(props["effort"][i]), 20.0 if name.startswith("xhand_") else 90.0)
    gym.set_actor_dof_properties(env, robot, props)
    lower = torch.tensor(props["lower"], dtype=torch.float32)
    upper = torch.tensor(props["upper"], dtype=torch.float32)
    hold_targets = _load_hold_targets(list(dof_names), lower, upper)

    lift_ik = MoveLiftIK(ASSET_ROOT / ROBOT_ASSET_FILE, list(dof_names), lower, upper)
    lift_targets = lift_ik.solve_for_box_center_z(hold_targets, desired_box_center_z)
    print("Move test 2: test1 route with actual lift-joint IK")
    print(f"  target_support_z={target_support_z:.3f} desired_box_center_z={desired_box_center_z:.3f}")
    print(f"  palm_z hold={lift_ik.palm_z(hold_targets):.3f} solved={lift_ik.palm_z(lift_targets):.3f}")
    for name in ("first_leg_pitch_joint", "second_leg_pitch_joint", "waist_pitch_joint"):
        idx = list(dof_names).index(name)
        print(f"  {name}: hold={float(hold_targets[idx]):.3f} solved={float(lift_targets[idx]):.3f}")

    dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    dof_state["pos"] = hold_targets.numpy()
    dof_state["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot, hold_targets.numpy())

    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)

    frames = []
    for route in scene_plan.routes:
        for start, goal in zip(route.waypoints, route.waypoints[1:]):
            for pose in sample_root_trajectory(start.pose, goal.pose):
                frames.append((route.side, goal.label, pose))
        frames.extend((route.side, route.waypoints[-1].label, route.waypoints[-1].pose) for _ in range(30))

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

    frame = 0
    try:
        while viewer is None or not gym.query_viewer_has_closed(viewer):
            side, label, pose = frames[frame % len(frames)]
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(gym, sim, root_states, robot_index, pose)
            targets = _blend_targets(hold_targets, lift_targets, _lift_alpha(label))
            gym.set_actor_dof_position_targets(env, robot, targets.numpy())
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
            if frame % 120 == 0:
                print(f"frame={frame} side={side} target={label} root=({pose.x:.2f},{pose.y:.2f},0.00)")
            frame += 1
            if args.max_frames > 0 and frame >= args.max_frames:
                break
            if args.headless and args.max_frames == 0 and frame >= len(frames):
                break
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
