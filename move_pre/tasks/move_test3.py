#!/usr/bin/env python3
"""Move test 3: top-of-second-box EMS scene with actual lift-joint IK.

This test keeps arms/hands locked at q_hold, keeps actor root_z at 0, and solves
the URDF lift joints so the hand/held-box height matches the placement target.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from isaacgym import gymapi  # type: ignore
    from isaacgym import gymtorch  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on local Isaac Gym install
    raise SystemExit("Isaac Gym Python package is not importable. Activate the gym environment first.") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("PyTorch is required for move_test3.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move_pre.planning import (  # noqa: E402
    BOX_POSE,
    BOX_SIZE,
    PALLET_SIZE,
    PALLET_SURFACE_Z,
    PALLET_THICKNESS,
    TABLE_POSE,
    TABLE_SIZE,
    BoxPlacement,
    EmptySpace,
    Pose,
    StackScene,
    compute_pct_like_ems,
    pallet_center_near_table,
    sample_root_trajectory,
)
from move_pre.mobile_ik import MoveLiftIK, target_box_center_z  # noqa: E402
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


MOVE_TEST2_STAND_OFF = 0.72
MOVE_TEST2_TMP_RETREAT = 0.80
MOVE_TEST2_SAFE_MARGIN = 1.20
TEST_TITLE = "Move test 3: top-of-second-box lift-joint IK scene"
TEST_DESCRIPTION = "Run the top-of-second-box height/lift IK scene."


@dataclass(frozen=True)
class HeightTestScene:
    name: str
    description: str
    stack_scene: StackScene
    target_support_z: float
    final_pose: Pose
    tmp_pose: Pose
    waypoints: tuple[tuple[str, Pose], ...]
    show_target_box: bool = True


def _target_world_center(stack: StackScene) -> tuple[float, float, float]:
    cx, cy, cz = stack.pallet_center
    local = stack.next_box.center_local
    return (
        cx - stack.pallet_size[0] * 0.5 + local[0],
        cy - stack.pallet_size[1] * 0.5 + local[1],
        cz + local[2],
    )


def _box_world_pose(stack: StackScene, placement: BoxPlacement) -> tuple[float, float, float]:
    local = placement.center_local
    return (
        stack.pallet_center[0] - stack.pallet_size[0] * 0.5 + local[0],
        stack.pallet_center[1] - stack.pallet_size[1] * 0.5 + local[1],
        stack.pallet_center[2] + local[2],
    )


def solve_root_pose_for_target(stack: StackScene, target_support_z: float) -> tuple[Pose, Pose]:
    """Plan root x/y/yaw only; height is solved by lift-joint IK."""

    target_x, target_y, _ = _target_world_center(stack)
    final_x = stack.pallet_center[0] - stack.pallet_size[0] * 0.5 - MOVE_TEST2_STAND_OFF
    final_y = target_y
    yaw = math.atan2(target_y - final_y, target_x - final_x)
    final_pose = Pose(final_x, final_y, 0.0, yaw)
    tmp_pose = Pose(final_x - MOVE_TEST2_TMP_RETREAT, final_y, 0.0, yaw)
    return final_pose, tmp_pose


def _table_retreat_pose(start: Pose) -> Pose:
    dx = start.x - TABLE_POSE[0]
    dy = start.y - TABLE_POSE[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy, length = -1.0, 0.0, 1.0
    return Pose(start.x + dx / length * 0.90, start.y + dy / length * 0.90, start.z, start.yaw)


def build_waypoints(stack: StackScene, final_pose: Pose, tmp_pose: Pose) -> tuple[tuple[str, Pose], ...]:
    start = Pose(0.0, 0.0, 0.0, 0.0)
    table_retreat = _table_retreat_pose(start)
    safe_x = stack.pallet_center[0] - stack.pallet_size[0] * 0.5 - MOVE_TEST2_SAFE_MARGIN
    safe_y = stack.pallet_center[1] - stack.pallet_size[1] * 0.5 - MOVE_TEST2_SAFE_MARGIN
    safe_1 = Pose(table_retreat.x, safe_y, 0.0, 0.0)
    safe_2 = Pose(safe_x, safe_y, 0.0, 0.0)
    tmp_xy = Pose(tmp_pose.x, safe_y, 0.0, math.atan2(tmp_pose.y - safe_y, tmp_pose.x - safe_x))
    tmp_low = Pose(tmp_pose.x, tmp_pose.y, 0.0, tmp_pose.yaw)
    return (
        ("table_start", start),
        ("table_retreat", table_retreat),
        ("safe_axis_1", safe_1),
        ("safe_corner", safe_2),
        ("tmp_axis", tmp_xy),
        ("tmp_low", tmp_low),
        ("target", final_pose),
    )


def _make_stack_scene(
    preset_boxes: tuple[BoxPlacement, ...],
    target_box: BoxPlacement,
) -> StackScene:
    spaces = compute_pct_like_ems(preset_boxes)
    fits_ems = any(
        all(target_box.min_corner[i] >= space.min_corner[i] for i in range(3))
        and all(target_box.max_corner[i] <= space.max_corner[i] for i in range(3))
        for space in spaces
    )
    if not fits_ems:
        raise ValueError(f"Target {target_box.name} is not contained in any EMS leaf")
    return StackScene(
        pallet_center=pallet_center_near_table(),
        pallet_size=PALLET_SIZE,
        pallet_surface_z=PALLET_SURFACE_Z,
        preset_boxes=preset_boxes,
        next_box=target_box,
        empty_spaces=spaces,
        selected_leaf=EmptySpace(target_box.min_corner, target_box.max_corner),
    )


def build_height_test_scenes() -> tuple[HeightTestScene, ...]:
    scene_1_stack = _make_stack_scene(
        (
            BoxPlacement("scene1_box_1_0p20", (0.0, 0.0, 0.0), (0.20, 0.20, 0.20)),
            BoxPlacement("scene1_box_2_0p20", (0.20, 0.0, 0.0), (0.20, 0.20, 0.20)),
        ),
        BoxPlacement("scene1_box_3_target_top_of_0p20", (0.20, 0.0, 0.20), BOX_SIZE),
    )
    scene_1_support_z = PALLET_SURFACE_Z + 0.20
    scene_1_final, scene_1_tmp = solve_root_pose_for_target(scene_1_stack, scene_1_support_z)

    return (
        HeightTestScene(
            "scene1_top_of_0p20",
            "third 0.2m cube on the leaf above the second placed 0.2m cube",
            scene_1_stack,
            scene_1_support_z,
            scene_1_final,
            scene_1_tmp,
            build_waypoints(scene_1_stack, scene_1_final, scene_1_tmp),
        ),
    )


def _set_actor_root_pose(
    gym,
    sim,
    root_states: torch.Tensor,
    actor_index: int,
    pose: Pose,
) -> None:
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


def _create_scene_actors(gym, sim, env, scene: HeightTestScene, robot_asset):
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), f"robot_{scene.name}", 0, 1)

    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, scene.stack_scene.pallet_size[0], scene.stack_scene.pallet_size[1], PALLET_THICKNESS, fixed_opts)
    target_marker_thickness = 0.01
    target_asset = gym.create_box(
        sim,
        BOX_SIZE[0],
        BOX_SIZE[1],
        BOX_SIZE[2] if scene.show_target_box else target_marker_thickness,
        fixed_opts,
    )

    box_opts = gymapi.AssetOptions()
    box_opts.density = BOX_MASS / (BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2])
    source_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], box_opts)

    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), f"table_{scene.name}", 0, 0)
    source = gym.create_actor(env, source_asset, _make_transform(*BOX_POSE), f"source_box_{scene.name}", 0, 0)
    pallet_z = scene.stack_scene.pallet_surface_z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(
        env,
        pallet_asset,
        _make_transform(scene.stack_scene.pallet_center[0], scene.stack_scene.pallet_center[1], pallet_z),
        f"pallet_{scene.name}",
        0,
        0,
    )

    preset_actors = []
    for placement in scene.stack_scene.preset_boxes:
        sx, sy, sz = placement.size
        opts = gymapi.AssetOptions()
        opts.density = PRESET_BOX_MASS / (sx * sy * sz)
        asset = gym.create_box(sim, sx, sy, sz, opts)
        preset_actors.append(
            gym.create_actor(env, asset, _make_transform(*_box_world_pose(scene.stack_scene, placement)), placement.name, 0, 0)
        )

    if scene.show_target_box:
        target_pose = _box_world_pose(scene.stack_scene, scene.stack_scene.next_box)
    else:
        center = scene.stack_scene.next_box.center_local
        target_pose = (
            scene.stack_scene.pallet_center[0] - scene.stack_scene.pallet_size[0] * 0.5 + center[0],
            scene.stack_scene.pallet_center[1] - scene.stack_scene.pallet_size[1] * 0.5 + center[1],
            scene.target_support_z + target_marker_thickness * 0.5,
        )
    target = gym.create_actor(
        env,
        target_asset,
        _make_transform(*target_pose),
        f"target_marker_{scene.name}",
        0,
        0,
    )

    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, source, (0.9, 0.05, 0.02))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    for actor in preset_actors:
        _set_color(gym, env, actor, (0.15, 0.25, 0.85))
    _set_color(gym, env, target, (0.95, 0.75, 0.10))

    _set_shape_friction(gym, env, robot, 2.0)
    _set_shape_friction(gym, env, table, 1.2)
    _set_shape_friction(gym, env, source, 3.0)
    _set_shape_friction(gym, env, pallet, 1.4)
    for actor in preset_actors:
        _set_shape_friction(gym, env, actor, 1.8)
    return robot


def main() -> None:
    parser = argparse.ArgumentParser(description=TEST_DESCRIPTION)
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs forever.")
    args = parser.parse_args()

    scenes = build_height_test_scenes()
    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    robot_asset = load_robot_asset(gym, sim)

    envs = []
    robots = []
    for scene in scenes:
        env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.5, 6.5, 2.5), len(scenes))
        envs.append(env)
        robots.append(_create_scene_actors(gym, sim, env, scene, robot_asset))

    dof_names = gym.get_asset_dof_names(robot_asset)
    robot_hold_targets = []
    robot_lift_targets = []
    lift_summaries = []
    for env, robot in zip(envs, robots):
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
        robot_hold_targets.append(hold_targets)
        dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
        dof_state["pos"] = hold_targets.numpy()
        dof_state["vel"].fill(0.0)
        gym.set_actor_dof_states(env, robot, dof_state, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, robot, hold_targets.numpy())

    for scene, hold_targets in zip(scenes, robot_hold_targets):
        lift_ik = MoveLiftIK(MOVE_ROOT / "assets" / "integrated" / "tianyi_xhand_move.urdf", list(dof_names), lower, upper)
        desired_z = target_box_center_z(scene.target_support_z, BOX_SIZE[2])
        lift_targets = lift_ik.solve_for_box_center_z(hold_targets, desired_z)
        robot_lift_targets.append(lift_targets)
        lift_summaries.append((desired_z, lift_ik.palm_z(hold_targets), lift_ik.palm_z(lift_targets), lift_targets))

    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_indices = [gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM) for env, robot in zip(envs, robots)]

    trajectories = []
    for scene in scenes:
        frames = []
        for (start_label, start_pose), (goal_label, goal_pose) in zip(scene.waypoints, scene.waypoints[1:]):
            for pose in sample_root_trajectory(start_pose, goal_pose):
                frames.append((goal_label, pose))
        frames.extend((scene.waypoints[-1][0], scene.waypoints[-1][1]) for _ in range(60))
        trajectories.append(frames)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        first_stack = scenes[0].stack_scene
        gym.viewer_camera_look_at(
            viewer,
            envs[0],
            gymapi.Vec3(first_stack.pallet_center[0] + 2.2, first_stack.pallet_center[1] - 2.2, 1.8),
            gymapi.Vec3(first_stack.pallet_center[0], first_stack.pallet_center[1], 0.55),
        )

    print(TEST_TITLE)
    for scene in scenes:
        print(f"  {scene.name}: {scene.description}")
        print(f"    placed_boxes={len(scene.stack_scene.preset_boxes)} target_visual={'box' if scene.show_target_box else 'thin_footprint_marker'}")
        for placement in scene.stack_scene.preset_boxes:
            print(f"      placed {placement.name}: min={placement.min_corner} size={placement.size}")
        print(f"    target_min={scene.stack_scene.next_box.min_corner} size={scene.stack_scene.next_box.size}")
        print(f"    target_support_z={scene.target_support_z:.3f}")
        desired_z, hold_z, solved_z, lift_targets = lift_summaries[scenes.index(scene)]
        print(f"    desired_box_center_z={desired_z:.3f} palm_z_hold={hold_z:.3f} palm_z_solved={solved_z:.3f}")
        print(f"    planned_final_root=({scene.final_pose.x:.3f},{scene.final_pose.y:.3f},0.000) yaw={math.degrees(scene.final_pose.yaw):.1f}deg")
        for joint_name in ("first_leg_pitch_joint", "second_leg_pitch_joint", "waist_pitch_joint"):
            idx = list(dof_names).index(joint_name)
            print(f"    {joint_name}={float(lift_targets[idx]):.3f}")
        for label, pose in scene.waypoints:
            print(f"      {label}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} yaw={math.degrees(pose.yaw):.1f}deg")

    frame = 0
    try:
        while viewer is None or not gym.query_viewer_has_closed(viewer):
            gym.refresh_actor_root_state_tensor(sim)
            for actor_index, trajectory in zip(robot_indices, trajectories):
                _, pose = trajectory[min(frame, len(trajectory) - 1)]
                _set_actor_root_pose(gym, sim, root_states, actor_index, pose)
            for env, robot, hold_targets, lift_targets, trajectory in zip(envs, robots, robot_hold_targets, robot_lift_targets, trajectories):
                label, _ = trajectory[min(frame, len(trajectory) - 1)]
                alpha = 1.0 if label in {"target", "tmp_low"} else 0.0
                targets = hold_targets * (1.0 - alpha) + lift_targets * alpha
                gym.set_actor_dof_position_targets(env, robot, targets.numpy())

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
            if frame % 120 == 0:
                status = []
                for scene, trajectory in zip(scenes, trajectories):
                    label, pose = trajectory[min(frame, len(trajectory) - 1)]
                    status.append(f"{scene.name}:{label}=({pose.x:.2f},{pose.y:.2f},{pose.z:.2f})")
                print(f"frame={frame} " + " ".join(status))
            frame += 1
            if args.max_frames > 0 and frame >= args.max_frames:
                break
            if args.headless and args.max_frames == 0 and all(frame >= len(trajectory) for trajectory in trajectories):
                break
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
