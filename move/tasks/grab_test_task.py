#!/usr/bin/env python3
"""Full-process grab/move/place IK simulation.

This task uses the offline IK chain from ``move/grab_test.py`` and plays it in
Isaac Gym:

1. palm-side grasp with fingers pointing to +X;
2. lift and switch to the move-state lift joints;
3. move to the corrected test3 pre-place stance;
4. keep root fixed and place the box with move-URDF arm IK.

The box is physical until the palms reach the side faces. After a short hold,
the task attaches the box kinematically to the current two-palm grasp frame,
moves it to the move_test3 target release point, then detaches it slightly above the stack
so it can settle without hand/box interpenetration.
"""

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
    raise SystemExit("PyTorch is required for grab_test_task.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move.grab_test import SIM_PLACE_STAND_OFF, _target_world_center, run as build_grab_plan  # noqa: E402
from move.planning import (  # noqa: E402
    BOX_POSE,
    BOX_SIZE,
    PALLET_THICKNESS,
    TABLE_POSE,
    TABLE_SIZE,
    BoxPlacement,
    Pose,
)
from move.tasks.move_test1 import (  # noqa: E402
    BOX_MASS,
    PRESET_BOX_MASS,
    _make_transform,
    _set_color,
    _set_shape_friction,
    create_sim,
    load_robot_asset,
)
from move.utils import PALM_CENTER_Z, PALM_SURFACE_X  # noqa: E402


PICK_SEGMENT_FRAMES = 90
MOVE_SETTLE_FRAMES = 60
MOVE_APPROACH_FRAMES = 120
MOVE_LIFT_FRAMES = 120
PLACE_HANDOFF_FRAMES = 60
PLACE_SEGMENT_FRAMES = 70
FINAL_HOLD_FRAMES = 180
ATTACH_AFTER_PICK_FRAMES = 250
MOBILE_DOF_NAMES = {
    "first_leg_pitch_joint",
    "second_leg_pitch_joint",
    "waist_pitch_joint",
    "body_yaw_joint",
}


def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return a * (1.0 - alpha) + b * alpha


def _smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _lerp_pose(a: Pose, b: Pose, alpha: float) -> Pose:
    return Pose(
        a.x * (1.0 - alpha) + b.x * alpha,
        a.y * (1.0 - alpha) + b.y * alpha,
        a.z * (1.0 - alpha) + b.z * alpha,
        a.yaw * (1.0 - alpha) + b.yaw * alpha,
    )


def _smooth_timeline_joint_steps(
    timeline: list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]],
    max_joint_step: float,
) -> list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]]:
    if not timeline:
        return timeline
    smoothed = [timeline[0]]
    for phase, root_pose, q, box_center, box_theta in timeline[1:]:
        prev_phase, prev_root, prev_q, prev_center, prev_theta = smoothed[-1]
        del prev_phase
        max_delta = float(torch.max(torch.abs(q - prev_q)).item())
        steps = max(1, int(math.ceil(max_delta / max_joint_step)))
        for step in range(1, steps + 1):
            alpha = _smoothstep(step / float(steps))
            center = (
                prev_center[0] * (1.0 - alpha) + box_center[0] * alpha,
                prev_center[1] * (1.0 - alpha) + box_center[1] * alpha,
                prev_center[2] * (1.0 - alpha) + box_center[2] * alpha,
            )
            smoothed.append(
                (
                    phase,
                    _lerp_pose(prev_root, root_pose, alpha),
                    _lerp_tensor(prev_q, q, alpha),
                    center,
                    prev_theta * (1.0 - alpha) + box_theta * alpha,
                )
            )
    return smoothed


def _expand_key_targets(targets: torch.Tensor, frames_per_segment: int) -> list[torch.Tensor]:
    frames: list[torch.Tensor] = []
    if targets.shape[0] == 0:
        return frames
    frames.append(targets[0].clone())
    for start, goal in zip(targets[:-1], targets[1:]):
        for i in range(1, frames_per_segment + 1):
            alpha = _smoothstep(i / float(frames_per_segment))
            frames.append(_lerp_tensor(start, goal, alpha))
    return frames


def _finger_close_value(dof_name: str, close_amount: float) -> float:
    if "back" in dof_name:
        return -0.10 * close_amount
    if "thumb" in dof_name:
        return 0.18 * close_amount
    if dof_name.endswith("_joint1") or dof_name.endswith("_joint2"):
        return 0.25 * close_amount
    if "bend" in dof_name:
        return 0.10 * close_amount
    return 0.0


def _apply_pick_finger_closure(frames: list[torch.Tensor], dof_names: list[str]) -> None:
    # Keep the first physical test palm-only. Curling fingers before palm
    # contact is stable has been observed to kick the 0.2m cube sideways.
    return
    close_start = 3 * PICK_SEGMENT_FRAMES
    close_end = 5 * PICK_SEGMENT_FRAMES
    name_to_index = {name: i for i, name in enumerate(dof_names)}
    hand_names = [name for name in dof_names if name.startswith("xhand_left_") or name.startswith("xhand_right_")]
    for frame_index, q in enumerate(frames):
        if frame_index < close_start:
            continue
        alpha = _smoothstep((frame_index - close_start) / float(max(1, close_end - close_start)))
        for name in hand_names:
            q[name_to_index[name]] = _finger_close_value(name, alpha)


def _reorder_targets(targets: torch.Tensor, source_names: list[str], target_names: list[str]) -> torch.Tensor:
    source_index = {name: i for i, name in enumerate(source_names)}
    indices = [source_index[name] for name in target_names]
    if targets.ndim == 1:
        return targets[torch.tensor(indices, dtype=torch.long)].contiguous()
    return targets[:, torch.tensor(indices, dtype=torch.long)].contiguous()


def _expand_centers(
    centers: list[tuple[float, float, float]],
    frames_per_segment: int,
) -> list[tuple[float, float, float]]:
    frames: list[tuple[float, float, float]] = []
    if not centers:
        return frames
    frames.append(centers[0])
    for start, goal in zip(centers[:-1], centers[1:]):
        for i in range(1, frames_per_segment + 1):
            alpha = _smoothstep(i / float(frames_per_segment))
            frames.append(
                (
                    start[0] * (1.0 - alpha) + goal[0] * alpha,
                    start[1] * (1.0 - alpha) + goal[1] * alpha,
                    start[2] * (1.0 - alpha) + goal[2] * alpha,
                )
            )
    return frames


def _pose_from_dict(data: dict[str, float]) -> Pose:
    return Pose(float(data["x"]), float(data["y"]), float(data["z"]), float(data["yaw"]))


def _transform_local_point(root: Pose, local: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(root.yaw)
    s = math.sin(root.yaw)
    return (
        root.x + c * local[0] - s * local[1],
        root.y + s * local[0] + c * local[1],
        root.z + local[2],
    )


def _box_world_pose(stack, placement: BoxPlacement) -> tuple[float, float, float]:
    local = placement.center_local
    return (
        stack.pallet_center[0] - stack.pallet_size[0] * 0.5 + local[0],
        stack.pallet_center[1] - stack.pallet_size[1] * 0.5 + local[1],
        stack.pallet_center[2] + local[2],
    )


def _set_actor_root_pose(
    gym,
    sim,
    root_states: torch.Tensor,
    actor_index: int,
    position: tuple[float, float, float],
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> None:
    state = root_states[actor_index]
    state[0] = position[0]
    state[1] = position[1]
    state[2] = position[2]
    yaw_half = yaw * 0.5
    pitch_half = pitch * 0.5
    sy = math.sin(yaw_half)
    cy = math.cos(yaw_half)
    sp = math.sin(pitch_half)
    cp = math.cos(pitch_half)
    state[3] = -sy * sp
    state[4] = cy * sp
    state[5] = sy * cp
    state[6] = cy * cp
    state[7:13] = 0.0
    indices = torch.tensor([actor_index], dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(indices),
        int(indices.numel()),
    )


def _set_task_collision_filters(gym, env, robot, box) -> None:
    """Let the box collide with Xhand links, not the larger arm links."""

    box_shape_props = gym.get_actor_rigid_shape_properties(env, box)
    for prop in box_shape_props:
        prop.filter = 1
    gym.set_actor_rigid_shape_properties(env, box, box_shape_props)

    robot_shape_props = gym.get_actor_rigid_shape_properties(env, robot)
    body_names = gym.get_actor_rigid_body_names(env, robot)
    shape_ranges = gym.get_actor_rigid_body_shape_indices(env, robot)
    for body_name, shape_range in zip(body_names, shape_ranges):
        body_filter = 2 if "xhand" in body_name else 1
        for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
            robot_shape_props[shape_idx].filter = body_filter
    gym.set_actor_rigid_shape_properties(env, robot, robot_shape_props)


def _set_hand_box_collision_enabled(gym, env, robot, box, enabled: bool) -> None:
    box_shape_props = gym.get_actor_rigid_shape_properties(env, box)
    for prop in box_shape_props:
        prop.filter = 1 if enabled else 2
    gym.set_actor_rigid_shape_properties(env, box, box_shape_props)

    robot_shape_props = gym.get_actor_rigid_shape_properties(env, robot)
    body_names = gym.get_actor_rigid_body_names(env, robot)
    shape_ranges = gym.get_actor_rigid_body_shape_indices(env, robot)
    for body_name, shape_range in zip(body_names, shape_ranges):
        body_filter = 2 if "xhand" in body_name else 1
        for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
            robot_shape_props[shape_idx].filter = body_filter
    gym.set_actor_rigid_shape_properties(env, robot, robot_shape_props)


def _set_actor_collision_filter(gym, env, actor, collision_filter: int) -> None:
    shape_props = gym.get_actor_rigid_shape_properties(env, actor)
    for prop in shape_props:
        prop.filter = collision_filter
    gym.set_actor_rigid_shape_properties(env, actor, shape_props)


def _actor_center(gym, env, actor) -> tuple[float, float, float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    pose = states["pose"][0]["p"]
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]))


def _actor_yaw_pitch(gym, env, actor) -> tuple[float, float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    q = states["pose"][0]["r"]
    rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
    pitch = math.asin(min(max(-rot[2][0], -1.0), 1.0))
    yaw = math.atan2(rot[1][0], rot[0][0])
    return yaw, pitch


def _body_center(gym, env, actor, body_name: str) -> tuple[float, float, float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    names = gym.get_actor_rigid_body_names(env, actor)
    index = names.index(body_name)
    pose = states["pose"][index]["p"]
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]))


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> tuple[tuple[float, float, float], ...]:
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def _rotate_yaw_pitch(local: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float, float]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x_pitch = cp * local[0] + sp * local[2]
    y_pitch = local[1]
    z_pitch = -sp * local[0] + cp * local[2]
    return (
        cy * x_pitch - sy * y_pitch,
        sy * x_pitch + cy * y_pitch,
        z_pitch,
    )


def _inverse_rotate_yaw_pitch(world: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float, float]:
    cy = math.cos(-yaw)
    sy = math.sin(-yaw)
    x_yaw = cy * world[0] - sy * world[1]
    y_yaw = sy * world[0] + cy * world[1]
    z_yaw = world[2]
    cp = math.cos(-pitch)
    sp = math.sin(-pitch)
    return (
        cp * x_yaw + sp * z_yaw,
        y_yaw,
        -sp * x_yaw + cp * z_yaw,
    )


def _palm_center(gym, env, actor, body_name: str) -> tuple[float, float, float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    names = gym.get_actor_rigid_body_names(env, actor)
    index = names.index(body_name)
    pose = states["pose"][index]
    p = pose["p"]
    q = pose["r"]
    rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
    return (
        float(p["x"]) + rot[0][0] * PALM_SURFACE_X + rot[0][2] * PALM_CENTER_Z,
        float(p["y"]) + rot[1][0] * PALM_SURFACE_X + rot[1][2] * PALM_CENTER_Z,
        float(p["z"]) + rot[2][0] * PALM_SURFACE_X + rot[2][2] * PALM_CENTER_Z,
    )


def _finger_dir(gym, env, actor, body_name: str) -> tuple[float, float, float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    names = gym.get_actor_rigid_body_names(env, actor)
    index = names.index(body_name)
    q = states["pose"][index]["r"]
    rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
    return (rot[0][2], rot[1][2], rot[2][2])


def _grasp_center(gym, env, actor) -> tuple[float, float, float]:
    left = _palm_center(gym, env, actor, "xhand_left_left_hand_link")
    right = _palm_center(gym, env, actor, "xhand_right_right_hand_link")
    return (
        0.5 * (left[0] + right[0]),
        0.5 * (left[1] + right[1]),
        0.5 * (left[2] + right[2]),
    )


def _grasp_theta(gym, env, actor, root_yaw: float) -> float:
    left = _finger_dir(gym, env, actor, "xhand_left_left_hand_link")
    right = _finger_dir(gym, env, actor, "xhand_right_right_hand_link")
    world_x = 0.5 * (left[0] + right[0])
    world_y = 0.5 * (left[1] + right[1])
    z = 0.5 * (left[2] + right[2])
    c = math.cos(-root_yaw)
    s = math.sin(-root_yaw)
    x = c * world_x - s * world_y
    return math.atan2(-z, x)


def _add_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dof_error_summary(gym, env, robot, q: torch.Tensor, dof_names: list[str]) -> tuple[float, str]:
    states = gym.get_actor_dof_states(env, robot, gymapi.STATE_POS)
    current = torch.tensor(states["pos"], dtype=torch.float32)
    errors = torch.abs(current - q)
    max_error = float(torch.max(errors).item())
    order = torch.argsort(errors, descending=True)[:3]
    parts = [f"{dof_names[int(i)]}:{float(errors[int(i)]):.3f}" for i in order]
    return max_error, ",".join(parts)


def _lock_mobile_dof_state(gym, env, robot, dof_names: list[str]) -> None:
    """Kinematically hold move-only body DOFs fixed during the flip-style pick."""

    states = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    for index, name in enumerate(dof_names):
        if name in MOBILE_DOF_NAMES:
            states["pos"][index] = 0.0
            states["vel"][index] = 0.0
    gym.set_actor_dof_states(env, robot, states, gymapi.STATE_ALL)


def _set_robot_dof_state(gym, env, robot, q: torch.Tensor) -> None:
    states = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    states["pos"] = q.numpy()
    states["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, states, gymapi.STATE_ALL)


def _create_actors(gym, sim, env, scene, robot_asset):
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), "grab_test_robot", 0, 1)

    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, scene.stack_scene.pallet_size[0], scene.stack_scene.pallet_size[1], PALLET_THICKNESS, fixed_opts)

    box_opts = gymapi.AssetOptions()
    box_opts.density = BOX_MASS / (BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2])
    source_asset = gym.create_box(sim, BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], box_opts)

    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    source = gym.create_actor(env, source_asset, _make_transform(*BOX_POSE), "carried_box", 0, 0)
    pallet_z = scene.stack_scene.pallet_surface_z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(
        env,
        pallet_asset,
        _make_transform(scene.stack_scene.pallet_center[0], scene.stack_scene.pallet_center[1], pallet_z),
        "pallet",
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

    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, source, (0.9, 0.05, 0.02))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    for actor in preset_actors:
        _set_color(gym, env, actor, (0.15, 0.25, 0.85))

    _set_shape_friction(gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05)
    _set_shape_friction(gym, env, table, 2.5)
    _set_shape_friction(gym, env, source, 8.0, rolling_friction=0.08, torsion_friction=0.08)
    _set_shape_friction(gym, env, pallet, 1.4)
    for actor in preset_actors:
        _set_shape_friction(gym, env, actor, 1.8)
    _set_task_collision_filters(gym, env, robot, source)

    return robot, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full-process grab/move/place IK simulation.")
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs forever.")
    parser.add_argument("--stand-off", type=float, default=SIM_PLACE_STAND_OFF, help="Corrected final pre-place stand-off.")
    parser.add_argument("--attach-after-pick-frames", type=int, default=ATTACH_AFTER_PICK_FRAMES, help="Frame index at which the grasped box is kinematically attached.")
    args = parser.parse_args()

    report, plan = build_grab_plan(place_mode="move", stand_off=args.stand_off)
    if not report.pick_feasible or not report.place_feasible:
        raise RuntimeError(
            "Generated IK plan is infeasible: "
            f"pick_error={report.pick_max_error:.4f}, place_error={report.place_max_error:.4f}"
        )

    scene = plan["scene"]
    pick_targets = plan["pick_targets"]
    place_targets = plan["place_targets"]
    move_approach_target = plan["move_approach_target"]
    move_lift_target = plan["move_lift_target"]
    root_route = [(_label, _pose_from_dict(pose_dict)) for _label, pose_dict in plan["root_route"]]
    if "place_pose_path_world" in plan:
        place_pose_path_labeled = [(label, tuple(center), float(theta)) for label, center, theta in plan["place_pose_path_world"]]
    else:
        place_pose_path_labeled = [(label, tuple(center), 0.0) for label, center in plan["place_center_path_world"]]
    place_center_path_labeled = [(label, center) for label, center, _theta in place_pose_path_labeled]
    place_center_path = [center for _label, center in place_center_path_labeled]
    target_center = _target_world_center(scene)
    release_center = place_center_path[-1]
    pre_place_center = tuple(plan.get("place_handoff_center_world", place_center_path[0]))

    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    robot_asset = load_robot_asset(gym, sim)
    dof_names = list(gym.get_asset_dof_names(robot_asset))

    pick_targets = _reorder_targets(pick_targets, list(plan["move_dof_names"]), dof_names)
    place_targets = _reorder_targets(place_targets, list(plan["place_dof_names"]), dof_names)
    release_target = _reorder_targets(plan["release_target"], list(plan["place_dof_names"]), dof_names)
    move_approach_target = _reorder_targets(move_approach_target, list(plan["move_dof_names"]), dof_names)
    move_lift_target = _reorder_targets(move_lift_target, list(plan["move_dof_names"]), dof_names)

    pick_dof_frames = _expand_key_targets(pick_targets, PICK_SEGMENT_FRAMES)
    _apply_pick_finger_closure(pick_dof_frames, dof_names)
    pick_centers = [tuple(center) for center in plan["pick_box_centers"]]
    pick_box_frames = _expand_centers(pick_centers, PICK_SEGMENT_FRAMES)

    root_frames = root_route + [(root_route[-1][0], root_route[-1][1]) for _ in range(MOVE_SETTLE_FRAMES)]
    move_dof_frames = []
    move_box_frames = []
    move_box_thetas = []
    move_phase_labels = []
    carry_start = pick_box_frames[-1]
    carry_local_start = (
        carry_start[0],
        carry_start[1],
        carry_start[2],
    )
    carry_local_final = (
        pre_place_center[0] - root_route[-1][1].x,
        pre_place_center[1] - root_route[-1][1].y,
        pre_place_center[2] - root_route[-1][1].z,
    )
    carry_local_approach = (
        carry_local_final[0],
        carry_local_final[1],
        float(plan["move_approach_box_center_z"]) - root_route[-1][1].z,
    )
    move_approach_theta = float(plan.get("move_approach_box_theta", 0.0))
    move_preplace_theta = float(plan.get("move_preplace_box_theta", move_approach_theta))
    for label, root_pose in root_frames:
        move_phase_labels.append(label)
        move_dof_frames.append(pick_targets[-1])
        move_box_frames.append(_transform_local_point(root_pose, carry_local_start))
        move_box_thetas.append(0.0)
    final_move_root = root_route[-1][1]
    for i in range(1, MOVE_APPROACH_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_APPROACH_FRAMES))
        move_phase_labels.append("lift_to_approach")
        move_dof_frames.append(_lerp_tensor(pick_targets[-1], move_approach_target, alpha))
        carry_local = (
            carry_local_start[0] * (1.0 - alpha) + carry_local_approach[0] * alpha,
            carry_local_start[1] * (1.0 - alpha) + carry_local_approach[1] * alpha,
            carry_local_start[2] * (1.0 - alpha) + carry_local_approach[2] * alpha,
        )
        move_box_frames.append(_transform_local_point(final_move_root, carry_local))
        move_box_thetas.append(move_approach_theta * alpha)
    for i in range(1, MOVE_LIFT_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_LIFT_FRAMES))
        move_phase_labels.append("lift_to_preplace")
        move_dof_frames.append(_lerp_tensor(move_approach_target, move_lift_target, alpha))
        carry_local = (
            carry_local_approach[0] * (1.0 - alpha) + carry_local_final[0] * alpha,
            carry_local_approach[1] * (1.0 - alpha) + carry_local_final[1] * alpha,
            carry_local_approach[2] * (1.0 - alpha) + carry_local_final[2] * alpha,
        )
        move_box_frames.append(_transform_local_point(final_move_root, carry_local))
        move_box_thetas.append(move_approach_theta * (1.0 - alpha) + move_preplace_theta * alpha)

    if bool(plan.get("place_targets_are_dense", False)):
        place_dof_frames = [place_targets[i].clone() for i in range(place_targets.shape[0])]
        place_box_frames_labeled = place_pose_path_labeled
    else:
        place_dof_frames = _expand_key_targets(place_targets, PLACE_SEGMENT_FRAMES)
        place_box_frames = _expand_centers(place_center_path, PLACE_SEGMENT_FRAMES)
        place_box_frames_labeled = [("place", center, 0.0) for center in place_box_frames]
    final_root_pose = root_route[-1][1]

    timeline = []
    for q, center in zip(pick_dof_frames, pick_box_frames):
        timeline.append(("pick", Pose(0.0, 0.0, 0.0, 0.0), q, center, 0.0))
    move_root_poses = [pose for _label, pose in root_frames] + [final_move_root for _ in range(MOVE_APPROACH_FRAMES + MOVE_LIFT_FRAMES)]
    for label, root_pose, q, center, theta in zip(move_phase_labels, move_root_poses, move_dof_frames, move_box_frames, move_box_thetas):
        timeline.append((f"move:{label}", root_pose, q, center, theta))
    if place_dof_frames:
        move_end_q = move_dof_frames[-1] if move_dof_frames else pick_dof_frames[-1]
        move_end_center = move_box_frames[-1] if move_box_frames else pick_box_frames[-1]
        move_end_theta = move_box_thetas[-1] if move_box_thetas else 0.0
        first_label, first_center, first_theta = place_box_frames_labeled[0]
        for i in range(1, PLACE_HANDOFF_FRAMES + 1):
            alpha = _smoothstep(i / float(PLACE_HANDOFF_FRAMES))
            center = (
                move_end_center[0] * (1.0 - alpha) + first_center[0] * alpha,
                move_end_center[1] * (1.0 - alpha) + first_center[1] * alpha,
                move_end_center[2] * (1.0 - alpha) + first_center[2] * alpha,
            )
            timeline.append(
                (
                    f"place_handoff:{first_label}",
                    final_root_pose,
                    _lerp_tensor(move_end_q, place_dof_frames[0], alpha),
                    center,
                    move_end_theta * (1.0 - alpha) + first_theta * alpha,
                )
            )
    for q, (label, center, theta) in zip(place_dof_frames, place_box_frames_labeled):
        timeline.append((f"place:{label}", final_root_pose, q, center, theta))
    release_hold_q = release_target
    timeline.append(("release", final_root_pose, release_hold_q, release_center, 0.0))
    for _ in range(FINAL_HOLD_FRAMES):
        timeline.append(("post_release_free_fall", final_root_pose, release_hold_q, release_center, 0.0))
    timeline = _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.5, 6.5, 2.5), 1)
    robot, source = _create_actors(gym, sim, env, scene, robot_asset)

    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        if name.startswith("xhand_"):
            props["stiffness"][i] = 120.0
            props["damping"][i] = 12.0
            props["effort"][i] = max(float(props["effort"][i]), 35.0)
        else:
            props["stiffness"][i] = 520.0
            props["damping"][i] = 52.0
            props["effort"][i] = max(float(props["effort"][i]), 180.0)
    gym.set_actor_dof_properties(env, robot, props)

    initial_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    initial_state["pos"] = timeline[0][2].numpy()
    initial_state["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, initial_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot, timeline[0][2].numpy())

    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
    source_index = gym.get_actor_index(env, source, gymapi.DOMAIN_SIM)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(2.0, -2.2, 1.45),
            gymapi.Vec3(0.0, 0.0, 0.80),
        )

    print("grab_test_task full-process IK simulation")
    print(f"  stand_off={args.stand_off:.3f} place_mode=move")
    print("  box_mode=contact_then_kinematic_attach")
    print(f"  pick_error={report.pick_max_error:.4f} place_error={report.place_max_error:.4f}")
    print(
        "  target_world_center="
        f"({target_center[0]:.3f},{target_center[1]:.3f},{target_center[2]:.3f}) "
        f"release_center=({release_center[0]:.3f},{release_center[1]:.3f},{release_center[2]:.3f}) "
        f"frames={len(timeline)}"
    )

    frame = 0
    attached = False
    released = False
    place_state_reported = False
    attach_offset_local: tuple[float, float, float] | None = None
    attach_yaw_offset = 0.0
    attach_theta_offset = 0.0
    try:
        while viewer is None or not gym.query_viewer_has_closed(viewer):
            phase, root_pose, q, box_center, box_theta = timeline[min(frame, len(timeline) - 1)]
            if not place_state_reported and phase.startswith("place:"):
                place_state_reported = True
                print(
                    "grab_test_place_state "
                    "root_locked=true arms_hands_active=true "
                    f"dense_cartesian_ik_frames={len(place_dof_frames)}"
                )
            if not attached and not released and frame >= args.attach_after_pick_frames:
                attached = True
                actual_center = _actor_center(gym, env, source)
                actual_yaw, actual_pitch = _actor_yaw_pitch(gym, env, source)
                grasp_center = _grasp_center(gym, env, robot)
                grasp_theta = _grasp_theta(gym, env, robot, root_pose.yaw)
                attach_offset_local = _inverse_rotate_yaw_pitch(
                    _sub_vec(actual_center, grasp_center),
                    root_pose.yaw,
                    grasp_theta,
                )
                attach_yaw_offset = actual_yaw - root_pose.yaw
                attach_theta_offset = actual_pitch - grasp_theta
                _set_hand_box_collision_enabled(gym, env, robot, source, enabled=False)
                print(
                    "grab_test_attach "
                    f"frame={frame} phase={phase} "
                    f"offset_local=({attach_offset_local[0]:.3f},{attach_offset_local[1]:.3f},{attach_offset_local[2]:.3f}) "
                    f"yaw_offset={math.degrees(attach_yaw_offset):.1f}deg "
                    f"theta_offset={math.degrees(attach_theta_offset):.1f}deg"
                )
            release_now = attached and not released and phase == "release"
            if release_now:
                gym.refresh_actor_root_state_tensor(sim)
                released = True
                attached = False
                actual_release = _actor_center(gym, env, source)
                print(
                    "grab_test_release "
                    f"frame={frame} release_from_current_pose=true hands_retreating=true "
                    f"actual_box=({actual_release[0]:.3f},{actual_release[1]:.3f},{actual_release[2]:.3f})"
                )

            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(gym, sim, root_states, robot_index, (root_pose.x, root_pose.y, root_pose.z), root_pose.yaw)
            _set_robot_dof_state(gym, env, robot, q)
            if phase == "pick":
                _lock_mobile_dof_state(gym, env, robot, dof_names)
            gym.set_actor_dof_position_targets(env, robot, q.numpy())

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if attached and attach_offset_local is not None:
                gym.refresh_actor_root_state_tensor(sim)
                grasp_center = _grasp_center(gym, env, robot)
                grasp_theta = _grasp_theta(gym, env, robot, root_pose.yaw)
                attached_center = _add_vec(
                    grasp_center,
                    _rotate_yaw_pitch(attach_offset_local, root_pose.yaw, grasp_theta),
                )
                attached_theta = grasp_theta + attach_theta_offset
                _set_actor_root_pose(gym, sim, root_states, source_index, attached_center, root_pose.yaw + attach_yaw_offset, attached_theta)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            if frame % 120 == 0:
                actual_box = _actor_center(gym, env, source)
                left_hand = _palm_center(gym, env, robot, "xhand_left_left_hand_link")
                right_hand = _palm_center(gym, env, robot, "xhand_right_right_hand_link")
                contacts = gym.get_env_rigid_contacts(env)
                q_err, q_err_joints = _dof_error_summary(gym, env, robot, q, dof_names)
                print(
                    f"frame={frame} phase={phase} "
                    f"root=({root_pose.x:.2f},{root_pose.y:.2f},{root_pose.z:.2f}) "
                    f"box_theta={math.degrees(box_theta):.1f}deg "
                    f"planned_box=({box_center[0]:.2f},{box_center[1]:.2f},{box_center[2]:.2f}) "
                    f"actual_box=({actual_box[0]:.2f},{actual_box[1]:.2f},{actual_box[2]:.2f}) "
                    f"left_palm=({left_hand[0]:.2f},{left_hand[1]:.2f},{left_hand[2]:.2f}) "
                    f"right_palm=({right_hand[0]:.2f},{right_hand[1]:.2f},{right_hand[2]:.2f}) "
                    f"palm_dist=({ _distance(left_hand, actual_box):.3f},{ _distance(right_hand, actual_box):.3f}) "
                    f"contacts={len(contacts)} q_err={q_err:.3f} top_q_err={q_err_joints} "
                    f"attached={attached} released={released}"
                )

            frame += 1
            if args.max_frames > 0 and frame >= args.max_frames:
                break
            if args.headless and args.max_frames == 0 and frame >= len(timeline):
                break
    finally:
        actual_box = _actor_center(gym, env, source)
        error = _distance(actual_box, target_center)
        print(
            "grab_test_result "
            f"actual_box=({actual_box[0]:.4f},{actual_box[1]:.4f},{actual_box[2]:.4f}) "
            f"target=({target_center[0]:.4f},{target_center[1]:.4f},{target_center[2]:.4f}) "
            f"error={error:.4f} attached={attached} released={released}"
        )
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
