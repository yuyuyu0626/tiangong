"""Continuous multi-box Isaac Gym simulation for move_pro.

PCT selects each target online. The robot motion, IK helpers, timeline
construction, and attach/release behavior come from the original move project.
Unlike the first move_pro prototype, one simulation and one viewer are kept
alive for the complete box sequence.
"""

from __future__ import annotations

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch  # type: ignore
import torch

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

import move.grab_test as _move_grab
import move.tasks.grab_test_task as _grab_task
from move.grab_test import OfflineTest3Scene, SIM_PLACE_STAND_OFF
from move.planning import (
    BOX_POSE,
    PALLET_SIZE,
    PALLET_SURFACE_Z,
    PALLET_THICKNESS,
    TABLE_POSE,
    TABLE_SIZE,
    BoxPlacement,
    EmptySpace,
    StackScene,
    pallet_center_near_table,
    sample_root_trajectory,
)
from move.tasks.grab_test_task import (
    ATTACH_AFTER_PICK_FRAMES,
    FINAL_HOLD_FRAMES,
    MOVE_APPROACH_FRAMES,
    MOVE_LIFT_FRAMES,
    MOVE_SETTLE_FRAMES,
    PICK_SEGMENT_FRAMES,
    PLACE_HANDOFF_FRAMES,
    _actor_center,
    _actor_yaw_pitch,
    _add_vec,
    _distance,
    _expand_centers,
    _expand_key_targets,
    _grasp_center,
    _grasp_theta,
    _inverse_rotate_yaw_pitch,
    _lock_mobile_dof_state,
    _pose_from_dict,
    _reorder_targets,
    _rotate_yaw_pitch,
    _set_actor_collision_filter,
    _set_actor_root_pose,
    _set_hand_box_collision_enabled,
    _set_robot_dof_state,
    _set_task_collision_filters,
    _smooth_timeline_joint_steps,
    _smoothstep,
    _sub_vec,
    _transform_local_point,
)
from move.tasks.move_test1 import (
    _make_transform,
    _set_color,
    _set_shape_friction,
    create_sim,
    load_robot_asset,
)

from move_pro.config import BOX_MASS, MOVE_URDF
from move_pro.integrator import BoxTask, MoveProIntegrator, MoveProPlan


TimelineFrame = Tuple[
    str,
    object,
    torch.Tensor,
    Tuple[float, float, float],
    float,
]


@dataclass
class PreparedBox:
    task: BoxTask
    scene: OfflineTest3Scene
    timeline: list[TimelineFrame]
    source_pose: tuple[float, float, float]
    pick_error: float
    place_error: float
    stand_off: float


def _source_pose(box_size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        BOX_POSE[0],
        BOX_POSE[1],
        TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + box_size[2] * 0.5,
    )


def _make_bpp_scene(
    target_world_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    stand_off: float,
    preset_boxes: tuple[BoxPlacement, ...],
) -> OfflineTest3Scene:
    sx, sy, sz = box_size
    pallet_center = pallet_center_near_table()
    local_center = (
        target_world_center[0] - (pallet_center[0] - PALLET_SIZE[0] * 0.5),
        target_world_center[1] - (pallet_center[1] - PALLET_SIZE[1] * 0.5),
        target_world_center[2] - PALLET_SURFACE_Z,
    )
    target_min = (
        local_center[0] - sx * 0.5,
        local_center[1] - sy * 0.5,
        local_center[2] - sz * 0.5,
    )
    target_box = BoxPlacement("bpp_target", target_min, box_size)

    # The thin support is planning-only. The real pallet actor supplies the
    # physical support in the shared simulation.
    floor_support = BoxPlacement(
        "_pallet_support",
        (0.0, 0.0, 0.0),
        (PALLET_SIZE[0], PALLET_SIZE[1], 0.001),
    )
    stack = StackScene(
        pallet_center=pallet_center,
        pallet_size=PALLET_SIZE,
        pallet_surface_z=PALLET_SURFACE_Z,
        preset_boxes=(floor_support, *preset_boxes),
        next_box=target_box,
        empty_spaces=(),
        selected_leaf=EmptySpace(target_box.min_corner, target_box.max_corner),
    )
    final_pose, tmp_pose = _move_grab._solve_test3_root_pose(stack, stand_off)
    waypoints = _move_grab._build_test3_waypoints(stack, final_pose, tmp_pose)
    return OfflineTest3Scene(
        name="bpp_target",
        stack_scene=stack,
        target_support_z=target_world_center[2] - sz * 0.5,
        final_pose=final_pose,
        tmp_pose=tmp_pose,
        waypoints=waypoints,
    )


@contextmanager
def _dynamic_move_box(scene: OfflineTest3Scene, box_size, source_pose):
    """Temporarily inject one variable-size box into move's planner."""

    original = (
        _move_grab.BOX_SIZE,
        _move_grab.BOX_POSE,
        _move_grab.MOVE_URDF,
        _move_grab.build_offline_test3_scene,
    )
    _move_grab.BOX_SIZE = tuple(box_size)
    _move_grab.BOX_POSE = tuple(source_pose)
    _move_grab.MOVE_URDF = MOVE_URDF
    _move_grab.build_offline_test3_scene = lambda stand_off=0.0: scene
    try:
        yield
    finally:
        (
            _move_grab.BOX_SIZE,
            _move_grab.BOX_POSE,
            _move_grab.MOVE_URDF,
            _move_grab.build_offline_test3_scene,
        ) = original


def _build_timeline(plan: dict[str, object], dof_names: list[str]) -> list[TimelineFrame]:
    pick_targets = _reorder_targets(
        plan["pick_targets"], list(plan["move_dof_names"]), dof_names
    )
    place_targets = _reorder_targets(
        plan["place_targets"], list(plan["place_dof_names"]), dof_names
    )
    release_target = _reorder_targets(
        plan["release_target"], list(plan["place_dof_names"]), dof_names
    )
    move_approach_target = _reorder_targets(
        plan["move_approach_target"], list(plan["move_dof_names"]), dof_names
    )
    move_lift_target = _reorder_targets(
        plan["move_lift_target"], list(plan["move_dof_names"]), dof_names
    )

    pick_frames = _expand_key_targets(pick_targets, PICK_SEGMENT_FRAMES)
    pick_centers = _expand_centers(
        [tuple(center) for center in plan["pick_box_centers"]],
        PICK_SEGMENT_FRAMES,
    )
    root_route = [
        ("", _pose_from_dict(pose)) for _, pose in plan["root_route"]
    ]
    root_frames = []
    for (_, start), (_, goal) in zip(root_route, root_route[1:]):
        if start != goal:
            root_frames.extend(sample_root_trajectory(start, goal))
    if root_frames:
        root_frames.extend([root_frames[-1]] * MOVE_SETTLE_FRAMES)
    if not root_frames:
        raise RuntimeError("move produced an empty root route")

    pose_path = plan.get(
        "place_pose_path_world",
        [(label, center, 0.0) for label, center in plan["place_center_path_world"]],
    )
    target_center = _move_grab._target_world_center(plan["scene"])
    pre_place = plan.get("place_handoff_center_world", pose_path[0][1])
    carry_start = pick_centers[-1]
    final_root = root_frames[-1]
    carry_final = (
        pre_place[0] - final_root.x,
        pre_place[1] - final_root.y,
        pre_place[2] - final_root.z,
    )
    carry_approach = (
        carry_final[0],
        carry_final[1],
        float(plan.get("move_approach_box_center_z", target_center[2]))
        - final_root.z,
    )
    approach_theta = float(plan.get("move_approach_box_theta", 0.0))
    preplace_theta = float(
        plan.get("move_preplace_box_theta", approach_theta)
    )

    move_q = []
    move_centers = []
    move_thetas = []
    for root in root_frames:
        move_q.append(pick_targets[-1])
        move_centers.append(_transform_local_point(root, carry_start))
        move_thetas.append(0.0)
    for index in range(1, MOVE_APPROACH_FRAMES + 1):
        alpha = _smoothstep(index / MOVE_APPROACH_FRAMES)
        move_q.append(
            pick_targets[-1] * (1.0 - alpha)
            + move_approach_target * alpha
        )
        local = tuple(
            carry_start[axis] * (1.0 - alpha)
            + carry_approach[axis] * alpha
            for axis in range(3)
        )
        move_centers.append(_transform_local_point(final_root, local))
        move_thetas.append(approach_theta * alpha)
    for index in range(1, MOVE_LIFT_FRAMES + 1):
        alpha = _smoothstep(index / MOVE_LIFT_FRAMES)
        move_q.append(
            move_approach_target * (1.0 - alpha) + move_lift_target * alpha
        )
        local = tuple(
            carry_approach[axis] * (1.0 - alpha)
            + carry_final[axis] * alpha
            for axis in range(3)
        )
        move_centers.append(_transform_local_point(final_root, local))
        move_thetas.append(
            approach_theta * (1.0 - alpha) + preplace_theta * alpha
        )

    timeline: list[TimelineFrame] = []
    for q, center in zip(pick_frames, pick_centers):
        timeline.append(("pick", _move_grab.Pose(0, 0, 0, 0), q, center, 0.0))
    all_roots = list(root_frames) + [final_root] * (
        MOVE_APPROACH_FRAMES + MOVE_LIFT_FRAMES
    )
    for root, q, center, theta in zip(
        all_roots, move_q, move_centers, move_thetas
    ):
        timeline.append(("move", root, q, center, theta))

    dense_place_targets = [
        place_targets[index].clone() for index in range(place_targets.shape[0])
    ]
    if dense_place_targets:
        start_q = move_q[-1]
        start_center = move_centers[-1]
        start_theta = move_thetas[-1]
        first_center, first_theta = pose_path[0][1], pose_path[0][2]
        for index in range(1, PLACE_HANDOFF_FRAMES + 1):
            alpha = _smoothstep(index / PLACE_HANDOFF_FRAMES)
            center = tuple(
                start_center[axis] * (1.0 - alpha)
                + first_center[axis] * alpha
                for axis in range(3)
            )
            timeline.append(
                (
                    "place_handoff",
                    final_root,
                    start_q * (1.0 - alpha)
                    + dense_place_targets[0] * alpha,
                    center,
                    start_theta * (1.0 - alpha) + first_theta * alpha,
                )
            )
    for q, (_, center, theta) in zip(dense_place_targets, pose_path):
        timeline.append(("place", final_root, q, center, theta))

    release_center = pose_path[-1][1]
    timeline.append(("release", final_root, release_target, release_center, 0.0))
    timeline.extend(
        ("post_release", final_root, release_target, release_center, 0.0)
        for _ in range(FINAL_HOLD_FRAMES)
    )

    # 返回路径：放完箱子后机器人沿来路反向退回桌子起点，而不是瞬间闪回。
    # 复用 move 的 root_route（table→pallet），反向采样得到 pallet→table。
    # 返回途中手臂从 release 姿态平滑收回到待命姿态 pick_frames[0]（每个箱子
    # timeline 的起始姿态），避免一直前伸着退回去。
    return_route = [pose for _, pose in reversed(root_route)]
    return_frames: list = []
    for start, goal in zip(return_route, return_route[1:]):
        if start != goal:
            return_frames.extend(sample_root_trajectory(start, goal))
    home_q = pick_frames[0]
    total_return = len(return_frames)
    # 手臂在返回前段（前 60%）完成收回，之后保持待命姿态稳定行进。
    retract_frames = max(1, int(total_return * 0.6))
    for index, root in enumerate(return_frames):
        alpha = _smoothstep(min(1.0, index / retract_frames))
        q = release_target * (1.0 - alpha) + home_q * alpha
        timeline.append(("return", root, q, release_center, 0.0))

    return _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)


def _prepare_boxes(
    plan: MoveProPlan,
    dof_names: list[str],
    stand_off: float,
) -> list[PreparedBox]:
    prepared = []
    preset_boxes: list[BoxPlacement] = []
    pallet_center = pallet_center_near_table()
    for task in plan.box_tasks:
        if not task.placement.feasible:
            continue
        box_size = task.world_size
        source_pose = _source_pose(box_size)
        candidate_offsets = (
            stand_off,
            0.42,
            0.50,
            0.60,
            0.68,
            0.72,
            0.80,
            0.90,
        )
        attempts = []
        seen_offsets = set()
        for candidate_offset in candidate_offsets:
            candidate_offset = round(float(candidate_offset), 3)
            if candidate_offset in seen_offsets:
                continue
            seen_offsets.add(candidate_offset)
            candidate_scene = _make_bpp_scene(
                task.world_target,
                box_size,
                candidate_offset,
                tuple(preset_boxes),
            )
            with _dynamic_move_box(
                candidate_scene, box_size, source_pose
            ):
                candidate_report, candidate_plan = _move_grab.run(
                    place_mode="move", stand_off=candidate_offset
                )
            attempts.append(
                (
                    candidate_report.pick_max_error
                    + candidate_report.place_max_error,
                    candidate_offset,
                    candidate_scene,
                    candidate_report,
                    candidate_plan,
                )
            )
            if (
                candidate_report.pick_feasible
                and candidate_report.place_feasible
            ):
                break

        feasible_attempts = [
            attempt
            for attempt in attempts
            if attempt[3].pick_feasible and attempt[3].place_feasible
        ]
        if not feasible_attempts:
            details = ", ".join(
                f"{offset:.2f}:"
                f"{attempt_report.pick_max_error:.4f}/"
                f"{attempt_report.place_max_error:.4f}"
                for _, offset, _, attempt_report, _ in attempts
            )
            raise RuntimeError(
                f"IK infeasible for box {task.index}: "
                f"stand_off attempts [{details}]"
            )
        _, selected_offset, scene, report, move_plan = min(
            feasible_attempts, key=lambda attempt: attempt[0]
        )
        print(
            f"  prepared box={task.index} stand_off={selected_offset:.2f} "
            f"pick={report.pick_max_error:.4f} "
            f"place={report.place_max_error:.4f}"
        )
        prepared.append(
            PreparedBox(
                task=task,
                scene=scene,
                timeline=_build_timeline(move_plan, dof_names),
                source_pose=source_pose,
                pick_error=report.pick_max_error,
                place_error=report.place_max_error,
                stand_off=selected_offset,
            )
        )
        target = task.world_target
        preset_boxes.append(
            BoxPlacement(
                f"placed_{task.index}",
                (
                    target[0]
                    - pallet_center[0]
                    + PALLET_SIZE[0] * 0.5
                    - box_size[0] * 0.5,
                    target[1]
                    - pallet_center[1]
                    + PALLET_SIZE[1] * 0.5
                    - box_size[1] * 0.5,
                    target[2]
                    - PALLET_SURFACE_Z
                    - box_size[2] * 0.5,
                ),
                box_size,
            )
        )
    return prepared


BOX_COLORS = (
    (0.92, 0.18, 0.08),
    (0.95, 0.52, 0.05),
    (0.95, 0.78, 0.08),
    (0.28, 0.72, 0.22),
    (0.08, 0.62, 0.72),
    (0.12, 0.38, 0.88),
    (0.48, 0.24, 0.82),
    (0.82, 0.20, 0.62),
)


def _parked_pose(index: int, box_size) -> tuple[float, float, float]:
    column = index % 8
    row = index // 8
    return (-2.0 - column * 0.55, -2.0 - row * 0.55, box_size[2] * 0.5)


def _create_shared_scene(gym, sim, env, robot_asset, prepared):
    robot = gym.create_actor(
        env,
        robot_asset,
        _make_transform(0.0, 0.0, 0.0),
        "move_pro_robot",
        0,
        1,
    )
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    table_asset = gym.create_box(sim, *TABLE_SIZE, fixed)
    pallet_asset = gym.create_box(
        sim, PALLET_SIZE[0], PALLET_SIZE[1], PALLET_THICKNESS, fixed
    )
    table = gym.create_actor(
        env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0
    )
    pallet_center = pallet_center_near_table()
    pallet = gym.create_actor(
        env,
        pallet_asset,
        _make_transform(
            pallet_center[0],
            pallet_center[1],
            PALLET_SURFACE_Z - PALLET_THICKNESS * 0.5,
        ),
        "pallet",
        0,
        0,
    )

    boxes = []
    for index, item in enumerate(prepared):
        sx, sy, sz = item.task.world_size
        options = gymapi.AssetOptions()
        options.density = BOX_MASS / (sx * sy * sz)
        asset = gym.create_box(sim, sx, sy, sz, options)
        box = gym.create_actor(
            env,
            asset,
            _make_transform(*_parked_pose(index, item.task.world_size)),
            f"move_pro_box_{item.task.index:03d}",
            0,
            0,
        )
        boxes.append(box)
        _set_color(gym, env, box, BOX_COLORS[index % len(BOX_COLORS)])
        _set_shape_friction(
            gym, env, box, 8.0, rolling_friction=0.08, torsion_friction=0.08
        )

    _set_color(gym, env, table, (0.88, 0.88, 0.88))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    _set_shape_friction(
        gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05
    )
    _set_shape_friction(gym, env, table, 2.5)
    _set_shape_friction(gym, env, pallet, 1.4)
    return robot, boxes


def _configure_robot(gym, env, robot, dof_names):
    properties = gym.get_actor_dof_properties(env, robot)
    properties["driveMode"].fill(gymapi.DOF_MODE_POS)
    for index, name in enumerate(dof_names):
        if "xhand_" in name:
            properties["stiffness"][index] = 120.0
            properties["damping"][index] = 12.0
            properties["effort"][index] = max(
                float(properties["effort"][index]), 35.0
            )
        else:
            properties["stiffness"][index] = 520.0
            properties["damping"][index] = 52.0
            properties["effort"][index] = max(
                float(properties["effort"][index]), 180.0
            )
    gym.set_actor_dof_properties(env, robot, properties)


def _play_box(
    gym,
    sim,
    env,
    robot,
    box,
    root_states,
    robot_index: int,
    box_index: int,
    dof_names: list[str],
    item: PreparedBox,
    viewer,
    fast: bool,
    global_frame: int,
    max_frames: int,
) -> tuple[int, bool]:
    _set_actor_root_pose(
        gym, sim, root_states, box_index, item.source_pose, 0.0, 0.0
    )
    _set_task_collision_filters(gym, env, robot, box)
    _set_hand_box_collision_enabled(gym, env, robot, box, enabled=True)

    attached = False
    released = False
    attach_offset = None
    attach_yaw_offset = 0.0
    attach_theta_offset = 0.0
    for local_frame, (phase, root, q, _center, _theta) in enumerate(
        item.timeline
    ):
        frame = global_frame + local_frame
        if max_frames > 0 and frame >= max_frames:
            return frame, False
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            return frame, False

        if (
            not attached
            and not released
            and phase == "pick"
            and local_frame >= ATTACH_AFTER_PICK_FRAMES
        ):
            attached = True
            actual_center = _actor_center(gym, env, box)
            actual_yaw, actual_pitch = _actor_yaw_pitch(gym, env, box)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root.yaw)
            attach_offset = _inverse_rotate_yaw_pitch(
                _sub_vec(actual_center, grasp_center),
                root.yaw,
                grasp_theta,
            )
            attach_yaw_offset = actual_yaw - root.yaw
            attach_theta_offset = actual_pitch - grasp_theta
            _set_hand_box_collision_enabled(
                gym, env, robot, box, enabled=False
            )
            print(
                f"move_pro_attach box={item.task.index} frame={frame} "
                f"offset=({attach_offset[0]:.3f},"
                f"{attach_offset[1]:.3f},{attach_offset[2]:.3f})"
            )

        if attached and not released and phase == "release":
            released = True
            attached = False
            actual = _actor_center(gym, env, box)
            print(
                f"move_pro_release box={item.task.index} frame={frame} "
                f"actual=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f})"
            )

        gym.refresh_actor_root_state_tensor(sim)
        _set_actor_root_pose(
            gym,
            sim,
            root_states,
            robot_index,
            (root.x, root.y, root.z),
            root.yaw,
        )
        # 释放后把箱子运动学冻结在 BPP 目标位姿：在物理 step 之前先设定位姿
        # 并清零速度，让物理引擎对它没有净作用，消除"物理推→snap 拉回"的反复横跳
        # （抖动）。z 即 BPP 目标，箱底正好贴支撑面，不会嵌入。
        # return 阶段也保持锚定，箱子留在原位不被带走。
        if released:
            _set_actor_root_pose(
                gym, sim, root_states, box_index, item.task.world_target, 0.0, 0.0
            )
        _set_robot_dof_state(gym, env, robot, q)
        if phase == "pick":
            _lock_mobile_dof_state(gym, env, robot, dof_names)
        gym.set_actor_dof_position_targets(env, robot, q.numpy())
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if attached and attach_offset is not None:
            gym.refresh_actor_root_state_tensor(sim)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root.yaw)
            center = _add_vec(
                grasp_center,
                _rotate_yaw_pitch(attach_offset, root.yaw, grasp_theta),
            )
            _set_actor_root_pose(
                gym,
                sim,
                root_states,
                box_index,
                center,
                root.yaw + attach_yaw_offset,
                grasp_theta + attach_theta_offset,
            )

        # 物理 step 之后再设一次，保证目标位姿是该帧最后的权威位姿。
        if released:
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(
                gym, sim, root_states, box_index, item.task.world_target, 0.0, 0.0
            )

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if not fast:
                gym.sync_frame_time(sim)

        if local_frame % 240 == 0:
            actual = _actor_center(gym, env, box)
            print(
                f"frame={frame} box={item.task.index} phase={phase} "
                f"actual=({actual[0]:.2f},{actual[1]:.2f},{actual[2]:.2f}) "
                f"attached={attached} released={released}"
            )

    actual = _actor_center(gym, env, box)
    error = _distance(actual, item.task.world_target)
    _set_actor_collision_filter(gym, env, box, 0)
    print(
        f"move_pro_box_result box={item.task.index} "
        f"actual=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) "
        f"target=({item.task.world_target[0]:.3f},"
        f"{item.task.world_target[1]:.3f},{item.task.world_target[2]:.3f}) "
        f"error={error:.4f}"
    )
    return global_frame + len(item.timeline), True


class MoveProSimulator:
    """Run all PCT-selected boxes in one persistent move simulation."""

    def __init__(
        self,
        method: str = "LSAH",
        container_size=(10, 10, 16),
        stand_off: float = SIM_PLACE_STAND_OFF,
    ):
        self.stand_off = stand_off
        self.integrator = MoveProIntegrator(
            method=method,
            container_size=container_size,
            stand_off=stand_off,
        )

    def run(
        self,
        box_sequence,
        sizes_are_pct=True,
        headless=False,
        fast=False,
        max_frames=0,
    ):
        plan = self.integrator.build_plan(
            box_sequence, compute_ik=False, sizes_are_pct=sizes_are_pct
        )
        print(plan.summary())

        gym = gymapi.acquire_gym()
        sim = create_sim(gym)
        if sim is None:
            raise RuntimeError("Isaac Gym failed to create the simulation")
        viewer = None
        try:
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0, 0, 1)
            gym.add_ground(sim, plane)
            robot_asset = load_robot_asset(gym, sim)
            if robot_asset is None:
                raise RuntimeError(f"failed to load robot URDF from {MOVE_URDF}")
            dof_names = list(gym.get_asset_dof_names(robot_asset))
            if not dof_names:
                raise RuntimeError("robot asset contains no active DOFs")

            print("Preparing move IK plans for all boxes...")
            prepared = _prepare_boxes(plan, dof_names, self.stand_off)
            if not prepared:
                raise RuntimeError("PCT produced no executable boxes")

            env = gym.create_env(
                sim,
                gymapi.Vec3(-8.0, -5.0, 0.0),
                gymapi.Vec3(7.0, 7.0, 3.0),
                1,
            )
            robot, boxes = _create_shared_scene(
                gym, sim, env, robot_asset, prepared
            )
            _configure_robot(gym, env, robot, dof_names)
            first_q = prepared[0].timeline[0][2]
            _set_robot_dof_state(gym, env, robot, first_q)
            gym.set_actor_dof_position_targets(env, robot, first_q.numpy())

            gym.prepare_sim(sim)
            root_states = gymtorch.wrap_tensor(
                gym.acquire_actor_root_state_tensor(sim)
            )
            robot_index = gym.get_actor_index(
                env, robot, gymapi.DOMAIN_SIM
            )
            box_indices = [
                gym.get_actor_index(env, box, gymapi.DOMAIN_SIM)
                for box in boxes
            ]

            if not headless:
                viewer = gym.create_viewer(sim, gymapi.CameraProperties())
                if viewer is None:
                    raise RuntimeError("failed to create Isaac Gym viewer")
                pallet_center = pallet_center_near_table()
                gym.viewer_camera_look_at(
                    viewer,
                    env,
                    gymapi.Vec3(
                        pallet_center[0] + 1.8,
                        pallet_center[1] - 2.2,
                        1.7,
                    ),
                    gymapi.Vec3(
                        pallet_center[0], pallet_center[1], 0.65
                    ),
                )

            global_frame = 0
            completed = 0
            for item, box, box_index in zip(
                prepared, boxes, box_indices
            ):
                print(
                    f"\nmove_pro_start_box box={item.task.index} "
                    f"size={item.task.world_size} "
                    f"target={item.task.world_target} "
                    f"stand_off={item.stand_off:.2f} "
                    f"ik=({item.pick_error:.4f},{item.place_error:.4f})"
                )
                global_frame, keep_running = _play_box(
                    gym,
                    sim,
                    env,
                    robot,
                    box,
                    root_states,
                    robot_index,
                    box_index,
                    dof_names,
                    item,
                    viewer,
                    fast,
                    global_frame,
                    max_frames,
                )
                if not keep_running:
                    break
                completed += 1
            print(
                f"move_pro_result completed={completed}/{len(prepared)} "
                f"frames={global_frame}"
            )
            return plan
        finally:
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
