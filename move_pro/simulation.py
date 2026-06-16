"""Continuous multi-box Isaac Gym simulation for move_pro.

PCT selects each target online (variable-size boxes). Everything else — pick
keyframes, root routes, place path, timeline, attach/release/playback, and the
scene layout — is reused from move's task1_2 placement layer via
``move_pro.task1_2_adapter``. This keeps move_pro's non-decision behavior
byte-aligned with the polished task1_2 stacking demo.
"""

from __future__ import annotations

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch  # type: ignore

from contextlib import contextmanager
from dataclasses import dataclass

import move.tasks.task1_2 as t12
from move.planning import (
    PALLET_SIZE,
    PALLET_SURFACE_Z,
    PALLET_THICKNESS,
    TABLE_POSE,
    TABLE_SIZE,
    pallet_center_near_table,
)
from move.tasks.grab_test_task import (
    _set_robot_dof_state,
    _actor_center,
    _actor_yaw_pitch,
    _grasp_center,
    _grasp_theta,
    _inverse_rotate_yaw_pitch,
    _rotate_yaw_pitch,
    _add_vec,
    _sub_vec,
    _distance,
    _lock_mobile_dof_state,
    _set_actor_root_pose,
    _set_actor_collision_filter,
    _set_hand_box_collision_enabled,
    _set_task_collision_filters,
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
from move_pro.task1_2_adapter import build_pct_box_plan


@dataclass
class PreparedBox:
    task: BoxTask
    plan: object               # task1_2.BoxPlan
    timeline: list
    pick_error: float
    place_error: float
    stand_off: float
    side: str


# 机器人可从托盘四侧接近放置。move_pro 早期硬编码只用 -X 侧，导致够不到托盘远端。
# 按目标到托盘四边的距离排序接近侧（离哪边近优先从哪侧接近，手臂前伸最短最易 IK 可行），
# 四侧都作回退，保证整个托盘可达。
PLACE_SIDES = ("-X", "+X", "-Y", "+Y")


def _ordered_sides(target_world_center) -> tuple[str, ...]:
    pallet_center = pallet_center_near_table()
    cx, cy, _ = pallet_center
    hx, hy = PALLET_SIZE[0] * 0.5, PALLET_SIZE[1] * 0.5
    tx, ty = target_world_center[0], target_world_center[1]
    dist = {
        "+X": abs((cx + hx) - tx),
        "-X": abs(tx - (cx - hx)),
        "+Y": abs((cy + hy) - ty),
        "-Y": abs(ty - (cy - hy)),
    }
    return tuple(sorted(PLACE_SIDES, key=lambda s: dist[s]))


# 站距候选：从近到远尝试，第一个 IK 可行的即采用（与早期 move_pro 一致）。
_CANDIDATE_OFFSETS = (0.42, 0.50, 0.55, 0.60, 0.68, 0.72, 0.80, 0.90)


def _prepare_boxes(plan: MoveProPlan, asset_dof_names: list[str]) -> list[PreparedBox]:
    """为每个 PCT 目标，遍历四侧 × 站距，选 IK 可行且误差最小的 BoxPlan。"""
    prepared: list[PreparedBox] = []
    for task in plan.box_tasks:
        if not task.placement.feasible:
            continue
        target = task.world_target
        box_size = task.world_size
        attempts = []
        found = False
        for side in _ordered_sides(target):
            for stand_off in _CANDIDATE_OFFSETS:
                try:
                    box_plan = build_pct_box_plan(target, box_size, side, stand_off)
                except Exception as exc:  # IK / 规划失败，换下一组
                    attempts.append((float("inf"), side, stand_off, repr(exc), None))
                    continue
                err = box_plan.pick_max_error + box_plan.place_max_error
                feasible = (
                    box_plan.pick_max_error <= t12.PICK_ERROR_LIMIT
                    and box_plan.place_max_error <= t12.PLACE_ERROR_LIMIT
                )
                attempts.append((err, side, stand_off, box_plan, feasible))
                if feasible:
                    found = True
                    break
            if found:
                break

        feasible_attempts = [a for a in attempts if a[4] is True]
        if not feasible_attempts:
            details = ", ".join(
                f"{s}@{o:.2f}:"
                + (f"{p.pick_max_error:.3f}/{p.place_max_error:.3f}"
                   if hasattr(p, "pick_max_error") else str(p))
                for _e, s, o, p, _f in attempts
            )
            raise RuntimeError(
                f"IK infeasible for box {task.index}: attempts [{details}]"
            )
        _err, side, stand_off, box_plan, _f = min(
            feasible_attempts, key=lambda a: a[0]
        )
        box_plan = t12._reorder_box_plan(box_plan, asset_dof_names)
        timeline = t12.build_timeline(box_plan)
        print(
            f"  prepared box={task.index} side={side} stand_off={stand_off:.2f} "
            f"pick={box_plan.pick_max_error:.4f} place={box_plan.place_max_error:.4f} "
            f"frames={len(timeline)}"
        )
        prepared.append(
            PreparedBox(
                task=task,
                plan=box_plan,
                timeline=timeline,
                pick_error=box_plan.pick_max_error,
                place_error=box_plan.place_max_error,
                stand_off=stand_off,
                side=side,
            )
        )
    return prepared


# 箱子停泊位（远离工作区，依次排开），与 task1_2 _parked_box_pose 同风格。
def _parked_pose(index: int, box_size) -> tuple[float, float, float]:
    return (
        t12.BOX_PARK_X - t12.BOX_PARK_SPACING * index,
        -1.2,
        TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + box_size[2] * 0.5,
    )


def _create_scene(gym, sim, env, robot_asset, prepared):
    """对齐 task1_2._create_static_scene：桌、托盘、机器人，箱子从桌边源位出现。

    move_pro 的箱子尺寸各异，因此每个箱子单独建 actor（task1_2 是等大箱共享 asset）。
    其余（位置、颜色、摩擦、碰撞过滤）对齐 task1_2。
    """
    robot = gym.create_actor(
        env, robot_asset, _make_transform(0.0, 0.0, 0.0), "move_pro_robot", 0, 1
    )
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    table_asset = gym.create_box(sim, *TABLE_SIZE, fixed)
    pallet_center = pallet_center_near_table()
    pallet_asset = gym.create_box(
        sim, PALLET_SIZE[0], PALLET_SIZE[1], PALLET_THICKNESS, fixed
    )
    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)

    boxes = []
    for index, item in enumerate(prepared):
        sx, sy, sz = item.task.world_size
        opts = gymapi.AssetOptions()
        opts.density = BOX_MASS / (sx * sy * sz)
        asset = gym.create_box(sim, sx, sy, sz, opts)
        # 第一个箱子摆在源位，其余停泊（与 task1_2 一致：播放时再 reset 到源位）。
        pose = _source_pose(item.task.world_size) if index == 0 else _parked_pose(index, item.task.world_size)
        box = gym.create_actor(
            env, asset, _make_transform(*pose), f"move_pro_box_{item.task.index:03d}", 0, 0
        )
        boxes.append(box)
        _set_color(gym, env, box, BOX_COLORS[index % len(BOX_COLORS)])
        _set_shape_friction(
            gym, env, box,
            t12.BOX_CONTACT_FRICTION,
            rolling_friction=t12.BOX_CONTACT_ROLLING_FRICTION,
            torsion_friction=t12.BOX_CONTACT_TORSION_FRICTION,
        )

    pallet_z = PALLET_SURFACE_Z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(
        env, pallet_asset,
        _make_transform(pallet_center[0], pallet_center[1], pallet_z),
        "pallet", 0, 0,
    )
    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    _set_shape_friction(gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05)
    _set_shape_friction(gym, env, table, 2.5)
    _set_shape_friction(gym, env, pallet, t12.PALLET_CONTACT_FRICTION)
    return robot, boxes


def _source_pose(box_size):
    """箱子源位姿，箱底贴桌面（与 task1_2.SOURCE_BOX_POSE 同 x/y，z 随箱高）。"""
    return (
        t12.SOURCE_BOX_POSE[0],
        t12.SOURCE_BOX_POSE[1],
        TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + box_size[2] * 0.5,
    )


BOX_COLORS = (
    (0.92, 0.18, 0.08), (0.95, 0.52, 0.05), (0.95, 0.78, 0.08), (0.28, 0.72, 0.22),
    (0.08, 0.62, 0.72), (0.12, 0.38, 0.88), (0.48, 0.24, 0.82), (0.82, 0.20, 0.62),
)


@contextmanager
def _box_source(box_size):
    """临时把 task1_2.SOURCE_BOX_POSE 的 z 改为当前箱高对应值。

    task1_2._run_box_timeline 用模块级 SOURCE_BOX_POSE reset 箱子到桌边源位；
    move_pro 箱子尺寸各异，需让每个箱子按自身高度贴桌面，抓取才一致。
    """
    original = t12.SOURCE_BOX_POSE
    t12.SOURCE_BOX_POSE = _source_pose(box_size)
    try:
        yield
    finally:
        t12.SOURCE_BOX_POSE = original


def _play_box(gym, sim, env, robot, box, root_states, robot_index, box_index,
              item, viewer, fast, global_frame, max_frames, placed):
    """播放单个箱子的 timeline（对齐 task1_2._run_box_timeline 的抓取/搬运/释放），
    并在释放后把箱子运动学锁定到 BPP 目标位姿。

    task1_2 的放置参数是为 0.4m 等大箱调的，靠物理自然落稳；move_pro 的箱子尺寸
    各异（含很扁/很小的箱），自由落体会翻滚/弹飞。释放后锁定到目标位姿（清零速度）
    保证确定性落位，且不影响抓取/搬运/路线等其余对齐 task1_2 的行为。

    placed: 之前已放好的箱子 [(box_index, target), ...]。每帧把它们钉回各自目标位姿，
    防止当前箱子/机器人搬运时撞飞已放好的箱子（撞出托盘、挂边缘等）。
    """
    plan = item.plan
    timeline = item.timeline
    target = item.task.world_target
    render_every = 8 if fast else 1
    _set_task_collision_filters(gym, env, robot, box)
    t12._reset_box_to_source(gym, sim, root_states, 0, box_index, plan.source_box_yaw)

    def _pin_placed():
        for placed_index, placed_target in placed:
            _set_actor_root_pose(gym, sim, root_states, placed_index, placed_target, 0.0, 0.0)

    attached = False
    released = False
    attach_offset = None
    attach_yaw_offset = 0.0
    attach_theta_offset = 0.0
    pick_started = False
    pick_count = 0
    for local_frame, (phase, root, q, _center, _theta) in enumerate(timeline):
        frame = global_frame + local_frame
        if max_frames > 0 and local_frame >= max_frames:
            return frame, False
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            return frame, False

        if phase == "pick" and not pick_started:
            t12._reset_box_to_source(gym, sim, root_states, 0, box_index, plan.source_box_yaw)
            pick_started = True
        if phase == "pick":
            pick_count += 1

        if not attached and not released and phase == "pick" and pick_count >= t12.ATTACH_AFTER_PICK_FRAMES:
            attached = True
            actual_center = _actor_center(gym, env, box)
            actual_yaw, actual_pitch = _actor_yaw_pitch(gym, env, box)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root.yaw)
            attach_offset = _inverse_rotate_yaw_pitch(
                _sub_vec(actual_center, grasp_center), root.yaw, grasp_theta)
            attach_yaw_offset = actual_yaw - root.yaw
            attach_theta_offset = actual_pitch - grasp_theta
            _set_hand_box_collision_enabled(gym, env, robot, box, enabled=False)

        if attached and not released and (phase == "release" or phase == "place:place_hold"):
            released = True
            attached = False

        gym.refresh_actor_root_state_tensor(sim)
        # 释放后在物理 step 前先锁定到目标，避免抖动/穿插。
        if released:
            _set_actor_root_pose(gym, sim, root_states, box_index, target, 0.0, 0.0)
        _pin_placed()  # 物理 step 前钉住所有已放箱子
        _set_actor_root_pose(gym, sim, root_states, robot_index, (root.x, root.y, root.z), root.yaw)
        _set_robot_dof_state(gym, env, robot, q)
        if phase == "pick":
            _lock_mobile_dof_state(gym, env, robot, plan.dof_names)
        gym.set_actor_dof_position_targets(env, robot, q.numpy())
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if attached and attach_offset is not None:
            gym.refresh_actor_root_state_tensor(sim)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root.yaw)
            center = _add_vec(grasp_center, _rotate_yaw_pitch(attach_offset, root.yaw, grasp_theta))
            _set_actor_root_pose(gym, sim, root_states, box_index, center,
                                 root.yaw + attach_yaw_offset, grasp_theta + attach_theta_offset)
        # 释放后物理 step 之后再锁定一次，保证目标位姿是该帧最后权威。
        if released:
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(gym, sim, root_states, box_index, target, 0.0, 0.0)
        _pin_placed()  # 物理 step 后再钉一次，已放箱子全程不被撞动

        if viewer is not None and local_frame % render_every == 0:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if not fast:
                gym.sync_frame_time(sim)

    _set_actor_collision_filter(gym, env, box, 0)
    actual = _actor_center(gym, env, box)
    error = _distance(actual, target)
    print(
        f"move_pro_box_result box={item.task.index} "
        f"actual=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) "
        f"target=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) error={error:.4f}"
    )
    # 当前箱子放完，加入已放列表，后续箱子放置时它会被钉住。
    placed.append((box_index, target))
    return global_frame + len(timeline), True


class MoveProSimulator:
    """在一个持久仿真里依次放完所有 PCT 决策的箱子（复用 task1_2 执行层）。"""

    def __init__(self, method: str = "LSAH", container_size=(10, 10, 16)):
        self.integrator = MoveProIntegrator(method=method, container_size=container_size)

    def run(self, box_sequence, sizes_are_pct=True, headless=False, fast=False, max_frames=0):
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
            ground = gymapi.PlaneParams()
            ground.normal = gymapi.Vec3(0, 0, 1)
            gym.add_ground(sim, ground)
            robot_asset = load_robot_asset(gym, sim)
            if robot_asset is None:
                raise RuntimeError(f"failed to load robot URDF from {MOVE_URDF}")
            dof_names = list(gym.get_asset_dof_names(robot_asset))
            if not dof_names:
                raise RuntimeError("robot asset contains no active DOFs")

            print("Preparing task1_2 box plans for all PCT targets...")
            prepared = _prepare_boxes(plan, dof_names)
            if not prepared:
                raise RuntimeError("PCT produced no executable boxes")

            env = gym.create_env(
                sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.8, 6.8, 3.0), 1
            )
            robot, boxes = _create_scene(gym, sim, env, robot_asset, prepared)
            t12._configure_robot_dofs(gym, env, robot, dof_names)
            first_q = prepared[0].timeline[0][2]
            _set_robot_dof_state(gym, env, robot, first_q)
            gym.set_actor_dof_position_targets(env, robot, first_q.numpy())

            gym.prepare_sim(sim)
            root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
            robot_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
            box_indices = [
                gym.get_actor_index(env, box, gymapi.DOMAIN_SIM) for box in boxes
            ]

            if not headless:
                viewer = gym.create_viewer(sim, gymapi.CameraProperties())
                if viewer is None:
                    raise RuntimeError("failed to create Isaac Gym viewer")
                pc = pallet_center_near_table()
                gym.viewer_camera_look_at(
                    viewer, env,
                    gymapi.Vec3(pc[0] + 1.8, pc[1] - 2.2, 1.7),
                    gymapi.Vec3(pc[0], pc[1], 0.65),
                )

            global_frame = 0
            placed = []
            completed = 0
            for item, box, box_index in zip(prepared, boxes, box_indices):
                print(
                    f"\nmove_pro_start_box box={item.task.index} "
                    f"size={item.task.world_size} target={item.task.world_target} "
                    f"side={item.side} stand_off={item.stand_off:.2f}"
                )
                remaining = 0 if max_frames == 0 else max(0, max_frames - global_frame)
                if max_frames > 0 and remaining == 0:
                    break
                with _box_source(item.task.world_size):
                    global_frame, keep_running = _play_box(
                        gym, sim, env, robot, box, root_states,
                        robot_index, box_index, item, viewer, fast,
                        global_frame, remaining, placed,
                    )
                if not keep_running:
                    break
                completed += 1
                # 复测所有已放箱子相对各自目标的漂移，验证未被后续放置撞动。
                drifts = [
                    (idx, _distance(_actor_center(gym, env, boxes[box_indices.index(idx)]), tgt))
                    for idx, tgt in placed
                ]
                if drifts:
                    max_drift = max(d for _i, d in drifts)
                    print(f"move_pro_placed_drift after_box={item.task.index} "
                          f"n={len(drifts)} max_drift={max_drift:.4f}")
            print(f"move_pro_result completed={completed}/{len(prepared)} frames={global_frame}")
            return plan
        finally:
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
