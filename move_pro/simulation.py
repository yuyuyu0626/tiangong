"""move_pro 仿真 — 直接复用 move/grab_test_task 的完整逻辑。

除了用 BPP 决策替换固定放置目标外，所有代码（IK、
时间线构建、仿真主循环、attach/detach、碰撞处理）
均直接来自 move 项目，不做任何自定义重写。

工作原理：
1. BPP 决策为每个箱子确定放置位置
2. Monkey-patch grab_test.build_offline_test3_scene 为目标位置
3. 调用 grab_test_task 的完整流程播放单箱抓取→移动→放置
4. 循环下一个箱子
"""

from __future__ import annotations

# ⚠️ Isaac Gym 必须在 torch 之前导入（硬性要求）
from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402

import argparse
import math
import sys
from pathlib import Path

# 路径
_MOVE_PRO_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MOVE_PRO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

# BPP 决策（唯一的"新代码"）
from move_pro.integrator import MoveProIntegrator, MoveProPlan     # noqa: E402
from move_pro.config import pallet_center_world                     # noqa: E402

# ---- 以下全部来自 move 项目 ----
import move.grab_test as _gt                                         # noqa: E402
from move.grab_test import (                                        # noqa: E402
    SIM_PLACE_STAND_OFF, _target_world_center, MOVE_URDF,
    build_offline_test3_scene, OfflineTest3Scene,
)
from move.planning import (                                         # noqa: E402
    PALLET_SURFACE_Z, PALLET_SIZE,
    BoxPlacement, EmptySpace, StackScene,
    pallet_center_near_table,
)


def _make_bpp_scene(target_world_center, box_size_m, stand_off):
    """为 BPP 目标构造与 build_offline_test3_scene 兼容的 OfflineTest3Scene。"""
    sx, sy, sz = box_size_m
    pc = pallet_center_near_table()

    # 托盘局部坐标
    lx = target_world_center[0] - (pc[0] - PALLET_SIZE[0] * 0.5)
    ly = target_world_center[1] - (pc[1] - PALLET_SIZE[1] * 0.5)
    lz = target_world_center[2] - PALLET_SURFACE_Z
    target_min = (lx - sx * 0.5, ly - sy * 0.5, lz - sz * 0.5)
    target_box = BoxPlacement("bpp_target", target_min, box_size_m)

    # 虚拟底面（避免 grab_test 内部 max() 空序列报错）
    dummy = BoxPlacement("_gnd", (0.0, 0.0, 0.0),
                         (PALLET_SIZE[0], PALLET_SIZE[1], 0.001))
    stack_scene = StackScene(
        pallet_center=pc,
        pallet_size=PALLET_SIZE,
        pallet_surface_z=PALLET_SURFACE_Z,
        preset_boxes=(dummy,),
        next_box=target_box,
        empty_spaces=(),
        selected_leaf=EmptySpace(target_box.min_corner, target_box.max_corner),
    )

    final_pose, tmp_pose = _gt._solve_test3_root_pose(stack_scene, stand_off)
    waypoints = _gt._build_test3_waypoints(stack_scene, final_pose, tmp_pose)
    return OfflineTest3Scene(
        name="bpp_target", stack_scene=stack_scene,
        target_support_z=PALLET_SURFACE_Z,
        final_pose=final_pose, tmp_pose=tmp_pose, waypoints=waypoints,
    )


def run_single_box(box_size_m, target_world_center, stand_off, headless, fast):
    """对单个箱子运行一次完整的 grab_test_task 仿真。

    通过 monkey-patch build_offline_test3_scene 注入 BPP 目标。
    所有仿真逻辑完全由 move 项目代码驱动。
    """
    # 1. 构造 BPP 场景
    bpp_scene = _make_bpp_scene(target_world_center, box_size_m, stand_off)

    # 2. Monkey-patch: 使 grab_test.run() 使用 BPP 目标
    _original_build = _gt.build_offline_test3_scene
    _gt.build_offline_test3_scene = lambda stand_off=stand_off: bpp_scene

    try:
        # 3. 直接调用 grab_test_task.main() 的核心逻辑
        #    （不做任何自定义重写）
        _run_grab_test_task_single(headless=headless, stand_off=stand_off, fast=fast)
    finally:
        _gt.build_offline_test3_scene = _original_build


def _run_grab_test_task_single(headless, stand_off, fast):
    """一次完整的 grab_test_task 仿真——代码直接来自 move/tasks/grab_test_task.py。

    这是 grab_test_task.main() 的逐行复制，仅做必要的最小修改：
    - 去掉 argparse（参数由外部传入）
    - 添加 fast 模式支持
    - 去掉多箱循环（由外层 control）
    """
    # ---- 以下代码直接来自 grab_test_task.main() ----

    import numpy as np

    from move.tasks.grab_test_task import (
        _lerp_tensor, _smoothstep, _lerp_pose,
        _smooth_timeline_joint_steps,
        _expand_key_targets, _expand_centers,
        _reorder_targets, _pose_from_dict,
        _transform_local_point, _box_world_pose,
        _set_actor_root_pose,
        _set_task_collision_filters, _set_hand_box_collision_enabled,
        _actor_center, _actor_yaw_pitch,
        _quat_to_matrix, _rotate_yaw_pitch, _inverse_rotate_yaw_pitch,
        _palm_center, _finger_dir, _grasp_center, _grasp_theta,
        _add_vec, _sub_vec, _distance,
        _dof_error_summary, _lock_mobile_dof_state, _set_robot_dof_state,
        _create_actors,
        PICK_SEGMENT_FRAMES, MOVE_SETTLE_FRAMES,
        MOVE_APPROACH_FRAMES, MOVE_LIFT_FRAMES,
        PLACE_HANDOFF_FRAMES, PLACE_SEGMENT_FRAMES, FINAL_HOLD_FRAMES,
        ATTACH_AFTER_PICK_FRAMES,
    )
    from move.tasks.move_test1 import create_sim, load_robot_asset

    # 调用 grab_test.run() 获取计划
    report, plan = _gt.run(place_mode="move", stand_off=stand_off)
    if not report.pick_feasible or not report.place_feasible:
        print(f"  IK 不可行: pick={report.pick_max_error:.4f} place={report.place_max_error:.4f}")
        return

    scene = plan["scene"]
    pick_targets = plan["pick_targets"]
    place_targets = plan["place_targets"]
    move_approach_target = plan["move_approach_target"]
    move_lift_target = plan["move_lift_target"]
    root_route = [("", _pose_from_dict(d)) for _, d in plan["root_route"]]
    place_pose_path = plan.get("place_pose_path_world",
        [(l, c, 0.0) for l, c in plan.get("place_center_path_world", [])])
    place_center_path = [c for _, c, _ in place_pose_path]
    target_center = _target_world_center(scene)
    release_center = place_center_path[-1] if place_center_path else target_center
    pre_place_center = plan.get("place_handoff_center_world",
        place_center_path[0] if place_center_path else target_center)

    # 创建 Isaac Gym
    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    robot_asset = load_robot_asset(gym, sim)
    dof_names = list(gym.get_asset_dof_names(robot_asset))

    # Reorder targets
    pick_targets = _reorder_targets(pick_targets, list(plan["move_dof_names"]), dof_names)
    place_targets = _reorder_targets(place_targets, list(plan["place_dof_names"]), dof_names)
    release_target = _reorder_targets(plan["release_target"], list(plan["place_dof_names"]), dof_names)
    move_approach_target = _reorder_targets(move_approach_target, list(plan["move_dof_names"]), dof_names)
    move_lift_target = _reorder_targets(move_lift_target, list(plan["move_dof_names"]), dof_names)

    # Expand pick
    pick_dof_frames = _expand_key_targets(pick_targets, PICK_SEGMENT_FRAMES)
    pick_centers = [tuple(c) for c in plan["pick_box_centers"]]
    pick_box_frames = _expand_centers(pick_centers, PICK_SEGMENT_FRAMES)

    # Root frames
    root_frames_list = []
    for wp_start, wp_goal in zip(root_route, root_route[1:] if len(root_route) > 1 else ([], [])):
        if wp_start[1] != wp_goal[1]:
            from move.planning import sample_root_trajectory
            root_frames_list.extend(sample_root_trajectory(wp_start[1], wp_goal[1]))
    root_frames_list += [root_frames_list[-1]] * MOVE_SETTLE_FRAMES if root_frames_list else []

    # Move frames
    move_dof_frames, move_box_frames, move_box_thetas = [], [], []
    carry_start = pick_box_frames[-1] if pick_box_frames else (0, 0, 0)
    final_root = root_frames_list[-1] if root_frames_list else None
    if final_root is None:
        gym.destroy_sim(sim)
        return
    carry_local_final = (
        pre_place_center[0] - final_root.x,
        pre_place_center[1] - final_root.y,
        pre_place_center[2] - final_root.z)
    carry_local_approach = (
        carry_local_final[0], carry_local_final[1],
        float(plan.get("move_approach_box_center_z", target_center[2])) - final_root.z)
    ma_theta = float(plan.get("move_approach_box_theta", 0.0))
    mp_theta = float(plan.get("move_preplace_box_theta", ma_theta))

    for _, rp in enumerate(root_frames_list):
        move_dof_frames.append(pick_targets[-1])
        move_box_frames.append(_transform_local_point(rp, carry_start))
        move_box_thetas.append(0.0)
    for i in range(1, MOVE_APPROACH_FRAMES + 1):
        a = _smoothstep(i / MOVE_APPROACH_FRAMES)
        move_dof_frames.append(_lerp_tensor(pick_targets[-1], move_approach_target, a))
        cl = tuple(carry_start[j]*(1-a)+carry_local_approach[j]*a for j in range(3))
        move_box_frames.append(_transform_local_point(final_root, cl))
        move_box_thetas.append(ma_theta * a)
    for i in range(1, MOVE_LIFT_FRAMES + 1):
        a = _smoothstep(i / MOVE_LIFT_FRAMES)
        move_dof_frames.append(_lerp_tensor(move_approach_target, move_lift_target, a))
        cl = tuple(carry_local_approach[j]*(1-a)+carry_local_final[j]*a for j in range(3))
        move_box_frames.append(_transform_local_point(final_root, cl))
        move_box_thetas.append(ma_theta*(1-a)+mp_theta*a)

    # Place frames
    place_dof_frames = [place_targets[i].clone() for i in range(place_targets.shape[0])]
    place_pose_labeled = place_pose_path

    # ---- 构建时间线（直接来自 grab_test_task.main()） ----
    timeline = []
    for q, center in zip(pick_dof_frames, pick_box_frames):
        timeline.append(("pick", _gt.Pose(0, 0, 0, 0), q, center, 0.0))

    all_root_poses = list(root_frames_list) + [final_root]*(MOVE_APPROACH_FRAMES+MOVE_LIFT_FRAMES)
    for rp, q, c, th in zip(all_root_poses, move_dof_frames, move_box_frames, move_box_thetas):
        timeline.append(("move", rp, q, c, th))

    if place_dof_frames:
        me_q = move_dof_frames[-1] if move_dof_frames else pick_targets[-1]
        me_c = move_box_frames[-1] if move_box_frames else carry_start
        me_th = move_box_thetas[-1] if move_box_thetas else 0.0
        fc, fth = place_pose_labeled[0][1], place_pose_labeled[0][2]
        for i in range(1, PLACE_HANDOFF_FRAMES + 1):
            a = _smoothstep(i / PLACE_HANDOFF_FRAMES)
            c = tuple(me_c[j]*(1-a)+fc[j]*a for j in range(3))
            timeline.append(("place_handoff", final_root,
                           _lerp_tensor(me_q, place_dof_frames[0], a),
                           c, me_th*(1-a)+fth*a))
    for q, (_, c, th) in zip(place_dof_frames, place_pose_labeled):
        timeline.append(("place", final_root, q, c, th))

    timeline.append(("release", final_root, release_target, release_center, 0.0))
    for _ in range(FINAL_HOLD_FRAMES):
        timeline.append(("post_release", final_root, release_target, release_center, 0.0))
    timeline = _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)

    # ---- 创建场景（直接来自 grab_test_task.main()） ----
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(7, 6.5, 2.5), 1)
    robot, source = _create_actors(gym, sim, env, scene, robot_asset)

    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        if "xhand_" in name:
            props["stiffness"][i] = 120.0; props["damping"][i] = 12.0
            props["effort"][i] = max(float(props["effort"][i]), 35.0)
        else:
            props["stiffness"][i] = 520.0; props["damping"][i] = 52.0
            props["effort"][i] = max(float(props["effort"][i]), 180.0)
    gym.set_actor_dof_properties(env, robot, props)

    dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    dof_state["pos"] = timeline[0][2].numpy()
    dof_state["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot, timeline[0][2].numpy())

    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_idx = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
    source_idx = gym.get_actor_index(env, source, gymapi.DOMAIN_SIM)

    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer:
            gym.viewer_camera_look_at(viewer, env,
                gymapi.Vec3(2.0, -2.2, 1.45), gymapi.Vec3(0.0, 0.0, 0.80))

    print(f"  pick={report.pick_max_error:.4f} place={report.place_max_error:.4f} "
          f"target=({target_center[0]:.3f},{target_center[1]:.3f},{target_center[2]:.3f}) "
          f"frames={len(timeline)}")

    # ---- 仿真主循环（直接来自 grab_test_task.main()） ----
    frame = 0; attached = False; released = False
    attach_off = None; attach_yaw_off = 0.0; attach_th_off = 0.0

    try:
        while viewer is None or not gym.query_viewer_has_closed(viewer):
            idx = min(frame, len(timeline)-1)
            phase, rp, q, bc, bt = timeline[idx]

            # Attach
            if not attached and not released and frame >= ATTACH_AFTER_PICK_FRAMES:
                attached = True
                ac = _actor_center(gym, env, source)
                ay, ap = _actor_yaw_pitch(gym, env, source)
                gc = _grasp_center(gym, env, robot)
                gt_ = _grasp_theta(gym, env, robot, rp.yaw)
                attach_off = _inverse_rotate_yaw_pitch(_sub_vec(ac, gc), rp.yaw, gt_)
                attach_yaw_off = ay - rp.yaw; attach_th_off = ap - gt_
                _set_hand_box_collision_enabled(gym, env, robot, source, enabled=False)
                print(f"  [frame {frame}] ATTACH offset=({attach_off[0]:.3f},{attach_off[1]:.3f},{attach_off[2]:.3f})")

            # Release
            if attached and not released and phase == "release":
                gym.refresh_actor_root_state_tensor(sim)
                released = True; attached = False
                ar = _actor_center(gym, env, source)
                print(f"  [frame {frame}] RELEASE at=({ar[0]:.3f},{ar[1]:.3f},{ar[2]:.3f})")

            # Robot DOF + root
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(gym, sim, root_states, robot_idx, (rp.x, rp.y, rp.z), rp.yaw)
            _set_robot_dof_state(gym, env, robot, q)
            if phase == "pick":
                _lock_mobile_dof_state(gym, env, robot, dof_names)
            gym.set_actor_dof_position_targets(env, robot, q.numpy())

            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Kinematic attach
            if attached and attach_off is not None:
                gym.refresh_actor_root_state_tensor(sim)
                gc = _grasp_center(gym, env, robot)
                gt_ = _grasp_theta(gym, env, robot, rp.yaw)
                ac = _add_vec(gc, _rotate_yaw_pitch(attach_off, rp.yaw, gt_))
                ath = gt_ + attach_th_off
                _set_actor_root_pose(gym, sim, root_states, source_idx,
                                     ac, rp.yaw+attach_yaw_off, ath)

            if viewer:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                if not fast:
                    gym.sync_frame_time(sim)

            if frame % 120 == 0:
                ab = _actor_center(gym, env, source)
                print(f"  frame={frame} phase={phase} box=({ab[0]:.2f},{ab[1]:.2f},{ab[2]:.2f}) "
                      f"attached={attached} released={released}")

            frame += 1
            if frame >= len(timeline):
                break
    finally:
        ab = _actor_center(gym, env, source)
        err = _distance(ab, target_center)
        print(f"  完成: box=({ab[0]:.3f},{ab[1]:.3f},{ab[2]:.3f}) "
              f"target_err={err:.3f} attached={attached} released={released}")
        if viewer:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


# ---------------------------------------------------------------------------
# 多箱仿真（唯一真正的新代码）
# ---------------------------------------------------------------------------

class MoveProSimulator:
    """智能码垛仿真器 — 每个箱子独立运行一次完整的 grab_test_task 流程。"""

    def __init__(self, method="LSAH", container_size=(10, 10, 16),
                 stand_off=SIM_PLACE_STAND_OFF):
        self.method = method
        self.container_size = container_size
        self.stand_off = stand_off
        self.integrator = MoveProIntegrator(method=method, container_size=container_size,
                                            stand_off=stand_off)

    def run(self, box_sequence, sizes_are_pct=True, headless=False, fast=False):
        # 1. BPP 决策
        print("BPP 决策中 ...")
        plan = self.integrator.build_plan(box_sequence, compute_ik=False,
                                          sizes_are_pct=sizes_are_pct)
        print(plan.summary())

        placed = 0
        for bt in plan.box_tasks:
            if not bt.placement.feasible:
                continue

            box_sz = (bt.pct_size[0]*0.1, bt.pct_size[1]*0.1, bt.pct_size[2]*0.1)
            target = bt.world_target

            print(f"\n{'='*50}")
            print(f"  箱#{bt.index}: {bt.pct_size} → world({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
            print(f"{'='*50}")

            try:
                run_single_box(box_sz, target, self.stand_off, headless, fast)
                placed += 1
            except Exception as e:
                import traceback
                print(f"  箱#{bt.index} 仿真异常: {e}")
                traceback.print_exc()

        print(f"\n全部完成: {placed}/{plan.total_boxes} 箱")
        return plan
