#!/usr/bin/env python3
"""Online Tianyi robot palletizing task in one Isaac Gym episode.

This is the final-task path.  It does not use the keyframe renderer: boxes are
processed one at a time in a single sim, and PCT state is committed only after
robot execution validates the placement.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

try:
    from isaacgym import gymapi, gymtorch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Isaac Gym Python package is not importable. Use run_move_bpp_env.sh.") from exc

import imageio.v2 as imageio
import numpy as np
import torch

MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move.controllers.attached_box_controller import AttachedBoxController
from move.grab_test import GRASP_CONTACT_X_OFFSET, MOVE_URDF, SIM_PLACE_STAND_OFF, build_custom_stack_scene
from move.online_palletizing import BoxSpec, iter_online_items
from move.palletizing_runtime import (
    STAND_OFF_CANDIDATES,
    IK_CANDIDATE_LIMIT,
    CONTACT_GAP_LIMIT_M,
    PRE_ATTACH_DRIFT_LIMIT_M,
    PRE_ATTACH_YAW_LIMIT_RAD,
    PRE_ATTACH_CENTER_Z_LIMIT_M,
    PLACE_ERROR_LIMIT_M,
    YAW_ERROR_LIMIT_RAD,
    PLACE_ABOVE_XY_LIMIT_M,
    PLACE_ABOVE_YAW_LIMIT_RAD,
    PLACE_ABOVE_TILT_LIMIT_RAD,
    PLACE_ABOVE_MIN_BOTTOM_GAP_M,
    PLACE_TILT_WARNING_RAD,
    PLACE_DESCEND_XY_LIMIT_M,
    PLACE_DESCEND_XY_FAIL_LIMIT_M,
    PLACE_DESCEND_TILT_LIMIT_RAD,
    PLACE_DESCENT_SERVO_MAX_XY_M,
    PLACE_DESCENT_SERVO_MAX_YAW_RAD,
    PLACE_DESCENT_SERVO_STEP_XY_M,
    PLACE_DESCENT_SERVO_STEP_YAW_RAD,
    PLACE_DESCENT_SERVO_MAX_Z_M,
    PLACE_DESCENT_SERVO_STEP_Z_M,
    PLACE_SERVO_MAX_XY_M,
    PLACE_SERVO_MAX_YAW_RAD,
    PLACE_SERVO_STEP_XY_M,
    PLACE_SERVO_STEP_YAW_RAD,
    FINAL_PLACE_ERROR_LIMIT_M,
    PRE_RELEASE_ERROR_LIMIT_M,
    EARLY_RELEASE_ERROR_LIMIT_M,
    VERIFIED_KINEMATIC_ERROR_LIMIT_M,
    VERIFIED_KINEMATIC_YAW_LIMIT_RAD,
    BOTTOM_GAP_MIN_M,
    BOTTOM_GAP_MAX_M,
    FROZEN_AABB_OVERLAP_LIMIT_M,
    SOFT_OBSTACLE_OVERLAP_LIMIT_M,
    HARD_OBSTACLE_OVERLAP_LIMIT_M,
    SOFT_STACK_BOUNDARY_TOLERANCE_M,
    LOCAL_REPAIR_OVERLAP_LIMIT_M,
    LOCAL_REPAIR_MAX_XY_M,
    LOCAL_REPAIR_STEP_XY_M,
    LOCAL_REPAIR_BOTTOM_GAP_MIN_M,
    TARGET_OVERLAP_REPAIR_LIMIT_M,
    ACTUAL_AABB_INFLATION_M,
    STACK_INSIDE_GATE_TOLERANCE_M,
    SUPPORT_LAYER_TOLERANCE_M,
    SUPPORT_XY_OVERLAP_MIN_M,
    TARGET_ADJUST_MARGIN_M,
    TARGET_ADJUST_MAX_XY_M,
    STACK_BOUNDARY_MARGIN_M,
    CONTACT_EXPLOSION_ANGULAR_SPEED_RADPS,
    DEMO_YAW_ERROR_LIMIT_RAD,
    PLACE_HOLD_ATTACHED_FRAMES,
    SIDE_OPEN_NO_COLLISION_FRAMES,
    RETREAT_NO_COLLISION_FRAMES,
    POST_RELEASE_RETREAT_FRAMES,
    SETTLE_FRAMES,
    FINAL_LINEAR_SPEED_LIMIT_MPS,
    FINAL_ANGULAR_SPEED_LIMIT_RADPS,
    STACK_VOLUME_M3,
    DENSE_FILL_UTILIZATION_THRESHOLD,
    DENSE_FILL_COMMIT_ERROR_LIMIT_M,
    DENSE_FILL_ADAPTIVE_LIMITS,
    DEFAULT_MAX_CONSECUTIVE_NO_FIT_SKIPS,
    DEFAULT_MAX_INFINITE_ARRIVALS,
    DEFAULT_MAX_PCT_LEAF_CHECKS,
    DEFAULT_MAX_IK_LEAF_CHECKS,
    DEFAULT_FINE_FILL_UTILIZATION_THRESHOLD,
    DEFAULT_FINE_FILL_SKIP_THRESHOLD,
    FINE_FILL_SMALL_SIZES_M,
    FINE_FILL_TINY_SIZES_M,
    FINE_FILL_MICRO_SIZES_M,
    RETRYABLE_CANDIDATE_REASONS,
    _dense_fill_commit_error_limit,
    _sample_online_box_spec,
    ACTUAL_SCENE_INFEASIBLE_REASONS,
    ROBOT_NO_FEASIBLE_EXECUTION_REASONS,
    RUNTIME_TRACKING_REASONS,
    ExecutionResult,
    ExecutionCandidate,
    ActualPackingState,
    FrozenOverlapByRole,
)
from move.palletizing_geometry import (
    _box_aabb_xyzyaw,
    _aabb_overlap_depth,
    _boxplacement_overlap_depth,
    _build_actual_packing_state,
    _overlap_with_frozen_boxes,
    _overlap_with_frozen_boxes_by_role,
    _overlap_with_frozen_boxes_by_role_detail,
    _upper_frozen_cover,
    _completion_bucket,
    _place_descent_block_reason,
    _frozen_overlap_xy_repair_vector,
    _xy_overlap_depths,
    _xy_overlap_positive,
    _support_top_z_for_footprint,
    _local_center_to_world,
    _world_center_to_local,
    _boxplacement_from_frozen_pose,
    _frozen_box_metrics,
    _stack_bounds,
    _xy_half_extents_for_yaw,
    _aabb_inside_stack,
    _actor_inside_stack,
    _target_aabb_inside_stack,
    _actor_xy_inside_stack,
    _stack_boundary_xy_repair_vector,
    _target_from_robot_center,
    _adjust_robot_target_for_actual_frozen,
    _oriented_box_bottom_z,
    _matrix_yaw_pitch_roll,
    _aabb_interval_on_axis,
    _approach_blockage_penalty,
)
from move.palletizing_candidate_selection import _select_pct_execution_target
from move.palletizing_sim_helpers import (
    _actor_root_speeds,
    _build_timeline,
    _create_source_box,
    _create_static_scene,
    _fmt_mat,
    _fmt_vec,
    _freeze_placed_boxes,
    _offline_grasp_frame,
    _rotation_error_deg,
    _set_robot_box_collision_disabled,
    _setup_camera,
    _sync_sim_state,
    _write_trace_frame,
    _yaw_error_to_options,
)
from move.pct_policy_bridge import DEFAULT_PCT_ROOT, PCTOnlineController, PctCandidate, PctPlacement
from move.planning import BOX_POSE, PALLET_THICKNESS, TABLE_POSE, TABLE_SIZE, BoxPlacement, Pose
from move.robot_placement import wrap_to_pi
from move.tasks.grab_test_task import (
    FINAL_HOLD_FRAMES,
    MOVE_APPROACH_FRAMES,
    MOVE_LIFT_FRAMES,
    MOVE_SETTLE_FRAMES,
    PICK_SEGMENT_FRAMES,
    PLACE_HANDOFF_FRAMES,
    PLACE_SEGMENT_FRAMES,
    _actor_center,
    _actor_yaw_pitch,
    _apply_pick_finger_closure,
    _apply_timeline_hand_closure,
    _box_world_pose,
    _dof_error_summary,
    _grasp_center,
    _grasp_theta,
    _expand_centers,
    _expand_key_targets,
    _lerp_tensor,
    _lock_mobile_dof_state,
    _make_transform,
    _pose_from_dict,
    _reorder_targets,
    _set_actor_collision_filter,
    _set_actor_root_pose,
    _set_color,
    _set_robot_dof_state,
    _set_shape_friction,
    _set_task_collision_filters,
    _smooth_timeline_joint_steps,
    _smoothstep,
    _target_world_center,
    _transform_local_point,
    _with_hand_closure,
)
from move.tasks.move_test1 import BOX_MASS, PRESET_BOX_MASS, create_sim, load_robot_asset
from move.utils import LEFT_EE_LINK, PALM_CENTER_Z, PALM_SURFACE_X, RIGHT_EE_LINK, UrdfKinematics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online robot palletizing in one Isaac Gym episode.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pct-root", type=Path, default=DEFAULT_PCT_ROOT)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-video", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=Path("/2024233240/move/outputs/online_robot_palletizing/metrics.json"))
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=368)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument("--out-trace", type=Path, default=None)
    parser.add_argument("--trace-every", type=int, default=4)
    parser.add_argument("--max-sim-frames-per-item", type=int, default=0)
    parser.add_argument("--place-release-height", type=float, default=0.004)
    parser.add_argument(
        "--post-place-mode",
        choices=("verified_kinematic_freeze", "dynamic_settle_then_freeze"),
        default="verified_kinematic_freeze",
        help=(
            "Placement handling after pre-release geometry passes. "
            "verified_kinematic_freeze freezes immediately at the validated pose; "
            "dynamic_settle_then_freeze releases the current box as a dynamic body, "
            "lets it settle, then freezes the settled pose if validation succeeds."
        ),
    )
    parser.add_argument("--bottom-gap-min", type=float, default=BOTTOM_GAP_MIN_M)
    parser.add_argument("--bottom-gap-max", type=float, default=BOTTOM_GAP_MAX_M)
    parser.add_argument(
        "--obstacle-overlap-max",
        type=float,
        default=SOFT_OBSTACLE_OVERLAP_LIMIT_M,
        help="Maximum non-support frozen-box AABB overlap tolerated at verified placement. Larger overlaps still reject the execution candidate.",
    )
    parser.add_argument("--pct-top-k", type=int, default=8)
    parser.add_argument(
        "--target-utilization",
        type=float,
        default=0.0,
        help="Stop after a successful placement once actual frozen-box utilization reaches this value. 0 disables this stop condition.",
    )
    parser.add_argument(
        "--max-consecutive-no-fit-skips",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_NO_FIT_SKIPS,
        help="With max-items=0, skip no-fit arrivals and stop only after this many consecutive skipped items.",
    )
    parser.add_argument(
        "--max-arrivals",
        type=int,
        default=DEFAULT_MAX_INFINITE_ARRIVALS,
        help="Safety cap for generated arrivals when max-items=0. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-pct-leaf-checks",
        type=int,
        default=DEFAULT_MAX_PCT_LEAF_CHECKS,
        help="Maximum ranked PCT leaves to geometrically inspect per arrival. 0 scans all valid leaves.",
    )
    parser.add_argument(
        "--max-ik-leaf-checks",
        type=int,
        default=DEFAULT_MAX_IK_LEAF_CHECKS,
        help="Maximum geometrically plausible leaves to send through the expensive robot IK planner per arrival. 0 disables this budget.",
    )
    parser.add_argument(
        "--fine-fill",
        dest="fine_fill",
        action="store_true",
        default=True,
        help="When max-items=0 and target utilization is set, bias dense late-stage arrivals toward small legal sizes.",
    )
    parser.add_argument(
        "--no-fine-fill",
        dest="fine_fill",
        action="store_false",
        help="Disable dense late-stage fine-fill arrival sampling.",
    )
    parser.add_argument(
        "--fine-fill-utilization-threshold",
        type=float,
        default=DEFAULT_FINE_FILL_UTILIZATION_THRESHOLD,
        help="Actual utilization at which fine-fill sampling starts.",
    )
    parser.add_argument(
        "--fine-fill-skip-threshold",
        type=int,
        default=DEFAULT_FINE_FILL_SKIP_THRESHOLD,
        help="Consecutive no-fit skipped arrivals that trigger fine-fill sampling even before the utilization threshold.",
    )
    parser.add_argument(
        "--debug-metrics",
        type=Path,
        default=None,
        help="Optional path for full debug metrics. The main metrics file stays compact for delivery.",
    )
    args = parser.parse_args()

    pct = PCTOnlineController(args.model_path, args.pct_root, args.device)
    pct.reset(args.seed)
    placed: list[BoxPlacement] = []
    results: list[ExecutionResult] = []
    skipped_no_fit_results: list[ExecutionResult] = []
    total_arrivals = 0
    consecutive_no_fit_skips = 0
    item_sample_mode_counts: dict[str, int] = {}
    item_sample_mode_by_index: dict[int, str] = {}
    completion_reason = "running"

    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    robot_asset = load_robot_asset(gym, sim)

    # Build the static scene from an empty stack. Later target scenes reuse the same pallet geometry.
    empty_candidate = PctPlacement(0, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.0, 0.0, 0.0), (0.1, 0.1, 0.1), 0, 0.0)
    empty_target = BoxPlacement("dummy", empty_candidate.min_corner_m, empty_candidate.placed_size_m)
    empty_scene = build_custom_stack_scene("online_static", tuple(), empty_target, stand_off=SIM_PLACE_STAND_OFF)
    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.5, 6.5, 2.5), 1)
    robot = _create_static_scene(gym, sim, env, robot_asset, empty_scene.stack_scene)
    dof_names = list(gym.get_asset_dof_names(robot_asset))
    offline_kin = UrdfKinematics(MOVE_URDF)

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

    camera = _setup_camera(gym, env, args.camera_width, args.camera_height, empty_scene.stack_scene) if args.record_video else None
    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(2.0, -2.2, 1.45), gymapi.Vec3(0.0, 0.0, 0.80))

    video_frames = []
    box_records: list[dict] = []
    total_frames = 0
    trace_fh = None
    if args.out_trace is not None:
        args.out_trace.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = args.out_trace.open("w", encoding="utf-8")
    try:
        item_iter = iter_online_items(args.seed, args.max_items) if args.max_items > 0 else None
        item_rng = random.Random(args.seed)
        next_item_index = 0
        retry_box_spec: BoxSpec | None = None
        retry_excluded_leaves: dict[int, set[int]] = {}
        while True:
            if retry_box_spec is not None:
                box_spec = retry_box_spec
                retry_box_spec = None
                print(
                    "online_item_retry "
                    f"index={box_spec.item_index} excluded_leaves={sorted(retry_excluded_leaves.get(box_spec.item_index, set()))}",
                    flush=True,
                )
            else:
                if item_iter is not None:
                    try:
                        box_spec = next(item_iter)
                    except StopIteration:
                        break
                    sample_mode = "uniform_random"
                else:
                    current_utilization_for_sampling = float(_frozen_box_metrics(box_records, empty_scene.stack_scene)["utilization"])
                    box_spec, sample_mode = _sample_online_box_spec(
                        item_rng,
                        next_item_index,
                        current_utilization=current_utilization_for_sampling,
                        consecutive_no_fit_skips=consecutive_no_fit_skips,
                        fine_fill_enabled=bool(args.fine_fill and args.target_utilization > 0.0),
                        fine_fill_utilization_threshold=float(args.fine_fill_utilization_threshold),
                        fine_fill_skip_threshold=int(args.fine_fill_skip_threshold),
                    )
                    next_item_index += 1
                    item_sample_mode_by_index[int(box_spec.item_index)] = sample_mode
                    item_sample_mode_counts[sample_mode] = item_sample_mode_counts.get(sample_mode, 0) + 1
            sample_mode = item_sample_mode_by_index.get(int(box_spec.item_index), "uniform_random")
            print(
                f"online_item_arrived index={box_spec.item_index} size={box_spec.original_size_m} "
                f"sample_mode={sample_mode}",
                flush=True,
            )
            total_arrivals += 1
            if args.max_items <= 0 and args.max_arrivals > 0 and total_arrivals > int(args.max_arrivals):
                completion_reason = "max_arrivals_reached"
                print(
                    "online_stop "
                    f"reason={completion_reason} max_arrivals={args.max_arrivals} "
                    f"placed_count={sum(1 for r in results if r.success)} "
                    f"skipped_no_fit_count={len(skipped_no_fit_results)}",
                    flush=True,
                )
                break
            current_utilization_for_selection = float(
                _frozen_box_metrics(box_records, empty_scene.stack_scene)["utilization"]
            )
            effective_max_ik_leaf_checks = int(args.max_ik_leaf_checks)
            if args.max_items <= 0 and args.target_utilization > 0.0 and effective_max_ik_leaf_checks > 0:
                if current_utilization_for_selection >= 0.90:
                    effective_max_ik_leaf_checks = max(effective_max_ik_leaf_checks, 32)
                elif consecutive_no_fit_skips >= 3:
                    effective_max_ik_leaf_checks = min(max(effective_max_ik_leaf_checks, 8), 12)
                elif current_utilization_for_selection >= 0.70:
                    effective_max_ik_leaf_checks = max(effective_max_ik_leaf_checks, 16)
                elif consecutive_no_fit_skips >= 1:
                    effective_max_ik_leaf_checks = max(effective_max_ik_leaf_checks, 12)
                if effective_max_ik_leaf_checks != int(args.max_ik_leaf_checks):
                    print(
                        "online_adaptive_ik_budget "
                        f"item={box_spec.item_index} utilization={current_utilization_for_selection:.4f} "
                        f"consecutive_no_fit_skips={consecutive_no_fit_skips} "
                        f"base={int(args.max_ik_leaf_checks)} effective={effective_max_ik_leaf_checks}",
                        flush=True,
                    )
            pct_selection = _select_pct_execution_target(
                pct,
                box_spec,
                placed,
                box_records,
                empty_scene.stack_scene,
                args.place_release_height,
                args.pct_top_k,
                int(args.max_pct_leaf_checks),
                int(effective_max_ik_leaf_checks),
                excluded_leaf_indices=retry_excluded_leaves.get(box_spec.item_index, set()),
            )
            if pct_selection is None:
                skipped_no_fit_results.append(
                    ExecutionResult(False, "skipped_no_pct_leaf", box_spec.item_index, None, None, None, None, 0)
                )
                consecutive_no_fit_skips += 1
                retry_excluded_leaves.pop(box_spec.item_index, None)
                print(
                    "online_item_skipped "
                    f"item={box_spec.item_index} reason=skipped_no_pct_leaf "
                    f"consecutive_no_fit_skips={consecutive_no_fit_skips} "
                    f"skipped_no_fit_count={len(skipped_no_fit_results)} "
                    f"placed_count={sum(1 for r in results if r.success)}",
                    flush=True,
                )
                if (
                    args.max_items <= 0
                    and args.max_consecutive_no_fit_skips > 0
                    and consecutive_no_fit_skips >= int(args.max_consecutive_no_fit_skips)
                ):
                    completion_reason = "consecutive_no_fit_limit_reached"
                    print(
                        "online_stop "
                        f"reason={completion_reason} consecutive_no_fit_skips={consecutive_no_fit_skips} "
                        f"placed_count={sum(1 for r in results if r.success)} "
                        f"skipped_no_fit_count={len(skipped_no_fit_results)}",
                        flush=True,
                    )
                    break
                continue
            (
                candidate,
                pct_first_candidate,
                robot_target,
                target_adjustment_diag,
                plan_candidates,
                pct_first_candidate_supported,
                pct_fallback_used,
                pct_fallback_reason,
                pct_prefilter_rejections,
            ) = pct_selection
            if candidate is None or robot_target is None or target_adjustment_diag is None or not plan_candidates:
                skip_reason = f"skipped_{pct_fallback_reason or 'no_execution_candidates'}"
                skipped_no_fit_results.append(
                    ExecutionResult(
                        False,
                        skip_reason,
                        box_spec.item_index,
                        asdict(pct_first_candidate),
                        None,
                        None,
                        None,
                        0,
                        pct_first_candidate_supported=pct_first_candidate_supported,
                        pct_fallback_used=pct_fallback_used,
                        pct_fallback_reason=pct_fallback_reason,
                        pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                        pct_selected_leaf_index=pct_first_candidate.action_leaf_index,
                    )
                )
                consecutive_no_fit_skips += 1
                retry_excluded_leaves.pop(box_spec.item_index, None)
                print(
                    "online_item_skipped "
                    f"item={box_spec.item_index} reason={skip_reason} "
                    f"consecutive_no_fit_skips={consecutive_no_fit_skips} "
                    f"skipped_no_fit_count={len(skipped_no_fit_results)} "
                    f"placed_count={sum(1 for r in results if r.success)} "
                    f"pct_prefilter_rejections={json.dumps(pct_prefilter_rejections, separators=(',', ':'))}",
                    flush=True,
                )
                if (
                    args.max_items <= 0
                    and args.max_consecutive_no_fit_skips > 0
                    and consecutive_no_fit_skips >= int(args.max_consecutive_no_fit_skips)
                ):
                    completion_reason = "consecutive_no_fit_limit_reached"
                    print(
                        "online_stop "
                        f"reason={completion_reason} consecutive_no_fit_skips={consecutive_no_fit_skips} "
                        f"placed_count={sum(1 for r in results if r.success)} "
                        f"skipped_no_fit_count={len(skipped_no_fit_results)}",
                        flush=True,
                    )
                    break
                continue
            candidate_failures: list[dict] = []
            item_completed = False
            final_failure_result: ExecutionResult | None = None
            execution_size = tuple(float(v) for v in robot_target.execution_size_m)
            asset_size = tuple(float(v) for v in robot_target.actor_size_m)
            for candidate_index, (target, source_pose, source_yaw, source_pitch, target_yaw, report, plan, scene, exec_candidate) in enumerate(plan_candidates):
                print(
                    "online_plan_selected "
                    f"item={box_spec.item_index} candidate_index={candidate_index} candidate_count={len(plan_candidates)} "
                    f"orientation={robot_target.orientation_type} "
                    f"target_yaw={math.degrees(target_yaw):.2f}deg source_yaw={math.degrees(source_yaw):.2f}deg "
                    f"source_pitch={math.degrees(source_pitch):.2f}deg "
                    f"stand_off={exec_candidate.stand_off:.3f} "
                    f"approach_yaw={math.degrees(exec_candidate.approach_yaw):.2f}deg "
                    f"score={exec_candidate.score:.4f} "
                    f"corridor_penalty={float(exec_candidate.diagnostics.get('corridor_penalty', 0.0)):.4f} "
                    f"pick_err={report.pick_max_error:.4f} place_err={report.place_max_error:.4f} "
                    f"pick_feasible={report.pick_feasible} place_feasible={report.place_feasible}",
                    flush=True,
                )
                if not report.pick_feasible or not report.place_feasible:
                    reason = "ik_infeasible"
                    candidate_failures.append(
                        {
                            "candidate_index": candidate_index,
                            "pct_rank": target_adjustment_diag.get("pct_rank"),
                            "pct_leaf_index": candidate.action_leaf_index,
                            "reason": reason,
                            "approach_side": exec_candidate.diagnostics.get("approach_side"),
                            "approach_yaw": exec_candidate.approach_yaw,
                            "stand_off": exec_candidate.stand_off,
                            "target_yaw": target_yaw,
                            "score": exec_candidate.score,
                            "local_adjustment_xyz_m": [0.0, 0.0, 0.0],
                            "overlap_depth_m": None,
                            "bottom_gap_m": None,
                            "inside_1m_cube": None,
                            "diagnostics": exec_candidate.diagnostics,
                        }
                    )
                    if final_failure_result is None:
                        final_failure_result = ExecutionResult(
                            False,
                            reason,
                            box_spec.item_index,
                            asdict(candidate),
                            None,
                            None,
                            None,
                            0,
                            pct_first_candidate_supported=pct_first_candidate_supported,
                            pct_fallback_used=pct_fallback_used,
                            pct_fallback_reason=pct_fallback_reason,
                            pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                            pct_selected_leaf_index=candidate.action_leaf_index,
                        )
                    print(
                        "online_candidate_attempt_result "
                        f"item={box_spec.item_index} candidate_index={candidate_index} success=False reason={reason} "
                        f"approach_yaw={math.degrees(exec_candidate.approach_yaw):.2f}deg stand_off={exec_candidate.stand_off:.3f}",
                        flush=True,
                    )
                    continue

                preflip_asset = abs(float(source_pitch)) > math.radians(45.0)
                sim_asset_size = execution_size if preflip_asset else asset_size
                sim_source_pitch = 0.0 if preflip_asset else float(source_pitch)
                source_actor = _create_source_box(gym, sim, env, sim_asset_size, source_pose, source_yaw, box_spec.item_index, pitch=sim_source_pitch)
                box_record = {
                    "item_index": box_spec.item_index,
                    "actor": source_actor,
                    "actor_index": gym.get_actor_index(env, source_actor, gymapi.DOMAIN_SIM),
                    "asset_size_m": asset_size,
                    "sim_asset_size_m": sim_asset_size,
                    "size_m": execution_size,
                    "source_pitch": float(source_pitch),
                    "sim_source_pitch": float(sim_source_pitch),
                    "state": "SOURCE_DYNAMIC",
                }
                box_records.append(box_record)
                pct_payload = {
                    "target_center_m": list(robot_target.target_center_m),
                    "target_world_center_m": list(_target_world_center(scene)),
                    "target_yaw": target_yaw,
                    "target_yaw_options": list(robot_target.target_yaw_options),
                    "source_yaw": source_yaw,
                    "source_pitch": source_pitch,
                    "sim_source_pitch": sim_source_pitch,
                    "placed_size_m": list(robot_target.target_aabb_size_m),
                    "actor_size_m": list(robot_target.actor_size_m),
                    "execution_size_m": list(robot_target.execution_size_m),
                    "sim_asset_size_m": list(sim_asset_size),
                    "orientation_type": robot_target.orientation_type,
                    "leaf_index": robot_target.pct_leaf_index,
                    "pct_target_center_m": target_adjustment_diag["pct_target_center_m"],
                    "execution_target_center_m": target_adjustment_diag["execution_target_center_m"],
                    "target_adjustment_xy_m": target_adjustment_diag["target_adjustment_xy_m"],
                    "target_adjustment_dx_m": target_adjustment_diag["target_adjustment_dx_m"],
                    "target_adjustment_dy_m": target_adjustment_diag["target_adjustment_dy_m"],
                    "execution_candidate": {
                        "target_center": list(exec_candidate.target_center),
                        "target_yaw": exec_candidate.target_yaw,
                        "target_aabb_size": list(exec_candidate.target_aabb_size),
                        "actor_size": list(exec_candidate.actor_size),
                        "execution_size": list(exec_candidate.execution_size),
                        "approach_yaw": exec_candidate.approach_yaw,
                        "final_root": list(exec_candidate.final_root),
                        "stand_off": exec_candidate.stand_off,
                        "source_yaw": exec_candidate.source_yaw,
                        "source_pitch": exec_candidate.source_pitch,
                        "score": exec_candidate.score,
                        "reject_reason": exec_candidate.reject_reason,
                        "diagnostics": exec_candidate.diagnostics,
                    },
                }
                _set_task_collision_filters(gym, env, robot, source_actor)
                source_index = gym.get_actor_index(env, source_actor, gymapi.DOMAIN_SIM)
                root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
                timeline = _build_timeline(plan, dof_names)
                initial_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
                initial_state["pos"] = timeline[0][2].numpy()
                initial_state["vel"].fill(0.0)
                gym.set_actor_dof_states(env, robot, initial_state, gymapi.STATE_ALL)
                gym.set_actor_dof_position_targets(env, robot, timeline[0][2].numpy())
                controller = AttachedBoxController(gym, sim, env, robot, source_actor, execution_size)
                source_nominal_center = tuple(float(v) for v in source_pose)
                source_nominal_yaw = float(source_yaw)
                pre_attach_phases = {"pick:pick_pre_grasp", "pick:pick_straight_clamp", "pick:pick_compress", "pick:pick_compress_hold"}

                attached = False
                released = False
                pre_attach_gap = None
                item_failed_reason = None
                item_failed_place_error = None
                item_failed_yaw_error = None
                pre_release_error_value = None
                pre_release_yaw_error_value = None
                pre_release_pose = None
                pre_release_pitch_value = None
                pre_release_bottom_gap = None
                pre_release_overlap = None
                pre_release_support_top_z = None
                pre_release_actual_bottom_z = None
                pre_release_planned_center_z = None
                pre_release_error_xyz = None
                placement_mode = "not_released"
                freeze_pose_source = None
                contact_explosion = False
                release_root_pose = None
                release_q = None
                place_above_target_ok = False
                place_servo_dx = 0.0
                place_servo_dy = 0.0
                place_servo_dyaw = 0.0
                place_servo_dz = 0.0
                place_servo_latched = False
                place_servo_root_nominal = None
                place_servo_q_nominal = None
                descent_servo_latched = False
                descent_root_nominal = None
                descent_q_nominal = None
                descent_corr_dx = 0.0
                descent_corr_dy = 0.0
                descent_corr_dyaw = 0.0
                local_adjustment_xy_m = 0.0
                local_adjustment_z_m = 0.0
                local_adjusted = False
                item_frames = 0
                for frame, (phase, root_pose, q, _box_center, _box_theta) in enumerate(timeline):
                    if args.max_sim_frames_per_item and frame >= args.max_sim_frames_per_item:
                        break
                    raw_root_pose = root_pose
                    place_hold_servo_phase = phase == "place:place_above_target_hold"
                    descent_servo_phase = phase.startswith("place:place_descend_to_release")
                    post_descent_servo_phase = (
                        phase == "place:place_hold"
                        or phase == "place_hold_attached"
                        or phase == "pre_detach_sync"
                        or phase == "side_open_no_collision"
                    )
                    if place_hold_servo_phase and not place_servo_latched:
                        place_servo_root_nominal = raw_root_pose
                        place_servo_q_nominal = q.clone()
                        place_servo_dx = 0.0
                        place_servo_dy = 0.0
                        place_servo_dyaw = 0.0
                        place_servo_dz = 0.0
                        place_servo_latched = True
                        print(
                            "online_place_servo_latch "
                            f"item={box_spec.item_index} frame={frame} phase={phase} "
                            f"root_nominal=({raw_root_pose.x:.4f},{raw_root_pose.y:.4f},{raw_root_pose.z:.4f},{raw_root_pose.yaw:.4f})",
                            flush=True,
                        )
                    if place_hold_servo_phase and place_servo_latched:
                        root_pose = Pose(
                            raw_root_pose.x + place_servo_dx,
                            raw_root_pose.y + place_servo_dy,
                            raw_root_pose.z + place_servo_dz,
                            raw_root_pose.yaw + place_servo_dyaw,
                        )
                    elif descent_servo_phase:
                        if not descent_servo_latched:
                            descent_root_nominal = Pose(
                                raw_root_pose.x + place_servo_dx,
                                raw_root_pose.y + place_servo_dy,
                                raw_root_pose.z + place_servo_dz,
                                raw_root_pose.yaw + place_servo_dyaw,
                            )
                            descent_q_nominal = q.clone()
                            descent_corr_dx = 0.0
                            descent_corr_dy = 0.0
                            descent_corr_dyaw = 0.0
                            descent_servo_latched = True
                            print(
                                "online_place_descent_latch "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"root_nominal=({descent_root_nominal.x:.4f},{descent_root_nominal.y:.4f},{descent_root_nominal.z:.4f},{descent_root_nominal.yaw:.4f})",
                                flush=True,
                            )
                        root_pose = Pose(
                            descent_root_nominal.x + descent_corr_dx,
                            descent_root_nominal.y + descent_corr_dy,
                            raw_root_pose.z + place_servo_dz,
                            descent_root_nominal.yaw + descent_corr_dyaw,
                        )
                    elif post_descent_servo_phase and descent_servo_latched:
                        root_pose = Pose(
                            descent_root_nominal.x + descent_corr_dx,
                            descent_root_nominal.y + descent_corr_dy,
                            raw_root_pose.z + place_servo_dz,
                            descent_root_nominal.yaw + descent_corr_dyaw,
                        )
                    _sync_sim_state(gym, sim)
                    _set_actor_root_pose(gym, sim, root_states, robot_index, (root_pose.x, root_pose.y, root_pose.z), root_pose.yaw)
                    _set_robot_dof_state(gym, env, robot, q)
                    if phase.startswith("pick"):
                        _lock_mobile_dof_state(gym, env, robot, dof_names)
                    gym.set_actor_dof_position_targets(env, robot, q.numpy())
                    _freeze_placed_boxes(gym, sim, root_states, box_records)

                    gym.simulate(sim)
                    gym.fetch_results(sim, True)
                    _sync_sim_state(gym, sim)
                    # Re-apply the commanded robot state after the PhysX step
                    # before reading palm frames.  The box is still driven from
                    # the actual hand frame, but the trace controller no longer
                    # measures a transient drive lag as a grasp-frame failure.
                    _set_actor_root_pose(gym, sim, root_states, robot_index, (root_pose.x, root_pose.y, root_pose.z), root_pose.yaw)
                    _set_robot_dof_state(gym, env, robot, q)
                    gym.set_actor_dof_position_targets(env, robot, q.numpy())
                    _sync_sim_state(gym, sim)
                    _freeze_placed_boxes(gym, sim, root_states, box_records)
                    _sync_sim_state(gym, sim)

                    if (not attached) and (not released) and phase in pre_attach_phases:
                        actual_center = _actor_center(gym, env, source_actor)
                        actual_yaw, actual_pitch_pre = _actor_yaw_pitch(gym, env, source_actor)
                        gap = controller.contact_gap()
                        center_drift = math.sqrt(sum((actual_center[i] - source_nominal_center[i]) ** 2 for i in range(3)))
                        yaw_drift = abs(wrap_to_pi(actual_yaw - source_nominal_yaw))
                        preflip_mode = abs(float(source_pitch)) > math.radians(45.0)
                        yaw_gate_ok = True if preflip_mode else yaw_drift <= PRE_ATTACH_YAW_LIMIT_RAD
                        pre_attach_gap = gap.max_gap_m
                        print(
                            "online_pre_attach_diag "
                            f"item={box_spec.item_index} frame={frame} phase={phase} "
                            f"box_center=({actual_center[0]:.4f},{actual_center[1]:.4f},{actual_center[2]:.4f}) "
                            f"box_drift={center_drift:.4f} "
                            f"left_gap={gap.left_gap_m:.4f} right_gap={gap.right_gap_m:.4f} max_gap={gap.max_gap_m:.4f} "
                            f"centerline_x={gap.centerline_x_m:.4f} centerline_z={gap.centerline_z_m:.4f} "
                            f"centerline_x_target={GRASP_CONTACT_X_OFFSET:.4f} centerline_error={gap.centerline_error_m:.4f} "
                            f"box_yaw={math.degrees(actual_yaw):.2f}deg box_pitch={math.degrees(actual_pitch_pre):.2f}deg "
                            f"yaw_drift={math.degrees(yaw_drift):.2f}deg yaw_gate_ok={yaw_gate_ok}",
                            flush=True,
                        )
                        if center_drift > PRE_ATTACH_DRIFT_LIMIT_M:
                            item_failed_reason = "pre_attach_box_drift_too_large"
                            item_failed_place_error = center_drift
                            item_failed_yaw_error = yaw_drift
                            print(
                                "online_pre_attach_blocked "
                                f"item={box_spec.item_index} drift={center_drift:.4f} "
                                f"yaw_drift={math.degrees(yaw_drift):.2f}deg gap={gap.max_gap_m:.4f}",
                                flush=True,
                            )
                            break
                        center_z_ok = abs(gap.centerline_z_m) <= PRE_ATTACH_CENTER_Z_LIMIT_M
                        if gap.max_gap_m <= CONTACT_GAP_LIMIT_M and yaw_gate_ok and center_z_ok:
                            controller.attach_from_current_hand_frame(root_pose.yaw)
                            _set_robot_box_collision_disabled(gym, env, robot, source_actor)
                            controller.update_from_current_hand_frame(root_pose.yaw)
                            _sync_sim_state(gym, sim)
                            grasp_center = _grasp_center(gym, env, robot)
                            grasp_theta = _grasp_theta(gym, env, robot, root_pose.yaw)
                            grasp_pos, grasp_rot = controller.debug_grasp_frame()
                            box_pos, box_rot = controller.debug_box_pose_matrix()
                            box_in_grasp_pos, box_in_grasp_rot = controller.debug_box_in_grasp()
                            dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_POS)
                            actual_q = torch.tensor(dof_state["pos"], dtype=torch.float32)
                            offline_pos, offline_rot = _offline_grasp_frame(offline_kin, dof_names, actual_q, root_pose)
                            box_record["state"] = "HELD_ATTACHED"
                            attached = True
                            print(
                                "online_attach "
                                f"item={box_spec.item_index} phase={phase} gap={pre_attach_gap:.4f} drift={center_drift:.4f} "
                                f"actual_center=({actual_center[0]:.4f},{actual_center[1]:.4f},{actual_center[2]:.4f}) "
                                f"actual_yaw={math.degrees(actual_yaw):.2f}deg "
                                f"grasp_center=({grasp_center[0]:.4f},{grasp_center[1]:.4f},{grasp_center[2]:.4f}) "
                                f"grasp_theta={math.degrees(grasp_theta):.2f}deg "
                                f"root_yaw={math.degrees(root_pose.yaw):.2f}deg "
                                f"source_yaw={math.degrees(source_yaw):.2f}deg "
                                f"target_yaw={math.degrees(target_yaw):.2f}deg "
                                f"attach_offset_local=({controller.attach_offset_local[0]:.4f},{controller.attach_offset_local[1]:.4f},{controller.attach_offset_local[2]:.4f}) "
                                f"attach_yaw_offset={math.degrees(controller.attach_yaw_offset):.2f}deg "
                                f"centerline_x={gap.centerline_x_m:.4f} centerline_z={gap.centerline_z_m:.4f} "
                                f"hand_box_collision_disabled={controller.hand_box_collision_disabled()}",
                                flush=True,
                            )
                        elif gap.max_gap_m <= CONTACT_GAP_LIMIT_M and yaw_drift <= PRE_ATTACH_YAW_LIMIT_RAD and not center_z_ok:
                            print(
                                "online_pre_attach_centerline_blocked "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"centerline_z={gap.centerline_z_m:.4f} limit={PRE_ATTACH_CENTER_Z_LIMIT_M:.4f} "
                                f"centerline_x={gap.centerline_x_m:.4f} centerline_x_target={GRASP_CONTACT_X_OFFSET:.4f}",
                                flush=True,
                            )
                            print(
                                "online_attach_T "
                                f"item={box_spec.item_index} "
                                f"T_grasp_world_p={_fmt_vec(grasp_pos)} T_grasp_world_R={_fmt_mat(grasp_rot)} "
                                f"T_box_world_p={_fmt_vec(box_pos)} T_box_world_R={_fmt_mat(box_rot)} "
                                f"T_box_in_grasp_p={_fmt_vec(box_in_grasp_pos)} T_box_in_grasp_R={_fmt_mat(box_in_grasp_rot)} "
                                f"offline_grasp_p={_fmt_vec(offline_pos)} offline_grasp_R={_fmt_mat(offline_rot)} "
                                f"actual_offline_pos_err={float(np.linalg.norm(grasp_pos - offline_pos)):.4f} "
                                f"actual_offline_rot_err_deg={_rotation_error_deg(grasp_rot, offline_rot):.2f}",
                                flush=True,
                            )
                    if (not attached) and (not released) and phase.startswith("move:"):
                        item_failed_reason = "pre_attach_conditions_not_met"
                        item_failed_place_error = pre_attach_gap
                        item_failed_yaw_error = None
                        print(f"online_pre_attach_missing item={box_spec.item_index} phase={phase} last_gap={pre_attach_gap}", flush=True)
                        break

                    if attached:
                        controller.update_from_current_hand_frame(root_pose.yaw)
                        _sync_sim_state(gym, sim)
                        if frame % 60 == 0:
                            attached_center = _actor_center(gym, env, source_actor)
                            attached_yaw, _attached_pitch = _actor_yaw_pitch(gym, env, source_actor)
                            target_center_now = _target_world_center(scene)
                            pre_release_error = math.sqrt(sum((attached_center[i] - target_center_now[i]) ** 2 for i in range(3)))
                            pre_release_yaw_error = _yaw_error_to_options(attached_yaw, robot_target.target_yaw_options)
                            grasp_pos, grasp_rot = controller.debug_grasp_frame()
                            box_pos, box_rot = controller.debug_box_pose_matrix()
                            dof_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_POS)
                            actual_q = torch.tensor(dof_state["pos"], dtype=torch.float32)
                            offline_pos, offline_rot = _offline_grasp_frame(offline_kin, dof_names, actual_q, root_pose)
                            q_err, q_top = _dof_error_summary(gym, env, robot, q, dof_names)
                            print(
                                "online_attached_update "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"attached_center=({attached_center[0]:.4f},{attached_center[1]:.4f},{attached_center[2]:.4f}) "
                                f"attached_yaw={math.degrees(attached_yaw):.2f}deg "
                                f"target_center=({target_center_now[0]:.4f},{target_center_now[1]:.4f},{target_center_now[2]:.4f}) "
                                f"expected_center=({_box_center[0]:.4f},{_box_center[1]:.4f},{_box_center[2]:.4f}) "
                                f"target_yaw={math.degrees(target_yaw):.2f}deg "
                                f"pre_release_error={pre_release_error:.4f} "
                                f"pre_release_yaw_error={math.degrees(pre_release_yaw_error):.2f}deg "
                                f"q_err={q_err:.4f} top_q_err={q_top}",
                                flush=True,
                            )
                            actual_offline_pos_err = float(np.linalg.norm(grasp_pos - offline_pos))
                            actual_offline_rot_err = _rotation_error_deg(grasp_rot, offline_rot)
                            print(
                                "online_frame_consistency "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"actual_grasp_p={_fmt_vec(grasp_pos)} offline_grasp_p={_fmt_vec(offline_pos)} "
                                f"actual_offline_pos_err={actual_offline_pos_err:.4f} "
                                f"actual_offline_rot_err_deg={actual_offline_rot_err:.2f} "
                                f"box_p={_fmt_vec(box_pos)} box_R={_fmt_mat(box_rot)} "
                                f"expected_box_p=({_box_center[0]:.4f},{_box_center[1]:.4f},{_box_center[2]:.4f}) "
                                f"hand_box_collision_disabled={controller.hand_box_collision_disabled()}",
                                flush=True,
                            )
                            if (phase.startswith("move:") or phase.startswith("place")) and (actual_offline_pos_err > 0.75 or actual_offline_rot_err > 50.0 or q_err > 1.20):
                                item_failed_reason = "hand_frame_tracking_fail"
                                item_failed_place_error = actual_offline_pos_err
                                item_failed_yaw_error = pre_release_yaw_error
                                print(
                                    "online_candidate_rejected "
                                    f"item={box_spec.item_index} reason=hand_frame_tracking_fail "
                                    f"phase={phase} frame={frame} "
                                    f"actual_offline_pos_err={actual_offline_pos_err:.4f} "
                                    f"actual_offline_rot_err_deg={actual_offline_rot_err:.2f} q_err={q_err:.4f} "
                                    f"approach_yaw={math.degrees(exec_candidate.approach_yaw):.2f}deg "
                                    f"stand_off={exec_candidate.stand_off:.3f}",
                                    flush=True,
                                )
                                break
                    place_move_above_phase = phase.startswith("place:place_move_xy_above_target")
                    place_above_phase = phase == "place:place_above_target_hold"
                    descent_release_phase = phase.startswith("place:place_descend_to_release")
                    release_check_phase = attached and (
                        phase == "pre_detach_sync"
                        or phase.startswith("place:place_move_to_release")
                        or place_move_above_phase
                        or place_above_phase
                        or descent_release_phase
                        or phase == "place:place_hold"
                        or phase == "place_hold_attached"
                    )
                    if release_check_phase:
                        controller.update_from_current_hand_frame(root_pose.yaw)
                        _sync_sim_state(gym, sim)
                        actual = _actor_center(gym, env, source_actor)
                        actual_yaw, _actual_pitch = _actor_yaw_pitch(gym, env, source_actor)
                        target_center_now = _target_world_center(scene)
                        error_xyz = tuple(float(actual[i] - target_center_now[i]) for i in range(3))
                        pre_release_error = math.sqrt(sum(v * v for v in error_xyz))
                        pre_release_yaw_error = _yaw_error_to_options(actual_yaw, robot_target.target_yaw_options)
                        box_pos_matrix, box_rot_matrix = controller.debug_box_pose_matrix()
                        matrix_yaw, actual_pitch, actual_roll = _matrix_yaw_pitch_roll(box_rot_matrix)
                        xy_error = math.sqrt(error_xyz[0] * error_xyz[0] + error_xyz[1] * error_xyz[1])
                        tilt_abs = max(abs(actual_pitch), abs(actual_roll))
                        target_footprint_yaw = float(target_yaw)
                        support_top_z = _support_top_z_for_footprint(
                            target_center_now,
                            target_footprint_yaw,
                            execution_size,
                            box_records,
                            box_spec.item_index,
                            scene.stack_scene.pallet_surface_z,
                        )
                        box_bottom_z = _oriented_box_bottom_z(box_pos_matrix, box_rot_matrix, execution_size)
                        bottom_gap = box_bottom_z - support_top_z
                        planned_release_center_z = float(plan.get("release_box_center_z", float("nan")))
                        overlap = _overlap_with_frozen_boxes(actual, actual_yaw, execution_size, box_records, box_spec.item_index)
                        support_overlap, obstacle_overlap = _overlap_with_frozen_boxes_by_role(
                            actual,
                            actual_yaw,
                            execution_size,
                            box_records,
                            box_spec.item_index,
                            support_top_z,
                        )
                        stack_inside, stack_clearance = _actor_inside_stack(actual, actual_yaw, execution_size, scene.stack_scene, margin=0.0)
                        stack_xy_inside, stack_xy_clearance = _actor_xy_inside_stack(actual, actual_yaw, execution_size, scene.stack_scene, margin=0.0)
                        supported_freeze_center = (
                            float(actual[0]),
                            float(actual[1]),
                            float(support_top_z + execution_size[2] * 0.5),
                        )
                        supported_freeze_inside, supported_freeze_clearance = _actor_inside_stack(
                            supported_freeze_center,
                            actual_yaw,
                            execution_size,
                            scene.stack_scene,
                            margin=0.0,
                        )
                        target_release_inside, target_release_clearance = _actor_inside_stack(
                            target_center_now,
                            target_yaw,
                            execution_size,
                            scene.stack_scene,
                            margin=0.0,
                        )
                        release_linear_speed, release_angular_speed = _actor_root_speeds(gym, sim, root_states, source_index)
                        final_release_phase = phase == "pre_detach_sync"
                        release_error_limit = PRE_RELEASE_ERROR_LIMIT_M if (final_release_phase or descent_release_phase) else EARLY_RELEASE_ERROR_LIMIT_M
                        pose_ok = pre_release_error <= release_error_limit and pre_release_yaw_error <= YAW_ERROR_LIMIT_RAD
                        bottom_gap_ok = args.bottom_gap_min <= bottom_gap <= args.bottom_gap_max
                        overlap_ok = obstacle_overlap <= args.obstacle_overlap_max
                        geometry_ok = pose_ok and bottom_gap_ok and overlap_ok and supported_freeze_inside
                        release_window_phase = (
                            phase.startswith("place:place_descend_to_release")
                            or phase == "place:place_hold"
                            or phase == "place_hold_attached"
                            or final_release_phase
                        )
                        release_ok = geometry_ok and release_window_phase
                        if geometry_ok or final_release_phase or (frame % 30 == 0):
                            print(
                                "online_place_geom "
                                f"item={box_spec.item_index} frame={frame} phase_name={phase} "
                                f"desired_box_center=({_box_center[0]:.4f},{_box_center[1]:.4f},{_box_center[2]:.4f}) "
                                f"actual_box_center=({actual[0]:.4f},{actual[1]:.4f},{actual[2]:.4f}) "
                                f"target_center=({target_center_now[0]:.4f},{target_center_now[1]:.4f},{target_center_now[2]:.4f}) "
                                f"error_xyz=({error_xyz[0]:.4f},{error_xyz[1]:.4f},{error_xyz[2]:.4f}) "
                                f"xy_error={xy_error:.4f} "
                                f"yaw={math.degrees(matrix_yaw):.2f}deg pitch={math.degrees(actual_pitch):.2f}deg roll={math.degrees(actual_roll):.2f}deg "
                                f"yaw_error={math.degrees(pre_release_yaw_error):.2f}deg target_yaw={math.degrees(target_yaw):.2f}deg "
                                f"target_center_z={target_center_now[2]:.4f} planned_release_center_z={planned_release_center_z:.4f} "
                                f"actual_pre_release_center_z={actual[2]:.4f} "
                                f"actual_bottom_z={box_bottom_z:.4f} support_top_z={support_top_z:.4f} "
                                f"bottom_gap={bottom_gap:.4f} overlap={overlap:.4f} "
                                f"support_overlap={support_overlap:.4f} obstacle_overlap={obstacle_overlap:.4f} "
                                f"stack_inside={stack_inside} stack_clearance={stack_clearance:.4f} "
                                f"supported_freeze_inside={supported_freeze_inside} supported_freeze_clearance={supported_freeze_clearance:.4f} "
                                f"target_release_inside={target_release_inside} target_release_clearance={target_release_clearance:.4f} "
                                f"stack_xy_inside={stack_xy_inside} stack_xy_clearance={stack_xy_clearance:.4f} "
                                f"root_pose=({root_pose.x:.4f},{root_pose.y:.4f},{root_pose.z:.4f},{root_pose.yaw:.4f}) "
                                f"linear_speed={release_linear_speed:.4f} angular_speed={release_angular_speed:.4f}",
                                flush=True,
                            )
                        if place_above_phase:
                            if obstacle_overlap > FROZEN_AABB_OVERLAP_LIMIT_M and obstacle_overlap <= LOCAL_REPAIR_OVERLAP_LIMIT_M:
                                repair_dx, repair_dy, repair_initial, repair_remaining = _frozen_overlap_xy_repair_vector(
                                    actual,
                                    actual_yaw,
                                    execution_size,
                                    box_records,
                                    box_spec.item_index,
                                    max_total_xy=LOCAL_REPAIR_MAX_XY_M,
                                    support_top_z=support_top_z,
                                )
                                repair_norm = math.hypot(repair_dx, repair_dy)
                                if repair_norm > 1e-6:
                                    scale = min(1.0, LOCAL_REPAIR_STEP_XY_M / repair_norm)
                                    step_x = repair_dx * scale
                                    step_y = repair_dy * scale
                                    proposed_dx = place_servo_dx + step_x
                                    proposed_dy = place_servo_dy + step_y
                                    if math.hypot(proposed_dx, proposed_dy) <= LOCAL_REPAIR_MAX_XY_M:
                                        predicted_center = (actual[0] + step_x, actual[1] + step_y, actual[2])
                                        predicted_inside, _predicted_clearance = _actor_xy_inside_stack(
                                            predicted_center,
                                            actual_yaw,
                                            execution_size,
                                            scene.stack_scene,
                                            margin=0.0,
                                        )
                                        if predicted_inside:
                                            place_servo_dx = proposed_dx
                                            place_servo_dy = proposed_dy
                                            local_adjustment_xy_m = max(local_adjustment_xy_m, math.hypot(place_servo_dx, place_servo_dy))
                                            local_adjusted = True
                                            print(
                                                "online_place_local_repair_update "
                                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                                f"obstacle_overlap={obstacle_overlap:.4f} initial_overlap={repair_initial:.4f} "
                                                f"remaining_est={repair_remaining:.4f} "
                                                f"step=({step_x:.4f},{step_y:.4f}) "
                                                f"servo_xy=({place_servo_dx:.4f},{place_servo_dy:.4f})",
                                                flush=True,
                                            )
                            if (not stack_xy_inside or not supported_freeze_inside) and obstacle_overlap <= args.obstacle_overlap_max:
                                repair_dx, repair_dy, repair_norm = _stack_boundary_xy_repair_vector(
                                    actual,
                                    actual_yaw,
                                    execution_size,
                                    scene.stack_scene,
                                    max_total_xy=LOCAL_REPAIR_MAX_XY_M,
                                )
                                if repair_norm > 1e-6 and repair_norm <= LOCAL_REPAIR_MAX_XY_M:
                                    step_norm = math.hypot(repair_dx, repair_dy)
                                    scale = min(1.0, LOCAL_REPAIR_STEP_XY_M / max(step_norm, 1e-9))
                                    step_x = repair_dx * scale
                                    step_y = repair_dy * scale
                                    proposed_dx = place_servo_dx + step_x
                                    proposed_dy = place_servo_dy + step_y
                                    if math.hypot(proposed_dx, proposed_dy) <= LOCAL_REPAIR_MAX_XY_M:
                                        place_servo_dx = proposed_dx
                                        place_servo_dy = proposed_dy
                                        local_adjustment_xy_m = max(local_adjustment_xy_m, math.hypot(place_servo_dx, place_servo_dy))
                                        local_adjusted = True
                                        print(
                                            "online_place_boundary_repair_update "
                                            f"item={box_spec.item_index} frame={frame} phase={phase} "
                                            f"stack_clearance={stack_clearance:.4f} supported_clearance={supported_freeze_clearance:.4f} "
                                            f"repair=({repair_dx:.4f},{repair_dy:.4f}) step=({step_x:.4f},{step_y:.4f}) "
                                            f"servo_xy=({place_servo_dx:.4f},{place_servo_dy:.4f})",
                                            flush=True,
                                        )
                            yaw_signed_error = wrap_to_pi(target_yaw - matrix_yaw)
                            x_step = min(max((target_center_now[0] - actual[0]) * 0.20, -PLACE_SERVO_STEP_XY_M), PLACE_SERVO_STEP_XY_M)
                            y_step = min(max((target_center_now[1] - actual[1]) * 0.20, -PLACE_SERVO_STEP_XY_M), PLACE_SERVO_STEP_XY_M)
                            yaw_step = min(max(yaw_signed_error * 0.15, -PLACE_SERVO_STEP_YAW_RAD), PLACE_SERVO_STEP_YAW_RAD)
                            needs_edge_correction = (not supported_freeze_inside) and supported_freeze_clearance >= -LOCAL_REPAIR_MAX_XY_M
                            if (
                                xy_error > PLACE_ABOVE_XY_LIMIT_M * 0.5
                                or abs(yaw_signed_error) > math.radians(1.0)
                                or needs_edge_correction
                            ):
                                place_servo_dx = min(max(place_servo_dx + x_step, -PLACE_SERVO_MAX_XY_M), PLACE_SERVO_MAX_XY_M)
                                place_servo_dy = min(max(place_servo_dy + y_step, -PLACE_SERVO_MAX_XY_M), PLACE_SERVO_MAX_XY_M)
                                place_servo_dyaw = min(max(place_servo_dyaw + yaw_step, -PLACE_SERVO_MAX_YAW_RAD), PLACE_SERVO_MAX_YAW_RAD)
                                print(
                                    "online_place_servo_update "
                                    f"item={box_spec.item_index} frame={frame} phase={phase} "
                                    f"xy_error={xy_error:.4f} yaw_error={math.degrees(pre_release_yaw_error):.2f}deg "
                                    f"servo=({place_servo_dx:.4f},{place_servo_dy:.4f},{place_servo_dz:.4f},{math.degrees(place_servo_dyaw):.2f}deg)",
                                    flush=True,
                                )
                            if tilt_abs > PLACE_TILT_WARNING_RAD and frame % 30 == 0:
                                print(
                                    "online_place_tilt_warning "
                                    f"item={box_spec.item_index} frame={frame} phase={phase} "
                                    f"pitch={math.degrees(actual_pitch):.2f}deg roll={math.degrees(actual_roll):.2f}deg "
                                    f"limit_fail_deg=10.00",
                                    flush=True,
                                )
                            if (
                                xy_error <= PLACE_ABOVE_XY_LIMIT_M
                                and pre_release_yaw_error <= PLACE_ABOVE_YAW_LIMIT_RAD
                                and bottom_gap > PLACE_ABOVE_MIN_BOTTOM_GAP_M
                                and obstacle_overlap <= args.obstacle_overlap_max
                                and tilt_abs <= PLACE_ABOVE_TILT_LIMIT_RAD
                            ):
                                place_above_target_ok = True
                        elif place_move_above_phase and tilt_abs > PLACE_TILT_WARNING_RAD and frame % 30 == 0:
                            print(
                                "online_place_tilt_warning "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"pitch={math.degrees(actual_pitch):.2f}deg roll={math.degrees(actual_roll):.2f}deg "
                                f"limit_fail_deg=10.00",
                                flush=True,
                            )
                        if descent_release_phase:
                            if obstacle_overlap > FROZEN_AABB_OVERLAP_LIMIT_M and obstacle_overlap <= LOCAL_REPAIR_OVERLAP_LIMIT_M:
                                repair_dx, repair_dy, repair_initial, repair_remaining = _frozen_overlap_xy_repair_vector(
                                    actual,
                                    actual_yaw,
                                    execution_size,
                                    box_records,
                                    box_spec.item_index,
                                    max_total_xy=LOCAL_REPAIR_MAX_XY_M,
                                    support_top_z=support_top_z,
                                )
                                repair_norm = math.hypot(repair_dx, repair_dy)
                                if repair_norm > 1e-6:
                                    scale = min(1.0, LOCAL_REPAIR_STEP_XY_M / repair_norm)
                                    step_x = repair_dx * scale
                                    step_y = repair_dy * scale
                                    proposed_dx = descent_corr_dx + step_x
                                    proposed_dy = descent_corr_dy + step_y
                                    if math.hypot(proposed_dx, proposed_dy) <= LOCAL_REPAIR_MAX_XY_M:
                                        predicted_center = (actual[0] + step_x, actual[1] + step_y, actual[2])
                                        predicted_inside, _predicted_clearance = _actor_xy_inside_stack(
                                            predicted_center,
                                            actual_yaw,
                                            execution_size,
                                            scene.stack_scene,
                                            margin=0.0,
                                        )
                                        if predicted_inside:
                                            descent_corr_dx = proposed_dx
                                            descent_corr_dy = proposed_dy
                                            local_adjustment_xy_m = max(local_adjustment_xy_m, math.hypot(descent_corr_dx, descent_corr_dy))
                                            local_adjusted = True
                                            print(
                                                "online_place_descent_local_repair_update "
                                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                                f"obstacle_overlap={obstacle_overlap:.4f} initial_overlap={repair_initial:.4f} "
                                                f"remaining_est={repair_remaining:.4f} "
                                                f"step=({step_x:.4f},{step_y:.4f}) "
                                                f"descent_corr=({descent_corr_dx:.4f},{descent_corr_dy:.4f})",
                                                flush=True,
                                            )
                            yaw_signed_error = wrap_to_pi(target_yaw - matrix_yaw)
                            if (not stack_xy_inside or not supported_freeze_inside) and obstacle_overlap <= args.obstacle_overlap_max:
                                repair_dx, repair_dy, repair_norm = _stack_boundary_xy_repair_vector(
                                    actual,
                                    actual_yaw,
                                    execution_size,
                                    scene.stack_scene,
                                    max_total_xy=LOCAL_REPAIR_MAX_XY_M,
                                )
                                if repair_norm > 1e-6 and repair_norm <= LOCAL_REPAIR_MAX_XY_M:
                                    step_norm = math.hypot(repair_dx, repair_dy)
                                    scale = min(1.0, PLACE_DESCENT_SERVO_STEP_XY_M / max(step_norm, 1e-9))
                                    step_x = repair_dx * scale
                                    step_y = repair_dy * scale
                                    proposed_dx = descent_corr_dx + step_x
                                    proposed_dy = descent_corr_dy + step_y
                                    if math.hypot(proposed_dx, proposed_dy) <= LOCAL_REPAIR_MAX_XY_M:
                                        descent_corr_dx = proposed_dx
                                        descent_corr_dy = proposed_dy
                                        local_adjustment_xy_m = max(local_adjustment_xy_m, math.hypot(descent_corr_dx, descent_corr_dy))
                                        local_adjusted = True
                                        print(
                                            "online_place_descent_boundary_repair_update "
                                            f"item={box_spec.item_index} frame={frame} phase={phase} "
                                            f"stack_clearance={stack_clearance:.4f} supported_clearance={supported_freeze_clearance:.4f} "
                                            f"repair=({repair_dx:.4f},{repair_dy:.4f}) step=({step_x:.4f},{step_y:.4f}) "
                                            f"descent_corr=({descent_corr_dx:.4f},{descent_corr_dy:.4f})",
                                            flush=True,
                                        )
                            needs_descent_edge_correction = (not supported_freeze_inside) and supported_freeze_clearance >= -LOCAL_REPAIR_MAX_XY_M
                            if (
                                xy_error > PLACE_ABOVE_XY_LIMIT_M
                                or abs(yaw_signed_error) > math.radians(1.0)
                                or needs_descent_edge_correction
                            ):
                                x_step = min(max((target_center_now[0] - actual[0]) * 0.12, -PLACE_DESCENT_SERVO_STEP_XY_M), PLACE_DESCENT_SERVO_STEP_XY_M)
                                y_step = min(max((target_center_now[1] - actual[1]) * 0.12, -PLACE_DESCENT_SERVO_STEP_XY_M), PLACE_DESCENT_SERVO_STEP_XY_M)
                                yaw_step = min(max(yaw_signed_error * 0.10, -PLACE_DESCENT_SERVO_STEP_YAW_RAD), PLACE_DESCENT_SERVO_STEP_YAW_RAD)
                                descent_corr_dx = min(max(descent_corr_dx + x_step, -PLACE_DESCENT_SERVO_MAX_XY_M), PLACE_DESCENT_SERVO_MAX_XY_M)
                                descent_corr_dy = min(max(descent_corr_dy + y_step, -PLACE_DESCENT_SERVO_MAX_XY_M), PLACE_DESCENT_SERVO_MAX_XY_M)
                                descent_corr_dyaw = min(max(descent_corr_dyaw + yaw_step, -PLACE_DESCENT_SERVO_MAX_YAW_RAD), PLACE_DESCENT_SERVO_MAX_YAW_RAD)
                                print(
                                    "online_place_descent_servo_update "
                                    f"item={box_spec.item_index} frame={frame} phase={phase} "
                                    f"xy_error={xy_error:.4f} yaw_error={math.degrees(pre_release_yaw_error):.2f}deg "
                                    f"bottom_gap={bottom_gap:.4f} descent_corr=({descent_corr_dx:.4f},{descent_corr_dy:.4f},{math.degrees(descent_corr_dyaw):.2f}deg)",
                                    flush=True,
                                )
                            target_bottom_gap = 0.5 * (args.bottom_gap_min + args.bottom_gap_max)
                            if bottom_gap < args.bottom_gap_min and bottom_gap >= LOCAL_REPAIR_BOTTOM_GAP_MIN_M:
                                z_step = min((args.bottom_gap_min - bottom_gap) + 0.001, PLACE_DESCENT_SERVO_STEP_Z_M)
                                place_servo_dz = min(place_servo_dz + z_step, PLACE_DESCENT_SERVO_MAX_Z_M)
                                local_adjustment_z_m = max(local_adjustment_z_m, abs(place_servo_dz))
                                local_adjusted = local_adjusted or local_adjustment_z_m > 0.0
                                print(
                                    "online_place_descent_z_servo "
                                    f"item={box_spec.item_index} frame={frame} phase={phase} "
                                    f"mode=raise bottom_gap={bottom_gap:.4f} target_gap={target_bottom_gap:.4f} "
                                    f"z_step={z_step:.4f} place_servo_dz={place_servo_dz:.4f}",
                                    flush=True,
                                )
                            elif bottom_gap > args.bottom_gap_max:
                                z_step = min((bottom_gap - target_bottom_gap) * 0.50, PLACE_DESCENT_SERVO_STEP_Z_M)
                                place_servo_dz = max(place_servo_dz - z_step, -PLACE_DESCENT_SERVO_MAX_Z_M)
                                local_adjustment_z_m = max(local_adjustment_z_m, abs(place_servo_dz))
                                local_adjusted = local_adjusted or local_adjustment_z_m > 0.0
                                print(
                                    "online_place_descent_z_servo "
                                    f"item={box_spec.item_index} frame={frame} phase={phase} "
                                    f"mode=lower bottom_gap={bottom_gap:.4f} target_gap={target_bottom_gap:.4f} "
                                    f"z_step={z_step:.4f} place_servo_dz={place_servo_dz:.4f}",
                                    flush=True,
                                )
                        if descent_release_phase and not place_above_target_ok:
                            item_failed_reason = "place_above_target_not_aligned"
                            item_failed_place_error = pre_release_error
                            item_failed_yaw_error = pre_release_yaw_error
                            pre_release_error_value = pre_release_error
                            pre_release_yaw_error_value = pre_release_yaw_error
                            pre_release_bottom_gap = bottom_gap
                            pre_release_overlap = obstacle_overlap
                            pre_release_support_top_z = support_top_z
                            pre_release_actual_bottom_z = box_bottom_z
                            pre_release_planned_center_z = planned_release_center_z
                            pre_release_error_xyz = error_xyz
                            print(
                                "online_place_above_blocked "
                                f"item={box_spec.item_index} frame={frame} phase={phase} "
                                f"xy_error={xy_error:.4f} yaw_error={math.degrees(pre_release_yaw_error):.2f}deg "
                                f"pitch={math.degrees(actual_pitch):.2f}deg roll={math.degrees(actual_roll):.2f}deg "
                                f"servo=({place_servo_dx:.4f},{place_servo_dy:.4f},{place_servo_dz:.4f},{math.degrees(place_servo_dyaw):.2f}deg)",
                                flush=True,
                            )
                            break
                        descent_near_release_height = bottom_gap <= max(args.bottom_gap_max + 0.020, 0.030)
                        descent_stack_violation = descent_near_release_height and (
                            (not stack_xy_inside) or (not supported_freeze_inside)
                        )
                        descent_block_reason = _place_descent_block_reason(
                            xy_error=xy_error,
                            bottom_gap=bottom_gap,
                            obstacle_overlap=obstacle_overlap,
                            stack_xy_inside=stack_xy_inside,
                            supported_freeze_inside=supported_freeze_inside,
                            descent_near_release_height=descent_near_release_height,
                            tilt_abs=tilt_abs,
                            descent_corr_xy_m=math.hypot(descent_corr_dx, descent_corr_dy),
                            place_servo_dz=place_servo_dz,
                            bottom_gap_max=args.bottom_gap_max,
                        )
                        if descent_release_phase and descent_block_reason is not None:
                            item_failed_reason = descent_block_reason
                            item_failed_place_error = pre_release_error
                            item_failed_yaw_error = pre_release_yaw_error
                            pre_release_error_value = pre_release_error
                            pre_release_yaw_error_value = pre_release_yaw_error
                            pre_release_bottom_gap = bottom_gap
                            pre_release_overlap = obstacle_overlap
                            pre_release_support_top_z = support_top_z
                            pre_release_actual_bottom_z = box_bottom_z
                            pre_release_planned_center_z = planned_release_center_z
                            pre_release_error_xyz = error_xyz
                            print(
                                "online_place_descend_blocked "
                                f"item={box_spec.item_index} frame={frame} phase={phase} reason={descent_block_reason} "
                                f"xy_error={xy_error:.4f} bottom_gap={bottom_gap:.4f} overlap={overlap:.4f} "
                                f"support_overlap={support_overlap:.4f} obstacle_overlap={obstacle_overlap:.4f} "
                                f"support_overlap_allowed={support_overlap > 0.0} "
                                f"stack_inside={stack_inside} stack_clearance={stack_clearance:.4f} "
                                f"supported_freeze_inside={supported_freeze_inside} supported_freeze_clearance={supported_freeze_clearance:.4f} "
                                f"stack_xy_inside={stack_xy_inside} stack_xy_clearance={stack_xy_clearance:.4f} "
                                f"near_release_height={descent_near_release_height} "
                                f"pitch={math.degrees(actual_pitch):.2f}deg roll={math.degrees(actual_roll):.2f}deg "
                                f"servo=({place_servo_dx:.4f},{place_servo_dy:.4f},{place_servo_dz:.4f},{math.degrees(place_servo_dyaw):.2f}deg) "
                                f"descent_corr=({descent_corr_dx:.4f},{descent_corr_dy:.4f},{math.degrees(descent_corr_dyaw):.2f}deg) "
                                f"local_adjustment_xyz=({local_adjustment_xy_m:.4f},0.0000,{local_adjustment_z_m:.4f})",
                                flush=True,
                            )
                            break
                        if not release_ok:
                            if not final_release_phase:
                                pass
                            else:
                                item_failed_reason = "pre_release_pose_invalid"
                                item_failed_place_error = pre_release_error
                                item_failed_yaw_error = pre_release_yaw_error
                                pre_release_error_value = pre_release_error
                                pre_release_yaw_error_value = pre_release_yaw_error
                                pre_release_bottom_gap = bottom_gap
                                pre_release_overlap = obstacle_overlap
                                pre_release_support_top_z = support_top_z
                                pre_release_actual_bottom_z = box_bottom_z
                                pre_release_planned_center_z = planned_release_center_z
                                pre_release_error_xyz = error_xyz
                                print(
                                    "online_pre_release_blocked "
                                    f"item={box_spec.item_index} error={pre_release_error:.4f} "
                                    f"yaw_error={math.degrees(pre_release_yaw_error):.2f}deg "
                                    f"bottom_gap={bottom_gap:.4f} overlap={overlap:.4f} "
                                    f"support_overlap={support_overlap:.4f} obstacle_overlap={obstacle_overlap:.4f} "
                                    f"stack_inside={stack_inside} stack_clearance={stack_clearance:.4f} "
                                    f"supported_freeze_inside={supported_freeze_inside} supported_freeze_clearance={supported_freeze_clearance:.4f} "
                                    f"actual_bottom_z={box_bottom_z:.4f} support_top_z={support_top_z:.4f} error_xyz=({error_xyz[0]:.4f},{error_xyz[1]:.4f},{error_xyz[2]:.4f}) "
                                    f"linear_speed={release_linear_speed:.4f} angular_speed={release_angular_speed:.4f}",
                                    flush=True,
                                )
                                break
                        else:
                            pre_release_error_value = pre_release_error
                            pre_release_yaw_error_value = pre_release_yaw_error
                            pre_release_bottom_gap = bottom_gap
                            pre_release_overlap = obstacle_overlap
                            pre_release_support_top_z = support_top_z
                            pre_release_actual_bottom_z = box_bottom_z
                            pre_release_planned_center_z = planned_release_center_z
                            pre_release_error_xyz = error_xyz
                            level_freeze_center = supported_freeze_center
                            pre_release_pose = (level_freeze_center[0], level_freeze_center[1], level_freeze_center[2], float(actual_yaw))
                            pre_release_pitch_value = float(actual_pitch)
                            controller.update_from_current_hand_frame(root_pose.yaw)
                            controller.detach(match_zero_velocity=True)
                            if args.post_place_mode == "dynamic_settle_then_freeze":
                                box_record["state"] = "PLACED_DYNAMIC"
                                placement_mode = "dynamic_settle"
                                freeze_pose_source = None
                                print(
                                    "online_release "
                                    f"item={box_spec.item_index} phase={phase} "
                                    "placement_mode=dynamic_settle freeze_pose_source=None "
                                    "snap_to_pct_target=False "
                                    f"bottom_gap={bottom_gap:.4f} support_overlap={support_overlap:.4f} "
                                    f"obstacle_overlap={obstacle_overlap:.4f} retreat_frames={POST_RELEASE_RETREAT_FRAMES}",
                                    flush=True,
                                )
                            else:
                                _set_actor_root_pose(gym, sim, root_states, source_index, level_freeze_center, float(actual_yaw), float(actual_pitch))
                                box_record["state"] = "PLACED_FROZEN"
                                box_record["frozen_pose"] = [level_freeze_center[0], level_freeze_center[1], level_freeze_center[2], float(actual_yaw)]
                                box_record["frozen_pitch"] = float(actual_pitch)
                                placement_mode = "verified_kinematic_place"
                                freeze_pose_source = "actual_pre_release_xy_yaw_supported_z"
                                print(
                                    "online_release "
                                    f"item={box_spec.item_index} phase={phase} "
                                    "placement_mode=verified_kinematic_place "
                                    "freeze_pose_source=actual_pre_release_xy_yaw_supported_z "
                                    "snap_to_pct_target=False "
                                    f"bottom_gap={bottom_gap:.4f} support_overlap={support_overlap:.4f} "
                                    f"obstacle_overlap={obstacle_overlap:.4f} retreat_frames={POST_RELEASE_RETREAT_FRAMES}",
                                    flush=True,
                                )
                            attached = False
                            released = True
                            release_root_pose = root_pose
                            release_q = q.clone()
                            break

                    if trace_fh is not None and total_frames % max(1, args.trace_every) == 0:
                        _write_trace_frame(trace_fh, total_frames, box_spec.item_index, box_record["state"], root_pose, q, dof_names, gym, env, box_records, pct_payload)
                    if viewer is not None:
                        gym.step_graphics(sim)
                        gym.draw_viewer(viewer, sim, True)
                        gym.sync_frame_time(sim)
                    if camera is not None and total_frames % max(1, args.record_every) == 0:
                        gym.step_graphics(sim)
                        gym.render_all_camera_sensors(sim)
                        img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
                        frame_img = torch.as_tensor(img, dtype=torch.uint8).view(args.camera_height, args.camera_width, 4).cpu().numpy()[:, :, :3]
                        video_frames.append(frame_img.copy())
                    total_frames += 1
                    item_frames += 1
                    if frame % 240 == 0:
                        actual = _actor_center(gym, env, source_actor)
                        q_err, q_top = _dof_error_summary(gym, env, robot, q, dof_names)
                        print(f"online_frame item={box_spec.item_index} frame={frame} phase={phase} actual=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) q_err={q_err:.3f} top={q_top}", flush=True)

                if item_failed_reason is None and released and release_q is not None and release_root_pose is not None:
                    retreat_target_q = timeline[-1][2]
                    print(
                        "online_post_release_retreat "
                        f"item={box_spec.item_index} frames={POST_RELEASE_RETREAT_FRAMES} "
                        "box_pose_control=disabled robot_box_collision_disabled=True",
                        flush=True,
                    )
                    for retreat_idx in range(POST_RELEASE_RETREAT_FRAMES):
                        alpha = _smoothstep((retreat_idx + 1) / max(1, POST_RELEASE_RETREAT_FRAMES))
                        retreat_q = _lerp_tensor(release_q, retreat_target_q, alpha)
                        _freeze_placed_boxes(gym, sim, root_states, box_records)
                        _set_actor_root_pose(
                            gym,
                            sim,
                            root_states,
                            robot_index,
                            (release_root_pose.x, release_root_pose.y, release_root_pose.z),
                            release_root_pose.yaw,
                        )
                        _set_robot_dof_state(gym, env, robot, retreat_q)
                        gym.set_actor_dof_position_targets(env, robot, retreat_q.numpy())
                        gym.simulate(sim)
                        gym.fetch_results(sim, True)
                        _sync_sim_state(gym, sim)
                        _freeze_placed_boxes(gym, sim, root_states, box_records)
                        _sync_sim_state(gym, sim)
                        retreat_linear_speed, retreat_angular_speed = _actor_root_speeds(gym, sim, root_states, source_index)
                        if retreat_angular_speed > CONTACT_EXPLOSION_ANGULAR_SPEED_RADPS:
                            if not contact_explosion:
                                print(
                                    "online_contact_explosion "
                                    f"item={box_spec.item_index} retreat_frame={retreat_idx} "
                                    f"linear_speed={retreat_linear_speed:.4f} angular_speed={retreat_angular_speed:.4f}",
                                    flush=True,
                                )
                            contact_explosion = True
                        if trace_fh is not None and total_frames % max(1, args.trace_every) == 0:
                            _write_trace_frame(
                                trace_fh,
                                total_frames,
                                box_spec.item_index,
                                box_record["state"],
                                release_root_pose,
                                retreat_q,
                                dof_names,
                                gym,
                                env,
                                box_records,
                                pct_payload,
                            )
                        if viewer is not None:
                            gym.step_graphics(sim)
                            gym.draw_viewer(viewer, sim, True)
                            gym.sync_frame_time(sim)
                        if camera is not None and total_frames % max(1, args.record_every) == 0:
                            gym.step_graphics(sim)
                            gym.render_all_camera_sensors(sim)
                            img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
                            frame_img = torch.as_tensor(img, dtype=torch.uint8).view(args.camera_height, args.camera_width, 4).cpu().numpy()[:, :, :3]
                            video_frames.append(frame_img.copy())
                        total_frames += 1
                        item_frames += 1

                if item_failed_reason is None:
                    for settle_idx in range(SETTLE_FRAMES):
                        _freeze_placed_boxes(gym, sim, root_states, box_records)
                        gym.simulate(sim)
                        gym.fetch_results(sim, True)
                        _sync_sim_state(gym, sim)
                        _freeze_placed_boxes(gym, sim, root_states, box_records)
                        _sync_sim_state(gym, sim)
                        settle_linear_speed, settle_angular_speed = _actor_root_speeds(gym, sim, root_states, source_index)
                        if settle_angular_speed > CONTACT_EXPLOSION_ANGULAR_SPEED_RADPS:
                            if not contact_explosion:
                                print(
                                    "online_contact_explosion "
                                    f"item={box_spec.item_index} settle_frame={settle_idx} "
                                    f"linear_speed={settle_linear_speed:.4f} angular_speed={settle_angular_speed:.4f}",
                                    flush=True,
                                )
                            contact_explosion = True
                        if trace_fh is not None and total_frames % max(1, args.trace_every) == 0:
                            _write_trace_frame(trace_fh, total_frames, box_spec.item_index, box_record["state"], timeline[-1][1], timeline[-1][2], dof_names, gym, env, box_records, pct_payload)
                        total_frames += 1
                _sync_sim_state(gym, sim)
                actual = _actor_center(gym, env, source_actor)
                actual_yaw, actual_pitch_final = _actor_yaw_pitch(gym, env, source_actor)
                target_center = _target_world_center(scene)
                place_error = item_failed_place_error if item_failed_place_error is not None else math.sqrt(sum((actual[i] - target_center[i]) ** 2 for i in range(3)))
                final_xy_error = math.sqrt((actual[0] - target_center[0]) ** 2 + (actual[1] - target_center[1]) ** 2)
                yaw_error = item_failed_yaw_error if item_failed_yaw_error is not None else _yaw_error_to_options(actual_yaw, robot_target.target_yaw_options)
                linear_speed, angular_speed = _actor_root_speeds(gym, sim, root_states, source_index)
                velocity_ok = linear_speed <= FINAL_LINEAR_SPEED_LIMIT_MPS and angular_speed <= FINAL_ANGULAR_SPEED_LIMIT_RADPS
                final_bottom_ok = pre_release_bottom_gap is not None and args.bottom_gap_min <= pre_release_bottom_gap <= args.bottom_gap_max
                final_overlap_ok = pre_release_overlap is not None and pre_release_overlap <= args.obstacle_overlap_max
                final_stack_inside, final_stack_clearance = _actor_inside_stack(actual, actual_yaw, execution_size, scene.stack_scene, margin=0.0)
                final_geometry_ok = final_xy_error <= FINAL_PLACE_ERROR_LIMIT_M and final_bottom_ok and final_overlap_ok and final_stack_inside
                strict_success = item_failed_reason is None and released and final_geometry_ok and yaw_error <= YAW_ERROR_LIMIT_RAD and velocity_ok
                demo_success = item_failed_reason is None and released and final_geometry_ok and yaw_error <= DEMO_YAW_ERROR_LIMIT_RAD and velocity_ok

                fallback_ok = (
                    item_failed_reason is None
                    and released
                    and (not strict_success)
                    and contact_explosion
                    and pre_release_pose is not None
                    and pre_release_error_value is not None
                    and pre_release_yaw_error_value is not None
                    and pre_release_error_value < VERIFIED_KINEMATIC_ERROR_LIMIT_M
                    and pre_release_yaw_error_value < VERIFIED_KINEMATIC_YAW_LIMIT_RAD
                )
                if fallback_ok:
                    actual = (float(pre_release_pose[0]), float(pre_release_pose[1]), float(pre_release_pose[2]))
                    actual_yaw = float(pre_release_pose[3])
                    actual_pitch_final = float(pre_release_pitch_value if pre_release_pitch_value is not None else source_pitch)
                    _set_actor_root_pose(gym, sim, root_states, source_index, actual, actual_yaw, actual_pitch_final)
                    _sync_sim_state(gym, sim)
                    place_error = float(pre_release_error_value)
                    yaw_error = float(pre_release_yaw_error_value)
                    linear_speed = 0.0
                    angular_speed = 0.0
                    velocity_ok = True
                    strict_success = place_error <= FINAL_PLACE_ERROR_LIMIT_M and yaw_error <= YAW_ERROR_LIMIT_RAD
                    demo_success = place_error <= FINAL_PLACE_ERROR_LIMIT_M and yaw_error <= DEMO_YAW_ERROR_LIMIT_RAD
                    placement_mode = "verified_kinematic_place"
                    freeze_pose_source = "actual_pre_release_pose"
                    print(
                        "online_verified_kinematic_fallback "
                        f"item={box_spec.item_index} pre_release_error={place_error:.4f} "
                        f"pre_release_yaw_error={math.degrees(yaw_error):.2f}deg "
                        f"bottom_gap={pre_release_bottom_gap} overlap={pre_release_overlap}",
                        flush=True,
                    )
                elif strict_success and placement_mode == "verified_kinematic_place":
                    freeze_pose_source = freeze_pose_source or "actual_pre_release_pose"
                elif strict_success:
                    placement_mode = "dynamic_settle"
                    freeze_pose_source = "actual_settled_pose"
                elif placement_mode == "dynamic_settle":
                    placement_mode = "dynamic_settle_failed"

                success = strict_success
                dense_fill_active = float(args.target_utilization or 0.0) >= DENSE_FILL_UTILIZATION_THRESHOLD
                dense_fill_limit = _dense_fill_commit_error_limit(float(candidate.utilization_if_committed))
                if success and dense_fill_active and place_error > dense_fill_limit:
                    success = False
                    strict_success = False
                    demo_success = False
                    item_failed_reason = "dense_fill_actual_vs_pct_error"
                    print(
                        "online_dense_fill_commit_blocked "
                        f"item={box_spec.item_index} place_error={place_error:.4f} "
                        f"limit={dense_fill_limit:.4f} "
                        f"utilization_if_committed={candidate.utilization_if_committed:.4f} "
                        f"target_utilization={args.target_utilization:.3f}",
                        flush=True,
                    )
                print(
                    "online_final_validation "
                    f"item={box_spec.item_index} place_error={place_error:.4f} final_xy_error={final_xy_error:.4f} "
                    f"yaw_error={math.degrees(yaw_error):.2f}deg "
                    f"linear_speed={linear_speed:.4f} angular_speed={angular_speed:.4f} "
                    f"strict_success={strict_success} demo_success={demo_success} "
                    f"placement_mode={placement_mode} contact_explosion={contact_explosion} "
                    f"bottom_gap={pre_release_bottom_gap} overlap={pre_release_overlap} "
                    f"stack_inside={final_stack_inside} stack_clearance={final_stack_clearance:.4f}",
                    flush=True,
                )
                reason = "success" if success else (item_failed_reason or "placement_validation_failed")
                placement_dict = asdict(candidate)
                placement_dict["execution_target_center_m"] = list(robot_target.target_center_m)
                placement_dict["execution_target_adjustment"] = target_adjustment_diag
                placement_dict["pct_prefilter_rejections"] = pct_prefilter_rejections
                attempt_result = ExecutionResult(
                    success,
                    reason,
                    box_spec.item_index,
                    placement_dict,
                    place_error,
                    yaw_error,
                    pre_attach_gap,
                    item_frames,
                    strict_success,
                    demo_success,
                    linear_speed,
                    angular_speed,
                    placement_mode,
                    freeze_pose_source,
                    False,
                    pre_release_bottom_gap,
                    pre_release_overlap,
                    pre_release_error_value,
                    place_error,
                    contact_explosion,
                    pre_release_support_top_z,
                    pre_release_actual_bottom_z,
                    pre_release_planned_center_z,
                    pre_release_error_xyz,
                    pct_first_candidate_supported=pct_first_candidate_supported,
                    pct_fallback_used=pct_fallback_used,
                    pct_fallback_reason=pct_fallback_reason,
                    pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                    pct_selected_leaf_index=candidate.action_leaf_index,
                    local_adjustment_xy_m=local_adjustment_xy_m,
                    local_adjustment_z_m=local_adjustment_z_m,
                    local_adjusted=local_adjusted,
                )
                if success:
                    committed = pct.commit(candidate)
                    committed_dict = asdict(committed)
                    committed_dict["execution_target_center_m"] = list(robot_target.target_center_m)
                    committed_dict["execution_target_adjustment"] = target_adjustment_diag
                    committed_dict["pct_prefilter_rejections"] = pct_prefilter_rejections
                    box_record["state"] = "PLACED_FROZEN"
                    box_record["frozen_pose"] = [float(actual[0]), float(actual[1]), float(actual[2]), float(actual_yaw)]
                    box_record["frozen_pitch"] = float(actual_pitch_final)
                    placed.append(
                        _boxplacement_from_frozen_pose(
                            f"item_{box_spec.item_index:02d}_actual",
                            box_record["frozen_pose"],
                            tuple(float(v) for v in robot_target.target_aabb_size_m),
                            scene.stack_scene,
                        )
                    )
                    _set_actor_collision_filter(gym, env, source_actor, 0)
                    _freeze_placed_boxes(gym, sim, root_states, box_records)
                    _sync_sim_state(gym, sim)
                    if trace_fh is not None:
                        _write_trace_frame(
                            trace_fh,
                            total_frames,
                            box_spec.item_index,
                            box_record["state"],
                            timeline[-1][1],
                            timeline[-1][2],
                            dof_names,
                            gym,
                            env,
                            box_records,
                            pct_payload,
                        )
                    attempt_result = ExecutionResult(
                        True,
                        "success",
                        box_spec.item_index,
                        committed_dict,
                        place_error,
                        yaw_error,
                        pre_attach_gap,
                        item_frames,
                        strict_success,
                        demo_success,
                        linear_speed,
                        angular_speed,
                        placement_mode,
                        freeze_pose_source,
                        False,
                        pre_release_bottom_gap,
                        pre_release_overlap,
                        pre_release_error_value,
                        place_error,
                        contact_explosion,
                        pre_release_support_top_z,
                        pre_release_actual_bottom_z,
                        pre_release_planned_center_z,
                        pre_release_error_xyz,
                        pct_first_candidate_supported=pct_first_candidate_supported,
                        pct_fallback_used=pct_fallback_used,
                        pct_fallback_reason=pct_fallback_reason,
                        pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                        pct_selected_leaf_index=committed.action_leaf_index,
                        local_adjustment_xy_m=local_adjustment_xy_m,
                        local_adjustment_z_m=local_adjustment_z_m,
                        local_adjusted=local_adjusted,
                    )
                    results.append(attempt_result)
                    consecutive_no_fit_skips = 0
                    item_completed = True
                    retry_excluded_leaves.pop(box_spec.item_index, None)
                    if args.target_utilization > 0.0:
                        current_metrics = _frozen_box_metrics(box_records, empty_scene.stack_scene)
                        current_utilization = float(current_metrics["utilization"])
                        print(
                            "online_utilization_check "
                            f"item={box_spec.item_index} utilization={current_utilization:.4f} "
                            f"target={args.target_utilization:.4f}",
                            flush=True,
                        )
                        if current_utilization >= float(args.target_utilization):
                            completion_reason = "target_utilization_reached"
                    print(f"online_result item={box_spec.item_index} success=True strict_success={strict_success} demo_success={demo_success} reason=success error={place_error:.4f} yaw_error={math.degrees(yaw_error):.2f}deg placement_mode={placement_mode}", flush=True)
                    break

                candidate_failures.append(
                    {
                        "candidate_index": candidate_index,
                        "pct_rank": target_adjustment_diag.get("pct_rank"),
                        "pct_leaf_index": candidate.action_leaf_index,
                        "reason": reason,
                        "approach_side": exec_candidate.diagnostics.get("approach_side"),
                        "approach_yaw": exec_candidate.approach_yaw,
                        "stand_off": exec_candidate.stand_off,
                        "target_yaw": target_yaw,
                        "score": exec_candidate.score,
                        "place_error_m": float(place_error) if place_error is not None else None,
                        "yaw_error_rad": float(yaw_error) if yaw_error is not None else None,
                        "bottom_gap_m": pre_release_bottom_gap,
                        "overlap_m": pre_release_overlap,
                        "overlap_depth_m": pre_release_overlap,
                        "local_adjustment_xy_m": local_adjustment_xy_m,
                        "local_adjustment_z_m": local_adjustment_z_m,
                        "local_adjustment_xyz_m": [local_adjustment_xy_m, 0.0, local_adjustment_z_m],
                        "local_adjusted": local_adjusted,
                        "inside_1m_cube": final_stack_inside,
                        "diagnostics": exec_candidate.diagnostics,
                    }
                )
                final_failure_result = attempt_result
                print(
                    "online_candidate_attempt_result "
                    f"item={box_spec.item_index} candidate_index={candidate_index} success=False reason={reason} "
                    f"error={place_error:.4f} yaw_error={math.degrees(yaw_error):.2f}deg "
                    f"approach_yaw={math.degrees(exec_candidate.approach_yaw):.2f}deg stand_off={exec_candidate.stand_off:.3f} "
                    f"retryable={reason in RETRYABLE_CANDIDATE_REASONS}",
                    flush=True,
                )
                box_record["state"] = "NOT_SPAWNED"
                _set_actor_collision_filter(gym, env, source_actor, 99)
                _set_actor_root_pose(gym, sim, root_states, source_index, (-10.0, -10.0, -10.0), 0.0)
                _sync_sim_state(gym, sim)
                if reason in RETRYABLE_CANDIDATE_REASONS:
                    continue
                break
            if not item_completed:
                pct.reject(candidate)
                completion_reason = (final_failure_result.reason if final_failure_result is not None else "all_execution_candidates_failed")
                if final_failure_result is None:
                    final_failure_result = ExecutionResult(
                        False,
                        "all_execution_candidates_failed",
                        box_spec.item_index,
                        asdict(candidate),
                        None,
                        None,
                        None,
                        0,
                        pct_first_candidate_supported=pct_first_candidate_supported,
                        pct_fallback_used=pct_fallback_used,
                        pct_fallback_reason=pct_fallback_reason,
                        pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                        pct_selected_leaf_index=candidate.action_leaf_index,
                    )
                if final_failure_result.reason in RETRYABLE_CANDIDATE_REASONS:
                    excluded = retry_excluded_leaves.setdefault(box_spec.item_index, set())
                    excluded.add(int(candidate.action_leaf_index))
                    print(
                        "online_pct_leaf_execution_failed_retry "
                        f"item={box_spec.item_index} leaf={candidate.action_leaf_index} "
                        f"rank={target_adjustment_diag.get('pct_rank')} reason={final_failure_result.reason} "
                        f"excluded_leaves={sorted(excluded)} candidate_failures="
                        + json.dumps(candidate_failures, separators=(",", ":")),
                        flush=True,
                    )
                    completion_reason = "running"
                    retry_box_spec = box_spec
                    continue
                if args.max_items <= 0 and args.target_utilization > 0.0:
                    skip_reason = f"skipped_execution_infeasible:{final_failure_result.reason}"
                    skipped_no_fit_results.append(
                        ExecutionResult(
                            False,
                            skip_reason,
                            box_spec.item_index,
                            asdict(candidate),
                            None,
                            None,
                            None,
                            0,
                            pct_first_candidate_supported=pct_first_candidate_supported,
                            pct_fallback_used=pct_fallback_used,
                            pct_fallback_reason=pct_fallback_reason,
                            pct_first_leaf_index=pct_first_candidate.action_leaf_index,
                            pct_selected_leaf_index=candidate.action_leaf_index,
                        )
                    )
                    consecutive_no_fit_skips += 1
                    retry_excluded_leaves.pop(box_spec.item_index, None)
                    print(
                        "online_item_skipped "
                        f"item={box_spec.item_index} reason={skip_reason} "
                        f"consecutive_no_fit_skips={consecutive_no_fit_skips} "
                        f"skipped_no_fit_count={len(skipped_no_fit_results)} "
                        f"placed_count={sum(1 for r in results if r.success)} "
                        f"candidate_failures=" + json.dumps(candidate_failures, separators=(",", ":")),
                        flush=True,
                    )
                    if (
                        args.max_consecutive_no_fit_skips > 0
                        and consecutive_no_fit_skips >= int(args.max_consecutive_no_fit_skips)
                    ):
                        completion_reason = "consecutive_no_fit_limit_reached"
                        print(
                            "online_stop "
                            f"reason={completion_reason} consecutive_no_fit_skips={consecutive_no_fit_skips} "
                            f"placed_count={sum(1 for r in results if r.success)} "
                            f"skipped_no_fit_count={len(skipped_no_fit_results)}",
                            flush=True,
                        )
                        break
                    continue
                results.append(final_failure_result)
                print(
                    "online_result "
                    f"item={box_spec.item_index} success=False strict_success={final_failure_result.strict_success} "
                    f"demo_success={final_failure_result.demo_success} reason={final_failure_result.reason} "
                    f"pct_prefilter_rejections={json.dumps(pct_prefilter_rejections, separators=(',', ':'))} "
                    f"candidate_failures=" + json.dumps(candidate_failures, separators=(",", ":")),
                    flush=True,
                )
                break
            if completion_reason == "target_utilization_reached":
                print(
                    "online_stop "
                    f"reason={completion_reason} placed_count={sum(1 for r in results if r.success)} "
                    f"target_utilization={args.target_utilization:.4f}",
                    flush=True,
                )
                break
    finally:
        if trace_fh is not None:
            trace_fh.close()
            print(f"wrote_trace={args.out_trace}")
        if args.record_video and video_frames:
            args.record_video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.record_video, video_frames, fps=8)
            print(f"wrote_video={args.record_video} frames={len(video_frames)}")
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        placed_results = [r for r in results if r.success and r.place_error_m is not None]
        placement_mode_counts: dict[str, int] = {}
        for result in results:
            if result.success:
                placement_mode_counts[result.placement_mode] = placement_mode_counts.get(result.placement_mode, 0) + 1
        target_adjustments = [
            float((r.placement or {}).get("execution_target_adjustment", {}).get("target_adjustment_xy_m", 0.0) or 0.0)
            for r in results
            if r.placement is not None
        ]
        frozen_metrics = _frozen_box_metrics(box_records, empty_scene.stack_scene)
        actual_packing_state = _build_actual_packing_state(box_records, results, empty_scene.stack_scene)
        max_deviation_from_pct_m = max(
            (float(row["norm_m"]) for row in actual_packing_state.deviation_from_pct),
            default=0.0,
        )
        max_xy_deviation_from_pct_m = max(
            (float(row["xy_norm_m"]) for row in actual_packing_state.deviation_from_pct),
            default=0.0,
        )
        failure_reason = next((r.reason for r in reversed(results) if not r.success), "")
        if completion_reason == "running":
            completion_reason = "max_items_reached" if args.max_items > 0 else "ended"
        raw_completion_reason = completion_reason
        completion_reason = _completion_bucket(completion_reason)
        core_payload = {
            "seed": args.seed,
            "max_items": args.max_items,
            "max_arrivals": int(args.max_arrivals),
            "max_pct_leaf_checks": int(args.max_pct_leaf_checks),
            "max_ik_leaf_checks": int(args.max_ik_leaf_checks),
            "target_utilization": float(args.target_utilization),
            "completion_reason": completion_reason,
            "raw_completion_reason": raw_completion_reason,
            "num_arrived": total_arrivals,
            "arrived_count": total_arrivals,
            "num_placed": sum(1 for r in results if r.success),
            "placed_count": sum(1 for r in results if r.success),
            "skipped_no_fit_count": len(skipped_no_fit_results),
            "consecutive_no_fit_skips": consecutive_no_fit_skips,
            "max_consecutive_no_fit_skips": int(args.max_consecutive_no_fit_skips),
            "fine_fill_enabled": bool(args.fine_fill and args.target_utilization > 0.0 and args.max_items <= 0),
            "fine_fill_utilization_threshold": float(args.fine_fill_utilization_threshold),
            "fine_fill_skip_threshold": int(args.fine_fill_skip_threshold),
            "item_sample_mode_counts": item_sample_mode_counts,
            "item_sample_mode_by_index": {str(k): v for k, v in sorted(item_sample_mode_by_index.items())},
            "skipped_no_fit_items": [
                {
                    "item_index": int(r.item_index),
                    "reason": r.reason,
                    "first_leaf_index": r.pct_first_leaf_index,
                    "fallback_reason": r.pct_fallback_reason,
                }
                for r in skipped_no_fit_results
            ],
            "num_strict_success": sum(1 for r in results if r.strict_success),
            "num_demo_success": sum(1 for r in results if r.demo_success),
            "utilization": frozen_metrics["utilization"],
            "max_height_m": frozen_metrics["max_height_m"],
            "all_boxes_inside_1m_cube": frozen_metrics["all_boxes_inside_1m_cube"],
            "all_boxes_inside_1m_cube_strict": frozen_metrics["all_boxes_inside_1m_cube_strict"],
            "all_boxes_inside_1m_cube_soft": frozen_metrics["all_boxes_inside_1m_cube_soft"],
            "min_inside_clearance_m": frozen_metrics["min_inside_clearance_m"],
            "placement_mode_counts": placement_mode_counts,
            "failure_reason": failure_reason,
            "mean_place_error_m": (
                sum(float(r.place_error_m) for r in placed_results) / len(placed_results)
                if placed_results
                else None
            ),
            "max_place_error_m": (
                max(float(r.place_error_m) for r in placed_results)
                if placed_results
                else None
            ),
            "max_execution_target_adjustment_xy_m": max(target_adjustments, default=0.0),
            "max_local_adjustment_xy_m": max((float(r.local_adjustment_xy_m) for r in results), default=0.0),
            "max_local_adjustment_z_m": max((float(r.local_adjustment_z_m) for r in results), default=0.0),
            "num_local_adjusted": sum(1 for r in results if r.local_adjusted),
            "max_deviation_from_pct_m": max_deviation_from_pct_m,
            "max_xy_deviation_from_pct_m": max_xy_deviation_from_pct_m,
            "strict_yaw_threshold_deg": 5.0,
            "demo_yaw_threshold_deg": 8.0,
            "final_place_error_limit_m": FINAL_PLACE_ERROR_LIMIT_M,
            "dense_fill_utilization_threshold": DENSE_FILL_UTILIZATION_THRESHOLD,
            "dense_fill_commit_error_limit_m": DENSE_FILL_COMMIT_ERROR_LIMIT_M,
            "dense_fill_adaptive_limits": [
                {"utilization_lt": float(util), "error_limit_m": float(limit)}
                for util, limit in DENSE_FILL_ADAPTIVE_LIMITS
            ],
            "final_linear_speed_limit_mps": FINAL_LINEAR_SPEED_LIMIT_MPS,
            "final_angular_speed_limit_radps": FINAL_ANGULAR_SPEED_LIMIT_RADPS,
            "uses_live_isaac_gym_camera_recording": bool(args.record_video),
            "uses_offline_trace_rendering": args.out_trace is not None,
            "uses_planned_box_center_replay": False,
            "uses_keyframe_root_replay": False,
            "snap_to_pct_target": False,
            "all_future_boxes_spawned_at_start": False,
            "source_box_size_uses_original_size_m": True,
            "target_pose_uses_pct_placed_size_m": True,
            "palm_box_contact_checked_before_attach": True,
            "box_attached_to_actual_hand_frame": True,
            "pct_commit_after_execution_success": True,
            "actual_aware_pct_selection": True,
            "place_release_height_m": args.place_release_height,
            "bottom_gap_min_m": args.bottom_gap_min,
            "bottom_gap_max_m": args.bottom_gap_max,
            "strict_obstacle_overlap_limit_m": FROZEN_AABB_OVERLAP_LIMIT_M,
            "soft_obstacle_overlap_limit_m": SOFT_OBSTACLE_OVERLAP_LIMIT_M,
            "hard_obstacle_overlap_limit_m": HARD_OBSTACLE_OVERLAP_LIMIT_M,
            "soft_stack_boundary_tolerance_m": SOFT_STACK_BOUNDARY_TOLERANCE_M,
            "stack_inside_gate_tolerance_m": STACK_INSIDE_GATE_TOLERANCE_M,
            "verified_place_obstacle_overlap_max_m": args.obstacle_overlap_max,
            "ik_candidate_limit": IK_CANDIDATE_LIMIT,
        }
        debug_payload = {
            **core_payload,
            "placed_boxes": frozen_metrics["placed_boxes"],
            "actual_packing_state": asdict(actual_packing_state),
            "results": [asdict(r) for r in results],
        }
        args.metrics.write_text(json.dumps(core_payload, indent=2), encoding="utf-8")
        print(f"wrote_metrics={args.metrics}")
        if args.debug_metrics is not None:
            args.debug_metrics.parent.mkdir(parents=True, exist_ok=True)
            args.debug_metrics.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
            print(f"wrote_debug_metrics={args.debug_metrics}")
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
