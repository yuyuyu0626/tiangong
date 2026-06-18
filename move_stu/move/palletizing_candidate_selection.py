"""PCT leaf filtering and robot execution candidate planning."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np

from move.grab_test import build_custom_stack_scene, run as build_grab_plan
from move.online_palletizing import BoxSpec
from move.palletizing_geometry import *  # noqa: F403 - internal helpers are the selection surface.
from move.palletizing_runtime import *  # noqa: F403 - constants and data models are shared policy.
from move.pct_policy_bridge import PCTOnlineController
from move.planning import TABLE_POSE, TABLE_SIZE, BoxPlacement
from move.robot_placement import RobotPlacementTarget, robot_target_from_pct, wrap_to_pi

def source_center(size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (TABLE_POSE[0] - 0.10, TABLE_POSE[1], TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + size[2] * 0.5)

def _unique_yaws(values: tuple[float, ...] | list[float], eps: float = 1e-6) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        yaw = wrap_to_pi(float(value))
        if not any(abs(wrap_to_pi(yaw - old)) < eps for old in out):
            out.append(yaw)
    return tuple(out)


def _approach_yaw_options(target_yaw: float) -> tuple[float, ...]:
    # The PCT yaw is the box yaw.  The robot root/approach yaw is an execution
    # choice for reaching that same target, so include both target-relative and
    # world-cardinal directions.
    return _unique_yaws([
        target_yaw,
        target_yaw + math.pi * 0.5,
        target_yaw - math.pi * 0.5,
        target_yaw + math.pi,
        0.0,
        math.pi * 0.5,
        -math.pi * 0.5,
        math.pi,
    ])


APPROACH_SIDE_YAWS = {
    "-X": 0.0,
    "+X": math.pi,
    "-Y": math.pi * 0.5,
    "+Y": -math.pi * 0.5,
}


def _sorted_approach_sides(target: BoxPlacement, stack_size_xy: tuple[float, float] = (1.0, 1.0)) -> list[tuple[str, float]]:
    cx = float(target.min_corner[0] + target.size[0] * 0.5)
    cy = float(target.min_corner[1] + target.size[1] * 0.5)
    sx, sy = float(stack_size_xy[0]), float(stack_size_xy[1])
    distances = [
        ("-X", cx),
        ("+X", sx - cx),
        ("-Y", cy),
        ("+Y", sy - cy),
    ]
    distances.sort(key=lambda row: (row[1], row[0]))
    return distances


def _route_pose(value):
    if isinstance(value, tuple) and value and hasattr(value[-1], "x"):
        return value[-1]
    return value


def _root_route_length(scene) -> float:
    waypoints = [scene.tmp_pose, *scene.waypoints, scene.final_pose]
    total = 0.0
    last = None
    for value in waypoints:
        pose = _route_pose(value)
        cur = np.asarray([pose.x, pose.y], dtype=np.float64)
        if last is not None:
            total += float(np.linalg.norm(cur - last))
        last = cur
    return total


def _ik_prune_signature(diagnostics: dict, robot_target: RobotPlacementTarget) -> tuple:
    target_z_band = int(round(float(robot_target.target_center_m[2]) / 0.10))
    return (
        str(diagnostics.get("orientation_type", "")),
        str(diagnostics.get("approach_side", "")),
        int(round(float(diagnostics.get("target_yaw_deg", 0.0)) / 90.0)),
        int(round(float(diagnostics.get("approach_yaw_deg", 0.0)) / 90.0)),
        int(round(float(diagnostics.get("source_pitch_deg", 0.0)) / 90.0)),
        target_z_band,
    )


def _plan_candidates_for_item(
    box_spec: BoxSpec,
    robot_target: RobotPlacementTarget,
    candidate,
    placed: list[BoxPlacement],
    place_release_height: float,
    ik_prune_signatures: set[tuple] | None = None,
):
    del candidate
    target = _target_from_robot_center(robot_target)
    execution_size = tuple(float(v) for v in robot_target.execution_size_m)
    source = source_center(execution_size)
    source_xy = np.asarray([source[0], source[1]], dtype=np.float64)
    cheap_candidates: list[tuple[float, float, float, object, float, float, float, dict]] = []

    side_order = _sorted_approach_sides(target)
    for target_yaw_option in robot_target.target_yaw_options:
        target_yaw = float(target_yaw_option)
        for side_rank, (approach_side, side_distance) in enumerate(side_order):
            for stand_off in STAND_OFF_CANDIDATES:
                scene = build_custom_stack_scene(
                    f"online_item_{box_spec.item_index:02d}",
                    tuple(placed),
                    target,
                    stand_off=stand_off,
                    approach_side=approach_side,
                )
                corridor_penalty = _approach_blockage_penalty(target, placed, scene.final_pose.yaw, stand_off)
                route_length = _root_route_length(scene)
                root_xy = np.asarray([scene.final_pose.x, scene.final_pose.y], dtype=np.float64)
                root_source_distance = float(np.linalg.norm(root_xy - source_xy))
                yaw_delta = abs(wrap_to_pi(scene.final_pose.yaw - target_yaw))
                stand_off_penalty = abs(float(stand_off) - 0.28)
                side_penalty = float(side_rank) * 0.20
                source_yaw_options = _unique_yaws([0.0, wrap_to_pi(target_yaw - scene.final_pose.yaw)])
                for source_yaw in source_yaw_options:
                    for source_pitch in robot_target.source_pitch_options:
                        source_yaw_penalty = abs(wrap_to_pi(source_yaw)) * 0.03
                        source_pitch_penalty = 0.04 if abs(float(source_pitch)) > 1e-6 else 0.0
                        cheap_score = (
                            corridor_penalty
                            + side_penalty
                            + root_source_distance * 0.10
                            + route_length * 0.01
                            + stand_off_penalty * 0.05
                            + yaw_delta * 0.005
                            + source_yaw_penalty
                            + source_pitch_penalty
                        )
                        diagnostics = {
                            "approach_side": approach_side,
                            "approach_side_rank": int(side_rank),
                            "approach_side_distance": float(side_distance),
                            "target_yaw_deg": math.degrees(target_yaw),
                            "approach_yaw_deg": math.degrees(scene.final_pose.yaw),
                            "stand_off": float(stand_off),
                            "source_yaw_deg": math.degrees(source_yaw),
                            "source_pitch_deg": math.degrees(source_pitch),
                            "source_yaw_penalty": float(source_yaw_penalty),
                            "source_pitch_penalty": float(source_pitch_penalty),
                            "corridor_penalty": float(corridor_penalty),
                            "corridor_blocked": bool(corridor_penalty > 0.0),
                            "route_length": float(route_length),
                            "root_source_distance": float(root_source_distance),
                            "yaw_delta_deg": math.degrees(yaw_delta),
                            "orientation_type": robot_target.orientation_type,
                            "cheap_score": float(cheap_score),
                        }
                        cheap_candidates.append((cheap_score, target_yaw, scene.final_pose.yaw, scene, stand_off, source_yaw, float(source_pitch), diagnostics))

    cheap_candidates.sort(key=lambda row: row[0])
    print(
        "online_execution_candidate_prefilter "
        f"item={box_spec.item_index} "
        + json.dumps([row[7] for row in cheap_candidates[:8]], separators=(",", ":")),
        flush=True,
    )

    # Keep the cheapest unblocked execution variants directly.  Do not collapse
    # by (target_yaw, approach_yaw): different stand_off values can make the
    # same PCT target reachable or unstable in very different ways.
    ik_prune_signatures = ik_prune_signatures or set()
    unblocked_candidates = [row for row in cheap_candidates if not bool(row[7].get("corridor_blocked"))]
    before_ik_prune_count = len(unblocked_candidates)
    if ik_prune_signatures:
        unblocked_candidates = [
            row for row in unblocked_candidates
            if _ik_prune_signature(row[7], robot_target) not in ik_prune_signatures
        ]
        pruned_count = before_ik_prune_count - len(unblocked_candidates)
        if pruned_count > 0:
            print(
                "online_execution_candidate_ik_pruned "
                f"item={box_spec.item_index} pruned={pruned_count} remaining={len(unblocked_candidates)}",
                flush=True,
            )
    if not unblocked_candidates:
        print(
            "online_execution_candidate_prefilter_rejected "
            f"item={box_spec.item_index} reason=all_candidates_blocked_or_ik_pruned "
            f"candidate_count={len(cheap_candidates)}",
            flush=True,
        )
        return []
    selected_rows = unblocked_candidates[:IK_CANDIDATE_LIMIT]

    ik_diags: list[dict] = []
    bundles = []
    for cheap_score, target_yaw, _approach_yaw, scene, stand_off, source_yaw, source_pitch, diagnostics in selected_rows[:IK_CANDIDATE_LIMIT]:
        report, payload = build_grab_plan(
            place_mode="move",
            stand_off=stand_off,
            source_pose=source,
            source_yaw=source_yaw,
            box_size=execution_size,
            target_aabb_size=robot_target.target_aabb_size_m,
            target_yaw=target_yaw,
            scene=scene,
            place_release_height=place_release_height,
        )
        feasible_penalty = 0.0 if (report.pick_feasible and report.place_feasible) else 1000.0
        score = (
            float(cheap_score)
            + feasible_penalty
            + float(report.place_max_error) * 10.0
            + float(report.pick_max_error) * 5.0
        )
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "pick_feasible": bool(report.pick_feasible),
                "place_feasible": bool(report.place_feasible),
                "pick_err": float(report.pick_max_error),
                "place_err": float(report.place_max_error),
                "score": float(score),
            }
        )
        ik_diags.append(diagnostics)
        exec_candidate = ExecutionCandidate(
            target_center=tuple(float(v) for v in robot_target.target_center_m),
            target_yaw=float(target_yaw),
            target_aabb_size=tuple(float(v) for v in robot_target.target_aabb_size_m),
            actor_size=tuple(float(v) for v in robot_target.actor_size_m),
            execution_size=execution_size,
            approach_yaw=float(scene.final_pose.yaw),
            final_root=(float(scene.final_pose.x), float(scene.final_pose.y), float(scene.final_pose.z), float(scene.final_pose.yaw)),
            stand_off=float(stand_off),
            source_yaw=float(source_yaw),
            source_pitch=float(source_pitch),
            score=float(score),
            reject_reason=None if report.pick_feasible and report.place_feasible else "ik_infeasible",
            diagnostics=diagnostics,
        )
        bundles.append((score, target, source, source_yaw, source_pitch, target_yaw, report, payload, scene, exec_candidate))

    ik_diags.sort(key=lambda row: float(row["score"]))
    print(
        "online_execution_candidate_top "
        f"item={box_spec.item_index} "
        + json.dumps(ik_diags[:6], separators=(",", ":")),
        flush=True,
    )
    bundles.sort(key=lambda row: float(row[0]))
    return [row[1:] for row in bundles]


def _select_pct_execution_target(
    pct: PCTOnlineController,
    box_spec: BoxSpec,
    placed: list[BoxPlacement],
    box_records: list[dict],
    stack_scene,
    place_release_height: float,
    pct_top_k: int,
    max_pct_leaf_checks: int,
    max_ik_leaf_checks: int,
    excluded_leaf_indices: set[int] | None = None,
):
    excluded_leaf_indices = excluded_leaf_indices or set()
    alternatives = pct.ranked_leaf_candidates(box_spec.original_size_m)
    if not alternatives:
        return None
    if max_pct_leaf_checks > 0:
        alternatives = alternatives[: int(max_pct_leaf_checks)]
    del pct_top_k  # retained for CLI compatibility; actual-aware scan is budgeted separately.
    first_candidate = alternatives[0]
    first_supported = bool(robot_target_from_pct(first_candidate).target_yaw_options)
    rejections: list[dict] = []
    ik_prune_signatures: set[tuple] = set()
    ik_leaf_checks = 0

    for alt_index, candidate in enumerate(alternatives):
        if int(candidate.action_leaf_index) in excluded_leaf_indices:
            reason = "previous_execution_failed"
            policy_rank = int(getattr(candidate, "policy_rank", alt_index))
            policy_score = float(getattr(candidate, "policy_score", 0.0))
            rejections.append({"leaf": candidate.action_leaf_index, "rank": policy_rank, "reason": reason, "policy_score": policy_score})
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason}",
                flush=True,
            )
            continue
        robot_target = robot_target_from_pct(candidate)
        policy_rank = int(getattr(candidate, "policy_rank", alt_index))
        policy_score = float(getattr(candidate, "policy_score", 0.0))
        if not robot_target.target_yaw_options:
            reason = "unsupported_flip"
            rejections.append({"leaf": candidate.action_leaf_index, "rank": policy_rank, "reason": reason, "policy_score": policy_score})
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} original_size={candidate.original_size_m} placed_size={candidate.placed_size_m}",
                flush=True,
            )
            continue

        adjusted_target, target_adjustment_diag = _adjust_robot_target_for_actual_frozen(robot_target, box_records, stack_scene)
        target_adjustment_diag["policy_rank"] = policy_rank
        target_adjustment_diag["policy_score"] = policy_score
        if target_adjustment_diag.get("target_adjustment_limited"):
            reason = "target_adjustment_limited"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    "target_adjustment_xy_m": target_adjustment_diag.get("target_adjustment_xy_m"),
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} adjustment={target_adjustment_diag.get('target_adjustment_xy_m')}",
                flush=True,
            )
            continue

        adjusted_box = _target_from_robot_center(adjusted_target)
        target_overlap = _boxplacement_overlap_depth(adjusted_box, placed)
        target_adjustment_diag["ideal_target_overlap_m"] = float(target_overlap)
        if target_overlap > FROZEN_AABB_OVERLAP_LIMIT_M:
            target_adjustment_diag["target_overlap_repair_needed"] = True
            target_adjustment_diag["target_overlap_m"] = target_overlap
            print(
                "online_pct_candidate_local_repair_allowed "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"overlap={target_overlap:.4f} limit={TARGET_OVERLAP_REPAIR_LIMIT_M:.4f}",
                flush=True,
            )

        target_yaw = float(adjusted_target.target_yaw_options[0]) if adjusted_target.target_yaw_options else 0.0
        adjusted_world_center = _local_center_to_world(adjusted_target.target_center_m, stack_scene)
        support_top_z = _support_top_z_for_footprint(
            adjusted_world_center,
            target_yaw,
            adjusted_target.execution_size_m,
            box_records,
            box_spec.item_index,
            stack_scene.pallet_surface_z,
        )
        overlap_detail = _overlap_with_frozen_boxes_by_role_detail(
            adjusted_world_center,
            target_yaw,
            adjusted_target.execution_size_m,
            box_records,
            box_spec.item_index,
            support_top_z,
            inflation_m=ACTUAL_AABB_INFLATION_M,
        )
        support_overlap = overlap_detail.support_overlap_m
        obstacle_overlap = overlap_detail.raw_obstacle_overlap_m
        inflated_obstacle_overlap = overlap_detail.inflated_obstacle_overlap_m
        safety_only_overlap = overlap_detail.safety_only_overlap_m
        repair_dx = repair_dy = 0.0
        repair_initial = float(obstacle_overlap)
        repair_remaining = float(obstacle_overlap)
        if obstacle_overlap > FROZEN_AABB_OVERLAP_LIMIT_M:
            repair_dx, repair_dy, repair_initial, repair_remaining = _frozen_overlap_xy_repair_vector(
                adjusted_world_center,
                target_yaw,
                adjusted_target.execution_size_m,
                box_records,
                box_spec.item_index,
                max_total_xy=LOCAL_REPAIR_MAX_XY_M,
                support_top_z=support_top_z,
                inflation_m=0.0,
            )
            repair_norm = math.hypot(repair_dx, repair_dy)
            if repair_norm > 1e-6 and repair_norm <= LOCAL_REPAIR_MAX_XY_M and repair_remaining <= SOFT_OBSTACLE_OVERLAP_LIMIT_M:
                repaired_world = (
                    float(adjusted_world_center[0] + repair_dx),
                    float(adjusted_world_center[1] + repair_dy),
                    float(adjusted_world_center[2]),
                )
                repaired_local = _world_center_to_local(repaired_world, stack_scene)
                adjusted_target = replace(adjusted_target, target_center_m=tuple(float(v) for v in repaired_local))
                adjusted_world_center = repaired_world
                target_adjustment_diag["actual_local_repair_applied"] = True
                target_adjustment_diag["actual_local_repair_dx_m"] = float(repair_dx)
                target_adjustment_diag["actual_local_repair_dy_m"] = float(repair_dy)
                target_adjustment_diag["target_adjustment_xy_m"] = float(
                    math.hypot(
                        float(target_adjustment_diag.get("target_adjustment_dx_m", 0.0)) + repair_dx,
                        float(target_adjustment_diag.get("target_adjustment_dy_m", 0.0)) + repair_dy,
                    )
                )
                support_top_z = _support_top_z_for_footprint(
                    adjusted_world_center,
                    target_yaw,
                    adjusted_target.execution_size_m,
                    box_records,
                    box_spec.item_index,
                    stack_scene.pallet_surface_z,
                )
                overlap_detail = _overlap_with_frozen_boxes_by_role_detail(
                    adjusted_world_center,
                    target_yaw,
                    adjusted_target.execution_size_m,
                    box_records,
                    box_spec.item_index,
                    support_top_z,
                    inflation_m=ACTUAL_AABB_INFLATION_M,
                )
                support_overlap = overlap_detail.support_overlap_m
                obstacle_overlap = overlap_detail.raw_obstacle_overlap_m
                inflated_obstacle_overlap = overlap_detail.inflated_obstacle_overlap_m
                safety_only_overlap = overlap_detail.safety_only_overlap_m
        upper_cover = _upper_frozen_cover(
            adjusted_world_center,
            target_yaw,
            adjusted_target.execution_size_m,
            box_records,
            box_spec.item_index,
        )
        target_adjustment_diag["actual_target_support_overlap_m"] = float(support_overlap)
        target_adjustment_diag["actual_target_obstacle_overlap_m"] = float(obstacle_overlap)
        target_adjustment_diag["actual_target_raw_obstacle_overlap_m"] = float(obstacle_overlap)
        target_adjustment_diag["actual_target_inflated_obstacle_overlap_m"] = float(inflated_obstacle_overlap)
        target_adjustment_diag["actual_target_safety_only_overlap_m"] = float(safety_only_overlap)
        target_adjustment_diag["actual_target_obstacle_overlap_inflation_m"] = float(ACTUAL_AABB_INFLATION_M)
        target_adjustment_diag["actual_repair_initial_overlap_m"] = float(repair_initial)
        target_adjustment_diag["actual_repair_remaining_overlap_m"] = float(repair_remaining)
        target_adjustment_diag["actual_repair_dx_m"] = float(repair_dx)
        target_adjustment_diag["actual_repair_dy_m"] = float(repair_dy)
        target_adjustment_diag["upper_frozen_cover"] = upper_cover
        target_inside, target_inside_clearance = _target_aabb_inside_stack(
            adjusted_world_center,
            adjusted_target.target_aabb_size_m,
            stack_scene,
            margin=0.0,
            tolerance_m=SOFT_STACK_BOUNDARY_TOLERANCE_M,
        )
        target_adjustment_diag["actual_target_inside_1m_cube"] = bool(target_inside)
        target_adjustment_diag["actual_target_inside_clearance_m"] = float(target_inside_clearance)
        if not target_inside:
            reason = "inside_cube_violation"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    "inside_clearance_m": float(target_inside_clearance),
                    "target_adjustment_xy_m": target_adjustment_diag.get("target_adjustment_xy_m"),
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} clearance={target_inside_clearance:.4f}",
                flush=True,
            )
            continue
        if upper_cover is not None:
            reason = "target_under_existing_box"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    **upper_cover,
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} "
                f"cover_item={upper_cover['item_index']} "
                f"vertical_gap={upper_cover['vertical_gap_m']:.4f} "
                f"xy_overlap=({upper_cover['xy_overlap_x_m']:.4f},{upper_cover['xy_overlap_y_m']:.4f}) "
                f"target_top_z={upper_cover['target_top_z_m']:.4f} cover_bottom_z={upper_cover['cover_bottom_z_m']:.4f}",
                flush=True,
            )
            continue
        if obstacle_overlap > HARD_OBSTACLE_OVERLAP_LIMIT_M or (
            obstacle_overlap > SOFT_OBSTACLE_OVERLAP_LIMIT_M and repair_remaining > SOFT_OBSTACLE_OVERLAP_LIMIT_M
        ):
            reason = "actual_target_obstacle_overlap"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    "obstacle_overlap_m": float(obstacle_overlap),
                    "raw_obstacle_overlap_m": float(obstacle_overlap),
                    "inflated_obstacle_overlap_m": float(inflated_obstacle_overlap),
                    "safety_only_overlap_m": float(safety_only_overlap),
                    "support_overlap_m": float(support_overlap),
                    "inflation_m": float(ACTUAL_AABB_INFLATION_M),
                    "repair_dx_m": float(repair_dx),
                    "repair_dy_m": float(repair_dy),
                    "repair_remaining_overlap_m": float(repair_remaining),
                    "target_adjustment_xy_m": target_adjustment_diag.get("target_adjustment_xy_m"),
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} "
                f"reason={reason} raw_obstacle_overlap={obstacle_overlap:.4f} "
                f"inflated_obstacle_overlap={inflated_obstacle_overlap:.4f} "
                f"safety_only_overlap={safety_only_overlap:.4f} support_overlap={support_overlap:.4f} "
                f"inflation={ACTUAL_AABB_INFLATION_M:.4f} repair=({repair_dx:.4f},{repair_dy:.4f}) "
                f"remaining={repair_remaining:.4f} adjustment={target_adjustment_diag.get('target_adjustment_xy_m')}",
                flush=True,
            )
            continue
        if obstacle_overlap > FROZEN_AABB_OVERLAP_LIMIT_M:
            target_adjustment_diag["actual_target_soft_obstacle_overlap_allowed"] = True
            print(
                "online_pct_candidate_soft_overlap_allowed "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} raw_obstacle_overlap={obstacle_overlap:.4f} "
                f"soft_limit={SOFT_OBSTACLE_OVERLAP_LIMIT_M:.4f} hard_limit={HARD_OBSTACLE_OVERLAP_LIMIT_M:.4f} "
                f"repair=({repair_dx:.4f},{repair_dy:.4f}) remaining={repair_remaining:.4f}",
                flush=True,
            )
        target_adjustment_diag["pct_rank"] = int(policy_rank)
        target_adjustment_diag["pct_leaf_index"] = int(candidate.action_leaf_index)
        if max_ik_leaf_checks > 0 and ik_leaf_checks >= int(max_ik_leaf_checks):
            reason = "ik_leaf_budget_exhausted"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    "max_ik_leaf_checks": int(max_ik_leaf_checks),
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} "
                f"ik_leaf_checks={ik_leaf_checks} max_ik_leaf_checks={int(max_ik_leaf_checks)}",
                flush=True,
            )
            break
        ik_leaf_checks += 1
        plan_candidates = _plan_candidates_for_item(
            box_spec,
            adjusted_target,
            candidate,
            placed,
            place_release_height,
            ik_prune_signatures=ik_prune_signatures,
        )
        feasible_count = sum(1 for row in plan_candidates if row[5].pick_feasible and row[5].place_feasible)
        if feasible_count <= 0:
            new_pruned = 0
            for row in plan_candidates:
                exec_candidate = row[8]
                report = row[5]
                if (
                    bool(report.pick_feasible)
                    and not bool(report.place_feasible)
                    and float(report.place_max_error) > 0.20
                ):
                    signature = _ik_prune_signature(exec_candidate.diagnostics, adjusted_target)
                    if signature not in ik_prune_signatures:
                        ik_prune_signatures.add(signature)
                        new_pruned += 1
            reason = "no_ik_feasible_execution_candidate"
            rejections.append(
                {
                    "leaf": candidate.action_leaf_index,
                    "rank": policy_rank,
                    "reason": reason,
                    "policy_score": policy_score,
                    "ik_prune_signatures_added": int(new_pruned),
                    "ik_prune_signature_count": int(len(ik_prune_signatures)),
                }
            )
            print(
                "online_pct_candidate_rejected "
                f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
                f"policy_score={policy_score:.6f} reason={reason} candidate_count={len(plan_candidates)} "
                f"ik_prune_added={new_pruned} ik_prune_total={len(ik_prune_signatures)}",
                flush=True,
            )
            continue

        fallback_used = candidate.action_leaf_index != first_candidate.action_leaf_index
        fallback_reason = None
        if fallback_used:
            fallback_reason = "robot_execution_prefilter"
            if not first_supported:
                fallback_reason = "unsupported_flip"
            elif rejections:
                fallback_reason = str(rejections[0].get("reason") or fallback_reason)
        print(
            "online_pct_candidate_selected "
            f"item={box_spec.item_index} rank={policy_rank} leaf={candidate.action_leaf_index} "
            f"policy_score={policy_score:.6f} "
            f"first_leaf={first_candidate.action_leaf_index} fallback_used={fallback_used} "
            f"fallback_reason={fallback_reason} feasible_execution_candidates={feasible_count}",
            flush=True,
        )
        return (
            candidate,
            first_candidate,
            adjusted_target,
            target_adjustment_diag,
            plan_candidates,
            first_supported,
            fallback_used,
            fallback_reason,
            rejections,
        )

    pct.reject(first_candidate)
    final_reject_reason = "ik_leaf_budget_exhausted" if any(
        row.get("reason") == "ik_leaf_budget_exhausted" for row in rejections
    ) else (rejections[0]["reason"] if rejections else "no_pct_execution_candidate")
    print(
        "online_pct_all_candidates_rejected "
        f"item={box_spec.item_index} first_leaf={first_candidate.action_leaf_index} "
        f"ik_leaf_checks={ik_leaf_checks} max_ik_leaf_checks={int(max_ik_leaf_checks)} "
        + json.dumps(rejections[:12], separators=(",", ":")),
        flush=True,
    )
    return (
        None,
        first_candidate,
        None,
        None,
        [],
        first_supported,
        bool(rejections),
        final_reject_reason,
        rejections,
    )


__all__ = ["_select_pct_execution_target"]
