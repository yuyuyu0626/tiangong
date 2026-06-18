"""Geometry, support, overlap, and actual-stack helpers for palletizing."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from move.palletizing_runtime import (
    ACTUAL_AABB_INFLATION_M,
    FROZEN_AABB_OVERLAP_LIMIT_M,
    HARD_OBSTACLE_OVERLAP_LIMIT_M,
    LOCAL_REPAIR_BOTTOM_GAP_MIN_M,
    LOCAL_REPAIR_MAX_XY_M,
    LOCAL_REPAIR_OVERLAP_LIMIT_M,
    ACTUAL_SCENE_INFEASIBLE_REASONS,
    PLACE_DESCEND_TILT_LIMIT_RAD,
    PLACE_DESCEND_XY_FAIL_LIMIT_M,
    PLACE_DESCENT_SERVO_MAX_XY_M,
    PLACE_DESCENT_SERVO_MAX_Z_M,
    ROBOT_NO_FEASIBLE_EXECUTION_REASONS,
    RUNTIME_TRACKING_REASONS,
    SOFT_STACK_BOUNDARY_TOLERANCE_M,
    STACK_BOUNDARY_MARGIN_M,
    STACK_INSIDE_GATE_TOLERANCE_M,
    STACK_VOLUME_M3,
    SUPPORT_LAYER_TOLERANCE_M,
    SUPPORT_XY_OVERLAP_MIN_M,
    TARGET_ADJUST_MARGIN_M,
    TARGET_ADJUST_MAX_XY_M,
    FrozenOverlapByRole,
    ActualPackingState,
    ExecutionResult,
)
from move.planning import BoxPlacement
from move.robot_placement import RobotPlacementTarget

def _box_aabb_xyzyaw(center: tuple[float, float, float], size: tuple[float, float, float], yaw: float) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=np.float64) * 0.5
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    extent = np.asarray([c * half[0] + s * half[1], s * half[0] + c * half[1], half[2]], dtype=np.float64)
    center_np = np.asarray(center, dtype=np.float64)
    return center_np - extent, center_np + extent


def _aabb_overlap_depth(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    if bool(np.any(overlap <= 0.0)):
        return 0.0
    return float(np.min(overlap))


def _boxplacement_overlap_depth(target: BoxPlacement, placed: list[BoxPlacement]) -> float:
    target_min = np.asarray(target.min_corner, dtype=np.float64)
    target_max = target_min + np.asarray(target.size, dtype=np.float64)
    max_overlap = 0.0
    for other in placed:
        other_min = np.asarray(other.min_corner, dtype=np.float64)
        other_max = other_min + np.asarray(other.size, dtype=np.float64)
        max_overlap = max(max_overlap, _aabb_overlap_depth(target_min, target_max, other_min, other_max))
    return max_overlap


def _build_actual_packing_state(box_records: list[dict], results: list[ExecutionResult], stack_scene) -> ActualPackingState:
    actual_frozen_boxes: list[dict] = []
    actual_aabbs: list[dict] = []
    inflated_actual_aabbs: list[dict] = []
    pct_target_aabbs: list[dict] = []
    deviation_from_pct: list[dict] = []
    actual_occupancy_grid = [[[0 for _z in range(10)] for _y in range(10)] for _x in range(10)]
    result_by_item = {int(r.item_index): r for r in results if r.placement is not None}

    for record in box_records:
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        item_index = int(record.get("item_index", -1))
        frozen = record["frozen_pose"]
        center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        yaw = float(frozen[3])
        size = tuple(float(v) for v in record["size_m"])
        actual_min, actual_max = _box_aabb_xyzyaw(center, size, yaw)
        inflated_min = actual_min - ACTUAL_AABB_INFLATION_M
        inflated_max = actual_max + ACTUAL_AABB_INFLATION_M
        actual_frozen_boxes.append(
            {
                "item_index": item_index,
                "center_world_m": [float(v) for v in center],
                "yaw_rad": yaw,
                "pitch_rad": float(record.get("frozen_pitch", 0.0)),
                "size_m": [float(v) for v in size],
                "asset_size_m": [float(v) for v in record.get("asset_size_m", size)],
                "sim_asset_size_m": [float(v) for v in record.get("sim_asset_size_m", record.get("asset_size_m", size))],
            }
        )
        actual_aabbs.append(
            {
                "item_index": item_index,
                "min_world_m": [float(v) for v in actual_min.tolist()],
                "max_world_m": [float(v) for v in actual_max.tolist()],
            }
        )
        inflated_actual_aabbs.append(
            {
                "item_index": item_index,
                "inflation_m": float(ACTUAL_AABB_INFLATION_M),
                "min_world_m": [float(v) for v in inflated_min.tolist()],
                "max_world_m": [float(v) for v in inflated_max.tolist()],
            }
        )

        local_min = np.asarray(_world_center_to_local(tuple(inflated_min.tolist()), stack_scene), dtype=np.float64)
        local_max = np.asarray(_world_center_to_local(tuple(inflated_max.tolist()), stack_scene), dtype=np.float64)
        lo = np.floor(local_min / 0.1).astype(int)
        hi = np.ceil(local_max / 0.1).astype(int)
        lo = np.clip(lo, 0, 10)
        hi = np.clip(hi, 0, 10)
        for gx in range(int(lo[0]), int(hi[0])):
            for gy in range(int(lo[1]), int(hi[1])):
                for gz in range(int(lo[2]), int(hi[2])):
                    actual_occupancy_grid[gx][gy][gz] = 1

        result = result_by_item.get(item_index)
        placement = result.placement if result is not None else None
        if placement is None:
            continue
        pct_center_local = placement.get("target_center_m") or placement.get("center_m")
        if pct_center_local is None and "min_corner_m" in placement and "placed_size_m" in placement:
            pct_min = np.asarray(placement["min_corner_m"], dtype=np.float64)
            pct_size = np.asarray(placement["placed_size_m"], dtype=np.float64)
            pct_center_local = (pct_min + pct_size * 0.5).tolist()
        target_size = tuple(float(v) for v in placement.get("placed_size_m", placement.get("target_aabb_size_m", size)))
        if pct_center_local is not None:
            pct_world = _local_center_to_world(tuple(float(v) for v in pct_center_local), stack_scene)
            pct_min, pct_max = _box_aabb_xyzyaw(pct_world, target_size, yaw)
            pct_target_aabbs.append(
                {
                    "item_index": item_index,
                    "center_world_m": [float(v) for v in pct_world],
                    "min_world_m": [float(v) for v in pct_min.tolist()],
                    "max_world_m": [float(v) for v in pct_max.tolist()],
                }
            )
            delta = [float(center[i] - pct_world[i]) for i in range(3)]
            deviation_from_pct.append(
                {
                    "item_index": item_index,
                    "delta_world_m": delta,
                    "norm_m": float(np.linalg.norm(np.asarray(delta, dtype=np.float64))),
                    "xy_norm_m": float(math.hypot(delta[0], delta[1])),
                }
            )

    return ActualPackingState(actual_frozen_boxes, actual_aabbs, inflated_actual_aabbs, actual_occupancy_grid, pct_target_aabbs, deviation_from_pct)


def _overlap_with_frozen_boxes(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    inflation_m: float = 0.0,
) -> float:
    cur_min, cur_max = _box_aabb_xyzyaw(center, size, yaw)
    max_overlap = 0.0
    for record in box_records:
        if int(record.get("item_index", -1)) == int(current_item_index):
            continue
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        frozen = record["frozen_pose"]
        frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        frozen_yaw = float(frozen[3])
        other_min, other_max = _box_aabb_xyzyaw(frozen_center, tuple(record["size_m"]), frozen_yaw)
        if inflation_m > 0.0:
            other_min = other_min - float(inflation_m)
            other_max = other_max + float(inflation_m)
        max_overlap = max(max_overlap, _aabb_overlap_depth(cur_min, cur_max, other_min, other_max))
    return max_overlap


def _overlap_with_frozen_boxes_by_role(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    support_top_z: float,
    inflation_m: float = 0.0,
) -> tuple[float, float]:
    """Split frozen-box overlap into allowed support contact and blocking obstacles."""

    detail = _overlap_with_frozen_boxes_by_role_detail(
        center,
        yaw,
        size,
        box_records,
        current_item_index,
        support_top_z,
        inflation_m=inflation_m,
    )
    return detail.support_overlap_m, detail.raw_obstacle_overlap_m


def _overlap_with_frozen_boxes_by_role_detail(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    support_top_z: float,
    inflation_m: float = 0.0,
) -> FrozenOverlapByRole:
    """Split frozen-box overlap without treating safety inflation as penetration.

    Raw overlap is real AABB interpenetration. Inflated overlap only indicates
    loss of clearance after expanding frozen boxes for a safety margin.  Dense
    filling must not reject a PCT leaf solely because the safety inflation
    touches it; release-time validation still checks real geometry.
    """

    cur_min, cur_max = _box_aabb_xyzyaw(center, size, yaw)
    support_overlap = 0.0
    raw_obstacle_overlap = 0.0
    inflated_obstacle_overlap = 0.0
    for record in box_records:
        if int(record.get("item_index", -1)) == int(current_item_index):
            continue
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        frozen = record["frozen_pose"]
        frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        frozen_yaw = float(frozen[3])
        frozen_size = tuple(float(v) for v in record["size_m"])
        raw_other_min, raw_other_max = _box_aabb_xyzyaw(frozen_center, frozen_size, frozen_yaw)
        raw_overlap_vec = np.minimum(cur_max, raw_other_max) - np.maximum(cur_min, raw_other_min)
        raw_depth = float(np.min(raw_overlap_vec)) if not bool(np.any(raw_overlap_vec <= 0.0)) else 0.0
        inflated_depth = raw_depth
        if inflation_m > 0.0:
            inflated_min = raw_other_min - float(inflation_m)
            inflated_max = raw_other_max + float(inflation_m)
            inflated_overlap_vec = np.minimum(cur_max, inflated_max) - np.maximum(cur_min, inflated_min)
            inflated_depth = float(np.min(inflated_overlap_vec)) if not bool(np.any(inflated_overlap_vec <= 0.0)) else 0.0
        if raw_depth <= 0.0 and inflated_depth <= 0.0:
            continue
        frozen_top_z = float(raw_other_max[2])
        xy_support = bool(np.all(_xy_overlap_depths(cur_min, cur_max, raw_other_min, raw_other_max) > SUPPORT_XY_OVERLAP_MIN_M))
        is_support = xy_support and abs(frozen_top_z - float(support_top_z)) <= SUPPORT_LAYER_TOLERANCE_M
        if is_support:
            support_overlap = max(support_overlap, raw_depth)
        else:
            raw_obstacle_overlap = max(raw_obstacle_overlap, raw_depth)
            inflated_obstacle_overlap = max(inflated_obstacle_overlap, inflated_depth)
    return FrozenOverlapByRole(
        support_overlap_m=float(support_overlap),
        raw_obstacle_overlap_m=float(raw_obstacle_overlap),
        inflated_obstacle_overlap_m=float(inflated_obstacle_overlap),
        safety_only_overlap_m=float(max(0.0, inflated_obstacle_overlap - raw_obstacle_overlap)),
    )


def _upper_frozen_cover(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    *,
    min_xy_overlap: float = SUPPORT_XY_OVERLAP_MIN_M,
    z_tolerance: float = 0.002,
) -> dict | None:
    """Return the nearest already-placed box that vertically covers this target.

    A dense PCT leaf can be geometrically valid in the final 1m cube but still
    impossible for a sequential robot if it asks us to insert a box underneath
    another frozen box.  That is a placement-order issue, not a PhysX issue.
    """

    cur_min, cur_max = _box_aabb_xyzyaw(center, size, yaw)
    best: dict | None = None
    for record in box_records:
        if int(record.get("item_index", -1)) == int(current_item_index):
            continue
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        frozen = record["frozen_pose"]
        frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        frozen_yaw = float(frozen[3])
        frozen_size = tuple(float(v) for v in record["size_m"])
        other_min, other_max = _box_aabb_xyzyaw(frozen_center, frozen_size, frozen_yaw)
        overlap_xy = _xy_overlap_depths(cur_min, cur_max, other_min, other_max)
        if bool(np.any(overlap_xy <= min_xy_overlap)):
            continue
        vertical_gap = float(other_min[2] - cur_max[2])
        if vertical_gap <= z_tolerance:
            continue
        entry = {
            "item_index": int(record.get("item_index", -1)),
            "vertical_gap_m": vertical_gap,
            "xy_overlap_x_m": float(overlap_xy[0]),
            "xy_overlap_y_m": float(overlap_xy[1]),
            "target_top_z_m": float(cur_max[2]),
            "cover_bottom_z_m": float(other_min[2]),
            "cover_top_z_m": float(other_max[2]),
        }
        if best is None or vertical_gap < float(best["vertical_gap_m"]):
            best = entry
    return best


def _completion_bucket(reason: str | None) -> str:
    if not reason:
        return "ended"
    if reason in {
        "target_utilization_reached",
        "max_items_reached",
        "max_arrivals_reached",
        "consecutive_no_fit_limit_reached",
        "pct_no_feasible_leaf",
        "robot_no_feasible_execution_candidate",
        "actual_scene_infeasible",
        "runtime_tracking_fail",
    }:
        return reason
    if reason in {"no_pct_execution_target", "unsupported_flip", "target_overlap_with_frozen", "target_adjustment_limited"}:
        return "pct_no_feasible_leaf"
    if reason in ROBOT_NO_FEASIBLE_EXECUTION_REASONS:
        return "robot_no_feasible_execution_candidate"
    if reason in ACTUAL_SCENE_INFEASIBLE_REASONS:
        return "actual_scene_infeasible"
    if reason in RUNTIME_TRACKING_REASONS:
        return "runtime_tracking_fail"
    return reason


def _place_descent_block_reason(
    *,
    xy_error: float,
    bottom_gap: float,
    obstacle_overlap: float,
    stack_xy_inside: bool,
    supported_freeze_inside: bool,
    descent_near_release_height: bool,
    tilt_abs: float,
    descent_corr_xy_m: float,
    place_servo_dz: float,
    bottom_gap_max: float,
) -> str | None:
    if obstacle_overlap > LOCAL_REPAIR_OVERLAP_LIMIT_M:
        return "obstacle_overlap_forbidden"
    if descent_near_release_height and (not stack_xy_inside or not supported_freeze_inside):
        return "inside_cube_violation"
    if bottom_gap < LOCAL_REPAIR_BOTTOM_GAP_MIN_M:
        return "bottom_gap_too_low"
    if bottom_gap > bottom_gap_max and place_servo_dz <= -PLACE_DESCENT_SERVO_MAX_Z_M + 1e-6:
        return "bottom_gap_too_high"
    if tilt_abs > PLACE_DESCEND_TILT_LIMIT_RAD:
        return "place_descend_tilt_exceeded"
    if xy_error >= PLACE_DESCEND_XY_FAIL_LIMIT_M:
        if descent_corr_xy_m >= PLACE_DESCENT_SERVO_MAX_XY_M - 1e-4:
            return "servo_correction_exceeded"
        return "place_descend_xy_drift"
    return None


def _frozen_overlap_xy_repair_vector(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    *,
    max_total_xy: float,
    support_top_z: float | None = None,
    inflation_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return a small XY correction that separates current box from frozen boxes.

    The vector is in world coordinates and only considers boxes on the same
    vertical layer. It is intentionally bounded: larger conflicts should make
    the execution candidate fail rather than rewriting the PCT placement.
    """

    current = np.asarray(center, dtype=np.float64).copy()
    total = np.zeros(2, dtype=np.float64)
    if support_top_z is None:
        initial_overlap = _overlap_with_frozen_boxes(center, yaw, size, box_records, current_item_index, inflation_m=inflation_m)
    else:
        _support_overlap, initial_overlap = _overlap_with_frozen_boxes_by_role(
            center,
            yaw,
            size,
            box_records,
            current_item_index,
            support_top_z,
            inflation_m=inflation_m,
        )

    for _ in range(6):
        cur_min, cur_max = _box_aabb_xyzyaw(tuple(current.tolist()), size, yaw)
        best: tuple[float, int, float] | None = None
        for record in box_records:
            if int(record.get("item_index", -1)) == int(current_item_index):
                continue
            if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
                continue
            frozen = record["frozen_pose"]
            frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
            frozen_yaw = float(frozen[3])
            frozen_size = tuple(float(v) for v in record["size_m"])
            raw_other_min, raw_other_max = _box_aabb_xyzyaw(frozen_center, frozen_size, frozen_yaw)
            other_min = raw_other_min.copy()
            other_max = raw_other_max.copy()
            if inflation_m > 0.0:
                other_min = other_min - float(inflation_m)
                other_max = other_max + float(inflation_m)
            overlap = np.minimum(cur_max, other_max) - np.maximum(cur_min, other_min)
            if overlap[2] <= FROZEN_AABB_OVERLAP_LIMIT_M:
                continue
            if support_top_z is not None:
                frozen_top_z = float(raw_other_max[2])
                if abs(frozen_top_z - float(support_top_z)) <= SUPPORT_LAYER_TOLERANCE_M:
                    overlap_xy = _xy_overlap_depths(cur_min, cur_max, raw_other_min, raw_other_max)
                    if bool(np.all(overlap_xy > SUPPORT_XY_OVERLAP_MIN_M)):
                        continue
            if overlap[0] <= FROZEN_AABB_OVERLAP_LIMIT_M or overlap[1] <= FROZEN_AABB_OVERLAP_LIMIT_M:
                continue
            axis = 0 if overlap[0] <= overlap[1] else 1
            direction = -1.0 if current[axis] <= frozen_center[axis] else 1.0
            needed = float(overlap[axis] + TARGET_ADJUST_MARGIN_M)
            if best is None or needed > best[0]:
                best = (needed, axis, direction)
        if best is None:
            break
        needed, axis, direction = best
        proposed = total.copy()
        proposed[axis] += direction * needed
        norm = float(np.linalg.norm(proposed))
        if norm > max_total_xy:
            if norm <= 1e-9:
                break
            proposed *= max_total_xy / norm
            total = proposed
            current[:2] = np.asarray(center[:2], dtype=np.float64) + total
            break
        total = proposed
        current[:2] = np.asarray(center[:2], dtype=np.float64) + total

    repaired_center = (float(current[0]), float(current[1]), float(current[2]))
    if support_top_z is None:
        remaining_overlap = _overlap_with_frozen_boxes(repaired_center, yaw, size, box_records, current_item_index, inflation_m=inflation_m)
    else:
        _support_overlap, remaining_overlap = _overlap_with_frozen_boxes_by_role(
            repaired_center,
            yaw,
            size,
            box_records,
            current_item_index,
            support_top_z,
            inflation_m=inflation_m,
        )
    return float(total[0]), float(total[1]), float(initial_overlap), float(remaining_overlap)


def _xy_overlap_depths(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.minimum(a_max[:2], b_max[:2]) - np.maximum(a_min[:2], b_min[:2])


def _xy_overlap_positive(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, eps: float = 1e-4) -> bool:
    overlap = _xy_overlap_depths(a_min, a_max, b_min, b_max)
    return bool(np.all(overlap > eps))


def _support_top_z_for_footprint(
    footprint_center: tuple[float, float, float],
    footprint_yaw: float,
    footprint_size: tuple[float, float, float],
    box_records: list[dict],
    current_item_index: int,
    pallet_surface_z: float,
) -> float:
    cur_min, cur_max = _box_aabb_xyzyaw(footprint_center, footprint_size, footprint_yaw)
    support_top_z = float(pallet_surface_z)
    planned_bottom_z = float(footprint_center[2]) - float(footprint_size[2]) * 0.5
    for record in box_records:
        if int(record.get("item_index", -1)) == int(current_item_index):
            continue
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        frozen = record["frozen_pose"]
        frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        frozen_yaw = float(frozen[3])
        frozen_size = tuple(float(v) for v in record["size_m"])
        other_min, other_max = _box_aabb_xyzyaw(frozen_center, frozen_size, frozen_yaw)
        overlap_xy = _xy_overlap_depths(cur_min, cur_max, other_min, other_max)
        if np.any(overlap_xy <= SUPPORT_XY_OVERLAP_MIN_M):
            continue
        frozen_top_z = float(frozen_center[2]) + frozen_size[2] * 0.5
        if abs(frozen_top_z - planned_bottom_z) <= SUPPORT_LAYER_TOLERANCE_M:
            support_top_z = max(support_top_z, frozen_top_z)
    return support_top_z


def _local_center_to_world(
    local_center: tuple[float, float, float],
    stack_scene,
) -> tuple[float, float, float]:
    return (
        float(stack_scene.pallet_center[0] - stack_scene.pallet_size[0] * 0.5 + local_center[0]),
        float(stack_scene.pallet_center[1] - stack_scene.pallet_size[1] * 0.5 + local_center[1]),
        float(stack_scene.pallet_center[2] + local_center[2]),
    )


def _world_center_to_local(
    world_center: tuple[float, float, float],
    stack_scene,
) -> tuple[float, float, float]:
    return (
        float(world_center[0] - (stack_scene.pallet_center[0] - stack_scene.pallet_size[0] * 0.5)),
        float(world_center[1] - (stack_scene.pallet_center[1] - stack_scene.pallet_size[1] * 0.5)),
        float(world_center[2] - stack_scene.pallet_center[2]),
    )


def _boxplacement_from_frozen_pose(
    name: str,
    frozen_pose: list[float] | tuple[float, float, float, float],
    aabb_size: tuple[float, float, float],
    stack_scene,
) -> BoxPlacement:
    local_center = _world_center_to_local(
        (float(frozen_pose[0]), float(frozen_pose[1]), float(frozen_pose[2])),
        stack_scene,
    )
    min_corner = tuple(float(local_center[i] - aabb_size[i] * 0.5) for i in range(3))
    return BoxPlacement(name, min_corner, tuple(float(v) for v in aabb_size))


def _frozen_box_metrics(box_records: list[dict], stack_scene) -> dict:
    placed_boxes = []
    total_volume = 0.0
    max_height = 0.0
    all_inside_strict = True
    all_inside_soft = True
    min_inside_clearance = float("inf")
    pallet_x_min = float(stack_scene.pallet_center[0] - stack_scene.pallet_size[0] * 0.5)
    pallet_y_min = float(stack_scene.pallet_center[1] - stack_scene.pallet_size[1] * 0.5)
    pallet_z_min = float(stack_scene.pallet_center[2])
    pallet_x_max = pallet_x_min + float(stack_scene.pallet_size[0])
    pallet_y_max = pallet_y_min + float(stack_scene.pallet_size[1])
    pallet_z_max = pallet_z_min + 1.0
    for record in box_records:
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        pose = [float(v) for v in record["frozen_pose"]]
        size = tuple(float(v) for v in record["size_m"])
        aabb_min, aabb_max = _box_aabb_xyzyaw((pose[0], pose[1], pose[2]), size, pose[3])
        total_volume += float(size[0] * size[1] * size[2])
        max_height = max(max_height, float(aabb_max[2] - pallet_z_min))
        clearances = (
            float(aabb_min[0]) - pallet_x_min,
            pallet_x_max - float(aabb_max[0]),
            float(aabb_min[1]) - pallet_y_min,
            pallet_y_max - float(aabb_max[1]),
            float(aabb_min[2]) - pallet_z_min,
            pallet_z_max - float(aabb_max[2]),
        )
        inside_clearance = min(clearances)
        min_inside_clearance = min(min_inside_clearance, float(inside_clearance))
        inside_strict = bool(inside_clearance >= -1e-5)
        inside_soft = bool(inside_clearance >= -SOFT_STACK_BOUNDARY_TOLERANCE_M)
        all_inside_strict = all_inside_strict and inside_strict
        all_inside_soft = all_inside_soft and inside_soft
        placed_boxes.append(
            {
                "item_index": int(record["item_index"]),
                "pose": pose,
                "size_m": list(size),
                "asset_size_m": list(record.get("asset_size_m", size)),
                "sim_asset_size_m": list(record.get("sim_asset_size_m", record.get("asset_size_m", size))),
                "frozen_pitch": float(record.get("frozen_pitch", 0.0)),
                "aabb_min_world_m": [float(v) for v in aabb_min.tolist()],
                "aabb_max_world_m": [float(v) for v in aabb_max.tolist()],
                "inside_1m_cube": bool(inside_soft),
                "inside_1m_cube_strict": bool(inside_strict),
                "inside_1m_cube_soft": bool(inside_soft),
                "inside_clearance_m": float(inside_clearance),
            }
        )
    return {
        "placed_boxes": placed_boxes,
        "utilization": float(total_volume / STACK_VOLUME_M3),
        "max_height_m": float(max_height),
        "all_boxes_inside_1m_cube": bool(all_inside_soft),
        "all_boxes_inside_1m_cube_strict": bool(all_inside_strict),
        "all_boxes_inside_1m_cube_soft": bool(all_inside_soft),
        "min_inside_clearance_m": float(min_inside_clearance if placed_boxes else 0.0),
        "soft_stack_boundary_tolerance_m": float(SOFT_STACK_BOUNDARY_TOLERANCE_M),
    }


def _stack_bounds(stack_scene) -> tuple[float, float, float, float, float, float]:
    pallet_x_min = float(stack_scene.pallet_center[0] - stack_scene.pallet_size[0] * 0.5)
    pallet_y_min = float(stack_scene.pallet_center[1] - stack_scene.pallet_size[1] * 0.5)
    pallet_z_min = float(stack_scene.pallet_center[2])
    return (
        pallet_x_min,
        pallet_x_min + float(stack_scene.pallet_size[0]),
        pallet_y_min,
        pallet_y_min + float(stack_scene.pallet_size[1]),
        pallet_z_min,
        pallet_z_min + 1.0,
    )


def _xy_half_extents_for_yaw(size: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    c = abs(math.cos(float(yaw)))
    s = abs(math.sin(float(yaw)))
    return (
        c * float(size[0]) * 0.5 + s * float(size[1]) * 0.5,
        s * float(size[0]) * 0.5 + c * float(size[1]) * 0.5,
    )


def _aabb_inside_stack(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    stack_scene,
    margin: float = 0.0,
    tolerance_m: float | None = None,
) -> tuple[bool, float]:
    tolerance = STACK_INSIDE_GATE_TOLERANCE_M if tolerance_m is None else float(tolerance_m)
    x_min, x_max, y_min, y_max, z_min, z_max = _stack_bounds(stack_scene)
    clearances = (
        float(aabb_min[0]) - (x_min + margin),
        (x_max - margin) - float(aabb_max[0]),
        float(aabb_min[1]) - (y_min + margin),
        (y_max - margin) - float(aabb_max[1]),
        float(aabb_min[2]) - z_min,
        z_max - float(aabb_max[2]),
    )
    min_clearance = min(clearances)
    return bool(min_clearance >= -tolerance), float(min_clearance)


def _actor_inside_stack(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    stack_scene,
    margin: float = 0.0,
    tolerance_m: float | None = None,
) -> tuple[bool, float]:
    aabb_min, aabb_max = _box_aabb_xyzyaw(center, size, yaw)
    return _aabb_inside_stack(aabb_min, aabb_max, stack_scene, margin=margin, tolerance_m=tolerance_m)


def _target_aabb_inside_stack(
    center: tuple[float, float, float],
    target_aabb_size: tuple[float, float, float],
    stack_scene,
    margin: float = 0.0,
    tolerance_m: float | None = None,
) -> tuple[bool, float]:
    half = np.asarray(target_aabb_size, dtype=np.float64) * 0.5
    center_np = np.asarray(center, dtype=np.float64)
    return _aabb_inside_stack(center_np - half, center_np + half, stack_scene, margin=margin, tolerance_m=tolerance_m)


def _actor_xy_inside_stack(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    stack_scene,
    margin: float = 0.0,
) -> tuple[bool, float]:
    aabb_min, aabb_max = _box_aabb_xyzyaw(center, size, yaw)
    x_min, x_max, y_min, y_max, _z_min, _z_max = _stack_bounds(stack_scene)
    clearances = (
        float(aabb_min[0]) - (x_min + margin),
        (x_max - margin) - float(aabb_max[0]),
        float(aabb_min[1]) - (y_min + margin),
        (y_max - margin) - float(aabb_max[1]),
    )
    min_clearance = min(clearances)
    return bool(min_clearance >= -STACK_INSIDE_GATE_TOLERANCE_M), float(min_clearance)


def _stack_boundary_xy_repair_vector(
    center: tuple[float, float, float],
    yaw: float,
    size: tuple[float, float, float],
    stack_scene,
    *,
    max_total_xy: float,
    target_clearance: float = 0.001,
) -> tuple[float, float, float]:
    """Small XY nudge that moves the current oriented AABB back inside the pallet bounds."""
    aabb_min, aabb_max = _box_aabb_xyzyaw(center, size, yaw)
    x_min, x_max, y_min, y_max, _z_min, _z_max = _stack_bounds(stack_scene)
    dx = 0.0
    dy = 0.0
    left_violation = (x_min + target_clearance) - float(aabb_min[0])
    right_violation = float(aabb_max[0]) - (x_max - target_clearance)
    bottom_violation = (y_min + target_clearance) - float(aabb_min[1])
    top_violation = float(aabb_max[1]) - (y_max - target_clearance)
    if left_violation > 0.0:
        dx += left_violation
    if right_violation > 0.0:
        dx -= right_violation
    if bottom_violation > 0.0:
        dy += bottom_violation
    if top_violation > 0.0:
        dy -= top_violation
    norm = math.hypot(dx, dy)
    if norm > max_total_xy:
        scale = max_total_xy / max(norm, 1e-9)
        dx *= scale
        dy *= scale
    return float(dx), float(dy), float(norm)


def _target_from_robot_center(robot_target: RobotPlacementTarget) -> BoxPlacement:
    size = tuple(float(v) for v in robot_target.target_aabb_size_m)
    center = tuple(float(v) for v in robot_target.target_center_m)
    min_corner = tuple(center[i] - size[i] * 0.5 for i in range(3))
    return BoxPlacement(f"item_{robot_target.item_index:02d}", min_corner, size)


def _adjust_robot_target_for_actual_frozen(
    robot_target: RobotPlacementTarget,
    box_records: list[dict],
    stack_scene,
) -> tuple[RobotPlacementTarget, dict]:
    """Shift the execution target minimally so actual frozen AABBs do not penetrate.

    PCT still owns the leaf and ideal target. This adjustment only compensates
    for small execution error from prior items, keeping the new box inside the
    1m x 1m pallet footprint and preserving the PCT target yaw/AABB.
    """

    size = tuple(float(v) for v in robot_target.target_aabb_size_m)
    actor_size = tuple(float(v) for v in robot_target.execution_size_m)
    original_local = np.asarray(robot_target.target_center_m, dtype=np.float64)
    current_world = np.asarray(_local_center_to_world(tuple(original_local), stack_scene), dtype=np.float64)
    original_world = current_world.copy()
    target_yaw = float(robot_target.target_yaw_options[0]) if robot_target.target_yaw_options else 0.0

    def clamp_inside_stack(world: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max, _z_min, _z_max = _stack_bounds(stack_scene)
        half_x, half_y = _xy_half_extents_for_yaw(actor_size, target_yaw)
        world[0] = min(max(float(world[0]), x_min + half_x + STACK_BOUNDARY_MARGIN_M), x_max - half_x - STACK_BOUNDARY_MARGIN_M)
        world[1] = min(max(float(world[1]), y_min + half_y + STACK_BOUNDARY_MARGIN_M), y_max - half_y - STACK_BOUNDARY_MARGIN_M)
        return world

    current_world = clamp_inside_stack(current_world)

    for _ in range(12):
        changed = False
        cur_min, cur_max = _box_aabb_xyzyaw(tuple(current_world.tolist()), size, target_yaw)
        for record in box_records:
            if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
                continue
            frozen = record["frozen_pose"]
            frozen_center = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
            frozen_yaw = float(frozen[3])
            frozen_size = tuple(float(v) for v in record["size_m"])
            other_min, other_max = _box_aabb_xyzyaw(frozen_center, frozen_size, frozen_yaw)
            overlap = np.minimum(cur_max, other_max) - np.maximum(cur_min, other_min)
            if bool(np.any(overlap <= FROZEN_AABB_OVERLAP_LIMIT_M)):
                continue
            # Only separate boxes that occupy the same vertical layer. Stacked
            # placements are handled by support_top_z and must remain allowed.
            if overlap[2] <= FROZEN_AABB_OVERLAP_LIMIT_M:
                continue
            if overlap[0] <= overlap[1]:
                direction = -1.0 if current_world[0] <= frozen_center[0] else 1.0
                current_world[0] += direction * float(overlap[0] + TARGET_ADJUST_MARGIN_M)
            else:
                direction = -1.0 if current_world[1] <= frozen_center[1] else 1.0
                current_world[1] += direction * float(overlap[1] + TARGET_ADJUST_MARGIN_M)
            changed = True

            current_world = clamp_inside_stack(current_world)
            cur_min, cur_max = _box_aabb_xyzyaw(tuple(current_world.tolist()), size, target_yaw)
        if not changed:
            break

    adjusted_local = np.asarray(_world_center_to_local(tuple(current_world.tolist()), stack_scene), dtype=np.float64)
    delta = current_world[:2] - original_world[:2]
    adjustment_m = float(np.linalg.norm(delta))
    diagnostics = {
        "pct_target_center_m": [float(v) for v in robot_target.target_center_m],
        "execution_target_center_m": [float(v) for v in adjusted_local.tolist()],
        "target_adjustment_xy_m": adjustment_m,
        "target_adjustment_dx_m": float(delta[0]),
        "target_adjustment_dy_m": float(delta[1]),
        "target_adjustment_limited": bool(adjustment_m > TARGET_ADJUST_MAX_XY_M),
    }
    if adjustment_m > 1e-6:
        print(
            "online_target_adjustment "
            f"item={robot_target.item_index} "
            f"pct_center=({robot_target.target_center_m[0]:.4f},{robot_target.target_center_m[1]:.4f},{robot_target.target_center_m[2]:.4f}) "
            f"exec_center=({adjusted_local[0]:.4f},{adjusted_local[1]:.4f},{adjusted_local[2]:.4f}) "
            f"delta_xy=({delta[0]:.4f},{delta[1]:.4f}) norm={adjustment_m:.4f}",
            flush=True,
        )
    if adjustment_m > TARGET_ADJUST_MAX_XY_M:
        return robot_target, diagnostics
    return replace(robot_target, target_center_m=tuple(float(v) for v in adjusted_local.tolist())), diagnostics


def _oriented_box_bottom_z(center: np.ndarray, rot: np.ndarray, size: tuple[float, float, float]) -> float:
    half = np.asarray(size, dtype=np.float64) * 0.5
    min_z = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.asarray([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                world = center + rot @ local
                min_z = min(min_z, float(world[2]))
    return min_z


def _matrix_yaw_pitch_roll(rot: np.ndarray) -> tuple[float, float, float]:
    value = min(max(-float(rot[2, 0]), -1.0), 1.0)
    pitch = math.asin(value)
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    return yaw, pitch, roll


def _aabb_interval_on_axis(center_xy: np.ndarray, half_xy: np.ndarray, axis_xy: np.ndarray) -> tuple[float, float]:
    c = float(np.dot(center_xy, axis_xy))
    r = abs(float(axis_xy[0])) * float(half_xy[0]) + abs(float(axis_xy[1])) * float(half_xy[1])
    return c - r, c + r


def _approach_blockage_penalty(target: BoxPlacement, placed: list[BoxPlacement], approach_yaw: float, stand_off: float) -> float:
    """Penalize approach corridors blocked by already placed boxes at the same vertical layer.

    PCT may legitimately choose adjacent or stacked AABBs.  This only scores the
    robot's approach side; it does not reject the PCT placement.
    """
    target_center = np.asarray([
        target.min_corner[0] + target.size[0] * 0.5,
        target.min_corner[1] + target.size[1] * 0.5,
    ], dtype=np.float64)
    target_half = np.asarray([target.size[0] * 0.5, target.size[1] * 0.5], dtype=np.float64)
    target_z_min = float(target.min_corner[2])
    target_z_max = float(target.min_corner[2] + target.size[2])

    # Root is placed at target - stand_off * approach_dir, so obstacles between
    # root and target lie in -approach_dir from the target.
    approach_dir = np.asarray([math.cos(approach_yaw), math.sin(approach_yaw)], dtype=np.float64)
    root_side_dir = -approach_dir
    lateral = np.asarray([-root_side_dir[1], root_side_dir[0]], dtype=np.float64)
    target_lat_min, target_lat_max = _aabb_interval_on_axis(target_center, target_half, lateral)
    target_front_min, target_front_max = _aabb_interval_on_axis(target_center, target_half, root_side_dir)

    penalty = 0.0
    for other in placed:
        other_z_min = float(other.min_corner[2])
        other_z_max = float(other.min_corner[2] + other.size[2])
        vertical_overlap = min(target_z_max, other_z_max) - max(target_z_min, other_z_min)
        if vertical_overlap <= 1e-4:
            continue
        other_center = np.asarray([
            other.min_corner[0] + other.size[0] * 0.5,
            other.min_corner[1] + other.size[1] * 0.5,
        ], dtype=np.float64)
        other_half = np.asarray([other.size[0] * 0.5, other.size[1] * 0.5], dtype=np.float64)
        other_lat_min, other_lat_max = _aabb_interval_on_axis(other_center, other_half, lateral)
        lateral_overlap = min(target_lat_max, other_lat_max) - max(target_lat_min, other_lat_min)
        if lateral_overlap <= -0.01:
            continue
        other_front_min, other_front_max = _aabb_interval_on_axis(other_center, other_half, root_side_dir)
        gap_from_target_side = other_front_min - target_front_max
        # Only boxes on the root side of the target can block the approach.
        # Adjacent boxes on the opposite side are legal PCT placements, not
        # corridor obstacles.
        if gap_from_target_side < -0.02:
            continue
        if gap_from_target_side > stand_off + 0.10:
            continue
        penalty += 10.0 + max(0.0, 0.05 - gap_from_target_side) * 100.0 + max(0.0, lateral_overlap) * 10.0
    return penalty

__all__ = ['_box_aabb_xyzyaw', '_aabb_overlap_depth', '_boxplacement_overlap_depth', '_build_actual_packing_state', '_overlap_with_frozen_boxes', '_overlap_with_frozen_boxes_by_role', '_overlap_with_frozen_boxes_by_role_detail', '_upper_frozen_cover', '_completion_bucket', '_place_descent_block_reason', '_frozen_overlap_xy_repair_vector', '_xy_overlap_depths', '_xy_overlap_positive', '_support_top_z_for_footprint', '_local_center_to_world', '_world_center_to_local', '_boxplacement_from_frozen_pose', '_frozen_box_metrics', '_stack_bounds', '_xy_half_extents_for_yaw', '_aabb_inside_stack', '_actor_inside_stack', '_target_aabb_inside_stack', '_actor_xy_inside_stack', '_stack_boundary_xy_repair_vector', '_target_from_robot_center', '_adjust_robot_target_for_actual_frozen', '_oriented_box_bottom_z', '_matrix_yaw_pitch_roll', '_aabb_interval_on_axis', '_approach_blockage_penalty']
