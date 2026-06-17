"""Runtime constants and small data models for online robot palletizing."""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

from move.online_palletizing import DISCRETE_SIZES_M, BoxSpec

STAND_OFF_CANDIDATES = (0.18, 0.22, 0.25, 0.28, 0.35, 0.40, 0.45, 0.50)
IK_CANDIDATE_LIMIT = int(os.environ.get("MOVE_IK_CANDIDATE_LIMIT", "8"))
CONTACT_GAP_LIMIT_M = 0.015
PRE_ATTACH_DRIFT_LIMIT_M = 0.020
PRE_ATTACH_YAW_LIMIT_RAD = math.radians(5.0)
PRE_ATTACH_CENTER_Z_LIMIT_M = 0.035
PLACE_ERROR_LIMIT_M = 0.035
YAW_ERROR_LIMIT_RAD = math.radians(5.0)
PLACE_ABOVE_XY_LIMIT_M = 0.015
PLACE_ABOVE_YAW_LIMIT_RAD = math.radians(3.0)
PLACE_ABOVE_TILT_LIMIT_RAD = math.radians(10.0)
PLACE_ABOVE_MIN_BOTTOM_GAP_M = 0.020
PLACE_TILT_WARNING_RAD = math.radians(3.0)
PLACE_DESCEND_XY_LIMIT_M = 0.020
PLACE_DESCEND_XY_FAIL_LIMIT_M = 0.030
PLACE_DESCEND_TILT_LIMIT_RAD = math.radians(10.0)
PLACE_DESCENT_SERVO_MAX_XY_M = 0.030
PLACE_DESCENT_SERVO_MAX_YAW_RAD = math.radians(3.0)
PLACE_DESCENT_SERVO_STEP_XY_M = 0.0018
PLACE_DESCENT_SERVO_STEP_YAW_RAD = math.radians(0.18)
PLACE_DESCENT_SERVO_MAX_Z_M = 0.220
PLACE_DESCENT_SERVO_STEP_Z_M = 0.0020
PLACE_SERVO_MAX_XY_M = 0.035
PLACE_SERVO_MAX_YAW_RAD = math.radians(5.0)
PLACE_SERVO_STEP_XY_M = 0.0025
PLACE_SERVO_STEP_YAW_RAD = math.radians(0.25)
FINAL_PLACE_ERROR_LIMIT_M = 0.020
PRE_RELEASE_ERROR_LIMIT_M = PLACE_ERROR_LIMIT_M
EARLY_RELEASE_ERROR_LIMIT_M = 0.018
VERIFIED_KINEMATIC_ERROR_LIMIT_M = 0.020
VERIFIED_KINEMATIC_YAW_LIMIT_RAD = math.radians(2.0)
BOTTOM_GAP_MIN_M = 0.002
BOTTOM_GAP_MAX_M = 0.006
FROZEN_AABB_OVERLAP_LIMIT_M = 0.002
SOFT_OBSTACLE_OVERLAP_LIMIT_M = 0.025
HARD_OBSTACLE_OVERLAP_LIMIT_M = 0.035
SOFT_STACK_BOUNDARY_TOLERANCE_M = 0.012
LOCAL_REPAIR_OVERLAP_LIMIT_M = 0.030
LOCAL_REPAIR_MAX_XY_M = 0.050
LOCAL_REPAIR_STEP_XY_M = 0.0020
LOCAL_REPAIR_BOTTOM_GAP_MIN_M = -0.006
TARGET_OVERLAP_REPAIR_LIMIT_M = 0.020
ACTUAL_AABB_INFLATION_M = 0.003
STACK_INSIDE_GATE_TOLERANCE_M = 0.002
SUPPORT_LAYER_TOLERANCE_M = 0.025
SUPPORT_XY_OVERLAP_MIN_M = 0.010
TARGET_ADJUST_MARGIN_M = 0.003
TARGET_ADJUST_MAX_XY_M = 0.050
STACK_BOUNDARY_MARGIN_M = 0.003
CONTACT_EXPLOSION_ANGULAR_SPEED_RADPS = 5.0
DEMO_YAW_ERROR_LIMIT_RAD = math.radians(8.0)
PLACE_HOLD_ATTACHED_FRAMES = 60
SIDE_OPEN_NO_COLLISION_FRAMES = 45
RETREAT_NO_COLLISION_FRAMES = 45
POST_RELEASE_RETREAT_FRAMES = 90
SETTLE_FRAMES = 90
FINAL_LINEAR_SPEED_LIMIT_MPS = 0.03
FINAL_ANGULAR_SPEED_LIMIT_RADPS = 0.10
STACK_VOLUME_M3 = 1.0
DENSE_FILL_UTILIZATION_THRESHOLD = 0.95
DENSE_FILL_COMMIT_ERROR_LIMIT_M = 0.010
DENSE_FILL_ADAPTIVE_LIMITS = (
    (0.80, 0.020),
    (0.90, 0.018),
    (0.97, 0.015),
    (0.995, 0.012),
    (1.01, 0.010),
)
DEFAULT_MAX_CONSECUTIVE_NO_FIT_SKIPS = 80
DEFAULT_MAX_INFINITE_ARRIVALS = 500
DEFAULT_MAX_PCT_LEAF_CHECKS = 0
DEFAULT_MAX_IK_LEAF_CHECKS = 8
DEFAULT_FINE_FILL_UTILIZATION_THRESHOLD = 0.80
DEFAULT_FINE_FILL_SKIP_THRESHOLD = 8
FINE_FILL_SMALL_SIZES_M = (0.1, 0.2, 0.3)
FINE_FILL_TINY_SIZES_M = (0.1, 0.2)
FINE_FILL_MICRO_SIZES_M = (0.1,)
RETRYABLE_CANDIDATE_REASONS = {
    "ik_infeasible",
    "hand_frame_tracking_fail",
    "place_above_target_not_aligned",
    "place_descend_geometry_violation",
    "place_descend_xy_drift",
    "bottom_gap_too_low",
    "bottom_gap_too_high",
    "obstacle_overlap_forbidden",
    "inside_cube_violation",
    "approach_corridor_blocked",
    "servo_correction_exceeded",
    "place_descend_tilt_exceeded",
    "pre_release_pose_invalid",
    "dense_fill_actual_vs_pct_error",
    "placement_validation_failed",
    "target_under_existing_box",
}


def _dense_fill_commit_error_limit(utilization_if_committed: float) -> float:
    """Tighten actual-vs-PCT error only as the container approaches full."""
    util = float(utilization_if_committed)
    for util_upper, limit in DENSE_FILL_ADAPTIVE_LIMITS:
        if util < float(util_upper):
            return float(limit)
    return float(DENSE_FILL_COMMIT_ERROR_LIMIT_M)


def _sample_online_box_spec(
    rng: random.Random,
    item_index: int,
    *,
    current_utilization: float,
    consecutive_no_fit_skips: int,
    fine_fill_enabled: bool,
    fine_fill_utilization_threshold: float,
    fine_fill_skip_threshold: int,
) -> tuple[BoxSpec, str]:
    """Sample the next online arrival.

    The normal mode preserves the original benchmark behavior: each dimension is
    uniformly sampled from the required discrete set.  Once the actual stack is
    already dense, waiting for a uniformly random large item wastes most trials,
    so an optional fine-fill mode biases later arrivals toward small legal sizes.
    """
    mode = "uniform_random"
    pool = tuple(float(v) for v in DISCRETE_SIZES_M)
    if fine_fill_enabled:
        util = float(current_utilization)
        skip_threshold = max(1, int(fine_fill_skip_threshold))
        skips = int(consecutive_no_fit_skips)
        if util >= 0.85 or skips >= max(3, skip_threshold // 3):
            mode = "fine_fill_micro"
            pool = FINE_FILL_MICRO_SIZES_M
        elif util >= 0.72 or skips >= 2:
            mode = "fine_fill_tiny"
            pool = FINE_FILL_TINY_SIZES_M
        elif util >= min(float(fine_fill_utilization_threshold), 0.70):
            mode = "fine_fill_small"
            pool = FINE_FILL_SMALL_SIZES_M
    return (
        BoxSpec(
            item_index=int(item_index),
            original_size_m=tuple(float(rng.choice(pool)) for _ in range(3)),
        ),
        mode,
    )
ACTUAL_SCENE_INFEASIBLE_REASONS = {
    "actual_target_obstacle_overlap",
    "bottom_gap_too_low",
    "bottom_gap_too_high",
    "obstacle_overlap_forbidden",
    "inside_cube_violation",
    "approach_corridor_blocked",
    "target_under_existing_box",
}
ROBOT_NO_FEASIBLE_EXECUTION_REASONS = {
    "ik_infeasible",
    "no_execution_candidates",
    "no_ik_feasible_execution_candidate",
    "all_execution_candidates_failed",
}
RUNTIME_TRACKING_REASONS = {
    "hand_frame_tracking_fail",
    "place_above_target_not_aligned",
    "place_descend_xy_drift",
    "servo_correction_exceeded",
    "place_descend_tilt_exceeded",
    "pre_release_pose_invalid",
    "dense_fill_actual_vs_pct_error",
    "placement_validation_failed",
}


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    reason: str
    item_index: int
    placement: dict | None
    place_error_m: float | None
    yaw_error_rad: float | None
    pre_attach_gap_m: float | None
    frames: int
    strict_success: bool = False
    demo_success: bool = False
    final_linear_speed_mps: float | None = None
    final_angular_speed_radps: float | None = None
    placement_mode: str = "not_released"
    freeze_pose_source: str | None = None
    snap_to_pct_target: bool = False
    bottom_gap_m: float | None = None
    overlap_with_frozen_boxes_m: float | None = None
    pre_release_error_m: float | None = None
    final_error_m: float | None = None
    contact_explosion: bool = False
    support_top_z_m: float | None = None
    actual_bottom_z_m: float | None = None
    planned_release_center_z_m: float | None = None
    pre_release_error_xyz_m: tuple[float, float, float] | None = None
    pct_first_candidate_supported: bool = True
    pct_fallback_used: bool = False
    pct_fallback_reason: str | None = None
    pct_first_leaf_index: int | None = None
    pct_selected_leaf_index: int | None = None
    local_adjustment_xy_m: float = 0.0
    local_adjustment_z_m: float = 0.0
    local_adjusted: bool = False


@dataclass(frozen=True)
class ExecutionCandidate:
    target_center: tuple[float, float, float]
    target_yaw: float
    target_aabb_size: tuple[float, float, float]
    actor_size: tuple[float, float, float]
    execution_size: tuple[float, float, float]
    approach_yaw: float
    final_root: tuple[float, float, float, float]
    stand_off: float
    source_yaw: float
    source_pitch: float
    score: float
    reject_reason: str | None
    diagnostics: dict


@dataclass(frozen=True)
class ActualPackingState:
    actual_frozen_boxes: list[dict]
    actual_aabbs: list[dict]
    inflated_actual_aabbs: list[dict]
    actual_occupancy_grid: list[list[list[int]]]
    pct_target_aabbs: list[dict]
    deviation_from_pct: list[dict]


@dataclass(frozen=True)
class FrozenOverlapByRole:
    support_overlap_m: float
    raw_obstacle_overlap_m: float
    inflated_obstacle_overlap_m: float
    safety_only_overlap_m: float

__all__ = ['STAND_OFF_CANDIDATES', 'IK_CANDIDATE_LIMIT', 'CONTACT_GAP_LIMIT_M', 'PRE_ATTACH_DRIFT_LIMIT_M', 'PRE_ATTACH_YAW_LIMIT_RAD', 'PRE_ATTACH_CENTER_Z_LIMIT_M', 'PLACE_ERROR_LIMIT_M', 'YAW_ERROR_LIMIT_RAD', 'PLACE_ABOVE_XY_LIMIT_M', 'PLACE_ABOVE_YAW_LIMIT_RAD', 'PLACE_ABOVE_TILT_LIMIT_RAD', 'PLACE_ABOVE_MIN_BOTTOM_GAP_M', 'PLACE_TILT_WARNING_RAD', 'PLACE_DESCEND_XY_LIMIT_M', 'PLACE_DESCEND_XY_FAIL_LIMIT_M', 'PLACE_DESCEND_TILT_LIMIT_RAD', 'PLACE_DESCENT_SERVO_MAX_XY_M', 'PLACE_DESCENT_SERVO_MAX_YAW_RAD', 'PLACE_DESCENT_SERVO_STEP_XY_M', 'PLACE_DESCENT_SERVO_STEP_YAW_RAD', 'PLACE_DESCENT_SERVO_MAX_Z_M', 'PLACE_DESCENT_SERVO_STEP_Z_M', 'PLACE_SERVO_MAX_XY_M', 'PLACE_SERVO_MAX_YAW_RAD', 'PLACE_SERVO_STEP_XY_M', 'PLACE_SERVO_STEP_YAW_RAD', 'FINAL_PLACE_ERROR_LIMIT_M', 'PRE_RELEASE_ERROR_LIMIT_M', 'EARLY_RELEASE_ERROR_LIMIT_M', 'VERIFIED_KINEMATIC_ERROR_LIMIT_M', 'VERIFIED_KINEMATIC_YAW_LIMIT_RAD', 'BOTTOM_GAP_MIN_M', 'BOTTOM_GAP_MAX_M', 'FROZEN_AABB_OVERLAP_LIMIT_M', 'SOFT_OBSTACLE_OVERLAP_LIMIT_M', 'HARD_OBSTACLE_OVERLAP_LIMIT_M', 'SOFT_STACK_BOUNDARY_TOLERANCE_M', 'LOCAL_REPAIR_OVERLAP_LIMIT_M', 'LOCAL_REPAIR_MAX_XY_M', 'LOCAL_REPAIR_STEP_XY_M', 'LOCAL_REPAIR_BOTTOM_GAP_MIN_M', 'TARGET_OVERLAP_REPAIR_LIMIT_M', 'ACTUAL_AABB_INFLATION_M', 'STACK_INSIDE_GATE_TOLERANCE_M', 'SUPPORT_LAYER_TOLERANCE_M', 'SUPPORT_XY_OVERLAP_MIN_M', 'TARGET_ADJUST_MARGIN_M', 'TARGET_ADJUST_MAX_XY_M', 'STACK_BOUNDARY_MARGIN_M', 'CONTACT_EXPLOSION_ANGULAR_SPEED_RADPS', 'DEMO_YAW_ERROR_LIMIT_RAD', 'PLACE_HOLD_ATTACHED_FRAMES', 'SIDE_OPEN_NO_COLLISION_FRAMES', 'RETREAT_NO_COLLISION_FRAMES', 'POST_RELEASE_RETREAT_FRAMES', 'SETTLE_FRAMES', 'FINAL_LINEAR_SPEED_LIMIT_MPS', 'FINAL_ANGULAR_SPEED_LIMIT_RADPS', 'STACK_VOLUME_M3', 'DENSE_FILL_UTILIZATION_THRESHOLD', 'DENSE_FILL_COMMIT_ERROR_LIMIT_M', 'DENSE_FILL_ADAPTIVE_LIMITS', 'DEFAULT_MAX_CONSECUTIVE_NO_FIT_SKIPS', 'DEFAULT_MAX_INFINITE_ARRIVALS', 'DEFAULT_MAX_PCT_LEAF_CHECKS', 'DEFAULT_MAX_IK_LEAF_CHECKS', 'DEFAULT_FINE_FILL_UTILIZATION_THRESHOLD', 'DEFAULT_FINE_FILL_SKIP_THRESHOLD', 'FINE_FILL_SMALL_SIZES_M', 'FINE_FILL_TINY_SIZES_M', 'FINE_FILL_MICRO_SIZES_M', 'RETRYABLE_CANDIDATE_REASONS', '_dense_fill_commit_error_limit', '_sample_online_box_spec', 'ACTUAL_SCENE_INFEASIBLE_REASONS', 'ROBOT_NO_FEASIBLE_EXECUTION_REASONS', 'RUNTIME_TRACKING_REASONS', 'ExecutionResult', 'ExecutionCandidate', 'ActualPackingState', 'FrozenOverlapByRole']
