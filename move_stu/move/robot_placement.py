#!/usr/bin/env python3
"""Robot-facing placement targets derived from PCT candidates."""
from __future__ import annotations

import math
from dataclasses import dataclass

from move.pct_policy_bridge import PctCandidate


@dataclass(frozen=True)
class OrientationClassification:
    type: str
    target_yaw_options: tuple[float, ...]
    source_pitch_options: tuple[float, ...]
    execution_size_m: tuple[float, float, float]
    supported: bool


@dataclass(frozen=True)
class RobotPlacementTarget:
    item_index: int
    actor_size_m: tuple[float, float, float]
    execution_size_m: tuple[float, float, float]
    target_aabb_size_m: tuple[float, float, float]
    target_center_m: tuple[float, float, float]
    target_yaw_options: tuple[float, ...]
    source_pitch_options: tuple[float, ...]
    orientation_type: str
    pct_leaf_index: int


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def classify_orientation(
    original: tuple[float, float, float],
    placed: tuple[float, float, float],
    eps: float = 1e-6,
) -> OrientationClassification:
    ox, oy, oz = original
    px, py, pz = placed
    if abs(px - ox) < eps and abs(py - oy) < eps and abs(pz - oz) < eps:
        return OrientationClassification("same", (0.0, math.pi), (0.0,), original, True)
    if abs(px - oy) < eps and abs(py - ox) < eps and abs(pz - oz) < eps:
        return OrientationClassification("yaw_90", (math.pi / 2.0, -math.pi / 2.0), (0.0,), original, True)
    if abs(px - oz) < eps and abs(py - oy) < eps and abs(pz - ox) < eps:
        # X/Z swap: prepare the object on its side in the feed area before
        # grasping. The physical actor remains the original box; execution
        # geometry uses the side-laid AABB.
        return OrientationClassification("preflip_xz", (0.0, math.pi), (math.pi / 2.0, -math.pi / 2.0), placed, True)
    if all(abs(a - b) < eps for a, b in zip(sorted(original), sorted(placed))):
        # General axis permutation.  The online robot task treats this as a
        # feed-area pre-adjustment: the box is prepared in the requested PCT
        # AABB before pickup, then the grasp/place chain executes that AABB
        # without requiring an in-hand 3D flip.
        return OrientationClassification("feed_reorient", (0.0, math.pi), (math.pi / 2.0,), placed, True)
    return OrientationClassification("unsupported_flip", tuple(), tuple(), placed, False)


def center_from_corners(
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple((float(min_corner[i]) + float(max_corner[i])) * 0.5 for i in range(3))  # type: ignore[return-value]


def robot_target_from_pct(candidate: PctCandidate) -> RobotPlacementTarget:
    orientation = classify_orientation(candidate.original_size_m, candidate.placed_size_m)
    return RobotPlacementTarget(
        item_index=candidate.item_index,
        actor_size_m=candidate.original_size_m,
        execution_size_m=orientation.execution_size_m,
        target_aabb_size_m=candidate.placed_size_m,
        target_center_m=center_from_corners(candidate.min_corner_m, candidate.max_corner_m),
        target_yaw_options=orientation.target_yaw_options,
        source_pitch_options=orientation.source_pitch_options,
        orientation_type=orientation.type,
        pct_leaf_index=candidate.action_leaf_index,
    )
