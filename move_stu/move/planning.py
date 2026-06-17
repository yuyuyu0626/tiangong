#!/usr/bin/env python3
"""Geometry planning for the palletizing move state.

The first move-state milestone is intentionally kinematic: arms and hands keep
the flip result, while the robot root translates, yaws, and changes height.
This module contains the deterministic geometry used by the Isaac Gym demo and
can be tested without Isaac Gym installed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable


TABLE_SIZE = (0.80, 0.60, 0.06)
TABLE_POSE = (0.60, 0.0, 0.60)
BOX_SIZE = (0.20, 0.20, 0.20)
BOX_POSE = (
    TABLE_POSE[0] - 0.15 + 0.05,
    TABLE_POSE[1],
    TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + BOX_SIZE[2] * 0.5,
)

PALLET_SIZE = (1.0, 1.0)
PALLET_SURFACE_Z = 0.10
PALLET_DIAGONAL_DISTANCE = 5.0
PALLET_THICKNESS = 0.02
STACK_HEIGHT_LIMIT = 1.0

PRESET_STACK_BOX_SIZE = (0.20, 0.20, 0.20)
TARGET_ON_SECOND_BOX_MIN = (0.20, 0.0, 0.20)

MOVE_STAND_OFF = 0.60
STACK_TMP_RETREAT = 0.80
TABLE_RETREAT = 0.90
SAFE_CORRIDOR_MARGIN = 1.20
DEFAULT_ROOT_Z = 0.0
DEFAULT_HEIGHT_CLEARANCE = 0.18
MIN_ROOT_Z = 0.0


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class BoxPlacement:
    name: str
    min_corner: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def max_corner(self) -> tuple[float, float, float]:
        return tuple(self.min_corner[i] + self.size[i] for i in range(3))  # type: ignore[return-value]

    @property
    def center_local(self) -> tuple[float, float, float]:
        return tuple(self.min_corner[i] + self.size[i] * 0.5 for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class EmptySpace:
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(self.max_corner[i] - self.min_corner[i] for i in range(3))  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        sx, sy, sz = self.size
        return sx * sy * sz


@dataclass(frozen=True)
class StackScene:
    pallet_center: tuple[float, float, float]
    pallet_size: tuple[float, float]
    pallet_surface_z: float
    preset_boxes: tuple[BoxPlacement, ...]
    next_box: BoxPlacement
    empty_spaces: tuple[EmptySpace, ...]
    selected_leaf: EmptySpace


@dataclass(frozen=True)
class StancePlan:
    side: str
    normal: tuple[float, float]
    face_center: tuple[float, float, float]
    pose: Pose
    tmp_pose: Pose
    height_command: float


@dataclass(frozen=True)
class RouteWaypoint:
    label: str
    pose: Pose


@dataclass(frozen=True)
class RoutePlan:
    side: str
    waypoints: tuple[RouteWaypoint, ...]


@dataclass(frozen=True)
class MoveScenePlan:
    start_pose: Pose
    stack_scene: StackScene
    stances: tuple[StancePlan, ...]
    routes: tuple[RoutePlan, ...]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate_angle(a: float, b: float, alpha: float) -> float:
    return normalize_angle(a + normalize_angle(b - a) * alpha)


def pallet_center_near_table() -> tuple[float, float, float]:
    """Place the pallet 5 m diagonally from the flip table center."""

    offset = PALLET_DIAGONAL_DISTANCE / math.sqrt(2.0)
    return (
        TABLE_POSE[0] + offset,
        TABLE_POSE[1] + offset,
        PALLET_SURFACE_Z,
    )


def _intersects(a: EmptySpace, b: BoxPlacement) -> bool:
    b_min = b.min_corner
    b_max = b.max_corner
    return all(a.min_corner[i] < b_max[i] and b_min[i] < a.max_corner[i] for i in range(3))


def _subtract_box(space: EmptySpace, box: BoxPlacement, min_size: float) -> list[EmptySpace]:
    if not _intersects(space, box):
        return [space]

    sx1, sy1, sz1 = space.min_corner
    sx2, sy2, sz2 = space.max_corner
    bx1, by1, bz1 = box.min_corner
    bx2, by2, bz2 = box.max_corner

    ix1 = max(sx1, bx1)
    iy1 = max(sy1, by1)
    iz1 = max(sz1, bz1)
    ix2 = min(sx2, bx2)
    iy2 = min(sy2, by2)
    iz2 = min(sz2, bz2)

    candidates = [
        EmptySpace((sx1, sy1, sz1), (ix1, sy2, sz2)),
        EmptySpace((ix2, sy1, sz1), (sx2, sy2, sz2)),
        EmptySpace((sx1, sy1, sz1), (sx2, iy1, sz2)),
        EmptySpace((sx1, iy2, sz1), (sx2, sy2, sz2)),
        EmptySpace((sx1, sy1, iz2), (sx2, sy2, sz2)),
    ]
    return [ems for ems in candidates if all(v >= min_size for v in ems.size)]


def _eliminate_inscribed(spaces: Iterable[EmptySpace]) -> tuple[EmptySpace, ...]:
    unique: list[EmptySpace] = []
    for space in spaces:
        if space not in unique:
            unique.append(space)

    kept: list[EmptySpace] = []
    for i, space in enumerate(unique):
        contained = False
        for j, other in enumerate(unique):
            if i == j:
                continue
            if all(space.min_corner[k] >= other.min_corner[k] for k in range(3)) and all(
                space.max_corner[k] <= other.max_corner[k] for k in range(3)
            ):
                contained = True
                break
        if not contained:
            kept.append(space)
    return tuple(kept)


def compute_pct_like_ems(
    placed_boxes: Iterable[BoxPlacement],
    pallet_size: tuple[float, float] = PALLET_SIZE,
    height_limit: float = STACK_HEIGHT_LIMIT,
    min_size: float = 0.05,
) -> tuple[EmptySpace, ...]:
    """Compute EMS leaves using the same split/eliminate idea as PCT_EMS."""

    spaces: tuple[EmptySpace, ...] = (
        EmptySpace((0.0, 0.0, 0.0), (pallet_size[0], pallet_size[1], height_limit)),
    )
    for box in placed_boxes:
        next_spaces: list[EmptySpace] = []
        for space in spaces:
            next_spaces.extend(_subtract_box(space, box, min_size))
        spaces = _eliminate_inscribed(next_spaces)
    return spaces


def leaf_candidates_for_box(
    spaces: Iterable[EmptySpace],
    box_size: tuple[float, float, float],
) -> tuple[BoxPlacement, ...]:
    """Generate PCT-style corner and center leaf placements for the next box."""

    sx, sy, sz = box_size
    candidates: set[tuple[float, float, float, float, float, float]] = set()
    for ems in spaces:
        ex1, ey1, ez1 = ems.min_corner
        ex2, ey2, ez2 = ems.max_corner
        if ex2 - ex1 < sx or ey2 - ey1 < sy or ez2 - ez1 < sz:
            continue
        points = (
            (ex1, ey1, ez1),
            (ex2 - sx, ey1, ez1),
            (ex1, ey2 - sy, ez1),
            (ex2 - sx, ey2 - sy, ez1),
            ((ex1 + ex2 - sx) * 0.5, ey1, ez1),
            (ex1, (ey1 + ey2 - sy) * 0.5, ez1),
            ((ex1 + ex2 - sx) * 0.5, ey2 - sy, ez1),
            (ex2 - sx, (ey1 + ey2 - sy) * 0.5, ez1),
            ((ex1 + ex2 - sx) * 0.5, (ey1 + ey2 - sy) * 0.5, ez1),
        )
        for point in points:
            rounded = tuple(round(v, 6) for v in (*point, sx, sy, sz))
            candidates.add(rounded)

    boxes = [
        BoxPlacement(f"candidate_{idx}", data[:3], data[3:])  # type: ignore[arg-type]
        for idx, data in enumerate(sorted(candidates), start=1)
    ]
    return tuple(boxes)


def choose_next_box_leaf(candidates: Iterable[BoxPlacement]) -> BoxPlacement:
    """Choose the lowest, most corner-supported leaf for the simple stack test."""

    ordered = sorted(
        candidates,
        key=lambda b: (
            b.min_corner[2],
            abs(b.min_corner[0]) + abs(b.min_corner[1]),
            b.min_corner[1],
            b.min_corner[0],
        ),
    )
    if not ordered:
        raise ValueError("No feasible PCT leaf found for the next box")
    selected = ordered[0]
    return BoxPlacement("box_3_target", selected.min_corner, selected.size)


def build_stack_scene(next_box_size: tuple[float, float, float] = BOX_SIZE) -> StackScene:
    """Build the pallet scene with the third cube explicitly placed on box 2.

    PCT would expose a top leaf above the second placed cube. For this stage we
    keep that target deterministic instead of selecting it algorithmically.
    """

    preset = (
        BoxPlacement("box_1_preset_0p20", (0.0, 0.0, 0.0), PRESET_STACK_BOX_SIZE),
        BoxPlacement("box_2_preset_0p20", (0.20, 0.0, 0.0), PRESET_STACK_BOX_SIZE),
    )
    spaces = compute_pct_like_ems(preset)
    selected = BoxPlacement("box_3_target_on_box_2_leaf", TARGET_ON_SECOND_BOX_MIN, next_box_size)
    leaf = EmptySpace(selected.min_corner, selected.max_corner)
    return StackScene(
        pallet_center=pallet_center_near_table(),
        pallet_size=PALLET_SIZE,
        pallet_surface_z=PALLET_SURFACE_Z,
        preset_boxes=preset,
        next_box=selected,
        empty_spaces=spaces,
        selected_leaf=leaf,
    )


def generate_stance_plans(
    stack_center: tuple[float, float, float],
    stack_size_xy: tuple[float, float] = PALLET_SIZE,
    stand_off: float = MOVE_STAND_OFF,
    target_support_z: float = PALLET_SURFACE_Z,
    placement_center_xy: tuple[float, float] | None = None,
    placement_size_xy: tuple[float, float] = BOX_SIZE[:2],
    z_box_rel_hold: float = BOX_POSE[2],
    start_root_z: float = DEFAULT_ROOT_Z,
    min_root_z: float = MIN_ROOT_Z,
) -> tuple[StancePlan, ...]:
    cx, cy, _ = stack_center
    if placement_center_xy is None:
        target_x, target_y = cx, cy
    else:
        target_x, target_y = placement_center_xy
    length, width = stack_size_xy
    side_defs = (
        ("+X", (1.0, 0.0), (cx + length * 0.5, target_y)),
        ("-X", (-1.0, 0.0), (cx - length * 0.5, target_y)),
        ("+Y", (0.0, 1.0), (target_x, cy + width * 0.5)),
        ("-Y", (0.0, -1.0), (target_x, cy - width * 0.5)),
    )
    desired_box_center_z = target_support_z + BOX_SIZE[2] * 0.5 + DEFAULT_HEIGHT_CLEARANCE
    raw_height_command = start_root_z + (desired_box_center_z - z_box_rel_hold)
    height_command = max(min_root_z, raw_height_command)

    stances: list[StancePlan] = []
    for side, normal, face_xy in side_defs:
        sx = face_xy[0] + normal[0] * stand_off
        sy = face_xy[1] + normal[1] * stand_off
        yaw = math.atan2(target_y - sy, target_x - sx)
        pose = Pose(sx, sy, height_command, yaw)
        tmp_pose = Pose(
            sx + normal[0] * STACK_TMP_RETREAT,
            sy + normal[1] * STACK_TMP_RETREAT,
            height_command,
            yaw,
        )
        stances.append(
            StancePlan(
                side=side,
                normal=normal,
                face_center=(face_xy[0], face_xy[1], target_support_z),
                pose=pose,
                tmp_pose=tmp_pose,
                height_command=height_command,
            )
        )
    return tuple(stances)


def _dedupe_waypoints(waypoints: Iterable[RouteWaypoint]) -> tuple[RouteWaypoint, ...]:
    deduped: list[RouteWaypoint] = []
    for waypoint in waypoints:
        if deduped:
            prev = deduped[-1].pose
            cur = waypoint.pose
            if (
                abs(prev.x - cur.x) < 1e-6
                and abs(prev.y - cur.y) < 1e-6
                and abs(prev.z - cur.z) < 1e-6
                and abs(normalize_angle(prev.yaw - cur.yaw)) < 1e-6
            ):
                continue
        deduped.append(waypoint)
    return tuple(deduped)


def _pose_toward(x: float, y: float, z: float, target: Pose, fallback_yaw: float) -> Pose:
    dx = target.x - x
    dy = target.y - y
    yaw = fallback_yaw if math.hypot(dx, dy) < 1e-6 else math.atan2(dy, dx)
    return Pose(x, y, z, yaw)


def source_retreat_pose(start_pose: Pose) -> Pose:
    """Back away from the source table before planning toward the pallet."""

    dx = start_pose.x - TABLE_POSE[0]
    dy = start_pose.y - TABLE_POSE[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy, length = -1.0, 0.0, 1.0
    x = start_pose.x + dx / length * TABLE_RETREAT
    y = start_pose.y + dy / length * TABLE_RETREAT
    yaw = math.atan2(BOX_POSE[1] - y, BOX_POSE[0] - x)
    return Pose(x, y, start_pose.z, yaw)


def _safe_corner(stack_scene: StackScene, start_pose: Pose) -> Pose:
    cx, cy, _ = stack_scene.pallet_center
    safe_x = cx - stack_scene.pallet_size[0] * 0.5 - SAFE_CORRIDOR_MARGIN
    safe_y = cy - stack_scene.pallet_size[1] * 0.5 - SAFE_CORRIDOR_MARGIN
    return Pose(safe_x, safe_y, start_pose.z, 0.0)


def _is_outside_x(pose: Pose, stack_scene: StackScene, margin: float = 0.05) -> bool:
    cx, _, _ = stack_scene.pallet_center
    x_min = cx - stack_scene.pallet_size[0] * 0.5 - margin
    x_max = cx + stack_scene.pallet_size[0] * 0.5 + margin
    return pose.x <= x_min or pose.x >= x_max


def _route_to_safe_corner(
    current: Pose,
    safe_corner: Pose,
    stack_scene: StackScene,
    side: str,
) -> tuple[RouteWaypoint, ...]:
    z = safe_corner.z
    if _is_outside_x(current, stack_scene):
        via_1 = _pose_toward(current.x, safe_corner.y, z, safe_corner, current.yaw)
        via_2 = _pose_toward(safe_corner.x, safe_corner.y, z, safe_corner, via_1.yaw)
    else:
        via_1 = _pose_toward(safe_corner.x, current.y, z, safe_corner, current.yaw)
        via_2 = _pose_toward(safe_corner.x, safe_corner.y, z, safe_corner, via_1.yaw)
    return _dedupe_waypoints(
        (
            RouteWaypoint(f"{side}_safe_axis_1", via_1),
            RouteWaypoint(f"{side}_safe_corner", via_2),
        )
    )


def _route_from_safe_corner_to_tmp(
    tmp: Pose,
    safe_corner: Pose,
    stack_scene: StackScene,
    side: str,
) -> tuple[RouteWaypoint, ...]:
    z = safe_corner.z
    if _is_outside_x(tmp, stack_scene):
        via_1 = _pose_toward(tmp.x, safe_corner.y, z, tmp, safe_corner.yaw)
    else:
        via_1 = _pose_toward(safe_corner.x, tmp.y, z, tmp, safe_corner.yaw)
    tmp_entry = Pose(tmp.x, tmp.y, tmp.z, tmp.yaw)
    return _dedupe_waypoints(
        (
            RouteWaypoint(f"{side}_tmp_axis_1", via_1),
            RouteWaypoint(f"{side}_tmp", tmp_entry),
        )
    )


def _route_to_tmp(
    current: Pose,
    tmp: Pose,
    safe_corner: Pose,
    stack_scene: StackScene,
    side: str,
) -> tuple[RouteWaypoint, ...]:
    return _dedupe_waypoints(
        (
            *_route_to_safe_corner(current, safe_corner, stack_scene, side),
            *_route_from_safe_corner_to_tmp(tmp, safe_corner, stack_scene, side),
        )
    )


def build_route_plans(
    start_pose: Pose,
    stack_scene: StackScene,
    stances: Iterable[StancePlan],
) -> tuple[RoutePlan, ...]:
    """Build straight-line keypoint routes with explicit retreat/approach points."""

    source_retreat = source_retreat_pose(start_pose)
    safe_corner = _safe_corner(stack_scene, start_pose)
    current = source_retreat
    routes: list[RoutePlan] = []
    for index, stance in enumerate(stances):
        prefix: list[RouteWaypoint] = []
        if index == 0:
            prefix.extend(
                (
                    RouteWaypoint("table_start", start_pose),
                    RouteWaypoint("table_retreat", source_retreat),
                )
            )
        prefix.extend(_route_to_tmp(current, stance.tmp_pose, safe_corner, stack_scene, stance.side))
        waypoints = _dedupe_waypoints(
            (
                *prefix,
                RouteWaypoint(f"{stance.side}_target", stance.pose),
                RouteWaypoint(f"{stance.side}_retreat", stance.tmp_pose),
            )
        )
        routes.append(RoutePlan(stance.side, waypoints))
        current = stance.tmp_pose
    return tuple(routes)


def build_move_scene_plan() -> MoveScenePlan:
    stack_scene = build_stack_scene()
    start_pose = Pose(0.0, 0.0, DEFAULT_ROOT_Z, 0.0)
    target_center = stack_scene.next_box.center_local
    placement_center_xy = (
        stack_scene.pallet_center[0] - 0.5 * stack_scene.pallet_size[0] + target_center[0],
        stack_scene.pallet_center[1] - 0.5 * stack_scene.pallet_size[1] + target_center[1],
    )
    stances = generate_stance_plans(
        stack_scene.pallet_center,
        stack_scene.pallet_size,
        target_support_z=stack_scene.pallet_surface_z + stack_scene.next_box.min_corner[2],
        placement_center_xy=placement_center_xy,
        placement_size_xy=stack_scene.next_box.size[:2],
        z_box_rel_hold=BOX_POSE[2],
        start_root_z=start_pose.z,
    )
    routes = build_route_plans(start_pose, stack_scene, stances)
    return MoveScenePlan(start_pose=start_pose, stack_scene=stack_scene, stances=stances, routes=routes)


def sample_root_trajectory(
    start: Pose,
    goal: Pose,
    linear_speed: float = 0.225,
    yaw_speed: float = 0.30,
    lift_speed: float = 0.075,
    dt: float = 1.0 / 60.0,
) -> tuple[Pose, ...]:
    distance_xy = math.hypot(goal.x - start.x, goal.y - start.y)
    distance_yaw = abs(normalize_angle(goal.yaw - start.yaw))
    distance_z = abs(goal.z - start.z)
    duration = max(
        distance_xy / max(linear_speed, 1e-6),
        distance_yaw / max(yaw_speed, 1e-6),
        distance_z / max(lift_speed, 1e-6),
        dt,
    )
    steps = max(2, int(math.ceil(duration / dt)) + 1)
    poses: list[Pose] = []
    for i in range(steps):
        alpha = i / (steps - 1)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        poses.append(
            Pose(
                x=start.x + (goal.x - start.x) * smooth,
                y=start.y + (goal.y - start.y) * smooth,
                z=start.z + (goal.z - start.z) * smooth,
                yaw=interpolate_angle(start.yaw, goal.yaw, smooth),
            )
        )
    return tuple(poses)


def _dataclass_to_dict(value):
    if isinstance(value, tuple):
        return [_dataclass_to_dict(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(value).items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the deterministic move-state geometry plan.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    plan = build_move_scene_plan()
    if args.json:
        print(json.dumps(_dataclass_to_dict(plan), indent=2, sort_keys=True))
        return

    stack = plan.stack_scene
    print("Move scene plan")
    print(f"  pallet_center_world={stack.pallet_center}")
    print(f"  pallet_surface_z={stack.pallet_surface_z:.3f} m")
    for box in stack.preset_boxes:
        print(f"  preset {box.name}: min={box.min_corner} size={box.size} max={box.max_corner}")
    print(f"  computed third box: min={stack.next_box.min_corner} size={stack.next_box.size} max={stack.next_box.max_corner}")
    print("  EMS leaves:")
    for ems in stack.empty_spaces:
        print(f"    min={ems.min_corner} max={ems.max_corner} size={ems.size}")
    print("  stance goals:")
    for stance in plan.stances:
        p = stance.pose
        tmp = stance.tmp_pose
        print(
            f"    {stance.side}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} "
            f"yaw={math.degrees(p.yaw):.1f}deg height={stance.height_command:.3f} "
            f"tmp=({tmp.x:.3f},{tmp.y:.3f},{tmp.z:.3f},{math.degrees(tmp.yaw):.1f}deg)"
        )
    print("  straight-line route keypoints:")
    for route in plan.routes:
        print(f"    route {route.side}:")
        for waypoint in route.waypoints:
            p = waypoint.pose
            print(
                f"      {waypoint.label}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} "
                f"yaw={math.degrees(p.yaw):.1f}deg"
            )


if __name__ == "__main__":
    main()
