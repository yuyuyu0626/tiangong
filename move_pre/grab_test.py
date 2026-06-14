#!/usr/bin/env python3
"""Offline grab/move/place IK chain for the next palletizing stage.

This script intentionally avoids Isaac Gym. It validates the kinematic handoff:

1. solve a two-palm grasp/lift with fingers pointing to +X;
2. switch to the move URDF and solve the move_test3 lift joints;
3. plan a conservative box-center placement path;
4. solve the two-palm embedded placement IK along that path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import torch


MOVE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move_pre.utils import (  # noqa: E402
    APPROACH_GAP_GRASP,
    APPROACH_GAP_PRE,
    APPROACH_GAP_READY,
    BOX_SIDE_CONTACT_Z_RATIO,
    IKFlipBoxController,
    LEFT_EE_LINK,
    PALM_BOX_CLEARANCE,
    PALM_CENTER_Z,
    PALM_SURFACE_X,
    RIGHT_EE_LINK,
    CartesianKeyframe,
    UrdfKinematics,
)
from move_pre.mobile_ik import MoveLiftIK, target_box_center_z  # noqa: E402
from move_pre.planning import (  # noqa: E402
    BOX_POSE,
    BOX_SIZE,
    PALLET_SIZE,
    PALLET_SURFACE_Z,
    SAFE_CORRIDOR_MARGIN,
    TABLE_POSE,
    BoxPlacement,
    EmptySpace,
    Pose,
    StackScene,
    compute_pct_like_ems,
    pallet_center_near_table,
    sample_root_trajectory,
)


MOVE_URDF = MOVE_ROOT / "assets" / "integrated" / "tianyi_xhand_move.urdf"
PICK_LIFT_HEIGHT = 0.20
PICK_READY_Z_OFFSET = 0.12
PICK_TOUCH_GAP = 0.050
PICK_COMPRESS_GAP = -PALM_BOX_CLEARANCE
PICK_CONTACT_X_OFFSET = 0.025
GRASP_BACK_OFFSET = 0.15
GRASP_CONTACT_X_OFFSET = PICK_CONTACT_X_OFFSET - GRASP_BACK_OFFSET
# The IK target is the palm feature point, while Isaac collision happens on
# hand meshes whose effective support area sits lower than that feature point.
# Aim slightly above the box side center so the actual contact is around the
# middle of the side face instead of the lower edge.
PICK_SIDE_CONTACT_Z_RATIO = 0.24
MOVE_APPROACH_HEIGHT = 0.15
MOVE_PREPLACE_HEIGHT = 0.16
PLACE_HOLD_FRAMES = 45
PLACE_UPRIGHT_FRAMES = 120
PLACE_CONTACT_GAP = -PALM_BOX_CLEARANCE
PLACE_RELEASE_HEIGHT = 0.030
PLACE_XY_SEGMENT_FRAMES = 180
PLACE_DESCEND_FRAMES = 180
TEST3_STAND_OFF = 0.72
SIM_PLACE_STAND_OFF = 0.35
TEST3_TMP_RETREAT = 0.80
PICK_ERROR_LIMIT = 0.05
PLACE_ERROR_LIMIT = 0.06


@dataclass(frozen=True)
class IkFrameReport:
    name: str
    left_error: float
    right_error: float
    max_error: float


@dataclass(frozen=True)
class GrabTestReport:
    pick_frames: int
    place_frames: int
    root_frames: int
    target_world_center: tuple[float, float, float]
    final_root_pose: tuple[float, float, float, float]
    target_box_center_z: float
    solved_palm_z: float
    pick_max_error: float
    place_max_error: float
    pick_feasible: bool
    place_feasible: bool
    place_feasible_with_flip_dofs: bool
    path_clearance_ok: bool
    output_file: str | None


@dataclass(frozen=True)
class OfflineTest3Scene:
    name: str
    stack_scene: StackScene
    target_support_z: float
    final_pose: Pose
    tmp_pose: Pose
    waypoints: tuple[tuple[str, Pose], ...]


def _parse_active_dofs(urdf_path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type == "fixed":
            continue
        names.append(joint.attrib["name"])
        limit = joint.find("limit")
        if joint_type == "continuous":
            lower.append(-math.pi)
            upper.append(math.pi)
        elif limit is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(float(limit.attrib.get("lower", "-3.141592653589793")))
            upper.append(float(limit.attrib.get("upper", "3.141592653589793")))
    return names, torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)


def _rot_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _pose_pair_for_box(
    center: np.ndarray,
    box_size: tuple[float, float, float],
    theta: float,
    side_gap: float,
    contact_z: float,
    contact_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left/right palm targets for side grasping a box.

    The palm normals match the box +/-Y side faces. Finger directions point to
    +X when theta is zero, which is the requested initial hand orientation.
    """

    half_y = box_size[1] * 0.5
    left_palm = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right_palm = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    fingers_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rot = _rot_y(theta)
    left_contact_local = np.array([contact_x, half_y + PALM_BOX_CLEARANCE + side_gap, contact_z], dtype=np.float64)
    right_contact_local = np.array([contact_x, -half_y - PALM_BOX_CLEARANCE - side_gap, contact_z], dtype=np.float64)
    return (
        center + rot @ left_contact_local,
        center + rot @ right_contact_local,
        rot @ left_palm,
        rot @ right_palm,
        rot @ fingers_forward,
        rot @ fingers_forward,
    )


def _make_keyframe(
    name: str,
    center: np.ndarray,
    box_size: tuple[float, float, float],
    theta: float,
    side_gap: float,
    contact_z: float,
    contact_x: float = 0.0,
) -> CartesianKeyframe:
    pose_pair = _pose_pair_for_box(center, box_size, theta, side_gap, contact_z, contact_x)
    return CartesianKeyframe(
        name,
        1,
        pose_pair[0],
        pose_pair[1],
        pose_pair[2],
        pose_pair[3],
        pose_pair[4],
        pose_pair[5],
        0.0,
        {},
    )


def _build_pick_keyframes() -> tuple[list[CartesianKeyframe], list[tuple[float, float, float]]]:
    """Build the flip-style physical grasp sequence.

    The hands keep finger_dir along +X throughout pick. They first stay wide and
    high, descend to the side contact height, pre-contact, compress into the box
    side faces, hold that compression, and only then lift.
    """

    box_center = np.asarray(BOX_POSE, dtype=np.float64)
    contact_z = BOX_SIZE[2] * PICK_SIDE_CONTACT_Z_RATIO
    high_center = box_center + np.array([0.0, 0.0, PICK_READY_Z_OFFSET], dtype=np.float64)
    lift_center = box_center + np.array([0.0, 0.0, PICK_LIFT_HEIGHT], dtype=np.float64)
    frames = [
        _make_keyframe("pick_ready_high", high_center, BOX_SIZE, 0.0, APPROACH_GAP_READY, contact_z + 0.02, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_descend_level", box_center, BOX_SIZE, 0.0, APPROACH_GAP_READY, contact_z + 0.02, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_pre_grasp", box_center, BOX_SIZE, 0.0, APPROACH_GAP_PRE, contact_z + 0.01, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_straight_clamp", box_center, BOX_SIZE, 0.0, PICK_TOUCH_GAP, contact_z, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_compress", box_center, BOX_SIZE, 0.0, PICK_COMPRESS_GAP, contact_z, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_compress_hold", box_center, BOX_SIZE, 0.0, PICK_COMPRESS_GAP, contact_z, GRASP_CONTACT_X_OFFSET),
        _make_keyframe("pick_lift", lift_center, BOX_SIZE, 0.0, PICK_COMPRESS_GAP, contact_z, GRASP_CONTACT_X_OFFSET),
    ]
    centers = [
        tuple(float(v) for v in high_center),
        BOX_POSE,
        BOX_POSE,
        BOX_POSE,
        BOX_POSE,
        BOX_POSE,
        tuple(float(v) for v in lift_center),
    ]
    return frames, centers


def _solve_keyframes(
    controller: IKFlipBoxController,
    frames: Iterable[CartesianKeyframe],
    iterations: int,
) -> tuple[torch.Tensor, list[IkFrameReport]]:
    targets: list[torch.Tensor] = []
    reports: list[IkFrameReport] = []
    for frame in frames:
        controller._solve_keyframe(frame, iterations=iterations, regularize_wrist=frame.name.startswith("pick_"))
        controller._apply_hand_close(frame.hand_close)
        controller._apply_extra_joints(frame.extra_joints)
        controller._clamp_all()
        q = torch.tensor(controller.q, dtype=torch.float32)
        targets.append(q)
        left_error = float(np.linalg.norm(controller._ee_feature("xhand_left_left_hand_link")[:3] - frame.left_pos))
        right_error = float(np.linalg.norm(controller._ee_feature("xhand_right_right_hand_link")[:3] - frame.right_pos))
        reports.append(IkFrameReport(frame.name, left_error, right_error, max(left_error, right_error)))
    return torch.stack(targets), reports


def _make_stack_scene(
    preset_boxes: tuple[BoxPlacement, ...],
    target_box: BoxPlacement,
) -> StackScene:
    spaces = compute_pct_like_ems(preset_boxes)
    fits_ems = any(
        all(target_box.min_corner[i] >= space.min_corner[i] for i in range(3))
        and all(target_box.max_corner[i] <= space.max_corner[i] for i in range(3))
        for space in spaces
    )
    if not fits_ems:
        raise ValueError(f"Target {target_box.name} is not contained in any EMS leaf")
    return StackScene(
        pallet_center=pallet_center_near_table(),
        pallet_size=PALLET_SIZE,
        pallet_surface_z=PALLET_SURFACE_Z,
        preset_boxes=preset_boxes,
        next_box=target_box,
        empty_spaces=spaces,
        selected_leaf=EmptySpace(target_box.min_corner, target_box.max_corner),
    )


def _placement_above_box_center(
    name: str,
    base_box: BoxPlacement,
    box_size: tuple[float, float, float],
) -> BoxPlacement:
    base_center = base_box.center_local
    target_center = (
        base_center[0],
        base_center[1],
        base_center[2] + box_size[2],
    )
    target_min = (
        target_center[0] - box_size[0] * 0.5,
        target_center[1] - box_size[1] * 0.5,
        target_center[2] - box_size[2] * 0.5,
    )
    return BoxPlacement(name, target_min, box_size)


def _target_world_center(scene) -> tuple[float, float, float]:
    local = scene.stack_scene.next_box.center_local
    stack = scene.stack_scene
    return (
        stack.pallet_center[0] - stack.pallet_size[0] * 0.5 + local[0],
        stack.pallet_center[1] - stack.pallet_size[1] * 0.5 + local[1],
        stack.pallet_center[2] + local[2],
    )


def _solve_test3_root_pose(stack: StackScene, stand_off: float) -> tuple[Pose, Pose]:
    target_x, target_y, _ = _target_world_center_for_stack(stack)
    final_x = stack.pallet_center[0] - stack.pallet_size[0] * 0.5 - stand_off
    final_y = target_y
    yaw = math.atan2(target_y - final_y, target_x - final_x)
    final_pose = Pose(final_x, final_y, 0.0, yaw)
    tmp_pose = Pose(final_x - TEST3_TMP_RETREAT, final_y, 0.0, yaw)
    return final_pose, tmp_pose


def _target_world_center_for_stack(stack: StackScene) -> tuple[float, float, float]:
    local = stack.next_box.center_local
    return (
        stack.pallet_center[0] - stack.pallet_size[0] * 0.5 + local[0],
        stack.pallet_center[1] - stack.pallet_size[1] * 0.5 + local[1],
        stack.pallet_center[2] + local[2],
    )


def _table_retreat_pose(start: Pose) -> Pose:
    dx = start.x - TABLE_POSE[0]
    dy = start.y - TABLE_POSE[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy, length = -1.0, 0.0, 1.0
    return Pose(start.x + dx / length * 0.90, start.y + dy / length * 0.90, start.z, start.yaw)


def _build_test3_waypoints(stack: StackScene, final_pose: Pose, tmp_pose: Pose) -> tuple[tuple[str, Pose], ...]:
    start = Pose(0.0, 0.0, 0.0, 0.0)
    table_retreat = _table_retreat_pose(start)
    safe_x = stack.pallet_center[0] - stack.pallet_size[0] * 0.5 - SAFE_CORRIDOR_MARGIN
    safe_y = stack.pallet_center[1] - stack.pallet_size[1] * 0.5 - SAFE_CORRIDOR_MARGIN
    safe_1 = Pose(table_retreat.x, safe_y, 0.0, 0.0)
    safe_2 = Pose(safe_x, safe_y, 0.0, 0.0)
    tmp_xy = Pose(tmp_pose.x, safe_y, 0.0, math.atan2(tmp_pose.y - safe_y, tmp_pose.x - safe_x))
    tmp_low = Pose(tmp_pose.x, tmp_pose.y, 0.0, tmp_pose.yaw)
    return (
        ("table_start", start),
        ("table_retreat", table_retreat),
        ("safe_axis_1", safe_1),
        ("safe_corner", safe_2),
        ("tmp_axis", tmp_xy),
        ("tmp_low", tmp_low),
        ("target", final_pose),
    )


def build_offline_test3_scene(stand_off: float = TEST3_STAND_OFF) -> OfflineTest3Scene:
    box_1 = BoxPlacement("scene1_box_1_0p20", (0.0, 0.0, 0.0), (0.20, 0.20, 0.20))
    box_2 = BoxPlacement("scene1_box_2_0p20", (0.20, 0.0, 0.0), (0.20, 0.20, 0.20))
    target_box = _placement_above_box_center("scene1_box_3_target_top_of_0p20", box_2, BOX_SIZE)
    stack = _make_stack_scene(
        (box_1, box_2),
        target_box,
    )
    final_pose, tmp_pose = _solve_test3_root_pose(stack, stand_off)
    target_support_z = PALLET_SURFACE_Z + box_2.center_local[2] + box_2.size[2] * 0.5
    return OfflineTest3Scene(
        name="scene1_top_of_0p20",
        stack_scene=stack,
        target_support_z=target_support_z,
        final_pose=final_pose,
        tmp_pose=tmp_pose,
        waypoints=_build_test3_waypoints(stack, final_pose, tmp_pose),
    )


def _world_to_root(point: tuple[float, float, float], root_pose: Pose) -> np.ndarray:
    dx = point[0] - root_pose.x
    dy = point[1] - root_pose.y
    c = math.cos(-root_pose.yaw)
    s = math.sin(-root_pose.yaw)
    return np.array([c * dx - s * dy, s * dx + c * dy, point[2] - root_pose.z], dtype=np.float64)


def _transform_local_point(root: Pose, local: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(root.yaw)
    s = math.sin(root.yaw)
    return (
        root.x + c * local[0] - s * local[1],
        root.y + s * local[0] + c * local[1],
        root.z + local[2],
    )


def _grasp_center_from_q(
    urdf_path: Path,
    dof_names: tuple[str, ...] | list[str],
    q: torch.Tensor,
) -> np.ndarray:
    kin = UrdfKinematics(urdf_path)
    q_map = {name: float(q[i]) for i, name in enumerate(dof_names)}

    def palm_center(link: str) -> np.ndarray:
        pose = kin.fk(link, q_map)
        rot = pose[:3, :3]
        return pose[:3, 3] + rot[:, 0] * PALM_SURFACE_X + rot[:, 2] * PALM_CENTER_Z

    return 0.5 * (palm_center(LEFT_EE_LINK) + palm_center(RIGHT_EE_LINK))


def _box_theta_from_q(
    urdf_path: Path,
    dof_names: tuple[str, ...] | list[str],
    q: torch.Tensor,
) -> float:
    """Infer the held box pitch from the two palm finger directions."""

    kin = UrdfKinematics(urdf_path)
    q_map = {name: float(q[i]) for i, name in enumerate(dof_names)}
    dirs = []
    for link in (LEFT_EE_LINK, RIGHT_EE_LINK):
        finger_dir = kin.fk(link, q_map)[:3, :3][:, 2]
        norm = np.linalg.norm(finger_dir[[0, 2]])
        if norm > 1e-8:
            dirs.append(finger_dir / np.linalg.norm(finger_dir))
    if not dirs:
        return 0.0
    avg = np.mean(np.stack(dirs), axis=0)
    return float(math.atan2(-avg[2], avg[0]))


def _append_segment(
    path: list[tuple[str, tuple[float, float, float]]],
    label: str,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    frames: int,
) -> None:
    for i in range(1, frames + 1):
        alpha = i / float(frames)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        path.append(
            (
                label,
                (
                    start[0] * (1.0 - alpha) + goal[0] * alpha,
                    start[1] * (1.0 - alpha) + goal[1] * alpha,
                    start[2] * (1.0 - alpha) + goal[2] * alpha,
                ),
            )
        )


def _append_pose_segment(
    path: list[tuple[str, tuple[float, float, float], float]],
    label: str,
    start_center: tuple[float, float, float],
    goal_center: tuple[float, float, float],
    start_theta: float,
    goal_theta: float,
    frames: int,
) -> None:
    for i in range(1, frames + 1):
        alpha = i / float(frames)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        path.append(
            (
                label,
                (
                    start_center[0] * (1.0 - alpha) + goal_center[0] * alpha,
                    start_center[1] * (1.0 - alpha) + goal_center[1] * alpha,
                    start_center[2] * (1.0 - alpha) + goal_center[2] * alpha,
                ),
                start_theta * (1.0 - alpha) + goal_theta * alpha,
            )
        )


def _lerp_q(start: torch.Tensor, goal: torch.Tensor, alpha: float) -> torch.Tensor:
    return start * (1.0 - alpha) + goal * alpha


def _smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _pose_path_key_indices(path: list[tuple[str, tuple[float, float, float], float]]) -> list[int]:
    if not path:
        return []
    indices = [0]
    for index in range(1, len(path)):
        if path[index][0] != path[index - 1][0]:
            indices.append(index - 1)
            indices.append(index)
    indices.append(len(path) - 1)

    unique: list[int] = []
    for index in indices:
        if 0 <= index < len(path) and (not unique or unique[-1] != index):
            unique.append(index)
    return unique


def _expand_targets_for_pose_path(
    key_indices: list[int],
    key_targets: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    if not key_indices:
        raise ValueError("Cannot expand targets for an empty pose path")
    if len(key_indices) != int(key_targets.shape[0]):
        raise ValueError("Key indices and key targets must have the same length")

    frames: list[torch.Tensor | None] = [None for _ in range(total_frames)]
    frames[key_indices[0]] = key_targets[0].clone()
    for key_pos in range(1, len(key_indices)):
        start_index = key_indices[key_pos - 1]
        goal_index = key_indices[key_pos]
        start_q = key_targets[key_pos - 1]
        goal_q = key_targets[key_pos]
        span = max(1, goal_index - start_index)
        for index in range(start_index + 1, goal_index + 1):
            alpha = _smoothstep((index - start_index) / float(span))
            frames[index] = _lerp_q(start_q, goal_q, alpha)

    last_q = key_targets[-1].clone()
    for index, q in enumerate(frames):
        if q is None:
            frames[index] = last_q.clone()
        else:
            last_q = q
    return torch.stack([q for q in frames if q is not None])


def _plan_center_path_world(
    scene,
    start_center: tuple[float, float, float],
    start_theta: float = 0.0,
) -> tuple[list[tuple[str, tuple[float, float, float]]], bool]:
    del start_theta
    target = _target_world_center(scene)
    release = (target[0], target[1], target[2] + PLACE_RELEASE_HEIGHT)
    preset_top = max(
        scene.stack_scene.pallet_surface_z + placement.max_corner[2]
        for placement in scene.stack_scene.preset_boxes
    )
    start = start_center
    path = [("place_handoff", start)]
    _append_segment(
        path,
        "place_move_to_release",
        start,
        release,
        PLACE_XY_SEGMENT_FRAMES * 2 + PLACE_DESCEND_FRAMES,
    )
    path.extend(("place_hold", release) for _ in range(PLACE_HOLD_FRAMES))
    clearance_ok = min(start[2], release[2]) >= preset_top
    return path, clearance_ok


def _plan_pose_path_world(
    scene,
    start_center: tuple[float, float, float],
    start_theta: float,
) -> tuple[list[tuple[str, tuple[float, float, float], float]], bool]:
    target = _target_world_center(scene)
    release = (target[0], target[1], target[2] + PLACE_RELEASE_HEIGHT)
    preset_top = max(
        scene.stack_scene.pallet_surface_z + placement.max_corner[2]
        for placement in scene.stack_scene.preset_boxes
    )
    start = start_center
    upright = start
    path = [("place_handoff_tilted", start, start_theta)]
    _append_pose_segment(path, "place_rotate_upright", start, upright, start_theta, 0.0, PLACE_UPRIGHT_FRAMES)
    _append_pose_segment(
        path,
        "place_move_to_release",
        upright,
        release,
        0.0,
        0.0,
        PLACE_XY_SEGMENT_FRAMES * 2 + PLACE_DESCEND_FRAMES,
    )
    path.extend(("place_hold", release, 0.0) for _ in range(PLACE_HOLD_FRAMES))
    clearance_ok = min(start[2], release[2]) >= preset_top
    return path, clearance_ok


def _root_route_frames(scene) -> list[tuple[str, Pose]]:
    frames: list[tuple[str, Pose]] = []
    for (start_label, start_pose), (goal_label, goal_pose) in zip(scene.waypoints, scene.waypoints[1:]):
        del start_label
        for pose in sample_root_trajectory(start_pose, goal_pose):
            frames.append((goal_label, pose))
    frames.extend((scene.waypoints[-1][0], scene.waypoints[-1][1]) for _ in range(30))
    return frames


def run(
    save_path: Path | None = None,
    place_mode: str = "move",
    stand_off: float = TEST3_STAND_OFF,
) -> tuple[GrabTestReport, dict[str, object]]:
    if place_mode != "move":
        raise ValueError(f"Unsupported place_mode: {place_mode}")
    move_names, move_lower, move_upper = _parse_active_dofs(MOVE_URDF)

    contact_z = BOX_SIZE[2] * BOX_SIDE_CONTACT_Z_RATIO
    pick_frames, pick_box_centers = _build_pick_keyframes()

    pick_controller = IKFlipBoxController(move_names, move_lower, move_upper, MOVE_URDF, BOX_POSE, BOX_SIZE)
    pick_controller._solve_keyframe(pick_frames[0], iterations=80, regularize_wrist=True)
    pick_targets, pick_reports = _solve_keyframes(pick_controller, pick_frames, iterations=16)
    pick_lift_q = pick_targets[-1]

    scene = build_offline_test3_scene(stand_off)
    target_center = _target_world_center(scene)
    release_box_center_z = target_box_center_z(scene.target_support_z, BOX_SIZE[2], clearance=PLACE_RELEASE_HEIGHT)
    move_approach_box_center_z = target_center[2] + MOVE_APPROACH_HEIGHT
    move_preplace_box_center_z = target_center[2] + MOVE_PREPLACE_HEIGHT
    pick_grasp_center = _grasp_center_from_q(MOVE_URDF, move_names, pick_lift_q)
    attach_offset_local = np.asarray(pick_box_centers[-1], dtype=np.float64) - pick_grasp_center
    move_lift_ik = MoveLiftIK(MOVE_URDF, move_names, move_lower, move_upper)
    move_approach_q = move_lift_ik.solve_for_box_center_z(
        pick_lift_q,
        move_approach_box_center_z - float(attach_offset_local[2]),
    )
    move_lift_q = move_lift_ik.solve_for_box_center_z(
        move_approach_q,
        move_preplace_box_center_z - float(attach_offset_local[2]),
    )
    move_approach_box_theta = _box_theta_from_q(MOVE_URDF, move_names, move_approach_q)
    move_preplace_box_theta = _box_theta_from_q(MOVE_URDF, move_names, move_lift_q)
    solved_palm_z = move_lift_ik.palm_z(move_lift_q)
    root_frames = _root_route_frames(scene)
    final_root = scene.final_pose
    move_grasp_center = _grasp_center_from_q(MOVE_URDF, move_names, move_lift_q)
    move_attached_center_local = move_grasp_center + attach_offset_local
    place_handoff_center = _transform_local_point(final_root, tuple(move_attached_center_local.tolist()))

    place_names = move_names
    place_lower = move_lower
    place_upper = move_upper
    place_urdf = MOVE_URDF
    place_seed = move_lift_q
    place_controller = IKFlipBoxController(place_names, place_lower, place_upper, place_urdf, BOX_POSE, BOX_SIZE)
    place_controller.q = place_seed.detach().cpu().numpy().astype(np.float64)

    pose_path_world, path_clearance_ok = _plan_pose_path_world(scene, place_handoff_center, move_preplace_box_theta)
    place_handoff_center = pose_path_world[0][1]
    place_key_indices = _pose_path_key_indices(pose_path_world)
    place_key_frames = []
    for path_index in place_key_indices:
        label, center_world, box_theta = pose_path_world[path_index]
        local_center = _world_to_root(center_world, final_root)
        place_key_frames.append(
            _make_keyframe(label, local_center, BOX_SIZE, box_theta, PLACE_CONTACT_GAP, contact_z, GRASP_CONTACT_X_OFFSET)
        )
    place_key_targets, place_reports = _solve_keyframes(place_controller, place_key_frames, iterations=80)
    place_targets = _expand_targets_for_pose_path(place_key_indices, place_key_targets, len(pose_path_world))
    release_local_center = _world_to_root(pose_path_world[-1][1], final_root)
    release_frame = _make_keyframe(
        "release_open",
        release_local_center,
        BOX_SIZE,
        0.0,
        APPROACH_GAP_READY,
        contact_z + 0.02,
        GRASP_CONTACT_X_OFFSET,
    )
    release_target, release_reports = _solve_keyframes(place_controller, [release_frame], iterations=80)

    payload: dict[str, object] = {
        "move_dof_names": move_names,
        "flip_dof_names": [],
        "place_mode": place_mode,
        "stand_off": stand_off,
        "scene": scene,
        "grasp_back_offset": GRASP_BACK_OFFSET,
        "grasp_contact_x_offset": GRASP_CONTACT_X_OFFSET,
        "place_dof_names": place_names,
        "pick_box_centers": pick_box_centers,
        "pick_targets": pick_targets,
        "move_approach_target": move_approach_q,
        "move_approach_box_center_z": move_approach_box_center_z,
        "move_approach_box_theta": move_approach_box_theta,
        "move_lift_target": move_lift_q,
        "move_preplace_box_center_z": move_preplace_box_center_z,
        "move_preplace_box_theta": move_preplace_box_theta,
        "move_attached_box_center_world": place_handoff_center,
        "release_box_center_z": release_box_center_z,
        "place_targets": place_targets,
        "place_key_indices": place_key_indices,
        "place_key_targets": place_key_targets,
        "release_target": release_target[0],
        "place_targets_are_dense": True,
        "place_handoff_center_world": place_handoff_center,
        "root_route": [(label, asdict(pose)) for label, pose in root_frames],
        "place_center_path_world": [(label, center) for label, center, _theta in pose_path_world],
        "place_pose_path_world": pose_path_world,
        "pick_reports": [asdict(report) for report in pick_reports],
        "place_reports": [asdict(report) for report in place_reports],
        "release_reports": [asdict(report) for report in release_reports],
    }
    if save_path is not None:
        torch.save(payload, save_path)

    report = GrabTestReport(
        pick_frames=len(pick_frames),
        place_frames=len(pose_path_world),
        root_frames=len(root_frames),
        target_world_center=_target_world_center(scene),
        final_root_pose=(final_root.x, final_root.y, final_root.z, final_root.yaw),
        target_box_center_z=move_preplace_box_center_z,
        solved_palm_z=solved_palm_z,
        pick_max_error=max(report.max_error for report in pick_reports),
        place_max_error=max(report.max_error for report in place_reports),
        pick_feasible=max(report.max_error for report in pick_reports) <= PICK_ERROR_LIMIT,
        place_feasible=max(report.max_error for report in place_reports) <= PLACE_ERROR_LIMIT,
        place_feasible_with_flip_dofs=False,
        path_clearance_ok=path_clearance_ok,
        output_file=str(save_path) if save_path is not None else None,
    )
    return report, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline grab/move/place IK chain for move_test3.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument("--save", type=Path, default=None, help="Optional .pt output path for the generated plan.")
    parser.add_argument("--place-mode", choices=("move",), default="move", help="Use move URDF DOFs for placement IK.")
    parser.add_argument("--stand-off", type=float, default=TEST3_STAND_OFF, help="Final placement stand-off distance in meters.")
    parser.add_argument(
        "--sim-feasible",
        action="store_true",
        help="Shortcut for the current feasible full-simulation setting: --place-mode move --stand-off 0.35.",
    )
    args = parser.parse_args()

    place_mode = "move" if args.sim_feasible else args.place_mode
    stand_off = SIM_PLACE_STAND_OFF if args.sim_feasible else args.stand_off
    report, _ = run(args.save, place_mode=place_mode, stand_off=stand_off)
    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return

    print("grab_test offline IK summary")
    print(f"  place_mode={place_mode} stand_off={stand_off:.3f}")
    print(f"  pick_frames={report.pick_frames} place_frames={report.place_frames} root_frames={report.root_frames}")
    print(
        "  target_world_center="
        f"({report.target_world_center[0]:.3f}, {report.target_world_center[1]:.3f}, {report.target_world_center[2]:.3f})"
    )
    print(
        "  final_root_pose="
        f"({report.final_root_pose[0]:.3f}, {report.final_root_pose[1]:.3f}, "
        f"{report.final_root_pose[2]:.3f}, yaw={math.degrees(report.final_root_pose[3]):.1f}deg)"
    )
    print(
        "  move_lift_height "
        f"target_box_center_z={report.target_box_center_z:.3f} solved_palm_z={report.solved_palm_z:.3f}"
    )
    print(f"  pick_max_palm_error={report.pick_max_error:.4f} m")
    print(f"  pick_feasible={report.pick_feasible}")
    print(f"  place_max_palm_error={report.place_max_error:.4f} m")
    print(f"  place_feasible={report.place_feasible}")
    print(f"  place_feasible_with_flip_dofs={report.place_feasible_with_flip_dofs}")
    print(f"  path_clearance_ok={report.path_clearance_ok}")
    if report.output_file is not None:
        print(f"  saved_plan={report.output_file}")


if __name__ == "__main__":
    main()
