"""Isaac Gym scene, timeline, and trace helpers for online palletizing."""
from __future__ import annotations

import json
import math

try:
    from isaacgym import gymapi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Isaac Gym Python package is not importable. Use run_move_bpp_env.sh.") from exc

import numpy as np
import torch

from move.planning import PALLET_THICKNESS, TABLE_POSE, TABLE_SIZE, Pose
from move.robot_placement import wrap_to_pi
from move.tasks.grab_test_task import (
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
    _expand_centers,
    _expand_key_targets,
    _lerp_tensor,
    _make_transform,
    _pose_from_dict,
    _reorder_targets,
    _set_actor_root_pose,
    _set_color,
    _set_shape_friction,
    _smooth_timeline_joint_steps,
    _smoothstep,
    _transform_local_point,
    _with_hand_closure,
)
from move.palletizing_runtime import PLACE_HOLD_ATTACHED_FRAMES, SIDE_OPEN_NO_COLLISION_FRAMES
from move.tasks.move_test1 import BOX_MASS
from move.utils import LEFT_EE_LINK, PALM_CENTER_Z, PALM_SURFACE_X, RIGHT_EE_LINK, UrdfKinematics

def _make_transform_yaw_pitch(
    x: float,
    y: float,
    z: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> "gymapi.Transform":
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(x), float(y), float(z))
    yaw_half = float(yaw) * 0.5
    pitch_half = float(pitch) * 0.5
    sy = math.sin(yaw_half)
    cy = math.cos(yaw_half)
    sp = math.sin(pitch_half)
    cp = math.cos(pitch_half)
    pose.r = gymapi.Quat(-sy * sp, cy * sp, sy * cp, cy * cp)
    return pose


def _yaw_error_to_options(actual_yaw: float, yaw_options: tuple[float, ...]) -> float:
    if not yaw_options:
        return abs(wrap_to_pi(actual_yaw))
    return min(abs(wrap_to_pi(actual_yaw - option)) for option in yaw_options)


def _actor_root_speeds(gym, sim, root_states: torch.Tensor, actor_index: int) -> tuple[float, float]:
    _sync_sim_state(gym, sim)
    vel = root_states[actor_index, 7:13].detach().cpu().numpy()
    linear = float(np.linalg.norm(vel[:3]))
    angular = float(np.linalg.norm(vel[3:]))
    return linear, angular


def _set_robot_box_collision_disabled(gym, env, robot, box) -> None:
    """Keep a just-released box colliding with the world, but not with robot links."""

    box_shape_props = gym.get_actor_rigid_shape_properties(env, box)
    for prop in box_shape_props:
        prop.filter = 2
    gym.set_actor_rigid_shape_properties(env, box, box_shape_props)

    robot_shape_props = gym.get_actor_rigid_shape_properties(env, robot)
    for prop in robot_shape_props:
        prop.filter = 2
    gym.set_actor_rigid_shape_properties(env, robot, robot_shape_props)


def _freeze_placed_boxes(gym, sim, root_states: torch.Tensor, box_records: list[dict]) -> None:
    for record in box_records:
        if record.get("state") != "PLACED_FROZEN" or "frozen_pose" not in record:
            continue
        frozen = record["frozen_pose"]
        actor_index = int(record["actor_index"])
        _set_actor_root_pose(
            gym,
            sim,
            root_states,
            actor_index,
            (float(frozen[0]), float(frozen[1]), float(frozen[2])),
            float(frozen[3]),
            float(record.get("frozen_pitch", 0.0)),
        )


def _normalize_np(vec: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return vec / norm


def _root_rot_z(root_pose: Pose) -> np.ndarray:
    c = math.cos(root_pose.yaw)
    s = math.sin(root_pose.yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _offline_grasp_frame(kin: UrdfKinematics, dof_names: list[str], q: torch.Tensor, root_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    q_map = {name: float(q[i]) for i, name in enumerate(dof_names)}
    root_r = _root_rot_z(root_pose)
    root_p = np.asarray([root_pose.x, root_pose.y, root_pose.z], dtype=np.float64)

    def palm(link: str) -> tuple[np.ndarray, np.ndarray]:
        pose = kin.fk(link, q_map)
        rot_local = pose[:3, :3]
        pos_local = pose[:3, 3] + rot_local[:, 0] * PALM_SURFACE_X + rot_local[:, 2] * PALM_CENTER_Z
        return root_p + root_r @ pos_local, root_r @ rot_local[:, 2]

    left, left_finger = palm(LEFT_EE_LINK)
    right, right_finger = palm(RIGHT_EE_LINK)
    origin = 0.5 * (left + right)
    y_axis = _normalize_np(left - right, (0.0, 1.0, 0.0))
    x_axis = left_finger + right_finger
    x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
    x_axis = _normalize_np(x_axis, (1.0, 0.0, 0.0))
    z_axis = _normalize_np(np.cross(x_axis, y_axis), (0.0, 0.0, 1.0))
    y_axis = _normalize_np(np.cross(z_axis, x_axis), (0.0, 1.0, 0.0))
    return origin, np.column_stack([x_axis, y_axis, z_axis])


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    trace = float(np.trace(a.T @ b))
    value = min(max((trace - 1.0) * 0.5, -1.0), 1.0)
    return math.degrees(math.acos(value))


def _sync_sim_state(gym, sim) -> None:
    gym.refresh_actor_root_state_tensor(sim)
    try:
        gym.refresh_rigid_body_state_tensor(sim)
    except Exception:
        pass


def _fmt_vec(vec: np.ndarray | tuple[float, ...]) -> str:
    values = [float(v) for v in vec]
    return "(" + ",".join(f"{v:.4f}" for v in values) + ")"


def _fmt_mat(rot: np.ndarray) -> str:
    return "[" + ";".join(",".join(f"{float(v):.3f}" for v in row) for row in rot) + "]"


def _create_static_scene(gym, sim, env, robot_asset, stack_scene):
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), "online_robot", 0, 1)
    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, stack_scene.pallet_size[0], stack_scene.pallet_size[1], PALLET_THICKNESS, fixed_opts)
    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    pallet_z = stack_scene.pallet_surface_z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(env, pallet_asset, _make_transform(stack_scene.pallet_center[0], stack_scene.pallet_center[1], pallet_z), "pallet", 0, 0)
    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    _set_shape_friction(gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05)
    _set_shape_friction(gym, env, table, 2.5)
    _set_shape_friction(gym, env, pallet, 1.4)
    return robot


def _create_source_box(
    gym,
    sim,
    env,
    size: tuple[float, float, float],
    pose: tuple[float, float, float],
    yaw: float,
    item_index: int,
    pitch: float = 0.0,
):
    opts = gymapi.AssetOptions()
    opts.density = BOX_MASS / max(size[0] * size[1] * size[2], 1e-6)
    if hasattr(opts, "linear_damping"):
        opts.linear_damping = 0.10
    if hasattr(opts, "angular_damping"):
        opts.angular_damping = 1.00
    asset = gym.create_box(sim, size[0], size[1], size[2], opts)
    actor = gym.create_actor(env, asset, _make_transform_yaw_pitch(*pose, yaw=yaw, pitch=pitch), f"source_item_{item_index:02d}", 0, 0)
    _set_color(gym, env, actor, (0.9, 0.05, 0.02))
    _set_shape_friction(gym, env, actor, 10.0, rolling_friction=0.12, torsion_friction=0.12)
    return actor


def _build_timeline(plan: dict, dof_names: list[str]):
    pick_targets = _reorder_targets(plan["pick_targets"], list(plan["move_dof_names"]), dof_names)
    place_targets = _reorder_targets(plan["place_targets"], list(plan["place_dof_names"]), dof_names)
    release_target = _reorder_targets(plan["release_target"], list(plan["place_dof_names"]), dof_names)
    move_approach_target = _reorder_targets(plan["move_approach_target"], list(plan["move_dof_names"]), dof_names)
    move_lift_target = _reorder_targets(plan["move_lift_target"], list(plan["move_dof_names"]), dof_names)

    pick_dof_frames = _expand_key_targets(pick_targets, PICK_SEGMENT_FRAMES)
    _apply_pick_finger_closure(pick_dof_frames, dof_names)
    pick_box_frames = _expand_centers([tuple(c) for c in plan["pick_box_centers"]], PICK_SEGMENT_FRAMES)

    root_route = [(_label, _pose_from_dict(pose_dict)) for _label, pose_dict in plan["root_route"]]
    root_frames = root_route + [(root_route[-1][0], root_route[-1][1]) for _ in range(MOVE_SETTLE_FRAMES)]
    final_root = root_route[-1][1]
    place_pose_path_labeled = [(label, tuple(center), float(theta)) for label, center, theta in plan["place_pose_path_world"]]
    place_center_path = [center for _label, center, _theta in place_pose_path_labeled]
    pre_place_center = tuple(plan.get("place_handoff_center_world", place_center_path[0]))

    move_dof_frames = []
    move_box_frames = []
    move_box_thetas = []
    move_phase_labels = []
    carry_start = pick_box_frames[-1]
    carry_local_start = (carry_start[0], carry_start[1], carry_start[2])
    carry_local_final = (pre_place_center[0] - final_root.x, pre_place_center[1] - final_root.y, pre_place_center[2] - final_root.z)
    carry_local_approach = (carry_local_final[0], carry_local_final[1], float(plan["move_approach_box_center_z"]) - final_root.z)
    move_approach_theta = float(plan.get("move_approach_box_theta", 0.0))
    move_preplace_theta = float(plan.get("move_preplace_box_theta", move_approach_theta))
    for label, root_pose in root_frames:
        move_phase_labels.append(label)
        move_dof_frames.append(pick_targets[-1])
        move_box_frames.append(_transform_local_point(root_pose, carry_local_start))
        move_box_thetas.append(0.0)
    for i in range(1, MOVE_APPROACH_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_APPROACH_FRAMES))
        move_phase_labels.append("lift_to_approach")
        move_dof_frames.append(_lerp_tensor(pick_targets[-1], move_approach_target, alpha))
        carry_local = tuple(carry_local_start[j] * (1.0 - alpha) + carry_local_approach[j] * alpha for j in range(3))
        move_box_frames.append(_transform_local_point(final_root, carry_local))
        move_box_thetas.append(move_approach_theta * alpha)
    for i in range(1, MOVE_LIFT_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_LIFT_FRAMES))
        move_phase_labels.append("lift_to_preplace")
        move_dof_frames.append(_lerp_tensor(move_approach_target, move_lift_target, alpha))
        carry_local = tuple(carry_local_approach[j] * (1.0 - alpha) + carry_local_final[j] * alpha for j in range(3))
        move_box_frames.append(_transform_local_point(final_root, carry_local))
        move_box_thetas.append(move_approach_theta * (1.0 - alpha) + move_preplace_theta * alpha)

    if bool(plan.get("place_targets_are_dense", False)):
        place_dof_frames = [place_targets[i].clone() for i in range(place_targets.shape[0])]
        place_box_frames_labeled = place_pose_path_labeled
    else:
        place_dof_frames = _expand_key_targets(place_targets, PLACE_SEGMENT_FRAMES)
        place_box_frames = _expand_centers(place_center_path, PLACE_SEGMENT_FRAMES)
        place_box_frames_labeled = [("place", center, 0.0) for center in place_box_frames]

    timeline = []
    pick_stage_names = (
        "pick_ready_high",
        "pick_descend_level",
        "pick_pre_grasp",
        "pick_straight_clamp",
        "pick_compress",
        "pick_compress_hold",
        "pick_lift",
    )
    for index, (q, center) in enumerate(zip(pick_dof_frames, pick_box_frames)):
        stage_index = min(index // PICK_SEGMENT_FRAMES, len(pick_stage_names) - 1)
        timeline.append((f"pick:{pick_stage_names[stage_index]}", Pose(0.0, 0.0, 0.0, 0.0), q, center, 0.0))
    move_root_poses = [pose for _label, pose in root_frames] + [final_root for _ in range(MOVE_APPROACH_FRAMES + MOVE_LIFT_FRAMES)]
    for label, root_pose, q, center, theta in zip(move_phase_labels, move_root_poses, move_dof_frames, move_box_frames, move_box_thetas):
        timeline.append((f"move:{label}", root_pose, q, center, theta))
    if place_dof_frames:
        move_end_q = move_dof_frames[-1]
        move_end_center = move_box_frames[-1]
        move_end_theta = move_box_thetas[-1]
        first_label, first_center, first_theta = place_box_frames_labeled[0]
        for i in range(1, PLACE_HANDOFF_FRAMES + 1):
            alpha = _smoothstep(i / float(PLACE_HANDOFF_FRAMES))
            center = tuple(move_end_center[j] * (1.0 - alpha) + first_center[j] * alpha for j in range(3))
            timeline.append((f"place_handoff:{first_label}", final_root, _lerp_tensor(move_end_q, place_dof_frames[0], alpha), center, move_end_theta * (1.0 - alpha) + first_theta * alpha))
    for q, (label, center, theta) in zip(place_dof_frames, place_box_frames_labeled):
        timeline.append((f"place:{label}", final_root, q, center, theta))
    release_center = place_center_path[-1]
    closed_place_q = _with_hand_closure(place_dof_frames[-1] if place_dof_frames else move_dof_frames[-1], dof_names, 1.0)
    for _ in range(PLACE_HOLD_ATTACHED_FRAMES):
        timeline.append(("place_hold_attached", final_root, closed_place_q, release_center, 0.0))
    timeline.append(("pre_detach_sync", final_root, closed_place_q, release_center, 0.0))
    for i in range(1, SIDE_OPEN_NO_COLLISION_FRAMES + 1):
        alpha = _smoothstep(i / float(SIDE_OPEN_NO_COLLISION_FRAMES))
        timeline.append(("side_open_no_collision", final_root, _lerp_tensor(closed_place_q, release_target, alpha), release_center, 0.0))
    timeline = _apply_timeline_hand_closure(timeline, dof_names)
    return _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)


def _setup_camera(gym, env, width: int, height: int, stack_scene):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.horizontal_fov = 88.0
    cam = gym.create_camera_sensor(env, props)
    mid_x = 0.5 * (TABLE_POSE[0] + stack_scene.pallet_center[0])
    mid_y = 0.5 * (TABLE_POSE[1] + stack_scene.pallet_center[1])
    gym.set_camera_location(cam, env, gymapi.Vec3(mid_x - 2.8, mid_y + 3.9, 3.2), gymapi.Vec3(mid_x, mid_y, 0.45))
    return cam



def _actor_pose_record(gym, env, actor) -> list[float]:
    center = _actor_center(gym, env, actor)
    yaw, pitch = _actor_yaw_pitch(gym, env, actor)
    return [float(center[0]), float(center[1]), float(center[2]), float(yaw), float(pitch)]


def _write_trace_frame(
    trace_fh,
    frame: int,
    item_index: int,
    phase: str,
    root_pose: Pose,
    q: torch.Tensor,
    dof_names: list[str],
    gym,
    env,
    box_records: list[dict],
    pct_payload: dict | None,
) -> None:
    if trace_fh is None:
        return
    boxes = []
    for record in box_records:
        state = record["state"]
        visible = state != "NOT_SPAWNED"
        if not visible:
            continue
        boxes.append(
            {
                "item_index": record["item_index"],
                "state": state,
                "size_m": list(record["size_m"]),
                "asset_size_m": list(record.get("asset_size_m", record["size_m"])),
                "sim_asset_size_m": list(record.get("sim_asset_size_m", record.get("asset_size_m", record["size_m"]))),
                "source_pitch": float(record.get("source_pitch", 0.0)),
                "sim_source_pitch": float(record.get("sim_source_pitch", record.get("source_pitch", 0.0))),
                "frozen_pitch": float(record.get("frozen_pitch", 0.0)),
                "pose": _actor_pose_record(gym, env, record["actor"]),
                "visible": True,
            }
        )
    trace_fh.write(
        json.dumps(
            {
                "frame": int(frame),
                "time": float(frame) / 60.0,
                "item_index": int(item_index),
                "phase": phase,
                "robot": {
                    "root_pose": [float(root_pose.x), float(root_pose.y), float(root_pose.z), float(root_pose.yaw)],
                    "dof_names": dof_names,
                    "dof_values": [float(v) for v in q.detach().cpu().view(-1).tolist()],
                },
                "boxes": boxes,
                "pct": pct_payload,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    trace_fh.flush()


__all__ = [
    "_make_transform_yaw_pitch",
    "_yaw_error_to_options",
    "_actor_root_speeds",
    "_set_robot_box_collision_disabled",
    "_freeze_placed_boxes",
    "_offline_grasp_frame",
    "_rotation_error_deg",
    "_sync_sim_state",
    "_fmt_vec",
    "_fmt_mat",
    "_create_static_scene",
    "_create_source_box",
    "_build_timeline",
    "_setup_camera",
    "_write_trace_frame",
]
