#!/usr/bin/env python3
"""Attach a carried box to the actual two-palm SE(3) frame in Isaac Gym."""
from __future__ import annotations

import math
from dataclasses import dataclass

from isaacgym import gymapi, gymtorch  # type: ignore
import numpy as np
import torch

from move.tasks.grab_test_task import (
    _actor_center,
    _actor_yaw_pitch,
    _finger_dir,
    _palm_center,
    _quat_to_matrix,
    _set_hand_box_collision_enabled,
)


LEFT_PALM_LINK = "xhand_left_left_hand_link"
RIGHT_PALM_LINK = "xhand_right_right_hand_link"


@dataclass(frozen=True)
class ContactGap:
    left_gap_m: float
    right_gap_m: float
    max_gap_m: float
    centerline_x_m: float
    centerline_z_m: float
    centerline_error_m: float


def _normalize(vec: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return vec / norm


def _matrix_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-12)) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-12)) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-12)) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _yaw_from_rot(rot: np.ndarray) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


class AttachedBoxController:
    def __init__(self, gym, sim, env, robot, box_actor, box_size: tuple[float, float, float]):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.robot = robot
        self.box_actor = box_actor
        self.box_size = tuple(float(v) for v in box_size)
        self.box_index = gym.get_actor_index(env, box_actor, gymapi.DOMAIN_SIM)
        self.root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        self.attached = False
        self.attach_offset_local: tuple[float, float, float] | None = None
        self.attach_yaw_offset = 0.0
        self.attach_theta_offset = 0.0
        self._box_pos_in_grasp: np.ndarray | None = None
        self._box_rot_in_grasp: np.ndarray | None = None

    def _body_pose_matrix(self, actor, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        states = self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)
        names = self.gym.get_actor_rigid_body_names(self.env, actor)
        index = names.index(body_name)
        pose = states["pose"][index]
        p = pose["p"]
        q = pose["r"]
        rot = np.asarray(_quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])), dtype=np.float64)
        pos = np.asarray([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64)
        return pos, rot

    def _box_pose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        states = self.gym.get_actor_rigid_body_states(self.env, self.box_actor, gymapi.STATE_ALL)
        pose = states["pose"][0]
        p = pose["p"]
        q = pose["r"]
        rot = np.asarray(_quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])), dtype=np.float64)
        pos = np.asarray([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64)
        return pos, rot

    def _grasp_frame(self) -> tuple[np.ndarray, np.ndarray]:
        left = np.asarray(_palm_center(self.gym, self.env, self.robot, LEFT_PALM_LINK), dtype=np.float64)
        right = np.asarray(_palm_center(self.gym, self.env, self.robot, RIGHT_PALM_LINK), dtype=np.float64)
        origin = 0.5 * (left + right)
        y_axis = _normalize(left - right, (0.0, 1.0, 0.0))
        left_finger = np.asarray(_finger_dir(self.gym, self.env, self.robot, LEFT_PALM_LINK), dtype=np.float64)
        right_finger = np.asarray(_finger_dir(self.gym, self.env, self.robot, RIGHT_PALM_LINK), dtype=np.float64)
        x_axis = left_finger + right_finger
        x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
        x_axis = _normalize(x_axis, (1.0, 0.0, 0.0))
        z_axis = _normalize(np.cross(x_axis, y_axis), (0.0, 0.0, 1.0))
        y_axis = _normalize(np.cross(z_axis, x_axis), (0.0, 1.0, 0.0))
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return origin, rot

    def _set_box_root_transform(self, pos: np.ndarray, rot: np.ndarray, zero_velocity: bool = True) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)
        state = self.root_states[self.box_index]
        state[0] = float(pos[0])
        state[1] = float(pos[1])
        state[2] = float(pos[2])
        qx, qy, qz, qw = _matrix_to_quat_xyzw(rot)
        state[3] = qx
        state[4] = qy
        state[5] = qz
        state[6] = qw
        if zero_velocity:
            state[7:13] = 0.0
        indices = torch.tensor([self.box_index], dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices),
            1,
        )


    def debug_grasp_frame(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grasp_frame()

    def debug_box_pose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return self._box_pose_matrix()

    def debug_box_in_grasp(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._box_pos_in_grasp, self._box_rot_in_grasp

    def hand_box_collision_disabled(self) -> bool:
        box_shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.box_actor)
        return all(int(prop.filter) == 2 for prop in box_shape_props)

    def contact_gap(self) -> ContactGap:
        center = _actor_center(self.gym, self.env, self.box_actor)
        yaw, _pitch = _actor_yaw_pitch(self.gym, self.env, self.box_actor)
        left = _palm_center(self.gym, self.env, self.robot, LEFT_PALM_LINK)
        right = _palm_center(self.gym, self.env, self.robot, RIGHT_PALM_LINK)
        mid = tuple((float(left[i]) + float(right[i])) * 0.5 for i in range(3))
        half_y = self.box_size[1] * 0.5

        def local_x(point):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            return math.cos(yaw) * dx + math.sin(yaw) * dy

        def local_y(point):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            return -math.sin(yaw) * dx + math.cos(yaw) * dy

        left_gap = abs(local_y(left)) - half_y
        right_gap = abs(local_y(right)) - half_y
        centerline_x = float(local_x(mid))
        centerline_z = float(mid[2] - center[2])
        centerline_error = float(math.hypot(centerline_x, centerline_z))
        return ContactGap(
            float(left_gap),
            float(right_gap),
            float(max(left_gap, right_gap)),
            centerline_x,
            centerline_z,
            centerline_error,
        )

    def attach_from_current_hand_frame(self, root_yaw: float, disable_hand_box_collision: bool = True) -> ContactGap:
        del root_yaw
        self.gym.refresh_actor_root_state_tensor(self.sim)
        box_pos, box_rot = self._box_pose_matrix()
        grasp_pos, grasp_rot = self._grasp_frame()
        self._box_pos_in_grasp = grasp_rot.T @ (box_pos - grasp_pos)
        self._box_rot_in_grasp = grasp_rot.T @ box_rot
        self.attach_offset_local = tuple(float(v) for v in self._box_pos_in_grasp)
        self.attach_yaw_offset = _yaw_from_rot(box_rot) - _yaw_from_rot(grasp_rot)
        self.attach_theta_offset = 0.0
        self.attached = True
        if disable_hand_box_collision:
            _set_hand_box_collision_enabled(self.gym, self.env, self.robot, self.box_actor, enabled=False)
        return self.contact_gap()

    def update_from_current_hand_frame(self, root_yaw: float) -> None:
        del root_yaw
        if not self.attached or self._box_pos_in_grasp is None or self._box_rot_in_grasp is None:
            return
        self.gym.refresh_actor_root_state_tensor(self.sim)
        grasp_pos, grasp_rot = self._grasp_frame()
        box_pos = grasp_pos + grasp_rot @ self._box_pos_in_grasp
        box_rot = grasp_rot @ self._box_rot_in_grasp
        self._set_box_root_transform(box_pos, box_rot, zero_velocity=True)

    def detach(self, match_zero_velocity: bool = True) -> None:
        self.attached = False
        if match_zero_velocity:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.root_states[self.box_index, 7:13] = torch.zeros(6, dtype=self.root_states.dtype)
            indices = torch.tensor([self.box_index], dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(indices),
                1,
            )
