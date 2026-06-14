"""Local IK and URDF helpers used by the move workflow.

This module intentionally duplicates the small kinematics surface that move
previously imported from ``flip.tasks.ik_controller``. Keeping it under
``move`` lets the move folder run without the flip package or flip assets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import torch


LEFT_ARM_JOINTS = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
]

RIGHT_ARM_JOINTS = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joinst",
]

LEFT_EE_LINK = "xhand_left_left_hand_link"
RIGHT_EE_LINK = "xhand_right_right_hand_link"

PALM_SURFACE_X = 0.025
PALM_CENTER_Z = -0.020
PALM_BOX_CLEARANCE = -0.032
BOX_SIDE_CONTACT_Z_RATIO = 0.18
APPROACH_GAP_READY = 0.16
APPROACH_GAP_PRE = 0.095
APPROACH_GAP_GRASP = 0.0
DEFAULT_IK_ITERATIONS = 6
APPROACH_WRIST_REGULARIZATION = 0.0


@dataclass
class JointInfo:
    """Minimal URDF joint information required for forward kinematics."""

    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class CartesianKeyframe:
    """Two-hand Cartesian IK target."""

    name: str
    duration: int
    left_pos: np.ndarray
    right_pos: np.ndarray
    left_palm_normal: np.ndarray
    right_palm_normal: np.ndarray
    left_finger_dir: np.ndarray
    right_finger_dir: np.ndarray
    hand_close: float
    extra_joints: Dict[str, float]


class UrdfKinematics:
    """Lightweight URDF forward kinematics for link world poses."""

    def __init__(self, urdf_path: Path):
        self.parents: Dict[str, str] = {}
        self.joints_by_child: Dict[str, JointInfo] = {}
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = np.zeros(3, dtype=np.float64)
            rpy = np.zeros(3, dtype=np.float64)
            if origin is not None:
                xyz = self._parse_vec(origin.attrib.get("xyz", "0 0 0"))
                rpy = self._parse_vec(origin.attrib.get("rpy", "0 0 0"))
            axis_node = joint.find("axis")
            axis = np.zeros(3, dtype=np.float64)
            if axis_node is not None:
                axis = self._parse_vec(axis_node.attrib.get("xyz", "0 0 0"))
            info = JointInfo(
                name=joint.attrib["name"],
                joint_type=joint.attrib.get("type", "fixed"),
                parent=parent,
                child=child,
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
            self.parents[child] = parent
            self.joints_by_child[child] = info

    @staticmethod
    def _parse_vec(text: str) -> np.ndarray:
        return np.asarray([float(v) for v in text.split()], dtype=np.float64)

    @staticmethod
    def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return rz @ ry @ rx

    @staticmethod
    def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        x, y, z = axis / norm
        c = math.cos(angle)
        s = math.sin(angle)
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _transform(rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = xyz
        return mat

    def _chain_to(self, link: str) -> List[JointInfo]:
        chain = []
        while link in self.joints_by_child:
            joint = self.joints_by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def fk(self, link: str, q_map: Dict[str, float]) -> np.ndarray:
        """Return the 4x4 world pose of ``link``."""
        mat = np.eye(4, dtype=np.float64)
        for joint in self._chain_to(link):
            mat = mat @ self._transform(self._rpy_to_rot(joint.rpy), joint.xyz)
            if joint.joint_type in {"revolute", "continuous"}:
                angle = float(q_map.get(joint.name, 0.0))
                mat = mat @ self._transform(self._axis_angle_to_rot(joint.axis, angle), np.zeros(3))
        return mat

    def position(self, link: str, q_map: Dict[str, float]) -> np.ndarray:
        return self.fk(link, q_map)[:3, 3]

    def rotation(self, link: str, q_map: Dict[str, float]) -> np.ndarray:
        return self.fk(link, q_map)[:3, :3]


class IKFlipBoxController:
    """Convert two-palm Cartesian targets into robot DOF targets.

    The name is kept for compatibility with existing move scripts. Unlike the
    flip task controller, this move-local version only contains the IK surface
    used by move/grab_test.py and does not own a flip keyframe state machine.
    """

    def __init__(
        self,
        dof_names: Sequence[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        urdf_path: Path,
        box_pose: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
    ):
        del box_pose, box_size
        self.dof_names = list(dof_names)
        self.name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        self.lower = lower.detach().cpu().numpy().astype(np.float64)
        self.upper = upper.detach().cpu().numpy().astype(np.float64)
        self.kin = UrdfKinematics(urdf_path)
        self.hand_dof_names = [
            name
            for name in self.dof_names
            if name.startswith("xhand_left_") or name.startswith("xhand_right_")
        ]

        self.q = np.zeros(len(self.dof_names), dtype=np.float64)
        self._apply_seed_posture()
        self.approach_wrist_reference = {
            name: float(self.q[self.name_to_index[name]])
            for name in [
                "wrist_pitch_l_joint",
                "wrist_roll_l_joint",
                "wrist_pitch_r_joint",
                "wrist_roll_r_joinst",
            ]
            if name in self.name_to_index
        }

    def initial_targets(self) -> torch.Tensor:
        return torch.tensor(self.q, dtype=torch.float32)

    def _apply_seed_posture(self) -> None:
        seed = {
            "shoulder_pitch_l_joint": -0.40,
            "shoulder_roll_l_joint": 0.15,
            "shoulder_yaw_l_joint": -0.40,
            "elbow_pitch_l_joint": -1.00,
            "elbow_yaw_l_joint": 0.00,
            "wrist_pitch_l_joint": 0.00,
            "wrist_roll_l_joint": -0.10,
            "shoulder_pitch_r_joint": -0.40,
            "shoulder_roll_r_joint": -0.15,
            "shoulder_yaw_r_joint": 0.40,
            "elbow_pitch_r_joint": -1.00,
            "elbow_yaw_r_joint": 0.00,
            "wrist_pitch_r_joint": 0.00,
            "wrist_roll_r_joinst": 0.10,
        }
        for name, value in seed.items():
            self._set_joint(name, value)
        self._clamp_all()

    def _solve_keyframe(
        self,
        keyframe: CartesianKeyframe,
        iterations: int,
        regularize_wrist: bool = False,
    ) -> None:
        self._solve_arm(
            LEFT_EE_LINK,
            LEFT_ARM_JOINTS,
            keyframe.left_pos,
            keyframe.left_palm_normal,
            keyframe.left_finger_dir,
            iterations,
            regularize_wrist,
        )
        self._solve_arm(
            RIGHT_EE_LINK,
            RIGHT_ARM_JOINTS,
            keyframe.right_pos,
            keyframe.right_palm_normal,
            keyframe.right_finger_dir,
            iterations,
            regularize_wrist,
        )

    def _solve_arm(
        self,
        ee_link: str,
        joint_names: Sequence[str],
        target_pos: np.ndarray,
        target_palm_normal: np.ndarray,
        target_finger_dir: np.ndarray,
        iterations: int,
        regularize_wrist: bool = False,
    ) -> None:
        active = [name for name in joint_names if name in self.name_to_index]
        if not active:
            return

        damping = 0.045
        eps = 1e-4
        for _ in range(iterations):
            current_feature = self._ee_feature(ee_link)
            target_feature = self._target_feature(target_pos, target_palm_normal, target_finger_dir)
            error = target_feature - current_feature
            pos_error = target_pos - current_feature[:3]
            palm_error = np.linalg.norm(target_palm_normal - current_feature[3:6] / self._orientation_scale())
            finger_error = np.linalg.norm(target_finger_dir - current_feature[6:9] / self._orientation_scale())
            if np.linalg.norm(pos_error) < 0.004 and palm_error < 0.08 and finger_error < 0.08:
                break

            jac = np.zeros((9, len(active)), dtype=np.float64)
            for col, joint_name in enumerate(active):
                idx = self.name_to_index[joint_name]
                old = self.q[idx]
                self.q[idx] = old + eps
                moved = self._ee_feature(ee_link)
                jac[:, col] = (moved - current_feature) / eps
                self.q[idx] = old

            lhs = jac @ jac.T + (damping**2) * np.eye(9)
            dq = jac.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq, -0.06, 0.06)
            for joint_name, delta in zip(active, dq):
                idx = self.name_to_index[joint_name]
                self.q[idx] += float(delta)
                self.q[idx] = np.clip(self.q[idx], self.lower[idx], self.upper[idx])
            if regularize_wrist and APPROACH_WRIST_REGULARIZATION > 0.0:
                self._regularize_approach_wrist(joint_names)

    def _regularize_approach_wrist(self, joint_names: Sequence[str]) -> None:
        gain = min(max(float(APPROACH_WRIST_REGULARIZATION), 0.0), 1.0)
        for name in joint_names:
            if "wrist" not in name or name not in self.approach_wrist_reference:
                continue
            idx = self.name_to_index[name]
            reference = self.approach_wrist_reference[name]
            self.q[idx] = (1.0 - gain) * self.q[idx] + gain * reference
            self.q[idx] = np.clip(self.q[idx], self.lower[idx], self.upper[idx])

    @staticmethod
    def _orientation_scale() -> float:
        return 0.20

    def _ee_feature(self, ee_link: str) -> np.ndarray:
        pose = self.kin.fk(ee_link, self._q_map())
        rot = pose[:3, :3]
        palm_normal = rot[:, 0]
        finger_dir = rot[:, 2]
        palm_center = pose[:3, 3] + palm_normal * PALM_SURFACE_X + finger_dir * PALM_CENTER_Z
        scale = self._orientation_scale()
        return np.concatenate([palm_center, palm_normal * scale, finger_dir * scale])

    def _target_feature(self, pos: np.ndarray, palm_normal: np.ndarray, finger_dir: np.ndarray) -> np.ndarray:
        scale = self._orientation_scale()
        return np.concatenate([pos, palm_normal * scale, finger_dir * scale])

    def _apply_hand_close(self, close_amount: float) -> None:
        del close_amount
        for dof_name in self.hand_dof_names:
            self._set_joint(dof_name, 0.0)

    def _apply_extra_joints(self, targets: Dict[str, float]) -> None:
        for name, value in targets.items():
            self._set_joint(name, value)

    def _set_joint(self, name: str, value: float) -> None:
        if name not in self.name_to_index:
            return
        idx = self.name_to_index[name]
        self.q[idx] = np.clip(float(value), self.lower[idx], self.upper[idx])

    def _clamp_all(self) -> None:
        self.q = np.maximum(np.minimum(self.q, self.upper), self.lower)

    def _q_map(self) -> Dict[str, float]:
        return {name: float(self.q[index]) for name, index in self.name_to_index.items()}
