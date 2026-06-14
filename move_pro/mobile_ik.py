"""Small URDF IK helpers for the move-state lift joints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from move.utils import (
    LEFT_EE_LINK,
    RIGHT_EE_LINK,
    PALM_CENTER_Z,
    PALM_SURFACE_X,
    UrdfKinematics,
)


LIFT_JOINTS = (
    "first_leg_pitch_joint",
    "second_leg_pitch_joint",
    "waist_pitch_joint",
)
PREPLACE_CLEARANCE = 0.02


class MoveLiftIK:
    """Solve the mobile-state lift joints while arms/hands stay at q_hold."""

    def __init__(
        self,
        urdf_path: Path,
        dof_names: Sequence[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
    ):
        self.kin = UrdfKinematics(urdf_path)
        self.dof_names = list(dof_names)
        self.name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        self.lower = lower.detach().cpu().numpy().astype(np.float64)
        self.upper = upper.detach().cpu().numpy().astype(np.float64)
        self.active = [name for name in LIFT_JOINTS if name in self.name_to_index]

    def palm_z(self, q: torch.Tensor) -> float:
        q_map = {name: float(q[i]) for i, name in enumerate(self.dof_names)}
        left_z = self._palm_center_z(LEFT_EE_LINK, q_map)
        right_z = self._palm_center_z(RIGHT_EE_LINK, q_map)
        return 0.5 * (left_z + right_z)

    def body_pitch(self, q: torch.Tensor) -> float:
        return self._body_pitch_np(q.detach().cpu().numpy().astype(np.float64))

    def body_x(self, q: torch.Tensor) -> float:
        return self._body_x_np(q.detach().cpu().numpy().astype(np.float64))

    def solve_for_box_center_z(
        self,
        hold_targets: torch.Tensor,
        target_box_center_z: float,
        iterations: int = 80,
        preserve_body_pitch: bool = False,
        pitch_weight: float = 0.35,
        preserve_body_x: bool = False,
        x_weight: float = 0.35,
        target_body_pitch: float | None = None,
        target_body_x: float | None = None,
    ) -> torch.Tensor:
        q = hold_targets.detach().cpu().numpy().astype(np.float64).copy()
        if not self.active:
            return hold_targets.clone()

        damping = 0.03
        eps = 1e-4
        max_step = 0.045
        target_pitch = self._body_pitch_np(q) if target_body_pitch is None else float(target_body_pitch)
        target_x = self._body_x_np(q) if target_body_x is None else float(target_body_x)
        for _ in range(iterations):
            current_z = self._palm_z_np(q)
            current_pitch = self._body_pitch_np(q) if preserve_body_pitch else 0.0
            current_x = self._body_x_np(q) if preserve_body_x else 0.0
            z_error = float(target_box_center_z - current_z)
            error_terms = [z_error]
            if preserve_body_pitch:
                pitch_error = float(target_pitch - current_pitch)
                error_terms.append(pitch_weight * pitch_error)
            else:
                pitch_error = 0.0
            if preserve_body_x:
                x_error = float(target_x - current_x)
                error_terms.append(x_weight * x_error)
            else:
                x_error = 0.0
            error = np.array(error_terms, dtype=np.float64)
            converged = (
                abs(z_error) < 0.003
                and (not preserve_body_pitch or abs(pitch_error) < 0.015)
                and (not preserve_body_x or abs(x_error) < 0.010)
            )
            if converged:
                break

            jac = np.zeros((error.shape[0], len(self.active)), dtype=np.float64)
            for col, joint_name in enumerate(self.active):
                idx = self.name_to_index[joint_name]
                old = q[idx]
                q[idx] = old + eps
                jac[0, col] = (self._palm_z_np(q) - current_z) / eps
                row = 1
                if preserve_body_pitch:
                    jac[row, col] = pitch_weight * (self._body_pitch_np(q) - current_pitch) / eps
                    row += 1
                if preserve_body_x:
                    jac[row, col] = x_weight * (self._body_x_np(q) - current_x) / eps
                q[idx] = old

            lhs = jac @ jac.T + (damping**2) * np.eye(jac.shape[0], dtype=np.float64)
            dq = jac.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq, -max_step, max_step)
            for joint_name, delta in zip(self.active, dq):
                idx = self.name_to_index[joint_name]
                q[idx] = np.clip(q[idx] + delta, self.lower[idx], self.upper[idx])

        return torch.tensor(q, dtype=torch.float32)

    def _palm_z_np(self, q: np.ndarray) -> float:
        q_map = {name: float(q[i]) for i, name in enumerate(self.dof_names)}
        return 0.5 * (
            self._palm_center_z(LEFT_EE_LINK, q_map)
            + self._palm_center_z(RIGHT_EE_LINK, q_map)
        )

    def _palm_center_z(self, link: str, q_map: dict[str, float]) -> float:
        pose = self.kin.fk(link, q_map)
        rot = pose[:3, :3]
        palm_center = pose[:3, 3] + rot[:, 0] * PALM_SURFACE_X + rot[:, 2] * PALM_CENTER_Z
        return float(palm_center[2])

    def _body_pitch_np(self, q: np.ndarray) -> float:
        q_map = {name: float(q[i]) for i, name in enumerate(self.dof_names)}
        pose = self.kin.fk("body_yaw_link", q_map)
        rot = pose[:3, :3]
        return float(np.arctan2(rot[0, 2], rot[0, 0]))

    def _body_x_np(self, q: np.ndarray) -> float:
        q_map = {name: float(q[i]) for i, name in enumerate(self.dof_names)}
        pose = self.kin.fk("body_yaw_link", q_map)
        return float(pose[0, 3])


def target_box_center_z(target_support_z: float, box_size_z: float, clearance: float = PREPLACE_CLEARANCE) -> float:
    return float(target_support_z + box_size_z * 0.5 + clearance)
