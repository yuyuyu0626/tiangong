#!/usr/bin/env python3
"""task1_2：标准 60 箱堆叠的调参版本。

脚本内容：
1. 垛型仍为 task1 的 3 x 5 x 4、共 60 箱；
2. 第 1、2 层保留 task1 路线，并加入第二层释放等待、角块收手和推入调试策略；
3. 第 3、4 层保留 task1 路线，并加入角块推入、第四层十字块抓取/抬升等调参策略；


运行方式：
    python -m move.tasks.task1_2
    python -m move.tasks.task1_2 --second-layer-proxy
    python -m move.tasks.task1_2 --third-fourth-proxy

模式说明：
    默认模式              跑完整 60 箱。
    --second-layer-proxy   只调试第 2 层，第 1 层用 1.0 x 1.0 x 0.4 m 代理块。
    --third-fourth-proxy   只调试第 3、4 层，第 1、2 层用 1.0 x 1.0 x 0.8 m 代理块。
    --fast/--fast-viewer   用于 viewer 加速播放。
    --viewer-render-every  后加2、4、8等参数调整渲染速度
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

try:
    from isaacgym import gymapi  # type: ignore
    from isaacgym import gymtorch  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on local Isaac Gym install
    raise SystemExit("Isaac Gym Python package is not importable. Activate the gym environment first.") from exc

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("NumPy and PyTorch are required for task1.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move.grab_test import (  # noqa: E402
    GRASP_CONTACT_X_OFFSET,
    MOVE_APPROACH_HEIGHT,
    MOVE_PREPLACE_HEIGHT,
    MOVE_URDF,
    PICK_LIFT_HEIGHT,
    PICK_READY_Z_OFFSET,
    PICK_SIDE_CONTACT_Z_RATIO,
    PICK_TOUCH_GAP,
    PLACE_CONTACT_GAP,
    PLACE_HOLD_FRAMES,
    PLACE_RELEASE_HEIGHT,
    PLACE_UPRIGHT_FRAMES,
    _box_theta_from_q,
    _grasp_center_from_q,
    _parse_active_dofs,
    _pose_path_key_indices,
    _solve_keyframes,
    _world_to_root,
)
from move.mobile_ik import MoveLiftIK  # noqa: E402
from move.planning import (  # noqa: E402
    PALLET_SURFACE_Z,
    PALLET_THICKNESS,
    SAFE_CORRIDOR_MARGIN,
    TABLE_POSE,
    TABLE_SIZE,
    Pose,
    pallet_center_near_table,
    sample_root_trajectory,
)
from move.tasks.grab_test_task import (  # noqa: E402
    ATTACH_AFTER_PICK_FRAMES,
    FINAL_HOLD_FRAMES,
    MOVE_APPROACH_FRAMES,
    MOVE_LIFT_FRAMES,
    MOVE_SETTLE_FRAMES,
    PICK_SEGMENT_FRAMES,
    PLACE_HANDOFF_FRAMES,
    _actor_center,
    _actor_yaw_pitch,
    _add_vec,
    _apply_pick_finger_closure,
    _distance,
    _dof_error_summary,
    _expand_key_targets,
    _grasp_center,
    _grasp_theta,
    _inverse_rotate_yaw_pitch,
    _lock_mobile_dof_state,
    _reorder_targets,
    _rotate_yaw_pitch,
    _set_actor_collision_filter,
    _set_actor_root_pose,
    _set_hand_box_collision_enabled,
    _set_robot_dof_state,
    _set_task_collision_filters,
    _smooth_timeline_joint_steps,
    _smoothstep,
    _sub_vec,
    _transform_local_point,
)
from move.tasks.move_test1 import (  # noqa: E402
    BOX_MASS,
    _make_transform,
    _set_color,
    _set_shape_friction,
    create_sim,
    load_robot_asset,
)
from move.utils import (  # noqa: E402
    APPROACH_GAP_PRE,
    APPROACH_GAP_READY,
    BOX_SIDE_CONTACT_Z_RATIO,
    IKFlipBoxController,
    LEFT_ARM_JOINTS,
    LEFT_EE_LINK,
    PALM_BOX_CLEARANCE,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_LINK,
)
# Embedded layer-3/4 waist-arm implementation.
# Defined as real Python code so task1.py stays standalone without source-string exec.
def _build_waist_stack_namespace() -> SimpleNamespace:
    from xml.etree import ElementTree as ET

    from move.utils import (
        LEFT_ARM_JOINTS,
        LEFT_EE_LINK,
        PALM_BOX_CLEARANCE,
        PALM_CENTER_Z,
        PALM_SURFACE_X,
        RIGHT_ARM_JOINTS,
        RIGHT_EE_LINK,
        UrdfKinematics,
    )
    from move.planning import (
        PALLET_SURFACE_Z,
        PALLET_THICKNESS,
        SAFE_CORRIDOR_MARGIN,
        TABLE_POSE,
        TABLE_SIZE,
        pallet_center_near_table,
        sample_root_trajectory,
    )

    ASSET_ROOT = MOVE_ROOT / "assets"
    ROBOT_ASSET_FILE = "integrated/tianyi_xhand_move.urdf"
    MOVE_URDF = ASSET_ROOT / ROBOT_ASSET_FILE

    COMPUTE_DEVICE_ID = 0
    GRAPHICS_DEVICE_ID = 0
    LIFT_JOINTS = (
        "first_leg_pitch_joint",
        "second_leg_pitch_joint",
        "waist_pitch_joint",
    )
    BODY_LINK = "body_yaw_link"
    LEFT_PALM_NORMAL = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    RIGHT_PALM_NORMAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    FINGER_FORWARD = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    STACK_SIZE = (1.0, 1.0, 1.6)
    GRID_X = 3
    GRID_Y = 5
    GRID_Z = 4
    STACK_BOX_SIZE = (STACK_SIZE[0] / GRID_X, STACK_SIZE[1] / GRID_Y, STACK_SIZE[2] / GRID_Z)
    PALLET_CENTER = pallet_center_near_table()
    SOURCE_BOX_POSE = (
        TABLE_POSE[0] - 0.10,
        TABLE_POSE[1],
        TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + STACK_BOX_SIZE[2] * 0.5,
    )
    DIRECT_OUTER_STAND_OFF = 0.30
    DIRECT_TMP_RETREAT = 0.85
    DIRECT_PREPLACE_CLEARANCE = 0.18
    STACK_CROSS_NEIGHBOR_STAND_OFF = 0.50
    THIRD_LAYER_CROSS_CORNER_STAND_OFF = 0.50
    THIRD_LAYER_CROSS_EXTENSION_STAND_OFF = 0.70
    THIRD_LAYER_OUTER_CORNER_STAND_OFF = 0.70
    FOURTH_LAYER_CROSS_CORNER_STAND_OFF = 0.50
    FOURTH_LAYER_CROSS_EXTENSION_STAND_OFF = 0.70
    FOURTH_LAYER_OUTER_CORNER_STAND_OFF = 0.70
    FOURTH_LAYER_CORNER_STAND_OFF_REDUCTION = 0.05
    STACK60_DIRECT_OUTER_STAND_OFF = 0.80
    STACK60_DIRECT_Y_SIDE_STAND_OFF = 0.68
    STACK60_DIRECT_Y_EXTENSION_STAND_OFF = 0.90
    STACK60_SECOND_LAYER_CENTER_STAND_OFF_REDUCTION = 0.05
    FOURTH_LAYER_CENTER_STAND_OFF_REDUCTION = 0.00
    STACK_CROSS_CENTER_RELEASE_CLEARANCE = 0.010
    THIRD_LAYER_X_NEIGHBOR_CENTER_RELEASE_EXTRA = -0.010
    FOURTH_LAYER_X_NEIGHBOR_CENTER_RELEASE_EXTRA = 0.020
    FOURTH_LAYER_CENTER_TARGET_FORWARD_OFFSET = 0.010
    FOURTH_LAYER_CENTER_CARRY_Z_REDUCTION = 0.100
    THIRD_LAYER_CENTER_GRIP_HEIGHT_FROM_BOTTOM = 0.10
    THIRD_LAYER_X_NEIGHBOR_GRIP_ROBOT_SIDE_OFFSET = 0.080
    THIRD_LAYER_X_NEIGHBOR_GRIP_HEIGHT_EXTRA = 0.100
    FOURTH_LAYER_X_NEIGHBOR_GRIP_HEIGHT_REDUCTION = 0.020
    FOURTH_LAYER_X_NEIGHBOR_LIFT_Z_REDUCTION = 0.100
    FOURTH_LAYER_X_NEIGHBOR_LIFT_FORWARD = 0.150
    PLACE_RELEASE_HEIGHT = 0.030
    STACK_RELEASE_HEIGHT = 0.040
    STACK_PRE_RELEASE_HOLD_FRAMES = 36
    STACK_OPEN_HAND_FRAMES = 80
    STACK_OPEN_HAND_LIFT_Z = 0.08
    STACK_RETURN_RECOVER_FRAMES = 160
    STACK_RETURN_HOLD_FRAMES = 30
    STACK_SEQUENCE_PLAN_IK_STRIDE = 10
    # More negative means the palm feature target moves back relative to the box
    # center, so the fingertips become the effective front contact.
    STACK_GRASP_CONTACT_X_OFFSET = 0.025 - 0.25
    STACK_PICK_READY_Z = 0.14
    STACK_PICK_LIFT_Z = 0.30
    BOX_MASS = 0.35
    PROXY_MASS = 8.0
    BOX_PARK_X = -6.0
    BOX_PARK_SPACING = 0.55
    FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE = 0.015
    FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE = 0.010
    FIRST_LAYER_Y_EXTENSION_RELEASE_CLEARANCE = 0.020
    FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE = 0.030
    FIRST_LAYER_INNER_HAND_X_CLEARANCE = 0.080
    FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE = 0.100
    THIRD_LAYER_CROSS_CORNER_RELEASE_CLEARANCE = 0.030
    THIRD_LAYER_CROSS_CORNER_HAND_CLEARANCE = 0.050
    FOURTH_LAYER_CORNER_HAND_CLEARANCE_EXTRA = 0.020
    FOURTH_LAYER_CORNER_FRONT_CLEARANCE_EXTRA = 0.020
    FIRST_LAYER_Y_ROW_BOXES = set(range(4, 16))
    FIRST_LAYER_Y_SWAPPED_BOXES = FIRST_LAYER_Y_ROW_BOXES
    FIRST_LAYER_INNER_EDGE_PUSH_BOXES = {6, 7, 8, 9, 12, 13, 14, 15}
    STACK_CROSS_CORNER_BOXES = {6, 7, 8, 9}
    STACK_CROSS_EXTENSION_BOXES = {10, 11}
    STACK_OUTER_CORNER_BOXES = {12, 13, 14, 15}
    STACK_CROSS_ORDER = (
        (1, 2),
        (0, 2),
        (2, 2),
        (1, 1),
        (1, 3),
    )
    STACK_REST_ORDER = (
        (0, 1),
        (2, 1),
        (0, 3),
        (2, 3),
        (1, 0),
        (1, 4),
        (0, 0),
        (2, 0),
        (0, 4),
        (2, 4),
    )


    @dataclass(frozen=True)
    class RootPose:
        x: float
        y: float
        z: float
        yaw: float


    @dataclass(frozen=True)
    class StackFrame:
        phase: str
        root: RootPose
        q: torch.Tensor
        box_center: tuple[float, float, float]
        attached: bool
        box_yaw: float = 0.0
        box_pitch: float = 0.0


    @dataclass(frozen=True)
    class StackCell:
        sequence: int
        layer: int
        ix: int
        iy: int

        @property
        def label(self) -> str:
            return f"box_{self.sequence:02d}_L{self.layer + 1}_x{self.ix}_y{self.iy}"


    @dataclass(frozen=True)
    class StackBoxPlan:
        cell: StackCell
        timeline: list[StackFrame]
        metadata: dict[str, object]


    @dataclass(frozen=True)
    class IkReport:
        pos_error: float
        palm_error: float = 0.0
        finger_error: float = 0.0
        body_pitch: float = 0.0
        body_x: float = 0.0
        body_z: float = 0.0


    def _load_gymapi():
        return gymapi


    def _load_gymtorch():
        return gymtorch


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


    def _seed_q(dof_names: list[str], lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(len(dof_names), dtype=torch.float32)
        name_to_index = {name: i for i, name in enumerate(dof_names)}
        seed = {
            "first_leg_pitch_joint": 0.0,
            "second_leg_pitch_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "body_yaw_joint": 0.0,
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
            if name in name_to_index:
                q[name_to_index[name]] = float(value)
        return torch.maximum(torch.minimum(q, upper), lower)


    def _q_map(dof_names: list[str], q: np.ndarray) -> dict[str, float]:
        return {name: float(q[i]) for i, name in enumerate(dof_names)}


    def _palm_feature(
        kin: UrdfKinematics,
        dof_names: list[str],
        q: np.ndarray,
        link: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = kin.fk(link, _q_map(dof_names, q))
        rot = pose[:3, :3]
        palm_normal = rot[:, 0]
        finger_dir = rot[:, 2]
        palm_center = pose[:3, 3] + palm_normal * PALM_SURFACE_X + finger_dir * PALM_CENTER_Z
        return palm_center, palm_normal, finger_dir


    def _body_pose_terms(
        kin: UrdfKinematics,
        dof_names: list[str],
        q: np.ndarray,
    ) -> tuple[float, float, float]:
        pose = kin.fk(BODY_LINK, _q_map(dof_names, q))
        rot = pose[:3, :3]
        body_x = float(pose[0, 3])
        body_z = float(pose[2, 3])
        body_pitch = float(np.arctan2(rot[0, 2], rot[0, 0]))
        return body_z, body_pitch, body_x


    def _lift_chain_origins(kin: UrdfKinematics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the x/z offsets that define the three-pitch upright lift chain."""

        chain = kin._chain_to(BODY_LINK)  # pylint: disable=protected-access
        origins_by_joint = {joint.name: joint.xyz for joint in chain}
        missing = [name for name in LIFT_JOINTS if name not in origins_by_joint]
        if missing:
            raise RuntimeError(f"Missing lift joints in {BODY_LINK} kinematic chain: {missing}")

        terminal = np.zeros(3, dtype=np.float64)
        after_last_lift = False
        for joint in chain:
            if joint.name == LIFT_JOINTS[-1]:
                after_last_lift = True
                continue
            if after_last_lift:
                terminal = terminal + joint.xyz
        return (
            origins_by_joint[LIFT_JOINTS[0]],
            origins_by_joint[LIFT_JOINTS[1]],
            origins_by_joint[LIFT_JOINTS[2]],
            terminal,
        )


    def _lift_xz_from_pitch_chain(
        origins: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        q1: np.ndarray,
        q2: np.ndarray,
        body_pitch: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        o1, o2, o3, terminal = origins
        a1 = q1
        a2 = q1 + q2
        c1 = np.cos(a1)
        s1 = np.sin(a1)
        c2 = np.cos(a2)
        s2 = np.sin(a2)
        cp = math.cos(float(body_pitch))
        sp = math.sin(float(body_pitch))
        body_x = (
            float(o1[0])
            + c1 * float(o2[0])
            + s1 * float(o2[2])
            + c2 * float(o3[0])
            + s2 * float(o3[2])
            + cp * float(terminal[0])
            + sp * float(terminal[2])
        )
        body_z = (
            float(o1[2])
            - s1 * float(o2[0])
            + c1 * float(o2[2])
            - s2 * float(o3[0])
            + c2 * float(o3[2])
            - sp * float(terminal[0])
            + cp * float(terminal[2])
        )
        return body_x, body_z


    def _seed_upright_lift_solution(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower_np: np.ndarray,
        upper_np: np.ndarray,
        q_start: np.ndarray,
        target_body_z: float,
        target_body_pitch: float,
        target_body_x: float,
        x_weight: float,
    ) -> tuple[np.ndarray, float]:
        """Pick a reachable upright waist/leg seed and clamp unreachable z targets."""

        name_to_index = {name: i for i, name in enumerate(dof_names)}
        idx1 = name_to_index[LIFT_JOINTS[0]]
        idx2 = name_to_index[LIFT_JOINTS[1]]
        idx3 = name_to_index[LIFT_JOINTS[2]]
        q1_values = np.linspace(lower_np[idx1], upper_np[idx1], 97, dtype=np.float64)
        q2_values = np.linspace(lower_np[idx2], upper_np[idx2], 97, dtype=np.float64)
        q1_grid, q2_grid = np.meshgrid(q1_values, q2_values, indexing="ij")
        q3_grid = float(target_body_pitch) - q1_grid - q2_grid
        valid = (q3_grid >= lower_np[idx3]) & (q3_grid <= upper_np[idx3])
        if not np.any(valid):
            return q_start.copy(), float(target_body_z)

        body_x_grid, body_z_grid = _lift_xz_from_pitch_chain(
            _lift_chain_origins(kin),
            q1_grid,
            q2_grid,
            target_body_pitch,
        )
        valid_z = body_z_grid[valid]
        effective_target_z = float(np.clip(float(target_body_z), float(np.min(valid_z)), float(np.max(valid_z))))

        motion_score = (
            np.abs(q1_grid - q_start[idx1])
            + np.abs(q2_grid - q_start[idx2])
            + np.abs(q3_grid - q_start[idx3])
        )
        score = (
            np.abs(body_z_grid - effective_target_z)
            + max(0.0, float(x_weight)) * np.abs(body_x_grid - float(target_body_x))
            + 0.006 * motion_score
        )
        score = np.where(valid, score, np.inf)
        best_flat = int(np.argmin(score))
        best = np.unravel_index(best_flat, score.shape)

        q_seed = q_start.copy()
        q_seed[idx1] = float(q1_grid[best])
        q_seed[idx2] = float(q2_grid[best])
        q_seed[idx3] = float(q3_grid[best])
        return q_seed, effective_target_z


    def solve_lift_body_z(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        target_body_z: float,
        iterations: int,
        target_body_pitch: float = 0.0,
        target_body_x: float | None = None,
        pitch_weight: float = 1.80,
        x_weight: float = 0.12,
    ) -> tuple[torch.Tensor, IkReport]:
        """Solve only the three waist/leg pitch joints for upright z motion."""

        q = q_start.detach().cpu().numpy().astype(np.float64).copy()
        lower_np = lower.detach().cpu().numpy().astype(np.float64)
        upper_np = upper.detach().cpu().numpy().astype(np.float64)
        name_to_index = {name: i for i, name in enumerate(dof_names)}
        active = [name for name in LIFT_JOINTS if name in name_to_index]
        if len(active) != len(LIFT_JOINTS):
            missing = sorted(set(LIFT_JOINTS) - set(active))
            raise RuntimeError(f"Missing lift joints in URDF active DOFs: {missing}")

        _initial_z, _initial_pitch, initial_x = _body_pose_terms(kin, dof_names, q)
        target_x = initial_x if target_body_x is None else float(target_body_x)
        eps = 1e-4
        max_step = 0.030
        requested_target_z = float(target_body_z)
        q, effective_target_z = _seed_upright_lift_solution(
            kin,
            dof_names,
            lower_np,
            upper_np,
            q,
            requested_target_z,
            float(target_body_pitch),
            target_x,
            x_weight,
        )

        def solve_stage(
            stage_iterations: int,
            include_x: bool,
            damping: float,
            z_tolerance: float,
            pitch_tolerance: float,
        ) -> None:
            nonlocal q
            for _ in range(max(1, int(stage_iterations))):
                current_z, current_pitch, current_x = _body_pose_terms(kin, dof_names, q)
                z_error = float(effective_target_z - current_z)
                pitch_error = float(target_body_pitch - current_pitch)
                x_error = float(target_x - current_x)
                if abs(z_error) < z_tolerance and abs(pitch_error) < pitch_tolerance:
                    if not include_x or abs(x_error) < 0.030:
                        break

                error_terms = [z_error, pitch_weight * pitch_error]
                if include_x:
                    error_terms.append(x_weight * x_error)
                error = np.array(error_terms, dtype=np.float64)
                jac = np.zeros((error.shape[0], len(active)), dtype=np.float64)
                for col, joint_name in enumerate(active):
                    idx = name_to_index[joint_name]
                    old = q[idx]
                    q[idx] = old + eps
                    moved_z, moved_pitch, moved_x = _body_pose_terms(kin, dof_names, q)
                    jac[0, col] = (moved_z - current_z) / eps
                    jac[1, col] = pitch_weight * (moved_pitch - current_pitch) / eps
                    if include_x:
                        jac[2, col] = x_weight * (moved_x - current_x) / eps
                    q[idx] = old
                lhs = jac @ jac.T + (damping**2) * np.eye(jac.shape[0], dtype=np.float64)
                dq = jac.T @ np.linalg.solve(lhs, error)
                dq = np.clip(dq, -max_step, max_step)
                for joint_name, delta in zip(active, dq):
                    idx = name_to_index[joint_name]
                    q[idx] = np.clip(q[idx] + delta, lower_np[idx], upper_np[idx])

        primary_iterations = max(1, int(iterations) * 2 // 3)
        refine_iterations = max(1, int(iterations) - primary_iterations)
        solve_stage(primary_iterations, include_x=False, damping=0.020, z_tolerance=0.0025, pitch_tolerance=0.006)
        solve_stage(refine_iterations, include_x=True, damping=0.030, z_tolerance=0.0030, pitch_tolerance=0.008)

        body_z, body_pitch, body_x = _body_pose_terms(kin, dof_names, q)
        # If the light x-preservation pass pulled the body away from upright, spend
        # a short final pass only on the two hard constraints.
        if abs(effective_target_z - body_z) > 0.004 or abs(target_body_pitch - body_pitch) > 0.010:
            solve_stage(max(12, int(iterations) // 4), include_x=False, damping=0.020, z_tolerance=0.0030, pitch_tolerance=0.007)

        body_z, body_pitch, body_x = _body_pose_terms(kin, dof_names, q)
        report = IkReport(
            pos_error=abs(float(requested_target_z - body_z)),
            body_pitch=body_pitch,
            body_x=body_x,
            body_z=body_z,
        )
        return torch.tensor(q, dtype=torch.float32), report


    def solve_arm_palm_target(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        side: str,
        target_pos: np.ndarray,
        iterations: int,
        orientation_scale: float = 0.20,
    ) -> tuple[torch.Tensor, IkReport]:
        """Solve one arm's 7 joints to a fixed-orientation palm target."""

        if side == "left":
            link = LEFT_EE_LINK
            arm_joints = LEFT_ARM_JOINTS
            target_palm_normal = LEFT_PALM_NORMAL
        elif side == "right":
            link = RIGHT_EE_LINK
            arm_joints = RIGHT_ARM_JOINTS
            target_palm_normal = RIGHT_PALM_NORMAL
        else:
            raise ValueError(f"Unsupported arm side: {side}")

        q = q_start.detach().cpu().numpy().astype(np.float64).copy()
        lower_np = lower.detach().cpu().numpy().astype(np.float64)
        upper_np = upper.detach().cpu().numpy().astype(np.float64)
        name_to_index = {name: i for i, name in enumerate(dof_names)}
        active = [name for name in arm_joints if name in name_to_index]
        if len(active) != len(arm_joints):
            missing = sorted(set(arm_joints) - set(active))
            raise RuntimeError(f"Missing {side} arm joints in URDF active DOFs: {missing}")

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_finger_dir = FINGER_FORWARD
        target_feature = np.concatenate(
            [
                target_pos,
                target_palm_normal * orientation_scale,
                target_finger_dir * orientation_scale,
            ]
        )
        damping = 0.045
        eps = 1e-4
        max_step = 0.060
        for _ in range(iterations):
            palm_pos, palm_normal, finger_dir = _palm_feature(kin, dof_names, q, link)
            current_feature = np.concatenate(
                [
                    palm_pos,
                    palm_normal * orientation_scale,
                    finger_dir * orientation_scale,
                ]
            )
            error = target_feature - current_feature
            pos_error = float(np.linalg.norm(target_pos - palm_pos))
            palm_error = float(np.linalg.norm(target_palm_normal - palm_normal))
            finger_error = float(np.linalg.norm(target_finger_dir - finger_dir))
            if pos_error < 0.0035 and palm_error < 0.060 and finger_error < 0.060:
                break
            jac = np.zeros((9, len(active)), dtype=np.float64)
            for col, joint_name in enumerate(active):
                idx = name_to_index[joint_name]
                old = q[idx]
                q[idx] = old + eps
                moved_pos, moved_normal, moved_finger = _palm_feature(kin, dof_names, q, link)
                moved_feature = np.concatenate(
                    [
                        moved_pos,
                        moved_normal * orientation_scale,
                        moved_finger * orientation_scale,
                    ]
                )
                jac[:, col] = (moved_feature - current_feature) / eps
                q[idx] = old
            lhs = jac @ jac.T + (damping**2) * np.eye(9, dtype=np.float64)
            dq = jac.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq, -max_step, max_step)
            for joint_name, delta in zip(active, dq):
                idx = name_to_index[joint_name]
                q[idx] = np.clip(q[idx] + delta, lower_np[idx], upper_np[idx])

        palm_pos, palm_normal, finger_dir = _palm_feature(kin, dof_names, q, link)
        report = IkReport(
            pos_error=float(np.linalg.norm(target_pos - palm_pos)),
            palm_error=float(np.linalg.norm(target_palm_normal - palm_normal)),
            finger_error=float(np.linalg.norm(target_finger_dir - finger_dir)),
        )
        return torch.tensor(q, dtype=torch.float32), report


    def fix_initial_hand_orientation(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_seed: torch.Tensor,
        iterations: int,
    ) -> tuple[torch.Tensor, IkReport, IkReport, tuple[float, float, float], tuple[float, float, float]]:
        """Make frame zero already satisfy the fixed palm orientation contract."""

        q_np = q_seed.detach().cpu().numpy().astype(np.float64)
        left_start, _left_normal, _left_finger = _palm_feature(kin, dof_names, q_np, LEFT_EE_LINK)
        right_start, _right_normal, _right_finger = _palm_feature(kin, dof_names, q_np, RIGHT_EE_LINK)
        q, left_report = solve_arm_palm_target(
            kin,
            dof_names,
            lower,
            upper,
            q_seed,
            "left",
            left_start,
            iterations,
        )
        q, right_report = solve_arm_palm_target(
            kin,
            dof_names,
            lower,
            upper,
            q,
            "right",
            right_start,
            iterations,
        )
        return (
            q,
            left_report,
            right_report,
            tuple(float(v) for v in left_start),
            tuple(float(v) for v in right_start),
        )


    def _smoothstep(alpha: float) -> float:
        alpha = min(max(float(alpha), 0.0), 1.0)
        return alpha * alpha * (3.0 - 2.0 * alpha)


    def _stack_plan_ik_stride(args: argparse.Namespace, return_home: bool) -> int:
        if not return_home:
            return 1
        return max(1, int(getattr(args, "stack_plan_ik_stride", STACK_SEQUENCE_PLAN_IK_STRIDE)))


    def _segment_key_frames(frame_count: int, stride: int) -> list[int]:
        frame_count = max(1, int(frame_count))
        stride = max(1, int(stride))
        keys = list(range(stride, frame_count + 1, stride))
        if not keys or keys[-1] != frame_count:
            keys.append(frame_count)
        return keys


    def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        return a * (1.0 - alpha) + b * alpha


    def _expand_segment(start: torch.Tensor, goal: torch.Tensor, frames: int) -> list[torch.Tensor]:
        return [_lerp_tensor(start, goal, _smoothstep(i / float(max(1, frames)))) for i in range(1, frames + 1)]


    def _build_lift_targets(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        initial_body_z: float,
        target_body_z: float,
        target_body_x: float,
        frames: int,
        iterations: int,
        arm_orientation_iterations: int,
    ) -> tuple[list[torch.Tensor], IkReport]:
        """Build a lift path whose intermediate frames keep body and palms upright."""

        targets: list[torch.Tensor] = []
        q = q_start.clone()
        report = IkReport(pos_error=0.0)
        frame_count = max(1, int(frames))
        step_iterations = max(18, int(iterations) // 3)
        for i in range(1, frame_count + 1):
            alpha = _smoothstep(i / float(frame_count))
            body_z = initial_body_z * (1.0 - alpha) + target_body_z * alpha
            q, report = solve_lift_body_z(
                kin,
                dof_names,
                lower,
                upper,
                q,
                body_z,
                iterations=iterations if i == frame_count else step_iterations,
                target_body_pitch=0.0,
                target_body_x=target_body_x,
            )
            q_np = q.detach().cpu().numpy().astype(np.float64)
            left_pos, _left_normal, _left_finger = _palm_feature(kin, dof_names, q_np, LEFT_EE_LINK)
            right_pos, _right_normal, _right_finger = _palm_feature(kin, dof_names, q_np, RIGHT_EE_LINK)
            q, _left_report = solve_arm_palm_target(
                kin,
                dof_names,
                lower,
                upper,
                q,
                "left",
                left_pos,
                arm_orientation_iterations,
            )
            q, _right_report = solve_arm_palm_target(
                kin,
                dof_names,
                lower,
                upper,
                q,
                "right",
                right_pos,
                arm_orientation_iterations,
            )
            targets.append(q.clone())
        return targets, report


    def _build_arm_waypoint_targets(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_lift: torch.Tensor,
        left_target: tuple[float, float, float] | None,
        right_target: tuple[float, float, float] | None,
        waypoints: int,
        iterations_per_waypoint: int,
    ) -> tuple[list[torch.Tensor], IkReport, IkReport, tuple[float, float, float], tuple[float, float, float]]:
        q_np = q_lift.detach().cpu().numpy().astype(np.float64)
        left_start, _left_normal, _left_finger = _palm_feature(kin, dof_names, q_np, LEFT_EE_LINK)
        right_start, _right_normal, _right_finger = _palm_feature(kin, dof_names, q_np, RIGHT_EE_LINK)
        if left_target is None:
            left_goal = left_start + np.array([0.12, 0.0, 0.0], dtype=np.float64)
        else:
            left_goal = np.asarray(left_target, dtype=np.float64)
        if right_target is None:
            right_goal = right_start + np.array([0.12, 0.0, 0.0], dtype=np.float64)
        else:
            right_goal = np.asarray(right_target, dtype=np.float64)

        q = q_lift.clone()
        arm_targets: list[torch.Tensor] = []
        left_report = IkReport(pos_error=0.0)
        right_report = IkReport(pos_error=0.0)
        for i in range(1, waypoints + 1):
            alpha = _smoothstep(i / float(max(1, waypoints)))
            left_pos = left_start * (1.0 - alpha) + left_goal * alpha
            right_pos = right_start * (1.0 - alpha) + right_goal * alpha
            q, left_report = solve_arm_palm_target(
                kin,
                dof_names,
                lower,
                upper,
                q,
                "left",
                left_pos,
                iterations_per_waypoint,
            )
            q, right_report = solve_arm_palm_target(
                kin,
                dof_names,
                lower,
                upper,
                q,
                "right",
                right_pos,
                iterations_per_waypoint,
            )
            arm_targets.append(q.clone())
        return (
            arm_targets,
            left_report,
            right_report,
            tuple(float(v) for v in left_goal),
            tuple(float(v) for v in right_goal),
        )


    def _world_to_root(point: tuple[float, float, float], root: RootPose) -> tuple[float, float, float]:
        dx = point[0] - root.x
        dy = point[1] - root.y
        c = math.cos(-root.yaw)
        s = math.sin(-root.yaw)
        return (
            c * dx - s * dy,
            s * dx + c * dy,
            point[2] - root.z,
        )


    def _root_to_world(root: RootPose, local: tuple[float, float, float]) -> tuple[float, float, float]:
        c = math.cos(root.yaw)
        s = math.sin(root.yaw)
        return (
            root.x + c * local[0] - s * local[1],
            root.y + s * local[0] + c * local[1],
            root.z + local[2],
        )


    def _stack_layer_sequence(sequence: int) -> int:
        return ((int(sequence) - 1) % (GRID_X * GRID_Y)) + 1


    def _stack_cell_layer_sequence(cell: StackCell) -> int:
        return _stack_layer_sequence(cell.sequence)


    def _stack_cell_from_order(layer: int, order_index: int, ix: int, iy: int) -> StackCell:
        return StackCell(int(layer) * GRID_X * GRID_Y + int(order_index), int(layer), int(ix), int(iy))


    def _stack_third_fourth_order() -> tuple[StackCell, ...]:
        """Order layers 3/4 to keep the fourth-layer center clear of third-layer arms."""

        cells: list[StackCell] = []
        center_ix, center_iy = STACK_CROSS_ORDER[0]
        cells.append(_stack_cell_from_order(2, 1, center_ix, center_iy))
        cells.append(_stack_cell_from_order(3, 1, center_ix, center_iy))
        for order_index, (ix, iy) in enumerate(STACK_CROSS_ORDER[1:], start=2):
            cells.append(_stack_cell_from_order(2, order_index, ix, iy))
        for order_index, (ix, iy) in enumerate(STACK_CROSS_ORDER[1:], start=2):
            cells.append(_stack_cell_from_order(3, order_index, ix, iy))
        for order_index, (ix, iy) in enumerate(STACK_REST_ORDER, start=len(STACK_CROSS_ORDER) + 1):
            cells.append(_stack_cell_from_order(2, order_index, ix, iy))
        for order_index, (ix, iy) in enumerate(STACK_REST_ORDER, start=len(STACK_CROSS_ORDER) + 1):
            cells.append(_stack_cell_from_order(3, order_index, ix, iy))
        return tuple(cells)


    def _stack_cell_target(cell: StackCell) -> tuple[float, float, float]:
        sx, sy, sz = STACK_BOX_SIZE
        local_x = (cell.ix + 0.5) * sx
        local_y = (cell.iy + 0.5) * sy
        local_z = (cell.layer + 0.5) * sz
        if cell.layer == 3 and _stack_cell_layer_sequence(cell) == 1:
            local_x += FOURTH_LAYER_CENTER_TARGET_FORWARD_OFFSET
        return (
            PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 + local_x,
            PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 + local_y,
            PALLET_SURFACE_Z + local_z,
        )


    def _stack_box_size_for_cell(cell: StackCell) -> tuple[float, float, float]:
        if _stack_cell_layer_sequence(cell) in FIRST_LAYER_Y_SWAPPED_BOXES:
            return (STACK_BOX_SIZE[1], STACK_BOX_SIZE[0], STACK_BOX_SIZE[2])
        return STACK_BOX_SIZE


    def _uses_stack_y_row_strategy(cell: StackCell) -> bool:
        return _stack_cell_layer_sequence(cell) in FIRST_LAYER_Y_ROW_BOXES


    def _uses_stack_inner_edge_clearance(cell: StackCell) -> bool:
        return _stack_cell_layer_sequence(cell) in FIRST_LAYER_INNER_EDGE_PUSH_BOXES


    def _uses_stack_cross_corner_inner_hand_hold(cell: StackCell) -> bool:
        return _stack_cell_layer_sequence(cell) in STACK_CROSS_CORNER_BOXES


    def _uses_third_layer_cross_corner(cell: StackCell) -> bool:
        return cell.layer == 2 and _stack_cell_layer_sequence(cell) in STACK_CROSS_CORNER_BOXES


    def _uses_third_layer_cross_extension(cell: StackCell) -> bool:
        return cell.layer == 2 and _stack_cell_layer_sequence(cell) in STACK_CROSS_EXTENSION_BOXES


    def _uses_third_layer_outer_corner(cell: StackCell) -> bool:
        return cell.layer == 2 and _stack_cell_layer_sequence(cell) in STACK_OUTER_CORNER_BOXES


    def _uses_fourth_layer_cross_corner(cell: StackCell) -> bool:
        return cell.layer == 3 and _stack_cell_layer_sequence(cell) in STACK_CROSS_CORNER_BOXES


    def _uses_fourth_layer_cross_extension(cell: StackCell) -> bool:
        return cell.layer == 3 and _stack_cell_layer_sequence(cell) in STACK_CROSS_EXTENSION_BOXES


    def _uses_fourth_layer_outer_corner(cell: StackCell) -> bool:
        return cell.layer == 3 and _stack_cell_layer_sequence(cell) in STACK_OUTER_CORNER_BOXES


    def _uses_stack_inner_hand_hold_after_release(cell: StackCell) -> bool:
        return (
            _uses_stack_cross_corner_inner_hand_hold(cell)
            or _uses_third_layer_outer_corner(cell)
            or _uses_fourth_layer_outer_corner(cell)
        )


    def _stack_inner_hand_side_for_cell(cell: StackCell) -> str | None:
        side = _stack_direct_side_for_cell(cell)
        if cell.ix == 0:
            return "right" if side == "-Y" else "left"
        if cell.ix == GRID_X - 1:
            return "left" if side == "-Y" else "right"
        return None


    def _stack_outer_hand_side_for_cell(cell: StackCell) -> str | None:
        inner_side = _stack_inner_hand_side_for_cell(cell)
        if inner_side == "left":
            return "right"
        if inner_side == "right":
            return "left"
        return None


    def _stack_direct_side_for_cell(cell: StackCell) -> str:
        if _uses_stack_y_row_strategy(cell):
            if cell.iy <= 1:
                return "-Y"
            if cell.iy >= GRID_Y - 2:
                return "+Y"
        if cell.ix == 0:
            return "-X"
        if cell.ix == GRID_X - 1:
            return "+X"
        if cell.iy <= 1:
            return "-Y"
        if cell.iy >= GRID_Y - 2:
            return "+Y"
        return "-X"


    def _stack_stand_off_for_cell(cell: StackCell, base_stand_off: float) -> float:
        layer_sequence = _stack_cell_layer_sequence(cell)
        if layer_sequence == 1:
            if cell.layer == 3:
                return max(0.0, float(base_stand_off) - FOURTH_LAYER_CENTER_STAND_OFF_REDUCTION)
            return float(base_stand_off)
        if layer_sequence in {2, 3, 4, 5}:
            return max(float(base_stand_off), STACK_CROSS_NEIGHBOR_STAND_OFF)
        if cell.layer == 2 and layer_sequence in {6, 7, 8, 9}:
            return max(float(base_stand_off), THIRD_LAYER_CROSS_CORNER_STAND_OFF)
        if _uses_third_layer_cross_extension(cell):
            return max(float(base_stand_off), THIRD_LAYER_CROSS_EXTENSION_STAND_OFF)
        if _uses_third_layer_outer_corner(cell):
            return max(float(base_stand_off), THIRD_LAYER_OUTER_CORNER_STAND_OFF)
        if _uses_fourth_layer_cross_corner(cell):
            return max(float(base_stand_off), FOURTH_LAYER_CROSS_CORNER_STAND_OFF - FOURTH_LAYER_CORNER_STAND_OFF_REDUCTION)
        if _uses_fourth_layer_cross_extension(cell):
            return max(float(base_stand_off), FOURTH_LAYER_CROSS_EXTENSION_STAND_OFF)
        if _uses_fourth_layer_outer_corner(cell):
            return max(float(base_stand_off), FOURTH_LAYER_OUTER_CORNER_STAND_OFF - FOURTH_LAYER_CORNER_STAND_OFF_REDUCTION)
        if _uses_stack_y_row_strategy(cell) and cell.iy in {0, GRID_Y - 1}:
            return max(float(base_stand_off), STACK60_DIRECT_Y_EXTENSION_STAND_OFF)
        if _stack_direct_side_for_cell(cell) in {"-Y", "+Y"}:
            return max(float(base_stand_off), STACK60_DIRECT_Y_SIDE_STAND_OFF)
        return max(float(base_stand_off), STACK60_DIRECT_OUTER_STAND_OFF)


    def _stack_y_release_clearance_for_cell(cell: StackCell) -> float:
        if _uses_fourth_layer_cross_corner(cell):
            return FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE + STACK_CROSS_CENTER_RELEASE_CLEARANCE + FOURTH_LAYER_CORNER_FRONT_CLEARANCE_EXTRA
        if _uses_fourth_layer_outer_corner(cell):
            return THIRD_LAYER_CROSS_CORNER_RELEASE_CLEARANCE + FOURTH_LAYER_CORNER_FRONT_CLEARANCE_EXTRA
        if (
            _uses_third_layer_cross_corner(cell)
            or _uses_third_layer_cross_extension(cell)
            or _uses_third_layer_outer_corner(cell)
            or _uses_fourth_layer_cross_extension(cell)
        ):
            return THIRD_LAYER_CROSS_CORNER_RELEASE_CLEARANCE
        if _uses_stack_inner_edge_clearance(cell):
            return FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE
        if cell.iy in {0, GRID_Y - 1}:
            return FIRST_LAYER_Y_EXTENSION_RELEASE_CLEARANCE
        return FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE


    def _stack_inner_hand_x_clearance_for_cell(cell: StackCell) -> float:
        if not _uses_stack_y_row_strategy(cell):
            return 0.0
        if (
            _uses_third_layer_cross_corner(cell)
            or _uses_third_layer_outer_corner(cell)
            or _uses_fourth_layer_cross_corner(cell)
            or _uses_fourth_layer_outer_corner(cell)
        ):
            hand_clearance = THIRD_LAYER_CROSS_CORNER_HAND_CLEARANCE
            if _uses_fourth_layer_cross_corner(cell) or _uses_fourth_layer_outer_corner(cell):
                hand_clearance += FOURTH_LAYER_CORNER_HAND_CLEARANCE_EXTRA
            if cell.ix == 0:
                return -hand_clearance
            if cell.ix == GRID_X - 1:
                return hand_clearance
        if cell.ix == 0:
            return -FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE if _uses_stack_inner_edge_clearance(cell) else -FIRST_LAYER_INNER_HAND_X_CLEARANCE
        if cell.ix == GRID_X - 1:
            return FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE if _uses_stack_inner_edge_clearance(cell) else FIRST_LAYER_INNER_HAND_X_CLEARANCE
        return 0.0


    def _stack_cross_center_release_offset(cell: StackCell) -> tuple[float, float]:
        layer_sequence = _stack_cell_layer_sequence(cell)
        clearance = STACK_CROSS_CENTER_RELEASE_CLEARANCE
        if cell.layer == 2 and layer_sequence in {2, 3}:
            clearance += THIRD_LAYER_X_NEIGHBOR_CENTER_RELEASE_EXTRA
        if cell.layer == 3 and layer_sequence in {2, 3}:
            clearance += FOURTH_LAYER_X_NEIGHBOR_CENTER_RELEASE_EXTRA
        if layer_sequence == 2:
            return (-clearance, 0.0)
        if layer_sequence == 3:
            return (clearance, 0.0)
        if layer_sequence == 4:
            return (0.0, -clearance)
        if layer_sequence == 5:
            return (0.0, clearance)
        return (0.0, 0.0)


    def _stack_release_center_for_cell(cell: StackCell, target_center: tuple[float, float, float]) -> tuple[float, float, float]:
        cross_dx, cross_dy = _stack_cross_center_release_offset(cell)
        target_center = (target_center[0] + cross_dx, target_center[1] + cross_dy, target_center[2])
        if _stack_cell_layer_sequence(cell) in {2, 3}:
            side = _stack_direct_side_for_cell(cell)
            if side == "-X":
                return (target_center[0] - FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE, target_center[1], target_center[2])
            if side == "+X":
                return (target_center[0] + FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE, target_center[1], target_center[2])
        if _uses_stack_y_row_strategy(cell):
            side = _stack_direct_side_for_cell(cell)
            y_clearance = _stack_y_release_clearance_for_cell(cell)
            x_clearance = _stack_inner_hand_x_clearance_for_cell(cell)
            if side == "-Y":
                return (target_center[0] + x_clearance, target_center[1] - y_clearance, target_center[2])
            if side == "+Y":
                return (target_center[0] + x_clearance, target_center[1] + y_clearance, target_center[2])
        return target_center


    def _stack_stance_center_for_cell(cell: StackCell, target_center: tuple[float, float, float]) -> tuple[float, float, float]:
        if _uses_third_layer_cross_corner(cell) or _uses_fourth_layer_cross_corner(cell):
            return _stack_release_center_for_cell(cell, target_center)
        return target_center


    def _parked_box_pose(index: int) -> tuple[float, float, float]:
        return (BOX_PARK_X - BOX_PARK_SPACING * int(index), -1.2, SOURCE_BOX_POSE[2])


    def _stack_center_target(target_layer: int) -> tuple[float, float, float]:
        return (
            PALLET_CENTER[0],
            PALLET_CENTER[1],
            PALLET_SURFACE_Z + (float(target_layer) + 0.5) * STACK_BOX_SIZE[2],
        )


    def _stack_proxy_size_for_target_layer(target_layer: int) -> tuple[float, float, float]:
        proxy_layers = max(0, min(int(target_layer), GRID_Z))
        return (STACK_SIZE[0], STACK_SIZE[1], STACK_BOX_SIZE[2] * proxy_layers)


    def _stack_sequence_for_center_target(target_layer: int) -> int:
        return int(target_layer) * GRID_X * GRID_Y + 1


    def _stack_final_root(target_center: tuple[float, float, float], stand_off: float) -> RootPose:
        final_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - float(stand_off)
        final_y = target_center[1]
        yaw = math.atan2(target_center[1] - final_y, target_center[0] - final_x)
        return RootPose(final_x, final_y, 0.0, yaw)


    def _stack_direct_root_pose(cell: StackCell, target_center: tuple[float, float, float], stand_off: float) -> tuple[RootPose, RootPose]:
        side = _stack_direct_side_for_cell(cell)
        if side == "-X":
            final_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - float(stand_off)
            final_y = target_center[1]
            normal = (-1.0, 0.0)
        elif side == "+X":
            final_x = PALLET_CENTER[0] + STACK_SIZE[0] * 0.5 + float(stand_off)
            final_y = target_center[1]
            normal = (1.0, 0.0)
        elif side == "-Y":
            final_x = target_center[0]
            final_y = PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 - float(stand_off)
            normal = (0.0, -1.0)
        elif side == "+Y":
            final_x = target_center[0]
            final_y = PALLET_CENTER[1] + STACK_SIZE[1] * 0.5 + float(stand_off)
            normal = (0.0, 1.0)
        else:
            raise ValueError(f"Unsupported stack side: {side}")
        yaw = math.atan2(target_center[1] - final_y, target_center[0] - final_x)
        final_root = RootPose(final_x, final_y, 0.0, yaw)
        tmp_root = RootPose(final_x + normal[0] * DIRECT_TMP_RETREAT, final_y + normal[1] * DIRECT_TMP_RETREAT, 0.0, yaw)
        return final_root, tmp_root


    def _stack_root_route(final_root: RootPose) -> list[tuple[str, RootPose]]:
        start = RootPose(0.0, 0.0, 0.0, 0.0)
        dx = start.x - TABLE_POSE[0]
        dy = start.y - TABLE_POSE[1]
        length = max(math.hypot(dx, dy), 1e-6)
        table_retreat = RootPose(start.x + dx / length * 0.90, start.y + dy / length * 0.90, 0.0, 0.0)
        safe_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - SAFE_CORRIDOR_MARGIN
        safe_y = PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 - SAFE_CORRIDOR_MARGIN
        tmp_root = RootPose(final_root.x - DIRECT_TMP_RETREAT, final_root.y, 0.0, final_root.yaw)
        keypoints = (
            ("table_start", start),
            ("table_retreat", table_retreat),
            ("safe_axis_1", RootPose(table_retreat.x, safe_y, 0.0, 0.0)),
            ("safe_corner", RootPose(safe_x, safe_y, 0.0, 0.0)),
            ("tmp_axis", RootPose(tmp_root.x, safe_y, 0.0, math.atan2(tmp_root.y - safe_y, tmp_root.x - safe_x))),
            ("tmp_low", tmp_root),
            ("target", final_root),
        )
        route: list[tuple[str, RootPose]] = []
        for (_start_label, start_pose), (goal_label, goal_pose) in zip(keypoints, keypoints[1:]):
            for pose in sample_root_trajectory(start_pose, goal_pose):  # type: ignore[arg-type]
                route.append((goal_label, RootPose(pose.x, pose.y, pose.z, pose.yaw)))
        route.extend((keypoints[-1][0], keypoints[-1][1]) for _ in range(30))
        return route


    def _stack_root_route_to(final_root: RootPose, tmp_root: RootPose, side: str) -> list[tuple[str, RootPose]]:
        start = RootPose(0.0, 0.0, 0.0, 0.0)
        dx = start.x - TABLE_POSE[0]
        dy = start.y - TABLE_POSE[1]
        length = max(math.hypot(dx, dy), 1e-6)
        table_retreat = RootPose(start.x + dx / length * 0.90, start.y + dy / length * 0.90, 0.0, 0.0)
        safe_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - SAFE_CORRIDOR_MARGIN
        safe_y = PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 - SAFE_CORRIDOR_MARGIN
        if side == "+Y":
            upper_safe_y = PALLET_CENTER[1] + STACK_SIZE[1] * 0.5 + SAFE_CORRIDOR_MARGIN
            keypoints = (
                ("table_start", start),
                ("table_retreat", table_retreat),
                ("safe_axis_1", RootPose(table_retreat.x, safe_y, 0.0, 0.0)),
                ("safe_corner", RootPose(safe_x, safe_y, 0.0, 0.0)),
                ("safe_plus_y", RootPose(safe_x, upper_safe_y, 0.0, math.pi * 0.5)),
                ("tmp_axis", RootPose(tmp_root.x, upper_safe_y, 0.0, math.atan2(tmp_root.y - upper_safe_y, tmp_root.x - safe_x))),
                ("tmp_low", tmp_root),
                ("target", final_root),
            )
        else:
            keypoints = (
                ("table_start", start),
                ("table_retreat", table_retreat),
                ("safe_axis_1", RootPose(table_retreat.x, safe_y, 0.0, 0.0)),
                ("safe_corner", RootPose(safe_x, safe_y, 0.0, 0.0)),
                ("tmp_axis", RootPose(tmp_root.x, safe_y, 0.0, math.atan2(tmp_root.y - safe_y, tmp_root.x - safe_x))),
                ("tmp_low", tmp_root),
                ("target", final_root),
            )
        route: list[tuple[str, RootPose]] = []
        for (_start_label, start_pose), (goal_label, goal_pose) in zip(keypoints, keypoints[1:]):
            for pose in sample_root_trajectory(start_pose, goal_pose):  # type: ignore[arg-type]
                route.append((goal_label, RootPose(pose.x, pose.y, pose.z, pose.yaw)))
        route.extend((keypoints[-1][0], keypoints[-1][1]) for _ in range(30))
        return route


    def _stack_return_route_from(route: list[tuple[str, RootPose]]) -> list[tuple[str, RootPose]]:
        if not route:
            return []
        home = RootPose(0.0, 0.0, 0.0, 0.0)
        frames = [("return_to_source", pose) for _label, pose in reversed(route)]
        reversed_poses = [pose for _label, pose in reversed(route)]
        for pose in sample_root_trajectory(reversed_poses[-1], home):  # type: ignore[arg-type]
            frames.append(("return_home", RootPose(pose.x, pose.y, pose.z, pose.yaw)))
        frames.extend(("return_hold", home) for _ in range(STACK_RETURN_HOLD_FRAMES))
        return frames


    def _box_grip_contact_z(box_size: tuple[float, float, float] = STACK_BOX_SIZE) -> float:
        return -box_size[2] * 0.5 + THIRD_LAYER_CENTER_GRIP_HEIGHT_FROM_BOTTOM


    def _box_grip_palm_targets_local(
        box_center_local: tuple[float, float, float],
        side_gap: float,
        contact_z: float,
        box_size: tuple[float, float, float] = STACK_BOX_SIZE,
        contact_forward: float = STACK_GRASP_CONTACT_X_OFFSET,
    ) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, cz = box_center_local
        half_y = box_size[1] * 0.5
        y_offset = half_y + float(side_gap)
        left = np.array(
            [cx + contact_forward, cy + y_offset, cz + contact_z],
            dtype=np.float64,
        )
        right = np.array(
            [cx + contact_forward, cy - y_offset, cz + contact_z],
            dtype=np.float64,
        )
        return left, right


    def solve_box_grip(
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        root: RootPose,
        box_center_world: tuple[float, float, float],
        side_gap: float,
        contact_z: float,
        iterations: int,
        box_size: tuple[float, float, float] = STACK_BOX_SIZE,
        contact_forward: float = STACK_GRASP_CONTACT_X_OFFSET,
    ) -> tuple[torch.Tensor, IkReport, IkReport]:
        box_center_local = _world_to_root(box_center_world, root)
        left_target, right_target = _box_grip_palm_targets_local(
            box_center_local,
            side_gap,
            contact_z,
            box_size=box_size,
            contact_forward=contact_forward,
        )
        q, left_report = solve_arm_palm_target(
            kin,
            dof_names,
            lower,
            upper,
            q_start,
            "left",
            left_target,
            iterations,
        )
        q, right_report = solve_arm_palm_target(
            kin,
            dof_names,
            lower,
            upper,
            q,
            "right",
            right_target,
            iterations,
        )
        return q, left_report, right_report


    def _append_box_grip_cartesian_segment(
        timeline: list[StackFrame],
        phase: str,
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        root: RootPose,
        box_start: tuple[float, float, float],
        box_goal: tuple[float, float, float],
        gap_start: float,
        gap_goal: float,
        contact_z: float,
        frames: int,
        attached: bool,
        iterations: int,
        box_size: tuple[float, float, float] = STACK_BOX_SIZE,
        contact_forward: float = STACK_GRASP_CONTACT_X_OFFSET,
        ik_stride: int = 1,
    ) -> tuple[torch.Tensor, list[IkReport]]:
        q = q_start.clone()
        reports: list[IkReport] = []
        frame_count = max(1, int(frames))
        prev_index = 0
        prev_q = q_start.clone()
        for key_index in _segment_key_frames(frame_count, ik_stride):
            key_alpha = _smoothstep(key_index / float(frame_count))
            key_center = (
                box_start[0] * (1.0 - key_alpha) + box_goal[0] * key_alpha,
                box_start[1] * (1.0 - key_alpha) + box_goal[1] * key_alpha,
                box_start[2] * (1.0 - key_alpha) + box_goal[2] * key_alpha,
            )
            key_gap = gap_start * (1.0 - key_alpha) + gap_goal * key_alpha
            q, left_report, right_report = solve_box_grip(
                kin,
                dof_names,
                lower,
                upper,
                q,
                root,
                key_center,
                side_gap=key_gap,
                contact_z=contact_z,
                iterations=iterations,
                box_size=box_size,
                contact_forward=contact_forward,
            )
            reports.extend([left_report, right_report])
            span = max(1, key_index - prev_index)
            for frame_index in range(prev_index + 1, key_index + 1):
                alpha = _smoothstep(frame_index / float(frame_count))
                center = (
                    box_start[0] * (1.0 - alpha) + box_goal[0] * alpha,
                    box_start[1] * (1.0 - alpha) + box_goal[1] * alpha,
                    box_start[2] * (1.0 - alpha) + box_goal[2] * alpha,
                )
                local_alpha = _smoothstep((frame_index - prev_index) / float(span))
                timeline.append(StackFrame(phase, root, _lerp_tensor(prev_q, q, local_alpha), center, attached))
            prev_q = q.clone()
            prev_index = key_index
        return q, reports


    def _append_outer_hand_open_segment(
        timeline: list[StackFrame],
        phase: str,
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        q_start: torch.Tensor,
        root: RootPose,
        box_start: tuple[float, float, float],
        box_goal: tuple[float, float, float],
        side: str,
        gap_start: float,
        gap_goal: float,
        contact_z: float,
        frames: int,
        iterations: int,
        box_size: tuple[float, float, float] = STACK_BOX_SIZE,
        contact_forward: float = STACK_GRASP_CONTACT_X_OFFSET,
        ik_stride: int = 1,
    ) -> tuple[torch.Tensor, list[IkReport]]:
        q = q_start.clone()
        reports: list[IkReport] = []
        frame_count = max(1, int(frames))
        prev_index = 0
        prev_q = q_start.clone()
        for key_index in _segment_key_frames(frame_count, ik_stride):
            key_alpha = _smoothstep(key_index / float(frame_count))
            key_center = (
                box_start[0] * (1.0 - key_alpha) + box_goal[0] * key_alpha,
                box_start[1] * (1.0 - key_alpha) + box_goal[1] * key_alpha,
                box_start[2] * (1.0 - key_alpha) + box_goal[2] * key_alpha,
            )
            key_gap = gap_start * (1.0 - key_alpha) + gap_goal * key_alpha
            box_center_local = _world_to_root(key_center, root)
            left_target, right_target = _box_grip_palm_targets_local(
                box_center_local,
                key_gap,
                contact_z,
                box_size=box_size,
                contact_forward=contact_forward,
            )
            target = left_target if side == "left" else right_target
            q, report = solve_arm_palm_target(
                kin,
                dof_names,
                lower,
                upper,
                q,
                side,
                target,
                iterations,
            )
            reports.append(report)
            span = max(1, key_index - prev_index)
            for frame_index in range(prev_index + 1, key_index + 1):
                alpha = _smoothstep(frame_index / float(frame_count))
                center = (
                    box_start[0] * (1.0 - alpha) + box_goal[0] * alpha,
                    box_start[1] * (1.0 - alpha) + box_goal[1] * alpha,
                    box_start[2] * (1.0 - alpha) + box_goal[2] * alpha,
                )
                local_alpha = _smoothstep((frame_index - prev_index) / float(span))
                timeline.append(StackFrame(phase, root, _lerp_tensor(prev_q, q, local_alpha), center, False))
            prev_q = q.clone()
            prev_index = key_index
        return q, reports


    def _append_q_segment(
        timeline: list[StackFrame],
        phase: str,
        root: RootPose,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        box_start: tuple[float, float, float],
        box_goal: tuple[float, float, float],
        frames: int,
        attached: bool,
    ) -> None:
        for i in range(1, max(1, int(frames)) + 1):
            alpha = _smoothstep(i / float(max(1, int(frames))))
            center = (
                box_start[0] * (1.0 - alpha) + box_goal[0] * alpha,
                box_start[1] * (1.0 - alpha) + box_goal[1] * alpha,
                box_start[2] * (1.0 - alpha) + box_goal[2] * alpha,
            )
            timeline.append(StackFrame(phase, root, _lerp_tensor(q_start, q_goal, alpha), center, attached))


    def _append_hold(
        timeline: list[StackFrame],
        phase: str,
        root: RootPose,
        q: torch.Tensor,
        box_center: tuple[float, float, float],
        frames: int,
        attached: bool,
    ) -> None:
        for _ in range(max(0, int(frames))):
            timeline.append(StackFrame(phase, root, q.clone(), box_center, attached))


    def create_sim(gym):
        gymapi_module = _load_gymapi()
        sim_params = gymapi_module.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi_module.UP_AXIS_Z
        sim_params.gravity = gymapi_module.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2
        sim_params.physx.use_gpu = False
        sim_params.use_gpu_pipeline = False
        return gym.create_sim(COMPUTE_DEVICE_ID, GRAPHICS_DEVICE_ID, gymapi_module.SIM_PHYSX, sim_params)


    def _make_transform(x: float, y: float, z: float, yaw: float = 0.0) -> "gymapi.Transform":
        gymapi_module = _load_gymapi()
        pose = gymapi_module.Transform()
        pose.p = gymapi_module.Vec3(float(x), float(y), float(z))
        half = yaw * 0.5
        pose.r = gymapi_module.Quat(0.0, 0.0, math.sin(half), math.cos(half))
        return pose


    def _set_actor_root_pose(
        gym,
        sim,
        root_states: torch.Tensor,
        actor_indices: torch.Tensor,
        actor_index: int,
        position: tuple[float, float, float],
        yaw: float = 0.0,
        pitch: float = 0.0,
        linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        state = root_states[actor_index]
        state[0] = position[0]
        state[1] = position[1]
        state[2] = position[2]
        yaw_half = yaw * 0.5
        pitch_half = pitch * 0.5
        sy = math.sin(yaw_half)
        cy = math.cos(yaw_half)
        sp = math.sin(pitch_half)
        cp = math.cos(pitch_half)
        state[3] = -sy * sp
        state[4] = cy * sp
        state[5] = sy * cp
        state[6] = cy * cp
        state[7] = linear_velocity[0]
        state[8] = linear_velocity[1]
        state[9] = linear_velocity[2]
        state[10:13] = 0.0
        gymtorch_module = _load_gymtorch()
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch_module.unwrap_tensor(root_states),
            gymtorch_module.unwrap_tensor(actor_indices),
            int(actor_indices.numel()),
        )


    def load_robot_asset(gym, sim):
        gymapi_module = _load_gymapi()
        options = gymapi_module.AssetOptions()
        options.fix_base_link = False
        options.disable_gravity = True
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = gymapi_module.DOF_MODE_POS
        options.replace_cylinder_with_capsule = True
        options.mesh_normal_mode = gymapi_module.COMPUTE_PER_VERTEX
        return gym.load_asset(sim, str(ASSET_ROOT), ROBOT_ASSET_FILE, options)


    def _set_color(gym, env, actor, color: tuple[float, float, float]) -> None:
        gymapi_module = _load_gymapi()
        gym.set_rigid_body_color(
            env,
            actor,
            0,
            gymapi_module.MESH_VISUAL_AND_COLLISION,
            gymapi_module.Vec3(float(color[0]), float(color[1]), float(color[2])),
        )


    def _set_actor_collision_filter(gym, env, actor, collision_filter: int) -> None:
        props = gym.get_actor_rigid_shape_properties(env, actor)
        for prop in props:
            prop.filter = int(collision_filter)
        gym.set_actor_rigid_shape_properties(env, actor, props)


    def _set_shape_friction(gym, env, actor, friction: float) -> None:
        props = gym.get_actor_rigid_shape_properties(env, actor)
        for prop in props:
            prop.friction = float(friction)
            prop.rolling_friction = 0.02
            prop.torsion_friction = 0.02
            prop.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, actor, props)


    def _set_box_hand_collision_enabled(gym, env, robot, box, enabled: bool) -> None:
        box_props = gym.get_actor_rigid_shape_properties(env, box)
        for prop in box_props:
            prop.filter = 1 if enabled else 2
        gym.set_actor_rigid_shape_properties(env, box, box_props)

        robot_props = gym.get_actor_rigid_shape_properties(env, robot)
        body_names = gym.get_actor_rigid_body_names(env, robot)
        shape_ranges = gym.get_actor_rigid_body_shape_indices(env, robot)
        for body_name, shape_range in zip(body_names, shape_ranges):
            body_filter = 2 if "xhand" in body_name else 1
            for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
                robot_props[shape_idx].filter = body_filter
        gym.set_actor_rigid_shape_properties(env, robot, robot_props)


    def _quat_to_matrix(x: float, y: float, z: float, w: float) -> tuple[tuple[float, float, float], ...]:
        return (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        )


    def _palm_pose_from_sim(
        gym,
        env,
        actor,
        body_name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        names = gym.get_actor_rigid_body_names(env, actor)
        index = names.index(body_name)
        pose = states["pose"][index]
        p = pose["p"]
        q = pose["r"]
        rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        center = (
            float(p["x"]) + rot[0][0] * PALM_SURFACE_X + rot[0][2] * PALM_CENTER_Z,
            float(p["y"]) + rot[1][0] * PALM_SURFACE_X + rot[1][2] * PALM_CENTER_Z,
            float(p["z"]) + rot[2][0] * PALM_SURFACE_X + rot[2][2] * PALM_CENTER_Z,
        )
        palm_normal = (rot[0][0], rot[1][0], rot[2][0])
        finger_dir = (rot[0][2], rot[1][2], rot[2][2])
        return center, palm_normal, finger_dir


    def _palm_center_from_sim(gym, env, actor, body_name: str) -> tuple[float, float, float]:
        center, _palm_normal, _finger_dir = _palm_pose_from_sim(gym, env, actor, body_name)
        return center


    def _finger_dir_from_sim(gym, env, actor, body_name: str) -> tuple[float, float, float]:
        _center, _palm_normal, finger_dir = _palm_pose_from_sim(gym, env, actor, body_name)
        return finger_dir


    def _actor_center_from_sim(gym, env, actor) -> tuple[float, float, float]:
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        pose = states["pose"][0]["p"]
        return (float(pose["x"]), float(pose["y"]), float(pose["z"]))


    def _actor_yaw_pitch_from_sim(gym, env, actor) -> tuple[float, float]:
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        q = states["pose"][0]["r"]
        rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        pitch = math.asin(min(max(-rot[2][0], -1.0), 1.0))
        yaw = math.atan2(rot[1][0], rot[0][0])
        return yaw, pitch


    def _grasp_center_from_sim(gym, env, robot) -> tuple[float, float, float]:
        left = _palm_center_from_sim(gym, env, robot, LEFT_EE_LINK)
        right = _palm_center_from_sim(gym, env, robot, RIGHT_EE_LINK)
        return (
            0.5 * (left[0] + right[0]),
            0.5 * (left[1] + right[1]),
            0.5 * (left[2] + right[2]),
        )


    def _grasp_theta_from_sim(gym, env, robot, root_yaw: float) -> float:
        left = _finger_dir_from_sim(gym, env, robot, LEFT_EE_LINK)
        right = _finger_dir_from_sim(gym, env, robot, RIGHT_EE_LINK)
        world_x = 0.5 * (left[0] + right[0])
        world_y = 0.5 * (left[1] + right[1])
        z = 0.5 * (left[2] + right[2])
        c = math.cos(-root_yaw)
        s = math.sin(-root_yaw)
        x = c * world_x - s * world_y
        return math.atan2(-z, x)


    def _rotate_yaw_pitch(local: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float, float]:
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        x_pitch = cp * local[0] + sp * local[2]
        y_pitch = local[1]
        z_pitch = -sp * local[0] + cp * local[2]
        return (
            cy * x_pitch - sy * y_pitch,
            sy * x_pitch + cy * y_pitch,
            z_pitch,
        )


    def _inverse_rotate_yaw_pitch(world: tuple[float, float, float], yaw: float, pitch: float) -> tuple[float, float, float]:
        cy = math.cos(-yaw)
        sy = math.sin(-yaw)
        x_yaw = cy * world[0] - sy * world[1]
        y_yaw = sy * world[0] + cy * world[1]
        z_yaw = world[2]
        cp = math.cos(-pitch)
        sp = math.sin(-pitch)
        return (
            cp * x_yaw + sp * z_yaw,
            y_yaw,
            -sp * x_yaw + cp * z_yaw,
        )


    def _add_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


    def _sub_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


    def _vec_error(a: tuple[float, float, float], b: np.ndarray) -> float:
        return math.sqrt((a[0] - float(b[0])) ** 2 + (a[1] - float(b[1])) ** 2 + (a[2] - float(b[2])) ** 2)


    def _body_terms_from_sim(gym, env, actor) -> tuple[float, float, float]:
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        names = gym.get_actor_rigid_body_names(env, actor)
        index = names.index(BODY_LINK)
        pose = states["pose"][index]
        p = pose["p"]
        q = pose["r"]
        rot = _quat_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        body_pitch = math.atan2(rot[0][2], rot[0][0])
        return float(p["z"]), body_pitch, float(p["x"])


    def _set_robot_dof_state(gym, env, robot, q: torch.Tensor) -> None:
        states = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
        states["pos"] = q.detach().cpu().numpy()
        states["vel"].fill(0.0)
        gym.set_actor_dof_states(env, robot, states, gymapi.STATE_ALL)


    def _parse_point(value: list[float] | None) -> tuple[float, float, float] | None:
        if value is None:
            return None
        return (float(value[0]), float(value[1]), float(value[2]))


    def _render_every(args: argparse.Namespace) -> int:
        explicit = getattr(args, "render_every", None)
        if explicit is not None:
            return max(1, int(explicit))
        viewer_explicit = getattr(args, "viewer_render_every", None)
        if viewer_explicit is not None:
            return max(1, int(viewer_explicit))
        return 4 if getattr(args, "fast", False) else 1


    def _frame_stride(args: argparse.Namespace) -> int:
        explicit = getattr(args, "frame_stride", None)
        if explicit is not None:
            return max(1, int(explicit))
        return 1


    def _draw_viewer_frame(gym, sim, viewer, args: argparse.Namespace, frame_index: int) -> None:
        if viewer is None:
            return
        if frame_index % _render_every(args) != 0:
            return
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        if not (getattr(args, "fast", False) or getattr(args, "fast_viewer", False)):
            gym.sync_frame_time(sim)


    def _stack_target_layer_from_args(args: argparse.Namespace) -> int:
        return 3 if getattr(args, "stack_fourth_layer_demo", False) else 2


    def _stack_body_dz_for_target_layer(args: argparse.Namespace, target_layer: int) -> float:
        if args.stack_body_dz is not None:
            return float(args.stack_body_dz)
        del target_layer
        # With the three pitch joints and an upright body constraint, this robot can
        # raise body_yaw_link by about 9.6 cm from the neutral posture. Higher layers
        # should be reached by the arms after this upright waist lift, not by asking
        # the waist IK for an unreachable body height.
        return 0.095


    def _build_stack_box_timeline(
        args: argparse.Namespace,
        kin: UrdfKinematics,
        dof_names: list[str],
        lower: torch.Tensor,
        upper: torch.Tensor,
        cell: StackCell,
        return_home: bool = False,
    ) -> tuple[list[StackFrame], dict[str, object]]:
        home = RootPose(0.0, 0.0, 0.0, 0.0)
        source_center = SOURCE_BOX_POSE
        target_center = _stack_cell_target(cell)
        stance_center = _stack_stance_center_for_cell(cell, target_center)
        stand_off = _stack_stand_off_for_cell(cell, args.stack_stand_off)
        final_root, tmp_root = _stack_direct_root_pose(cell, stance_center, stand_off)
        side = _stack_direct_side_for_cell(cell)
        root_route = _stack_root_route_to(final_root, tmp_root, side)
        box_size = _stack_box_size_for_cell(cell)
        contact_z = _box_grip_contact_z(box_size)
        contact_forward = STACK_GRASP_CONTACT_X_OFFSET
        if cell.layer in {2, 3} and _stack_cell_layer_sequence(cell) in {2, 3}:
            contact_forward -= THIRD_LAYER_X_NEIGHBOR_GRIP_ROBOT_SIDE_OFFSET
            contact_z += THIRD_LAYER_X_NEIGHBOR_GRIP_HEIGHT_EXTRA
            if cell.layer == 3:
                contact_z -= FOURTH_LAYER_X_NEIGHBOR_GRIP_HEIGHT_REDUCTION
        plan_ik_stride = _stack_plan_ik_stride(args, return_home)

        q_seed = _seed_q(dof_names, lower, upper)
        q0, init_left_report, init_right_report, _init_left_pos, _init_right_pos = fix_initial_hand_orientation(
            kin,
            dof_names,
            lower,
            upper,
            q_seed,
            iterations=args.init_arm_iterations,
        )

        timeline: list[StackFrame] = [StackFrame("init", home, q0.clone(), source_center, False)]
        reports: list[IkReport] = [init_left_report, init_right_report]
        lift_reports: list[IkReport] = []

        ready_center = (source_center[0], source_center[1], source_center[2] + STACK_PICK_READY_Z)
        pick_targets = (
            ("pick_ready", ready_center, 0.16, 70),
            ("pick_pre_grasp", source_center, 0.10, 70),
            ("pick_touch", source_center, 0.04, 55),
            ("pick_clamp", source_center, 0.00, 55),
        )
        q = q0
        ik_center = source_center
        ik_gap = 0.20
        for label, center, gap, frames in pick_targets:
            q, segment_reports = _append_box_grip_cartesian_segment(
                timeline,
                label,
                kin,
                dof_names,
                lower,
                upper,
                q,
                home,
                ik_center,
                center,
                ik_gap,
                gap,
                contact_z=contact_z,
                contact_forward=contact_forward,
                frames=frames,
                attached=False,
                iterations=args.arm_iterations,
                box_size=box_size,
                ik_stride=plan_ik_stride,
            )
            reports.extend(segment_reports)
            ik_center = center
            ik_gap = gap

        _append_hold(timeline, "attach_hold", home, q, source_center, 20, attached=True)

        q_np = q.detach().cpu().numpy().astype(np.float64)
        initial_body_z, _initial_body_pitch, initial_body_x = _body_pose_terms(kin, dof_names, q_np)
        target_body_z = initial_body_z + _stack_body_dz_for_target_layer(args, cell.layer)
        carry_source_z = max(source_center[2] + STACK_PICK_LIFT_Z, target_center[2] + DIRECT_PREPLACE_CLEARANCE)
        if cell.layer == 3 and _stack_cell_layer_sequence(cell) == 1:
            carry_source_z -= FOURTH_LAYER_CENTER_CARRY_Z_REDUCTION
        carry_source_center = (
            source_center[0],
            source_center[1],
            carry_source_z,
        )
        lift_frames = max(1, int(args.stack_lift_frames))
        prev_lift_index = 0
        prev_lift_q = q.clone()
        lift_report = IkReport(pos_error=0.0)
        for key_index in _segment_key_frames(lift_frames, plan_ik_stride):
            key_alpha = _smoothstep(key_index / float(lift_frames))
            key_lift_center = (
                source_center[0] * (1.0 - key_alpha) + carry_source_center[0] * key_alpha,
                source_center[1] * (1.0 - key_alpha) + carry_source_center[1] * key_alpha,
                source_center[2] * (1.0 - key_alpha) + carry_source_center[2] * key_alpha,
            )
            body_z = initial_body_z * (1.0 - key_alpha) + target_body_z * key_alpha
            q, lift_report = solve_lift_body_z(
                kin,
                dof_names,
                lower,
                upper,
                q,
                body_z,
                iterations=args.lift_iterations,
                target_body_pitch=0.0,
                target_body_x=initial_body_x,
            )
            q, left_report, right_report = solve_box_grip(
                kin,
                dof_names,
                lower,
                upper,
                q,
                home,
                key_lift_center,
                side_gap=0.0,
                contact_z=contact_z,
                contact_forward=contact_forward,
                iterations=args.arm_iterations,
                box_size=box_size,
            )
            lift_reports.append(lift_report)
            reports.extend([left_report, right_report])
            span = max(1, key_index - prev_lift_index)
            for frame_index in range(prev_lift_index + 1, key_index + 1):
                alpha = _smoothstep(frame_index / float(lift_frames))
                lift_center = (
                    source_center[0] * (1.0 - alpha) + carry_source_center[0] * alpha,
                    source_center[1] * (1.0 - alpha) + carry_source_center[1] * alpha,
                    source_center[2] * (1.0 - alpha) + carry_source_center[2] * alpha,
                )
                local_alpha = _smoothstep((frame_index - prev_lift_index) / float(span))
                timeline.append(StackFrame("lift_box_safe", home, _lerp_tensor(prev_lift_q, q, local_alpha), lift_center, True))
            prev_lift_q = q.clone()
            prev_lift_index = key_index

        q_carry = q.clone()
        carry_local = _world_to_root(carry_source_center, home)
        for label, root in root_route:
            timeline.append(StackFrame(f"move:{label}", root, q_carry.clone(), _root_to_world(root, carry_local), True))

        place_start = _root_to_world(final_root, carry_local)
        release_target = _stack_release_center_for_cell(cell, target_center)
        preplace_center = (release_target[0], release_target[1], target_center[2] + DIRECT_PREPLACE_CLEARANCE)
        release_center = (release_target[0], release_target[1], release_target[2] + float(args.stack_release_height))
        place_segments = (
            ("place_move_over_target", place_start, preplace_center, int(args.stack_place_xy_frames)),
            ("place_descend_to_release", preplace_center, release_center, int(args.stack_place_descend_frames)),
        )
        for label, start, goal, frames in place_segments:
            frame_count = max(1, frames)
            prev_place_index = 0
            prev_place_q = q.clone()
            for key_index in _segment_key_frames(frame_count, plan_ik_stride):
                key_alpha = _smoothstep(key_index / float(frame_count))
                key_center = (
                    start[0] * (1.0 - key_alpha) + goal[0] * key_alpha,
                    start[1] * (1.0 - key_alpha) + goal[1] * key_alpha,
                    start[2] * (1.0 - key_alpha) + goal[2] * key_alpha,
                )
                q, left_report, right_report = solve_box_grip(
                    kin,
                    dof_names,
                    lower,
                    upper,
                    q,
                    final_root,
                    key_center,
                    side_gap=0.0,
                    contact_z=contact_z,
                    contact_forward=contact_forward,
                    iterations=args.arm_iterations,
                    box_size=box_size,
                )
                reports.extend([left_report, right_report])
                span = max(1, key_index - prev_place_index)
                for frame_index in range(prev_place_index + 1, key_index + 1):
                    alpha = _smoothstep(frame_index / float(frame_count))
                    center = (
                        start[0] * (1.0 - alpha) + goal[0] * alpha,
                        start[1] * (1.0 - alpha) + goal[1] * alpha,
                        start[2] * (1.0 - alpha) + goal[2] * alpha,
                    )
                    local_alpha = _smoothstep((frame_index - prev_place_index) / float(span))
                    timeline.append(StackFrame(label, final_root, _lerp_tensor(prev_place_q, q, local_alpha), center, True))
                prev_place_q = q.clone()
                prev_place_index = key_index

        _append_hold(timeline, "place_hold_before_release", final_root, q, release_center, STACK_PRE_RELEASE_HOLD_FRAMES, attached=True)
        _append_hold(timeline, "release", final_root, q, release_center, 1, attached=False)
        open_hand_center = (release_center[0], release_center[1], release_center[2] + STACK_OPEN_HAND_LIFT_Z)
        outer_side = _stack_outer_hand_side_for_cell(cell)
        if _uses_stack_inner_hand_hold_after_release(cell) and outer_side is not None:
            q_open, open_reports = _append_outer_hand_open_segment(
                timeline,
                "open_outer_hand",
                kin,
                dof_names,
                lower,
                upper,
                q,
                final_root,
                release_center,
                open_hand_center,
                outer_side,
                0.0,
                0.18,
                contact_z + 0.02,
                STACK_OPEN_HAND_FRAMES,
                iterations=args.arm_iterations,
                box_size=box_size,
                contact_forward=contact_forward,
                ik_stride=plan_ik_stride,
            )
        else:
            q_open, open_reports = _append_box_grip_cartesian_segment(
                timeline,
                "open_hands",
                kin,
                dof_names,
                lower,
                upper,
                q,
                final_root,
                release_center,
                open_hand_center,
                0.0,
                0.18,
                contact_z + 0.02,
                frames=STACK_OPEN_HAND_FRAMES,
                attached=False,
                iterations=args.arm_iterations,
                box_size=box_size,
                contact_forward=contact_forward,
                ik_stride=plan_ik_stride,
            )
        reports.extend(open_reports)
        _append_hold(timeline, "settle", final_root, q_open, open_hand_center, args.hold_frames, attached=False)
        if return_home:
            for label, root in _stack_return_route_from(root_route):
                timeline.append(StackFrame(f"return:{label}", root, q_open.clone(), open_hand_center, False))
            for i in range(1, STACK_RETURN_RECOVER_FRAMES + 1):
                alpha = _smoothstep(i / float(STACK_RETURN_RECOVER_FRAMES))
                q_recover = _lerp_tensor(q_open, q0, alpha)
                timeline.append(StackFrame("return_recover_home", home, q_recover, open_hand_center, False))

        max_pos_error = max(report.pos_error for report in reports) if reports else 0.0
        max_palm_error = max(report.palm_error for report in reports) if reports else 0.0
        max_finger_error = max(report.finger_error for report in reports) if reports else 0.0
        max_body_z_error = max(report.pos_error for report in lift_reports) if lift_reports else 0.0
        metadata: dict[str, object] = {
            "source_center": source_center,
            "target_center": target_center,
            "stance_center": stance_center,
            "preplace_center": preplace_center,
            "release_center": release_center,
            "final_root": final_root,
            "target_layer": cell.layer,
            "target_sequence": cell.sequence,
            "target_label": cell.label,
            "box_size": box_size,
            "side": side,
            "stand_off": stand_off,
            "plan_ik_stride": plan_ik_stride,
            "proxy_size": _stack_proxy_size_for_target_layer(cell.layer),
            "contact_z": contact_z,
            "contact_forward": contact_forward,
            "max_pos_error": max_pos_error,
            "max_palm_error": max_palm_error,
            "max_finger_error": max_finger_error,
            "max_body_z_error": max_body_z_error,
            "target_body_z": target_body_z,
            "solved_body_z": lift_report.body_z,
            "body_z_error": lift_report.pos_error,
            "body_pitch": lift_report.body_pitch,
        }
        return timeline, metadata

    def _reset_stack_box_to_source(
        gym,
        sim,
        root_states: torch.Tensor,
        actor_indices: torch.Tensor,
        actor_index: int,
    ) -> None:
        _set_actor_root_pose(gym, sim, root_states, actor_indices, actor_index, SOURCE_BOX_POSE, 0.0, 0.0)


    def _run_stack_box_timeline(
        gym,
        sim,
        env,
        robot,
        box,
        root_states: torch.Tensor,
        robot_index: int,
        box_index: int,
        plan: StackBoxPlan,
        viewer,
        args: argparse.Namespace,
        global_frame_start: int,
        max_frames: int,
    ) -> tuple[int, bool]:
        robot_actor_indices = torch.tensor([robot_index], dtype=torch.int32)
        box_actor_indices = torch.tensor([box_index], dtype=torch.int32)
        _reset_stack_box_to_source(gym, sim, root_states, box_actor_indices, box_index)
        _set_box_hand_collision_enabled(gym, env, robot, box, enabled=True)

        released = False
        previous_attached = False
        attach_offset_local: tuple[float, float, float] | None = None
        attach_yaw_offset = 0.0
        attach_theta_offset = 0.0
        local_frame = 0
        while local_frame < len(plan.timeline):
            frame = plan.timeline[local_frame]
            frame_index = global_frame_start + local_frame
            if max_frames > 0 and frame_index >= max_frames:
                return frame_index, False
            gym.refresh_actor_root_state_tensor(sim)
            _set_actor_root_pose(
                gym,
                sim,
                root_states,
                robot_actor_indices,
                robot_index,
                (frame.root.x, frame.root.y, frame.root.z),
                frame.root.yaw,
            )
            if frame.attached:
                if not previous_attached:
                    actual_center = _actor_center_from_sim(gym, env, box)
                    actual_yaw, actual_pitch = _actor_yaw_pitch_from_sim(gym, env, box)
                    grasp_center = _grasp_center_from_sim(gym, env, robot)
                    grasp_theta = _grasp_theta_from_sim(gym, env, robot, frame.root.yaw)
                    attach_offset_local = _inverse_rotate_yaw_pitch(
                        _sub_vec(actual_center, grasp_center),
                        frame.root.yaw,
                        grasp_theta,
                    )
                    attach_yaw_offset = actual_yaw - frame.root.yaw
                    attach_theta_offset = actual_pitch - grasp_theta
                    _set_box_hand_collision_enabled(gym, env, robot, box, enabled=False)
                    print(
                        "waist_arm_stack_attach "
                        f"box={plan.cell.sequence} frame={frame_index} phase={frame.phase} "
                        f"offset=({attach_offset_local[0]:.3f},{attach_offset_local[1]:.3f},{attach_offset_local[2]:.3f})"
                    )
            elif not released and (frame.phase == "release" or previous_attached):
                released = True
                actual_release = _actor_center_from_sim(gym, env, box)
                print(
                    "waist_arm_stack_release "
                    f"box={plan.cell.sequence} frame={frame_index} phase={frame.phase} "
                    f"actual=({actual_release[0]:.3f},{actual_release[1]:.3f},{actual_release[2]:.3f})"
                )

            _set_robot_dof_state(gym, env, robot, frame.q)
            gym.set_actor_dof_position_targets(env, robot, frame.q.numpy())
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if frame.attached and attach_offset_local is not None:
                gym.refresh_actor_root_state_tensor(sim)
                grasp_center = _grasp_center_from_sim(gym, env, robot)
                grasp_theta = _grasp_theta_from_sim(gym, env, robot, frame.root.yaw)
                attached_center = _add_vec(
                    grasp_center,
                    _rotate_yaw_pitch(attach_offset_local, frame.root.yaw, grasp_theta),
                )
                _set_actor_root_pose(
                    gym,
                    sim,
                    root_states,
                    box_actor_indices,
                    box_index,
                    attached_center,
                    frame.root.yaw + attach_yaw_offset,
                    grasp_theta + attach_theta_offset,
                )
            if plan.cell.sequence >= getattr(args, "viewer_start_box", 1):
                _draw_viewer_frame(gym, sim, viewer, args, frame_index)

            if local_frame % 180 == 0 or local_frame == len(plan.timeline) - 1:
                actual_box = _actor_center_from_sim(gym, env, box)
                body_z, body_pitch, _body_x = _body_terms_from_sim(gym, env, robot)
                print(
                    f"frame={frame_index} box={plan.cell.sequence} phase={frame.phase} attached={frame.attached} "
                    f"planned=({frame.box_center[0]:.2f},{frame.box_center[1]:.2f},{frame.box_center[2]:.2f}) "
                    f"actual=({actual_box[0]:.2f},{actual_box[1]:.2f},{actual_box[2]:.2f}) "
                    f"body_z={body_z:.2f} pitch={math.degrees(body_pitch):.2f}deg"
                )

            previous_attached = frame.attached
            local_frame += _frame_stride(args)
        if released:
            _set_actor_collision_filter(gym, env, box, 0)
        return global_frame_start + len(plan.timeline), True

    return SimpleNamespace(**{name: value for name, value in locals().items() if not name.startswith("__")})


waist_stack = _build_waist_stack_namespace()


STACK_SIZE = (1.0, 1.0, 1.6)
GRID_X = 3
GRID_Y = 5
GRID_Z = 4
STACK_BOX_SIZE = (STACK_SIZE[0] / GRID_X, STACK_SIZE[1] / GRID_Y, STACK_SIZE[2] / GRID_Z)
FIRST_LAYER_PROXY_SIZE = (STACK_SIZE[0], STACK_SIZE[1], STACK_BOX_SIZE[2])
SECOND_LAYER_PROXY_SIZE = (STACK_SIZE[0], STACK_SIZE[1], STACK_BOX_SIZE[2] * 2.0)
STACK_BOX_DENSITY = BOX_MASS / (STACK_BOX_SIZE[0] * STACK_BOX_SIZE[1] * STACK_BOX_SIZE[2])
THIRD_LAYER_CENTER_SEQUENCE = GRID_X * GRID_Y * 2 + 1
THIRD_LAYER_CENTER_CONTACT_FRICTION = 1.2
THIRD_LAYER_CENTER_ROLLING_FRICTION = 0.01
THIRD_LAYER_CENTER_TORSION_FRICTION = 0.01
FOURTH_LAYER_CENTER_SEQUENCE = GRID_X * GRID_Y * 3 + 1
FOURTH_LAYER_CENTER_DENSITY_MULTIPLIER = 1.0
FOURTH_LAYER_CENTER_CONTACT_FRICTION = 1.2
FOURTH_LAYER_CENTER_ROLLING_FRICTION = 0.01
FOURTH_LAYER_CENTER_TORSION_FRICTION = 0.01
PALLET_CENTER = pallet_center_near_table()
SOURCE_BOX_POSE = (
    TABLE_POSE[0] - 0.10,
    TABLE_POSE[1],
    TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + STACK_BOX_SIZE[2] * 0.5,
)

DIRECT_STAND_OFF = 0.42
DIRECT_OUTER_STAND_OFF = 0.80
SECOND_LAYER_X_NEIGHBOR_STAND_OFF = 0.58
SECOND_LAYER_INNER_CORNER_PUSH_STAND_OFF_EXTRA = 0.150
DIRECT_Y_SIDE_STAND_OFF = 0.68
DIRECT_Y_EXTENSION_STAND_OFF = 0.90
SECOND_LAYER_Y_SIDE_STAND_OFF = DIRECT_Y_SIDE_STAND_OFF - 0.30
SECOND_LAYER_Y_EXTENSION_STAND_OFF = DIRECT_Y_EXTENSION_STAND_OFF - 0.20
SECOND_LAYER_Y_EXTENSION_STAND_OFF_REDUCTION = 0.000
DIRECT_TMP_RETREAT = 0.85
DIRECT_PREPLACE_CLEARANCE = 0.18
THIRD_LAYER_CENTER_GRIP_HEIGHT_FROM_BOTTOM = 0.10
RETURN_HOLD_FRAMES = 30
RETURN_RECOVER_FRAMES = 180
PLACE_MOVE_XY_FRAMES = 180
PLACE_DESCEND_FRAMES = 180
FIRST_LAYER_INNER_EDGE_PUSH_FRAMES = 160
FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES = 100
FIRST_LAYER_INNER_EDGE_PUSH_SETTLE_FRAMES = 40
FIRST_LAYER_OUTER_HAND_PREP_FRAMES = 45
PICK_ERROR_LIMIT = 0.06
PLACE_ERROR_LIMIT = 0.08
BOX_PARK_X = -6.0
BOX_PARK_SPACING = 0.55
SIDE_PICK_STAND_OFF = 0.55
SIDE_SOURCE_YAW = math.pi * 0.5
X_GRIP_ATTACH_OFFSET = (0.132, 0.0, -0.038)
FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE = 0.015
FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE = 0.010
FIRST_LAYER_Y_EXTENSION_RELEASE_CLEARANCE = 0.020
FIRST_LAYER_Y_ROW_BOXES = set(range(4, 16))
FIRST_LAYER_Y_SWAPPED_BOXES = FIRST_LAYER_Y_ROW_BOXES
FIRST_LAYER_INNER_EDGE_PUSH_BOXES = {6, 7, 8, 9, 12, 13, 14, 15}
FIRST_LAYER_Y_EXTENSION_BOXES = {10, 11, 12, 13, 14, 15}
FIRST_LAYER_Y_SWAPPED_CONTACT_FORWARD = GRASP_CONTACT_X_OFFSET - 0.030
SECOND_LAYER_INNER_ROW_PICK_FINGERTIP_OFFSET = 0.060
FIRST_LAYER_INNER_HAND_X_CLEARANCE = 0.080
FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE = 0.030
SECOND_LAYER_DIRECT_EDGE_RELEASE_CLEARANCE = 0.040
SECOND_LAYER_INNER_ROW_Y_RELEASE_CLEARANCE = 0.040
SECOND_LAYER_INNER_CORNER_Y_RELEASE_CLEARANCE = 0.020
SECOND_LAYER_Y_EXTENSION_RELEASE_CLEARANCE = 0.000
SECOND_LAYER_OUTER_CORNER_Y_RELEASE_CLEARANCE = 0.010
SECOND_LAYER_OUTER_CORNER_SECOND_Y_RELEASE_EXTRA = 0.010
SECOND_LAYER_Y_CROSS_RELEASE_CLEARANCE = FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE - 0.020
SECOND_LAYER_INNER_ROW_X_HAND_CLEARANCE = 0.100
FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE = 0.100
FIRST_LAYER_INNER_HAND_LIFT_ABOVE_TOP = 0.180
FIRST_LAYER_INNER_HAND_LIFT_MIN_DELTA = 0.140
FIRST_LAYER_INNER_HAND_LIFT_IK_ITERATIONS = 120
FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE = 0.020
FIRST_LAYER_OUTER_HAND_BODY_RETRACT_DISTANCE = 0.020
SECOND_LAYER_OUTER_HAND_BODY_RETRACT_DISTANCE = 0.050
FIRST_LAYER_OUTER_HAND_PUSH_EXTRA_DISTANCE = 0.050
FIRST_LAYER_OUTER_HAND_PUSH_BELOW_COM = 0.050
SECOND_LAYER_OUTER_HAND_PUSH_BELOW_COM = PLACE_RELEASE_HEIGHT + 0.080
FIRST_LAYER_OUTER_HAND_PUSH_IK_ITERATIONS = 120
FIRST_LAYER_OUTER_PUSH_OPEN_ALPHA = 0.35
SECOND_LAYER_EDGE_CORNER_PUSH_HAND_FORWARD_OFFSET = 0.050
SECOND_LAYER_EDGE_CORNER_LOWER_FORWARD_EXTRA = 0.050
SECOND_LAYER_EDGE_CORNER_PUSH_DISTANCE_REDUCTION = 0.020
SECOND_LAYER_OUTER_CORNER_SECOND_PUSH_DISTANCE_REDUCTION = 0.020
WAIST_LAYER_CORNER_PUSH_DISTANCE_EXTRA = 0.130
WAIST_THIRD_LAYER_CORNER_PUSH_DISTANCE_EXTRA = 0.020
WAIST_FOURTH_LAYER_CORNER_PUSH_DISTANCE_REDUCTION = -0.020
WAIST_LAYER_INNER_HAND_HORIZONTAL_RETRACT = 0.200
WAIST_FOURTH_LAYER_OUTER_HAND_BACK_RETRACT_EXTRA = 0.050
WAIST_LAYER_CORNER_PUSH_Z_LOWER = 0.050
BOX_CONTACT_FRICTION = 1.2
BOX_CONTACT_ROLLING_FRICTION = 0.01
BOX_CONTACT_TORSION_FRICTION = 0.01
BOX_CONTACT_RESTITUTION = 0.0
PALLET_CONTACT_FRICTION = 1.0


@dataclass(frozen=True)
class StackCell:
    """One target box in the 3 x 5 x 4 stack."""

    sequence: int
    layer: int
    ix: int
    iy: int
    mode: str

    @property
    def label(self) -> str:
        return f"box_{self.sequence:02d}_L{self.layer + 1}_x{self.ix}_y{self.iy}_{self.mode}"


@dataclass(frozen=True)
class BoxPlan:
    cell: StackCell
    dof_names: list[str]
    pick_targets: torch.Tensor
    pick_box_centers: list[tuple[float, float, float]]
    box_size: tuple[float, float, float]
    attach_offset_local: tuple[float, float, float]
    runtime_attach_offset_local: tuple[float, float, float]
    move_approach_target: torch.Tensor
    move_lift_target: torch.Tensor
    move_approach_box_center_z: float
    move_approach_box_theta: float
    move_preplace_box_center_z: float
    move_preplace_box_theta: float
    place_targets: torch.Tensor
    release_target: torch.Tensor
    pre_pick_route: list[tuple[str, Pose]]
    pick_root_pose: Pose
    source_box_yaw: float
    root_route: list[tuple[str, Pose]]
    return_route: list[tuple[str, Pose]]
    place_pose_path_world: list[tuple[str, tuple[float, float, float], float]]
    place_handoff_center_world: tuple[float, float, float]
    runtime_handoff_center_world: tuple[float, float, float]
    target_center: tuple[float, float, float]
    grip_axis: str
    place_box_yaw_local: float
    box_world_yaw: float
    keep_box_world_yaw: bool
    pick_max_error: float
    place_max_error: float


@dataclass(frozen=True)
class PlacedBoxPose:
    sequence: int
    actor_index: int
    target_center: tuple[float, float, float]
    position: tuple[float, float, float]
    yaw: float
    pitch: float


def build_first_layer_order() -> tuple[StackCell, ...]:
    """Return the requested first-layer debug order.

    Steps:
    1. center;
    2. four cross boxes;
    3. inner row edge boxes;
    4. two cross-extension boxes;
    5. remaining corners.
    """

    direct_cells = [
        (1, 2),
        (0, 2),
        (2, 2),
        (1, 1),
        (1, 3),
    ]
    push_cells = [
        (0, 1),
        (2, 1),
        (0, 3),
        (2, 3),
        (1, 0),
        (1, 4),
        (0, 0),
        (2, 0),
        (0, 4),
        (2, 4),
    ]
    cells: list[StackCell] = []
    for ix, iy in direct_cells:
        cells.append(StackCell(len(cells) + 1, 0, ix, iy, "direct"))
    for ix, iy in push_cells:
        cells.append(StackCell(len(cells) + 1, 0, ix, iy, "offset_push"))
    return tuple(cells)


def build_stack_order() -> tuple[StackCell, ...]:
    first_layer = build_first_layer_order()
    cells: list[StackCell] = []
    for layer in range(GRID_Z):
        for base in first_layer:
            cells.append(StackCell(len(cells) + 1, layer, base.ix, base.iy, base.mode))
    return tuple(cells)


def cell_center_world(cell: StackCell) -> tuple[float, float, float]:
    sx, sy, sz = STACK_BOX_SIZE
    local_x = (cell.ix + 0.5) * sx
    local_y = (cell.iy + 0.5) * sy
    local_z = (cell.layer + 0.5) * sz
    return (
        PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 + local_x,
        PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 + local_y,
        PALLET_SURFACE_Z + local_z,
    )


def _box_size_for_cell(cell: StackCell) -> tuple[float, float, float]:
    return _box_size_for_sequence(cell.sequence)


def _layer_sequence(sequence: int) -> int:
    return ((sequence - 1) % (GRID_X * GRID_Y)) + 1


def _cell_layer_sequence(cell: StackCell) -> int:
    return _layer_sequence(cell.sequence)


def _uses_first_layer_y_row_strategy(cell: StackCell) -> bool:
    return _cell_layer_sequence(cell) in FIRST_LAYER_Y_ROW_BOXES


def _supports_direct_box_plan(cell: StackCell) -> bool:
    return cell.mode == "direct" or _uses_first_layer_y_row_strategy(cell)


def _uses_inner_edge_push_strategy(cell: StackCell) -> bool:
    if cell.layer == 1:
        return False
    return _cell_layer_sequence(cell) in FIRST_LAYER_INNER_EDGE_PUSH_BOXES


def _uses_second_layer_direct_edge_strategy(cell: StackCell) -> bool:
    return cell.layer == 1 and _cell_layer_sequence(cell) in FIRST_LAYER_INNER_EDGE_PUSH_BOXES


def _uses_second_layer_inner_row_clearance_strategy(cell: StackCell) -> bool:
    return cell.layer == 1 and _cell_layer_sequence(cell) in {6, 7, 8, 9, 12, 13, 14, 15}


def _uses_second_layer_edge_corner_push_strategy(cell: StackCell) -> bool:
    return cell.layer == 1 and _cell_layer_sequence(cell) in FIRST_LAYER_INNER_EDGE_PUSH_BOXES


def _uses_third_layer_center_strategy(cell: StackCell) -> bool:
    return cell.layer == 2 and _cell_layer_sequence(cell) == 1


def _rot_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _make_grip_keyframe(
    name: str,
    center: np.ndarray,
    theta: float,
    side_gap: float,
    contact_z: float,
    contact_forward: float,
    grip_axis: str,
    box_size: tuple[float, float, float],
    box_yaw_local: float = 0.0,
    fingertip_offset: float = 0.0,
):
    """Build a two-palm keyframe for either Y-face or X-face gripping."""

    rot = _rot_z(box_yaw_local) @ _rot_y(theta)
    if grip_axis == "y":
        half_y = box_size[1] * 0.5
        left_contact_local = np.array([contact_forward, half_y + PALM_BOX_CLEARANCE + side_gap, contact_z], dtype=np.float64)
        right_contact_local = np.array([contact_forward, -half_y - PALM_BOX_CLEARANCE - side_gap, contact_z], dtype=np.float64)
        left_palm = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right_palm = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fingers_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif grip_axis == "x":
        half_x = box_size[0] * 0.5
        left_contact_local = np.array([half_x + PALM_BOX_CLEARANCE + side_gap, contact_forward, contact_z], dtype=np.float64)
        right_contact_local = np.array([-half_x - PALM_BOX_CLEARANCE - side_gap, contact_forward, contact_z], dtype=np.float64)
        left_palm = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        right_palm = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fingers_forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported grip axis: {grip_axis}")

    from move.utils import CartesianKeyframe

    fingertip_shift = rot @ fingers_forward * float(fingertip_offset)
    return CartesianKeyframe(
        name,
        1,
        center + rot @ left_contact_local - fingertip_shift,
        center + rot @ right_contact_local - fingertip_shift,
        rot @ left_palm,
        rot @ right_palm,
        rot @ fingers_forward,
        rot @ fingers_forward,
        0.0,
        {},
    )


def _build_pick_keyframes(
    grip_axis: str,
    source_center_local: tuple[float, float, float],
    box_yaw_local: float,
    box_size: tuple[float, float, float],
    contact_forward: float,
    fingertip_offset: float = 0.0,
    contact_z_offset: float = 0.0,
) -> tuple[list[object], list[tuple[float, float, float]]]:
    box_center = np.asarray(source_center_local, dtype=np.float64)
    contact_z = box_size[2] * PICK_SIDE_CONTACT_Z_RATIO + contact_z_offset
    high_center = box_center + np.array([0.0, 0.0, PICK_READY_Z_OFFSET], dtype=np.float64)
    lift_center = box_center + np.array([0.0, 0.0, PICK_LIFT_HEIGHT], dtype=np.float64)
    compress_gap = -PALM_BOX_CLEARANCE
    frames = [
        _make_grip_keyframe("pick_ready_high", high_center, 0.0, APPROACH_GAP_READY, contact_z + 0.02, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_descend_level", box_center, 0.0, APPROACH_GAP_READY, contact_z + 0.02, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_pre_grasp", box_center, 0.0, APPROACH_GAP_PRE, contact_z + 0.01, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_straight_clamp", box_center, 0.0, PICK_TOUCH_GAP, contact_z, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_compress", box_center, 0.0, compress_gap, contact_z, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_compress_hold", box_center, 0.0, compress_gap, contact_z, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
        _make_grip_keyframe("pick_lift", lift_center, 0.0, compress_gap, contact_z, contact_forward, grip_axis, box_size, box_yaw_local, fingertip_offset),
    ]
    centers = [
        tuple(float(v) for v in high_center),
        tuple(float(v) for v in box_center),
        tuple(float(v) for v in box_center),
        tuple(float(v) for v in box_center),
        tuple(float(v) for v in box_center),
        tuple(float(v) for v in box_center),
        tuple(float(v) for v in lift_center),
    ]
    return frames, centers


def _direct_side_for_cell(cell: StackCell) -> str:
    if _uses_first_layer_y_row_strategy(cell):
        if cell.iy <= 1:
            return "-Y"
        if cell.iy >= GRID_Y - 2:
            return "+Y"
    if cell.ix == 0:
        return "-X"
    if cell.ix == GRID_X - 1:
        return "+X"
    if cell.iy <= 1:
        return "-Y"
    if cell.iy >= GRID_Y - 2:
        return "+Y"
    return "-X"


def _grip_axis_for_cell(cell: StackCell) -> str:
    if _uses_first_layer_y_row_strategy(cell):
        return "y"
    if _direct_side_for_cell(cell) in {"-Y", "+Y"}:
        return "x"
    return "y"


def _source_box_yaw_for_cell(cell: StackCell) -> float:
    side = _direct_side_for_cell(cell)
    if _uses_first_layer_y_row_strategy(cell):
        return 0.0
    if side in {"-Y", "+Y"}:
        return SIDE_SOURCE_YAW
    return 0.0


def _contact_forward_for_cell(cell: StackCell) -> float:
    if _uses_first_layer_y_row_strategy(cell):
        return FIRST_LAYER_Y_SWAPPED_CONTACT_FORWARD
    return GRASP_CONTACT_X_OFFSET


def _pick_fingertip_offset_for_cell(cell: StackCell) -> float:
    if _uses_second_layer_inner_row_clearance_strategy(cell):
        return SECOND_LAYER_INNER_ROW_PICK_FINGERTIP_OFFSET
    return 0.0


def _grip_contact_z_for_cell(cell: StackCell, box_size: tuple[float, float, float]) -> float:
    if _uses_third_layer_center_strategy(cell):
        return -box_size[2] * 0.5 + THIRD_LAYER_CENTER_GRIP_HEIGHT_FROM_BOTTOM
    return box_size[2] * BOX_SIDE_CONTACT_Z_RATIO


def _pick_contact_z_offset_for_cell(cell: StackCell, box_size: tuple[float, float, float]) -> float:
    return _grip_contact_z_for_cell(cell, box_size) - box_size[2] * PICK_SIDE_CONTACT_Z_RATIO


def _move_approach_height_for_cell(cell: StackCell) -> float:
    return MOVE_APPROACH_HEIGHT


def _move_preplace_height_for_cell(cell: StackCell) -> float:
    return MOVE_PREPLACE_HEIGHT


def _lift_ik_iterations_for_cell(cell: StackCell) -> int:
    return 80


def _place_ik_iterations_for_cell(cell: StackCell) -> int:
    return 100


def _y_release_clearance_for_cell(cell: StackCell) -> float:
    layer_sequence = _cell_layer_sequence(cell)
    if cell.layer == 1 and layer_sequence in {4, 5}:
        return SECOND_LAYER_Y_CROSS_RELEASE_CLEARANCE
    if cell.layer == 1 and layer_sequence in {6, 7, 8, 9}:
        return SECOND_LAYER_INNER_CORNER_Y_RELEASE_CLEARANCE
    if cell.layer == 1 and layer_sequence in {10, 11}:
        return SECOND_LAYER_Y_EXTENSION_RELEASE_CLEARANCE
    if cell.layer == 1 and layer_sequence == 13:
        return SECOND_LAYER_OUTER_CORNER_Y_RELEASE_CLEARANCE + SECOND_LAYER_OUTER_CORNER_SECOND_Y_RELEASE_EXTRA
    if cell.layer == 1 and layer_sequence in {12, 13, 14, 15}:
        return SECOND_LAYER_OUTER_CORNER_Y_RELEASE_CLEARANCE
    if _uses_second_layer_inner_row_clearance_strategy(cell):
        return SECOND_LAYER_INNER_ROW_Y_RELEASE_CLEARANCE
    if _uses_second_layer_direct_edge_strategy(cell):
        return SECOND_LAYER_DIRECT_EDGE_RELEASE_CLEARANCE
    if _uses_inner_edge_push_strategy(cell):
        return FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE
    if cell.iy in {0, GRID_Y - 1}:
        return FIRST_LAYER_Y_EXTENSION_RELEASE_CLEARANCE
    return FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE


def _inner_hand_x_clearance_for_cell(cell: StackCell) -> float:
    if not _uses_first_layer_y_row_strategy(cell):
        return 0.0
    if _uses_second_layer_inner_row_clearance_strategy(cell):
        if cell.ix == 0:
            return -SECOND_LAYER_INNER_ROW_X_HAND_CLEARANCE
        if cell.ix == GRID_X - 1:
            return SECOND_LAYER_INNER_ROW_X_HAND_CLEARANCE
        return 0.0
    if _uses_second_layer_direct_edge_strategy(cell):
        return 0.0
    if cell.ix == 0:
        if _uses_inner_edge_push_strategy(cell):
            return -FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE
        return -FIRST_LAYER_INNER_HAND_X_CLEARANCE
    if cell.ix == GRID_X - 1:
        if _uses_inner_edge_push_strategy(cell):
            return FIRST_LAYER_INNER_EDGE_HAND_CLEARANCE
        return FIRST_LAYER_INNER_HAND_X_CLEARANCE
    return 0.0


def _inner_edge_stance_x_offset_for_cell(cell: StackCell) -> float:
    if _uses_second_layer_inner_row_clearance_strategy(cell):
        return _inner_hand_x_clearance_for_cell(cell)
    if not _uses_inner_edge_push_strategy(cell):
        return 0.0
    return _inner_hand_x_clearance_for_cell(cell)


def _stand_off_for_cell(cell: StackCell, base_stand_off: float) -> float:
    layer_sequence = _cell_layer_sequence(cell)
    if layer_sequence == 1:
        return base_stand_off
    if cell.layer == 1 and layer_sequence in {2, 3}:
        return max(base_stand_off, SECOND_LAYER_X_NEIGHBOR_STAND_OFF)
    if cell.layer == 1 and layer_sequence in {6, 7, 8, 9}:
        return max(base_stand_off, SECOND_LAYER_Y_SIDE_STAND_OFF) + SECOND_LAYER_INNER_CORNER_PUSH_STAND_OFF_EXTRA
    if cell.layer == 1 and _uses_first_layer_y_row_strategy(cell) and cell.iy in {0, GRID_Y - 1}:
        return max(0.0, max(base_stand_off, SECOND_LAYER_Y_EXTENSION_STAND_OFF) - SECOND_LAYER_Y_EXTENSION_STAND_OFF_REDUCTION)
    if cell.layer == 1 and _uses_first_layer_y_row_strategy(cell):
        return max(base_stand_off, SECOND_LAYER_Y_SIDE_STAND_OFF)
    if _uses_first_layer_y_row_strategy(cell) and cell.iy in {0, GRID_Y - 1}:
        return max(base_stand_off, DIRECT_Y_EXTENSION_STAND_OFF)
    if _direct_side_for_cell(cell) in {"-Y", "+Y"}:
        return max(base_stand_off, DIRECT_Y_SIDE_STAND_OFF)
    return max(base_stand_off, DIRECT_OUTER_STAND_OFF)


def _direct_root_pose(cell: StackCell, target_center: tuple[float, float, float], stand_off: float) -> tuple[Pose, Pose]:
    side = _direct_side_for_cell(cell)
    if side == "-X":
        final_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - stand_off
        final_y = target_center[1]
        normal = (-1.0, 0.0)
    elif side == "+X":
        final_x = PALLET_CENTER[0] + STACK_SIZE[0] * 0.5 + stand_off
        final_y = target_center[1]
        normal = (1.0, 0.0)
    elif side == "-Y":
        final_x = target_center[0]
        final_y = PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 - stand_off
        normal = (0.0, -1.0)
    elif side == "+Y":
        final_x = target_center[0]
        final_y = PALLET_CENTER[1] + STACK_SIZE[1] * 0.5 + stand_off
        normal = (0.0, 1.0)
    else:
        raise ValueError(f"Unsupported direct side: {side}")
    yaw = math.atan2(target_center[1] - final_y, target_center[0] - final_x)
    final_pose = Pose(final_x, final_y, 0.0, yaw)
    tmp_pose = Pose(final_x + normal[0] * DIRECT_TMP_RETREAT, final_y + normal[1] * DIRECT_TMP_RETREAT, 0.0, yaw)
    return final_pose, tmp_pose


def _table_retreat_pose(start: Pose) -> Pose:
    dx = start.x - TABLE_POSE[0]
    dy = start.y - TABLE_POSE[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy, length = -1.0, 0.0, 1.0
    return Pose(start.x + dx / length * 0.90, start.y + dy / length * 0.90, start.z, start.yaw)


def _pre_pick_route_to(pick_root: Pose) -> list[tuple[str, Pose]]:
    home = Pose(0.0, 0.0, 0.0, 0.0)
    if abs(home.x - pick_root.x) < 1e-6 and abs(home.y - pick_root.y) < 1e-6 and abs(home.yaw - pick_root.yaw) < 1e-6:
        return []
    return [("pre_pick_turn", pose) for pose in sample_root_trajectory(home, pick_root)]


def _root_route_to(start: Pose, final_pose: Pose, tmp_pose: Pose, side: str) -> list[tuple[str, Pose]]:
    table_retreat = _table_retreat_pose(start)
    safe_x = PALLET_CENTER[0] - STACK_SIZE[0] * 0.5 - SAFE_CORRIDOR_MARGIN
    safe_y = PALLET_CENTER[1] - STACK_SIZE[1] * 0.5 - SAFE_CORRIDOR_MARGIN
    if side == "+Y":
        upper_safe_y = PALLET_CENTER[1] + STACK_SIZE[1] * 0.5 + SAFE_CORRIDOR_MARGIN
        keypoints = (
            ("table_start", start),
            ("table_retreat", table_retreat),
            ("safe_axis_1", Pose(table_retreat.x, safe_y, 0.0, 0.0)),
            ("safe_corner", Pose(safe_x, safe_y, 0.0, 0.0)),
            ("safe_plus_y", Pose(safe_x, upper_safe_y, 0.0, math.pi * 0.5)),
            ("tmp_axis", Pose(tmp_pose.x, upper_safe_y, 0.0, math.atan2(tmp_pose.y - upper_safe_y, tmp_pose.x - safe_x))),
            ("tmp_low", tmp_pose),
            ("target", final_pose),
        )
        frames: list[tuple[str, Pose]] = []
        for (_start_label, start_pose), (goal_label, goal_pose) in zip(keypoints, keypoints[1:]):
            for pose in sample_root_trajectory(start_pose, goal_pose):
                frames.append((goal_label, pose))
        frames.extend((keypoints[-1][0], keypoints[-1][1]) for _ in range(30))
        return frames
    keypoints = (
        ("table_start", start),
        ("table_retreat", table_retreat),
        ("safe_axis_1", Pose(table_retreat.x, safe_y, 0.0, 0.0)),
        ("safe_corner", Pose(safe_x, safe_y, 0.0, 0.0)),
        ("tmp_axis", Pose(tmp_pose.x, safe_y, 0.0, math.atan2(tmp_pose.y - safe_y, tmp_pose.x - safe_x))),
        ("tmp_low", tmp_pose),
        ("target", final_pose),
    )
    frames: list[tuple[str, Pose]] = []
    for (_start_label, start_pose), (goal_label, goal_pose) in zip(keypoints, keypoints[1:]):
        for pose in sample_root_trajectory(start_pose, goal_pose):
            frames.append((goal_label, pose))
    frames.extend((keypoints[-1][0], keypoints[-1][1]) for _ in range(30))
    return frames


def _return_route_from(route: list[tuple[str, Pose]]) -> list[tuple[str, Pose]]:
    if not route:
        return []
    frames = [("return_to_source", pose) for _label, pose in reversed(route)]
    reversed_poses = [pose for _label, pose in reversed(route)]
    home = Pose(0.0, 0.0, 0.0, 0.0)
    for pose in sample_root_trajectory(reversed_poses[-1], home):
        frames.append(("return_home", pose))
    frames.extend(("return_hold", home) for _ in range(RETURN_HOLD_FRAMES))
    return frames


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
        alpha = _smoothstep(i / float(frames))
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


def _direct_place_pose_path(
    start_center: tuple[float, float, float],
    start_theta: float,
    target_center: tuple[float, float, float],
) -> list[tuple[str, tuple[float, float, float], float]]:
    release = (target_center[0], target_center[1], target_center[2] + PLACE_RELEASE_HEIGHT)
    preplace_z = max(start_center[2], target_center[2] + DIRECT_PREPLACE_CLEARANCE)
    preplace = (target_center[0], target_center[1], preplace_z)
    path = [("place_handoff_tilted", start_center, start_theta)]
    _append_pose_segment(path, "place_rotate_upright", start_center, start_center, start_theta, 0.0, PLACE_UPRIGHT_FRAMES)
    _append_pose_segment(path, "place_move_over_target", start_center, preplace, 0.0, 0.0, PLACE_MOVE_XY_FRAMES)
    _append_pose_segment(path, "place_descend_to_release", preplace, release, 0.0, 0.0, PLACE_DESCEND_FRAMES)
    path.extend(("place_hold", release, 0.0) for _ in range(PLACE_HOLD_FRAMES))
    return path


def _direct_release_center_for_cell(
    cell: StackCell,
    target_center: tuple[float, float, float],
) -> tuple[float, float, float]:
    if _cell_layer_sequence(cell) in {2, 3}:
        side = _direct_side_for_cell(cell)
        if side == "-X":
            return (
                target_center[0] - FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE,
                target_center[1],
                target_center[2],
            )
        if side == "+X":
            return (
                target_center[0] + FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE,
                target_center[1],
                target_center[2],
            )
    if _uses_first_layer_y_row_strategy(cell):
        side = _direct_side_for_cell(cell)
        y_clearance = _y_release_clearance_for_cell(cell)
        x_clearance = _inner_hand_x_clearance_for_cell(cell)
        if side == "-Y":
            return (
                target_center[0] + x_clearance,
                target_center[1] - y_clearance,
                target_center[2],
            )
        if side == "+Y":
            return (
                target_center[0] + x_clearance,
                target_center[1] + y_clearance,
                target_center[2],
            )
    return target_center


def build_direct_box_plan(cell: StackCell, stand_off: float) -> BoxPlan:
    if not _supports_direct_box_plan(cell):
        raise NotImplementedError(f"{cell.label} requires the offset-push primitive, which is scaffolded but not implemented yet.")

    move_names, move_lower, move_upper = _parse_active_dofs(MOVE_URDF)
    pick_grip_axis = _grip_axis_for_cell(cell)
    place_grip_axis = "y"
    box_size = _box_size_for_cell(cell)
    contact_forward = _contact_forward_for_cell(cell)
    pick_fingertip_offset = _pick_fingertip_offset_for_cell(cell)
    pick_contact_z_offset = _pick_contact_z_offset_for_cell(cell, box_size)
    target_center = cell_center_world(cell)
    final_root, _tmp_root = _direct_root_pose(cell, target_center, _stand_off_for_cell(cell, stand_off))
    stance_x_offset = _inner_edge_stance_x_offset_for_cell(cell)
    if abs(stance_x_offset) > 1e-6:
        final_root = Pose(final_root.x + stance_x_offset, final_root.y, final_root.z, final_root.yaw)
        _tmp_root = Pose(_tmp_root.x + stance_x_offset, _tmp_root.y, _tmp_root.z, _tmp_root.yaw)
    side = _direct_side_for_cell(cell)
    pick_root_pose = Pose(0.0, 0.0, 0.0, 0.0)
    source_box_yaw = _source_box_yaw_for_cell(cell)
    source_center_local = tuple(float(v) for v in _world_to_root(SOURCE_BOX_POSE, pick_root_pose))
    pick_box_yaw_local = source_box_yaw - pick_root_pose.yaw
    pick_frames, pick_box_centers = _build_pick_keyframes(
        pick_grip_axis,
        source_center_local,
        pick_box_yaw_local,
        box_size,
        contact_forward,
        pick_fingertip_offset,
        pick_contact_z_offset,
    )
    pick_controller = IKFlipBoxController(move_names, move_lower, move_upper, MOVE_URDF, SOURCE_BOX_POSE, box_size)
    pick_controller._solve_keyframe(pick_frames[0], iterations=100, regularize_wrist=True)
    pick_targets, pick_reports = _solve_keyframes(pick_controller, pick_frames, iterations=24)
    pick_lift_q = pick_targets[-1]

    pre_pick_route = _pre_pick_route_to(pick_root_pose)
    root_route = _root_route_to(pick_root_pose, final_root, _tmp_root, side)
    return_route = _return_route_from(root_route)
    keep_box_world_yaw = side in {"-Y", "+Y"} and not _uses_first_layer_y_row_strategy(cell)
    place_box_yaw_local = 0.0

    pick_grasp_center = _grasp_center_from_q(MOVE_URDF, move_names, pick_lift_q)
    attach_offset_local = np.asarray(pick_box_centers[-1], dtype=np.float64) - pick_grasp_center
    runtime_attach_offset_local = attach_offset_local
    if pick_grip_axis == "x":
        runtime_attach_offset_local = np.asarray(X_GRIP_ATTACH_OFFSET, dtype=np.float64)
    move_lift_ik = MoveLiftIK(MOVE_URDF, move_names, move_lower, move_upper)
    move_approach_box_center_z = target_center[2] + _move_approach_height_for_cell(cell)
    move_preplace_box_center_z = target_center[2] + _move_preplace_height_for_cell(cell)
    lift_ik_iterations = _lift_ik_iterations_for_cell(cell)
    move_approach_q = move_lift_ik.solve_for_box_center_z(
        pick_lift_q,
        move_approach_box_center_z - float(attach_offset_local[2]),
        iterations=lift_ik_iterations,
    )
    move_lift_q = move_lift_ik.solve_for_box_center_z(
        move_approach_q,
        move_preplace_box_center_z - float(attach_offset_local[2]),
        iterations=lift_ik_iterations,
    )
    move_approach_box_theta = _box_theta_from_q(MOVE_URDF, move_names, move_approach_q)
    move_preplace_box_theta = _box_theta_from_q(MOVE_URDF, move_names, move_lift_q)

    move_grasp_center = _grasp_center_from_q(MOVE_URDF, move_names, move_lift_q)
    move_attached_center_local = move_grasp_center + attach_offset_local
    runtime_attached_center_local = move_grasp_center + runtime_attach_offset_local
    place_handoff_center = _transform_local_point(final_root, tuple(move_attached_center_local.tolist()))
    runtime_handoff_center = _transform_local_point(final_root, tuple(runtime_attached_center_local.tolist()))
    release_center = _direct_release_center_for_cell(cell, target_center)
    place_path = _direct_place_pose_path(place_handoff_center, move_preplace_box_theta, release_center)

    place_seed_q = move_lift_q
    if pick_grip_axis != place_grip_axis:
        seed_frames, seed_box_centers = _build_pick_keyframes(
            place_grip_axis,
            source_center_local,
            0.0,
            box_size,
            contact_forward,
            pick_fingertip_offset,
            pick_contact_z_offset,
        )
        seed_controller = IKFlipBoxController(move_names, move_lower, move_upper, MOVE_URDF, SOURCE_BOX_POSE, box_size)
        seed_controller._solve_keyframe(seed_frames[0], iterations=100, regularize_wrist=True)
        seed_targets, _seed_reports = _solve_keyframes(seed_controller, seed_frames, iterations=60)
        seed_lift_q = seed_targets[-1]
        seed_grasp_center = _grasp_center_from_q(MOVE_URDF, move_names, seed_lift_q)
        seed_attach_offset = np.asarray(seed_box_centers[-1], dtype=np.float64) - seed_grasp_center
        seed_approach_q = move_lift_ik.solve_for_box_center_z(
            seed_lift_q,
            move_approach_box_center_z - float(seed_attach_offset[2]),
            iterations=lift_ik_iterations,
        )
        place_seed_q = move_lift_ik.solve_for_box_center_z(
            seed_approach_q,
            move_preplace_box_center_z - float(seed_attach_offset[2]),
            iterations=lift_ik_iterations,
        )

    place_controller = IKFlipBoxController(move_names, move_lower, move_upper, MOVE_URDF, SOURCE_BOX_POSE, box_size)
    place_controller.q = place_seed_q.detach().cpu().numpy().astype(np.float64)
    place_key_indices = _pose_path_key_indices(place_path)
    contact_z = _grip_contact_z_for_cell(cell, box_size)
    place_key_frames = []
    for path_index in place_key_indices:
        label, center_world, box_theta = place_path[path_index]
        local_center = _world_to_root(center_world, final_root)
        place_key_frames.append(
            _make_grip_keyframe(
                label,
                local_center,
                box_theta,
                PLACE_CONTACT_GAP,
                contact_z,
                contact_forward,
                place_grip_axis,
                box_size,
                place_box_yaw_local,
                pick_fingertip_offset,
            )
        )
    place_ik_iterations = _place_ik_iterations_for_cell(cell)
    place_key_targets, place_reports = _solve_keyframes(place_controller, place_key_frames, iterations=place_ik_iterations)

    place_targets = _expand_dense_targets(place_key_indices, place_key_targets, len(place_path))
    release_local_center = _world_to_root(place_path[-1][1], final_root)
    release_frame = _make_grip_keyframe(
        "release_open",
        release_local_center,
        0.0,
        APPROACH_GAP_READY,
        contact_z + 0.02,
        contact_forward,
        place_grip_axis,
        box_size,
        place_box_yaw_local,
        pick_fingertip_offset,
    )
    release_target, release_reports = _solve_keyframes(place_controller, [release_frame], iterations=place_ik_iterations)
    del release_reports

    return BoxPlan(
        cell=cell,
        dof_names=move_names,
        pick_targets=pick_targets,
        pick_box_centers=pick_box_centers,
        box_size=box_size,
        attach_offset_local=tuple(float(v) for v in attach_offset_local),
        runtime_attach_offset_local=tuple(float(v) for v in runtime_attach_offset_local),
        move_approach_target=move_approach_q,
        move_lift_target=move_lift_q,
        move_approach_box_center_z=move_approach_box_center_z,
        move_approach_box_theta=move_approach_box_theta,
        move_preplace_box_center_z=move_preplace_box_center_z,
        move_preplace_box_theta=move_preplace_box_theta,
        place_targets=place_targets,
        release_target=release_target[0],
        pre_pick_route=pre_pick_route,
        pick_root_pose=pick_root_pose,
        source_box_yaw=source_box_yaw,
        root_route=root_route,
        return_route=return_route,
        place_pose_path_world=place_path,
        place_handoff_center_world=place_handoff_center,
        runtime_handoff_center_world=runtime_handoff_center,
        target_center=target_center,
        grip_axis=pick_grip_axis,
        place_box_yaw_local=place_box_yaw_local,
        box_world_yaw=0.0,
        keep_box_world_yaw=keep_box_world_yaw,
        pick_max_error=max(report.max_error for report in pick_reports),
        place_max_error=max(report.max_error for report in place_reports),
    )


def _expand_dense_targets(key_indices: list[int], key_targets: torch.Tensor, total_frames: int) -> torch.Tensor:
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
            frames[index] = start_q * (1.0 - alpha) + goal_q * alpha
    last_q = key_targets[-1].clone()
    dense: list[torch.Tensor] = []
    for frame in frames:
        if frame is None:
            dense.append(last_q.clone())
        else:
            dense.append(frame)
            last_q = frame
    return torch.stack(dense)


def _side_dof_name(name: str) -> str | None:
    if name.startswith("xhand_left_") or "_l_" in name or name.endswith("_l_joint"):
        return "left"
    if name.startswith("xhand_right_") or "_r_" in name or name.endswith("_r_joint") or name.endswith("_r_joinst"):
        return "right"
    return None


def _blend_side_dofs(
    base: torch.Tensor,
    target: torch.Tensor,
    dof_names: list[str],
    side: str,
    alpha: float,
) -> torch.Tensor:
    q = base.clone()
    for index, name in enumerate(dof_names):
        if _side_dof_name(name) == side:
            q[index] = base[index] * (1.0 - alpha) + target[index] * alpha
    return q


def _dof_limits_for_names(dof_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    active_names, lower, upper = _parse_active_dofs(MOVE_URDF)
    if active_names == dof_names:
        return lower, upper
    return _reorder_targets(lower, active_names, dof_names), _reorder_targets(upper, active_names, dof_names)


def _inner_hand_cartesian_lift_q(
    q: torch.Tensor,
    dof_names: list[str],
    side: str | None,
    root_pose: Pose,
    cell: StackCell,
) -> torch.Tensor:
    if side is None:
        return q
    if side == "left":
        ee_link = LEFT_EE_LINK
        arm_joints = [name for name in LEFT_ARM_JOINTS if "wrist" not in name]
    elif side == "right":
        ee_link = RIGHT_EE_LINK
        arm_joints = [name for name in RIGHT_ARM_JOINTS if "wrist" not in name]
    else:
        return q

    lower, upper = _dof_limits_for_names(dof_names)
    controller = IKFlipBoxController(dof_names, lower, upper, MOVE_URDF, SOURCE_BOX_POSE, STACK_BOX_SIZE)
    controller.q = q.detach().cpu().numpy().astype(np.float64)
    current_feature = controller._ee_feature(ee_link)
    current_pos = current_feature[:3]
    palm_normal = current_feature[3:6] / controller._orientation_scale()
    finger_dir = current_feature[6:9] / controller._orientation_scale()
    layer_top_local_z = PALLET_SURFACE_Z + (cell.layer + 1) * STACK_BOX_SIZE[2] - root_pose.z
    target_pos = current_pos.copy()
    target_pos[2] = max(
        current_pos[2] + FIRST_LAYER_INNER_HAND_LIFT_MIN_DELTA,
        layer_top_local_z + FIRST_LAYER_INNER_HAND_LIFT_ABOVE_TOP,
    )
    controller._solve_arm(
        ee_link,
        arm_joints,
        target_pos,
        palm_normal,
        finger_dir,
        FIRST_LAYER_INNER_HAND_LIFT_IK_ITERATIONS,
        regularize_wrist=False,
    )
    lifted = torch.tensor(controller.q, dtype=q.dtype, device=q.device)
    for index, name in enumerate(dof_names):
        if _side_dof_name(name) == side and ("wrist" in name or name.startswith("xhand_")):
            lifted[index] = q[index]
    return lifted


def _world_xy_delta_to_root_local(root_pose: Pose, dx: float, dy: float) -> np.ndarray:
    c = math.cos(-root_pose.yaw)
    s = math.sin(-root_pose.yaw)
    return np.array([c * dx - s * dy, s * dx + c * dy, 0.0], dtype=np.float64)


def _hand_cartesian_feature(
    q: torch.Tensor,
    dof_names: list[str],
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = _dof_limits_for_names(dof_names)
    controller = IKFlipBoxController(dof_names, lower, upper, MOVE_URDF, SOURCE_BOX_POSE, STACK_BOX_SIZE)
    controller.q = q.detach().cpu().numpy().astype(np.float64)
    ee_link = LEFT_EE_LINK if side == "left" else RIGHT_EE_LINK
    feature = controller._ee_feature(ee_link)
    scale = controller._orientation_scale()
    return feature[:3], feature[3:6] / scale, feature[6:9] / scale


def _outer_hand_cartesian_move_q(
    q: torch.Tensor,
    dof_names: list[str],
    side: str | None,
    root_pose: Pose,
    world_dx: float,
    world_dy: float,
    world_z: float | None = None,
    target_palm_normal: np.ndarray | None = None,
    target_finger_dir: np.ndarray | None = None,
) -> torch.Tensor:
    if side is None:
        return q
    if side == "left":
        ee_link = LEFT_EE_LINK
        arm_joints = LEFT_ARM_JOINTS
    elif side == "right":
        ee_link = RIGHT_EE_LINK
        arm_joints = RIGHT_ARM_JOINTS
    else:
        return q

    lower, upper = _dof_limits_for_names(dof_names)
    controller = IKFlipBoxController(dof_names, lower, upper, MOVE_URDF, SOURCE_BOX_POSE, STACK_BOX_SIZE)
    controller.q = q.detach().cpu().numpy().astype(np.float64)
    current_feature = controller._ee_feature(ee_link)
    current_pos = current_feature[:3]
    palm_normal = target_palm_normal
    finger_dir = target_finger_dir
    if palm_normal is None:
        palm_normal = current_feature[3:6] / controller._orientation_scale()
    if finger_dir is None:
        finger_dir = current_feature[6:9] / controller._orientation_scale()
    target_pos = current_pos + _world_xy_delta_to_root_local(root_pose, world_dx, world_dy)
    if world_z is not None:
        target_pos[2] = world_z - root_pose.z
    controller._solve_arm(
        ee_link,
        arm_joints,
        target_pos,
        palm_normal,
        finger_dir,
        FIRST_LAYER_OUTER_HAND_PUSH_IK_ITERATIONS,
        regularize_wrist=False,
    )
    pushed = torch.tensor(controller.q, dtype=q.dtype, device=q.device)
    for index, name in enumerate(dof_names):
        if _side_dof_name(name) == side and name.startswith("xhand_"):
            pushed[index] = q[index]
    return pushed


def _inner_hand_side_for_cell(cell: StackCell) -> str | None:
    side = _direct_side_for_cell(cell)
    if cell.ix == 0:
        return "right" if side == "-Y" else "left"
    if cell.ix == GRID_X - 1:
        return "left" if side == "-Y" else "right"
    return None


def _outer_hand_side_for_cell(cell: StackCell) -> str | None:
    inner_side = _inner_hand_side_for_cell(cell)
    if inner_side == "left":
        return "right"
    if inner_side == "right":
        return "left"
    return None


def _rigid_body_side_from_name(name: str) -> str | None:
    if "left" in name or "_l_" in name or name.endswith("_l_link"):
        return "left"
    if "right" in name or "_r_" in name or name.endswith("_r_link"):
        return "right"
    return None


def _set_outer_hand_box_collision_enabled(gym, env, robot, box, outer_side: str) -> None:
    box_shape_props = gym.get_actor_rigid_shape_properties(env, box)
    for prop in box_shape_props:
        prop.filter = 1
    gym.set_actor_rigid_shape_properties(env, box, box_shape_props)

    robot_shape_props = gym.get_actor_rigid_shape_properties(env, robot)
    body_names = gym.get_actor_rigid_body_names(env, robot)
    shape_ranges = gym.get_actor_rigid_body_shape_indices(env, robot)
    for body_name, shape_range in zip(body_names, shape_ranges):
        body_side = _rigid_body_side_from_name(body_name)
        body_filter = 2 if "xhand" in body_name and body_side == outer_side else 1
        for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
            robot_shape_props[shape_idx].filter = body_filter
    gym.set_actor_rigid_shape_properties(env, robot, robot_shape_props)


def _inner_edge_push_world_delta(cell: StackCell) -> tuple[float, float]:
    x_step = -_inner_hand_x_clearance_for_cell(cell)
    if abs(x_step) > 1e-6:
        x_step += math.copysign(FIRST_LAYER_OUTER_HAND_PUSH_EXTRA_DISTANCE, x_step)
    side = _direct_side_for_cell(cell)
    y_step = 0.0
    if side == "-Y":
        y_step = _y_release_clearance_for_cell(cell) if cell.layer == 1 else FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE
    elif side == "+Y":
        y_step = -(_y_release_clearance_for_cell(cell) if cell.layer == 1 else FIRST_LAYER_INNER_EDGE_RELEASE_CLEARANCE)
    return (x_step, y_step)


def _outer_hand_body_retract_distance_for_cell(cell: StackCell) -> float:
    if cell.layer == 1:
        return SECOND_LAYER_OUTER_HAND_BODY_RETRACT_DISTANCE
    return FIRST_LAYER_OUTER_HAND_BODY_RETRACT_DISTANCE


def _outer_hand_push_below_com_for_cell(cell: StackCell) -> float:
    if cell.layer == 1:
        return SECOND_LAYER_OUTER_HAND_PUSH_BELOW_COM
    return FIRST_LAYER_OUTER_HAND_PUSH_BELOW_COM


def _ideal_outer_hand_push_orientation(
    cell: StackCell,
    outer_side: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if cell.layer != 1 or outer_side is None:
        return None
    if outer_side == "left":
        palm_normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    elif outer_side == "right":
        palm_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        return None
    finger_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return palm_normal, finger_dir


def _lerp_pose(start: Pose, goal: Pose, alpha: float) -> Pose:
    return Pose(
        start.x * (1.0 - alpha) + goal.x * alpha,
        start.y * (1.0 - alpha) + goal.y * alpha,
        start.z * (1.0 - alpha) + goal.z * alpha,
        start.yaw * (1.0 - alpha) + goal.yaw * alpha,
    )


def build_timeline(plan: BoxPlan) -> list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]]:
    pick_dof_frames = _expand_key_targets(plan.pick_targets, PICK_SEGMENT_FRAMES)
    _apply_pick_finger_closure(pick_dof_frames, plan.dof_names)
    pick_box_frames = _expand_centers(plan.pick_box_centers, PICK_SEGMENT_FRAMES)

    root_frames = plan.root_route + [(plan.root_route[-1][0], plan.root_route[-1][1]) for _ in range(MOVE_SETTLE_FRAMES)]
    final_move_root = plan.root_route[-1][1]
    carry_start = pick_box_frames[-1]
    carry_local_start = carry_start
    carry_local_final = (
        plan.runtime_handoff_center_world[0] - final_move_root.x,
        plan.runtime_handoff_center_world[1] - final_move_root.y,
        plan.runtime_handoff_center_world[2] - final_move_root.z,
    )
    carry_local_approach = (
        carry_local_final[0],
        carry_local_final[1],
        plan.move_approach_box_center_z - final_move_root.z,
    )

    move_phase_labels: list[str] = []
    move_dof_frames: list[torch.Tensor] = []
    move_box_frames: list[tuple[float, float, float]] = []
    move_box_thetas: list[float] = []
    for label, root_pose in root_frames:
        move_phase_labels.append(label)
        move_dof_frames.append(plan.pick_targets[-1])
        move_box_frames.append(_transform_local_point(root_pose, carry_local_start))
        move_box_thetas.append(0.0)
    for i in range(1, MOVE_APPROACH_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_APPROACH_FRAMES))
        move_phase_labels.append("lift_to_approach")
        move_dof_frames.append(plan.pick_targets[-1] * (1.0 - alpha) + plan.move_approach_target * alpha)
        carry_local = (
            carry_local_start[0] * (1.0 - alpha) + carry_local_approach[0] * alpha,
            carry_local_start[1] * (1.0 - alpha) + carry_local_approach[1] * alpha,
            carry_local_start[2] * (1.0 - alpha) + carry_local_approach[2] * alpha,
        )
        move_box_frames.append(_transform_local_point(final_move_root, carry_local))
        move_box_thetas.append(plan.move_approach_box_theta * alpha)
    for i in range(1, MOVE_LIFT_FRAMES + 1):
        alpha = _smoothstep(i / float(MOVE_LIFT_FRAMES))
        move_phase_labels.append("lift_to_preplace")
        move_dof_frames.append(plan.move_approach_target * (1.0 - alpha) + plan.move_lift_target * alpha)
        carry_local = (
            carry_local_approach[0] * (1.0 - alpha) + carry_local_final[0] * alpha,
            carry_local_approach[1] * (1.0 - alpha) + carry_local_final[1] * alpha,
            carry_local_approach[2] * (1.0 - alpha) + carry_local_final[2] * alpha,
        )
        move_box_frames.append(_transform_local_point(final_move_root, carry_local))
        move_box_thetas.append(plan.move_approach_box_theta * (1.0 - alpha) + plan.move_preplace_box_theta * alpha)

    place_box_frames_labeled = plan.place_pose_path_world
    timeline: list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]] = []
    first_pick_center = _transform_local_point(plan.pick_root_pose, pick_box_frames[0])
    for label, root_pose in plan.pre_pick_route:
        timeline.append((f"pre_pick:{label}", root_pose, plan.pick_targets[0], first_pick_center, 0.0))
    for q, center in zip(pick_dof_frames, pick_box_frames):
        timeline.append(("pick", plan.pick_root_pose, q, _transform_local_point(plan.pick_root_pose, center), 0.0))

    move_root_poses = [pose for _label, pose in root_frames] + [final_move_root for _ in range(MOVE_APPROACH_FRAMES + MOVE_LIFT_FRAMES)]
    for label, root_pose, q, center, theta in zip(move_phase_labels, move_root_poses, move_dof_frames, move_box_frames, move_box_thetas):
        timeline.append((f"move:{label}", root_pose, q, center, theta))

    move_end_q = move_dof_frames[-1]
    move_end_center = move_box_frames[-1]
    move_end_theta = move_box_thetas[-1]
    first_label, first_center, first_theta = place_box_frames_labeled[0]
    for i in range(1, PLACE_HANDOFF_FRAMES + 1):
        alpha = _smoothstep(i / float(PLACE_HANDOFF_FRAMES))
        center = (
            move_end_center[0] * (1.0 - alpha) + first_center[0] * alpha,
            move_end_center[1] * (1.0 - alpha) + first_center[1] * alpha,
            move_end_center[2] * (1.0 - alpha) + first_center[2] * alpha,
        )
        q = move_end_q * (1.0 - alpha) + plan.place_targets[0] * alpha
        timeline.append((f"place_handoff:{first_label}", final_move_root, q, center, move_end_theta * (1.0 - alpha) + first_theta * alpha))

    for q, (label, center, theta) in zip(plan.place_targets, place_box_frames_labeled):
        timeline.append((f"place:{label}", final_move_root, q, center, theta))

    release_center = place_box_frames_labeled[-1][1]
    recover_target = plan.pick_targets[0]
    if _uses_inner_edge_push_strategy(plan.cell):
        place_hold_q = plan.place_targets[-1]
        timeline.append(("release", final_move_root, place_hold_q, release_center, 0.0))
        for _ in range(FINAL_HOLD_FRAMES):
            timeline.append(("post_release_inner_edge_settle", final_move_root, place_hold_q, release_center, 0.0))

        inner_side = _inner_hand_side_for_cell(plan.cell)
        outer_side = _outer_hand_side_for_cell(plan.cell)
        push_ready_q = place_hold_q
        if inner_side is not None:
            push_ready_q = _inner_hand_cartesian_lift_q(push_ready_q, plan.dof_names, inner_side, final_move_root, plan.cell)
        if outer_side is not None:
            push_ready_q = _blend_side_dofs(
                push_ready_q,
                plan.release_target,
                plan.dof_names,
                outer_side,
                FIRST_LAYER_OUTER_PUSH_OPEN_ALPHA,
            )
        for i in range(1, FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES))
            q = place_hold_q * (1.0 - alpha) + push_ready_q * alpha
            timeline.append(("post_release_inner_hand_lift", final_move_root, q, release_center, 0.0))

        push_dx, push_dy = _inner_edge_push_world_delta(plan.cell)
        push_length = max(math.hypot(push_dx, push_dy), 1e-6)
        push_unit_x = push_dx / push_length
        push_unit_y = push_dy / push_length
        body_retract_distance = _outer_hand_body_retract_distance_for_cell(plan.cell)
        body_retract_x = -math.cos(final_move_root.yaw) * body_retract_distance
        body_retract_y = -math.sin(final_move_root.yaw) * body_retract_distance
        low_push_world_z = release_center[2] - _outer_hand_push_below_com_for_cell(plan.cell)
        outer_retract_q = push_ready_q
        outer_low_q = push_ready_q
        outer_contact_q = push_ready_q
        push_target_q = push_ready_q
        if outer_side is not None:
            _outer_pos, outer_palm_normal, outer_finger_dir = _hand_cartesian_feature(
                plan.release_target,
                plan.dof_names,
                outer_side,
            )
            ideal_push_orientation = _ideal_outer_hand_push_orientation(plan.cell, outer_side)
            if ideal_push_orientation is not None:
                outer_palm_normal, outer_finger_dir = ideal_push_orientation
            outer_retract_q = _outer_hand_cartesian_move_q(
                push_ready_q,
                plan.dof_names,
                outer_side,
                final_move_root,
                -push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
                -push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
                target_palm_normal=outer_palm_normal,
                target_finger_dir=outer_finger_dir,
            )
            outer_low_q = _outer_hand_cartesian_move_q(
                outer_retract_q,
                plan.dof_names,
                outer_side,
                final_move_root,
                body_retract_x,
                body_retract_y,
                low_push_world_z,
                target_palm_normal=outer_palm_normal,
                target_finger_dir=outer_finger_dir,
            )
            outer_contact_q = _outer_hand_cartesian_move_q(
                outer_low_q,
                plan.dof_names,
                outer_side,
                final_move_root,
                push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
                push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
                low_push_world_z,
                target_palm_normal=outer_palm_normal,
                target_finger_dir=outer_finger_dir,
            )
            push_target_q = _outer_hand_cartesian_move_q(
                outer_contact_q,
                plan.dof_names,
                outer_side,
                final_move_root,
                push_dx,
                push_dy,
                low_push_world_z,
                target_palm_normal=outer_palm_normal,
                target_finger_dir=outer_finger_dir,
            )
        for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
            q = push_ready_q * (1.0 - alpha) + outer_retract_q * alpha
            timeline.append(("post_release_outer_hand_retract", final_move_root, q, release_center, 0.0))
        for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
            q = outer_retract_q * (1.0 - alpha) + outer_low_q * alpha
            timeline.append(("post_release_outer_hand_lower", final_move_root, q, release_center, 0.0))
        for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
            q = outer_low_q * (1.0 - alpha) + outer_contact_q * alpha
            timeline.append(("post_release_outer_hand_contact", final_move_root, q, release_center, 0.0))
        for i in range(1, FIRST_LAYER_INNER_EDGE_PUSH_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_INNER_EDGE_PUSH_FRAMES))
            q = outer_contact_q * (1.0 - alpha) + push_target_q * alpha
            timeline.append(("post_release_outer_hand_push", final_move_root, q, release_center, 0.0))
        for _ in range(FIRST_LAYER_INNER_EDGE_PUSH_SETTLE_FRAMES):
            timeline.append(("post_release_push_settle", final_move_root, push_target_q, release_center, 0.0))

        for i in range(1, RETURN_RECOVER_FRAMES + 1):
            alpha = _smoothstep(i / float(RETURN_RECOVER_FRAMES))
            q = push_target_q * (1.0 - alpha) + recover_target * alpha
            timeline.append(("return_recover:upright", final_move_root, q, release_center, 0.0))
        for label, root_pose in plan.return_route:
            timeline.append((f"return:{label}", root_pose, recover_target, release_center, 0.0))
        return _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)

    if _uses_second_layer_inner_row_clearance_strategy(plan.cell):
        place_hold_q = plan.place_targets[-1]
        timeline.append(("release", final_move_root, place_hold_q, release_center, 0.0))
        for _ in range(FINAL_HOLD_FRAMES):
            timeline.append(("post_release_upper_layer_settle", final_move_root, place_hold_q, release_center, 0.0))

        inner_side = _inner_hand_side_for_cell(plan.cell)
        outer_side = _outer_hand_side_for_cell(plan.cell)
        lift_q = place_hold_q
        if inner_side is not None:
            lift_q = _inner_hand_cartesian_lift_q(lift_q, plan.dof_names, inner_side, final_move_root, plan.cell)
        if outer_side is not None:
            lift_q = _blend_side_dofs(
                lift_q,
                plan.release_target,
                plan.dof_names,
                outer_side,
                FIRST_LAYER_OUTER_PUSH_OPEN_ALPHA,
            )
        for i in range(1, FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES + 1):
            alpha = _smoothstep(i / float(FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES))
            q = place_hold_q * (1.0 - alpha) + lift_q * alpha
            timeline.append(("post_release_upper_layer_inner_hand_lift", final_move_root, q, release_center, 0.0))

        for i in range(1, RETURN_RECOVER_FRAMES + 1):
            alpha = _smoothstep(i / float(RETURN_RECOVER_FRAMES))
            q = lift_q * (1.0 - alpha) + recover_target * alpha
            timeline.append(("return_recover:upright", final_move_root, q, release_center, 0.0))
        for label, root_pose in plan.return_route:
            timeline.append((f"return:{label}", root_pose, recover_target, release_center, 0.0))
        return _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)

    timeline.append(("release", final_move_root, plan.release_target, release_center, 0.0))
    for _ in range(FINAL_HOLD_FRAMES):
        timeline.append(("post_release_free_fall", final_move_root, plan.release_target, release_center, 0.0))
    for i in range(1, RETURN_RECOVER_FRAMES + 1):
        alpha = _smoothstep(i / float(RETURN_RECOVER_FRAMES))
        q = plan.release_target * (1.0 - alpha) + recover_target * alpha
        timeline.append(("return_recover:upright", final_move_root, q, release_center, 0.0))
    for label, root_pose in plan.return_route:
        timeline.append((f"return:{label}", root_pose, recover_target, release_center, 0.0))
    return _smooth_timeline_joint_steps(timeline, max_joint_step=0.02)


def _delay_second_layer_release_until_hold_end(plan: BoxPlan) -> BoxPlan:
    if plan.cell.layer != 1:
        return plan
    place_path = [
        ("place_attached_hold", center, theta) if label == "place_hold" else (label, center, theta)
        for label, center, theta in plan.place_pose_path_world
    ]
    return replace(plan, place_pose_path_world=place_path)


def _replace_second_layer_edge_corner_post_release(
    plan: BoxPlan,
    timeline: list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]],
) -> list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]]:
    if not _uses_second_layer_edge_corner_push_strategy(plan.cell):
        return timeline

    try:
        release_index = next(index for index, frame in enumerate(timeline) if frame[0] == "release")
        retract_start = next(index for index, frame in enumerate(timeline) if frame[0] == "post_release_upper_layer_inner_hand_lift")
    except StopIteration:
        return timeline

    _release_phase, final_root, place_hold_q, release_center, _release_theta = timeline[release_index]
    inner_side = _inner_hand_side_for_cell(plan.cell)
    outer_side = _outer_hand_side_for_cell(plan.cell)
    push_ready_q = place_hold_q
    if inner_side is not None:
        _inner_pos, inner_palm_normal, inner_finger_dir = _hand_cartesian_feature(
            push_ready_q,
            plan.dof_names,
            inner_side,
        )
        push_ready_q = _outer_hand_cartesian_move_q(
            push_ready_q,
            plan.dof_names,
            inner_side,
            final_root,
            -math.cos(final_root.yaw) * WAIST_LAYER_INNER_HAND_HORIZONTAL_RETRACT,
            -math.sin(final_root.yaw) * WAIST_LAYER_INNER_HAND_HORIZONTAL_RETRACT,
            target_palm_normal=inner_palm_normal,
            target_finger_dir=inner_finger_dir,
        )
    if outer_side is not None:
        push_ready_q = _blend_side_dofs(
            push_ready_q,
            plan.release_target,
            plan.dof_names,
            outer_side,
            FIRST_LAYER_OUTER_PUSH_OPEN_ALPHA,
        )

    new_timeline = list(timeline[:retract_start])
    recover_target = plan.pick_targets[0]
    for i in range(1, FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES + 1):
        alpha = _smoothstep(i / float(FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES))
        q = place_hold_q * (1.0 - alpha) + push_ready_q * alpha
        new_timeline.append(("post_release_second_layer_inner_hand_retract", final_root, q, release_center, 0.0))

    push_dx, push_dy = _inner_edge_push_world_delta(plan.cell)
    push_length = max(math.hypot(push_dx, push_dy), 1e-6)
    push_unit_x = push_dx / push_length
    push_unit_y = push_dy / push_length
    push_distance = max(0.0, push_length - SECOND_LAYER_EDGE_CORNER_PUSH_DISTANCE_REDUCTION)
    if plan.cell.layer == 1 and _cell_layer_sequence(plan.cell) == 13:
        push_distance = max(0.0, push_distance - SECOND_LAYER_OUTER_CORNER_SECOND_PUSH_DISTANCE_REDUCTION)
    push_dx = push_unit_x * push_distance
    push_dy = push_unit_y * push_distance
    body_retract_distance = _outer_hand_body_retract_distance_for_cell(plan.cell)
    body_retract_x = -math.cos(final_root.yaw) * body_retract_distance
    body_retract_y = -math.sin(final_root.yaw) * body_retract_distance
    hand_forward_x = math.cos(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_PUSH_HAND_FORWARD_OFFSET
    hand_forward_y = math.sin(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_PUSH_HAND_FORWARD_OFFSET
    lower_forward_x = math.cos(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_LOWER_FORWARD_EXTRA
    lower_forward_y = math.sin(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_LOWER_FORWARD_EXTRA
    low_push_world_z = release_center[2] - _outer_hand_push_below_com_for_cell(plan.cell)
    outer_retract_q = push_ready_q
    outer_low_q = push_ready_q
    outer_contact_q = push_ready_q
    push_target_q = push_ready_q
    if outer_side is not None:
        _outer_pos, outer_palm_normal, outer_finger_dir = _hand_cartesian_feature(
            plan.release_target,
            plan.dof_names,
            outer_side,
        )
        ideal_push_orientation = _ideal_outer_hand_push_orientation(plan.cell, outer_side)
        if ideal_push_orientation is not None:
            outer_palm_normal, outer_finger_dir = ideal_push_orientation
        outer_retract_q = _outer_hand_cartesian_move_q(
            push_ready_q,
            plan.dof_names,
            outer_side,
            final_root,
            -push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            -push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        outer_low_q = _outer_hand_cartesian_move_q(
            outer_retract_q,
            plan.dof_names,
            outer_side,
            final_root,
            body_retract_x + hand_forward_x + lower_forward_x,
            body_retract_y + hand_forward_y + lower_forward_y,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        outer_contact_q = _outer_hand_cartesian_move_q(
            outer_low_q,
            plan.dof_names,
            outer_side,
            final_root,
            push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        push_target_q = _outer_hand_cartesian_move_q(
            outer_contact_q,
            plan.dof_names,
            outer_side,
            final_root,
            push_dx,
            push_dy,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )

    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = push_ready_q * (1.0 - alpha) + outer_retract_q * alpha
        new_timeline.append(("post_release_outer_hand_retract", final_root, q, release_center, 0.0))
    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = outer_retract_q * (1.0 - alpha) + outer_low_q * alpha
        new_timeline.append(("post_release_outer_hand_lower", final_root, q, release_center, 0.0))
    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = _smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = outer_low_q * (1.0 - alpha) + outer_contact_q * alpha
        new_timeline.append(("post_release_outer_hand_contact", final_root, q, release_center, 0.0))
    for i in range(1, FIRST_LAYER_INNER_EDGE_PUSH_FRAMES + 1):
        alpha = _smoothstep(i / float(FIRST_LAYER_INNER_EDGE_PUSH_FRAMES))
        q = outer_contact_q * (1.0 - alpha) + push_target_q * alpha
        new_timeline.append(("post_release_outer_hand_push", final_root, q, release_center, 0.0))
    for _ in range(FIRST_LAYER_INNER_EDGE_PUSH_SETTLE_FRAMES):
        new_timeline.append(("post_release_push_settle", final_root, push_target_q, release_center, 0.0))

    for i in range(1, RETURN_RECOVER_FRAMES + 1):
        alpha = _smoothstep(i / float(RETURN_RECOVER_FRAMES))
        q = push_target_q * (1.0 - alpha) + recover_target * alpha
        new_timeline.append(("return_recover:upright", final_root, q, release_center, 0.0))
    for label, root_pose in plan.return_route:
        new_timeline.append((f"return:{label}", root_pose, recover_target, release_center, 0.0))
    return _smooth_timeline_joint_steps(new_timeline, max_joint_step=0.02)


def build_task1_2_timeline(plan: BoxPlan) -> list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]]:
    return _replace_second_layer_edge_corner_post_release(plan, build_timeline(plan))


def _expand_centers(
    centers: list[tuple[float, float, float]],
    frames_per_segment: int,
) -> list[tuple[float, float, float]]:
    frames: list[tuple[float, float, float]] = []
    if not centers:
        return frames
    frames.append(centers[0])
    for start, goal in zip(centers[:-1], centers[1:]):
        for i in range(1, frames_per_segment + 1):
            alpha = _smoothstep(i / float(frames_per_segment))
            frames.append(
                (
                    start[0] * (1.0 - alpha) + goal[0] * alpha,
                    start[1] * (1.0 - alpha) + goal[1] * alpha,
                    start[2] * (1.0 - alpha) + goal[2] * alpha,
                )
            )
    return frames


def _parked_box_pose(index: int) -> tuple[float, float, float]:
    return (BOX_PARK_X - BOX_PARK_SPACING * index, -1.2, SOURCE_BOX_POSE[2])


def _box_size_for_sequence(sequence: int) -> tuple[float, float, float]:
    if _layer_sequence(sequence) in FIRST_LAYER_Y_SWAPPED_BOXES:
        return (STACK_BOX_SIZE[1], STACK_BOX_SIZE[0], STACK_BOX_SIZE[2])
    return STACK_BOX_SIZE


def _lower_layer_proxy_size(proxy_layers: int) -> tuple[float, float, float]:
    layers = max(0, min(int(proxy_layers), GRID_Z))
    if layers == 1:
        return FIRST_LAYER_PROXY_SIZE
    if layers == 2:
        return SECOND_LAYER_PROXY_SIZE
    return (STACK_SIZE[0], STACK_SIZE[1], STACK_BOX_SIZE[2] * layers)


def _create_static_scene(
    gym,
    sim,
    env,
    robot_asset,
    box_sequences: list[int],
    proxy_layers: int = 0,
):
    robot = gym.create_actor(env, robot_asset, _make_transform(0.0, 0.0, 0.0), "stack_60_robot", 0, 1)

    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, STACK_SIZE[0], STACK_SIZE[1], PALLET_THICKNESS, fixed_opts)
    lower_layer_proxy_asset = None
    proxy_size = _lower_layer_proxy_size(proxy_layers)
    if proxy_layers > 0:
        lower_layer_proxy_asset = gym.create_box(
            sim,
            proxy_size[0],
            proxy_size[1],
            proxy_size[2],
            fixed_opts,
        )

    box_opts = gymapi.AssetOptions()
    box_opts.density = STACK_BOX_DENSITY
    source_asset = gym.create_box(sim, STACK_BOX_SIZE[0], STACK_BOX_SIZE[1], STACK_BOX_SIZE[2], box_opts)
    swapped_source_asset = gym.create_box(sim, STACK_BOX_SIZE[1], STACK_BOX_SIZE[0], STACK_BOX_SIZE[2], box_opts)
    fourth_layer_center_box_opts = gymapi.AssetOptions()
    fourth_layer_center_box_opts.density = STACK_BOX_DENSITY * FOURTH_LAYER_CENTER_DENSITY_MULTIPLIER
    fourth_layer_center_asset = gym.create_box(
        sim,
        STACK_BOX_SIZE[0],
        STACK_BOX_SIZE[1],
        STACK_BOX_SIZE[2],
        fourth_layer_center_box_opts,
    )

    table = gym.create_actor(env, table_asset, _make_transform(*TABLE_POSE), "table", 0, 0)
    boxes = []
    for index, sequence in enumerate(box_sequences):
        pose = SOURCE_BOX_POSE if index == 0 else _parked_box_pose(index)
        if sequence == FOURTH_LAYER_CENTER_SEQUENCE:
            box_asset = fourth_layer_center_asset
        else:
            box_asset = swapped_source_asset if _layer_sequence(sequence) in FIRST_LAYER_Y_SWAPPED_BOXES else source_asset
        boxes.append(
            gym.create_actor(
                env,
                box_asset,
                _make_transform(*pose),
                f"stack_box_{sequence:02d}",
                0,
                0,
            )
    )
    pallet_z = PALLET_SURFACE_Z - PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(env, pallet_asset, _make_transform(PALLET_CENTER[0], PALLET_CENTER[1], pallet_z), "pallet", 0, 0)
    lower_layer_proxy = None
    if lower_layer_proxy_asset is not None:
        lower_layer_proxy_z = PALLET_SURFACE_Z + proxy_size[2] * 0.5
        lower_layer_proxy = gym.create_actor(
            env,
            lower_layer_proxy_asset,
            _make_transform(PALLET_CENTER[0], PALLET_CENTER[1], lower_layer_proxy_z),
            f"lower_{proxy_layers}_layer_proxy",
            0,
            0,
        )

    _set_color(gym, env, table, (1.0, 1.0, 1.0))
    for index, box in enumerate(boxes):
        layer_alpha = 0.15 + 0.12 * (index % 5)
        _set_color(gym, env, box, (0.9, layer_alpha, 0.02))
    _set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    if lower_layer_proxy is not None:
        _set_color(gym, env, lower_layer_proxy, (0.62, 0.62, 0.58))

    _set_shape_friction(gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05)
    _set_shape_friction(gym, env, table, 2.5)
    for sequence, box in zip(box_sequences, boxes):
        if sequence == THIRD_LAYER_CENTER_SEQUENCE:
            _set_contact_material(
                gym,
                env,
                box,
                THIRD_LAYER_CENTER_CONTACT_FRICTION,
                THIRD_LAYER_CENTER_ROLLING_FRICTION,
                THIRD_LAYER_CENTER_TORSION_FRICTION,
                BOX_CONTACT_RESTITUTION,
            )
            continue
        if sequence == FOURTH_LAYER_CENTER_SEQUENCE:
            _set_contact_material(
                gym,
                env,
                box,
                FOURTH_LAYER_CENTER_CONTACT_FRICTION,
                FOURTH_LAYER_CENTER_ROLLING_FRICTION,
                FOURTH_LAYER_CENTER_TORSION_FRICTION,
                BOX_CONTACT_RESTITUTION,
            )
            continue
        _set_contact_material(
            gym,
            env,
            box,
            BOX_CONTACT_FRICTION,
            BOX_CONTACT_ROLLING_FRICTION,
            BOX_CONTACT_TORSION_FRICTION,
            BOX_CONTACT_RESTITUTION,
        )
    _set_shape_friction(gym, env, pallet, PALLET_CONTACT_FRICTION)
    if lower_layer_proxy is not None:
        _set_shape_friction(gym, env, lower_layer_proxy, PALLET_CONTACT_FRICTION)
    if boxes:
        _set_task_collision_filters(gym, env, robot, boxes[0])
    return robot, boxes


def _set_contact_material(
    gym,
    env,
    actor,
    friction: float,
    rolling_friction: float,
    torsion_friction: float,
    restitution: float,
) -> None:
    props = gym.get_actor_rigid_shape_properties(env, actor)
    for prop in props:
        prop.friction = float(friction)
        prop.rolling_friction = float(rolling_friction)
        prop.torsion_friction = float(torsion_friction)
        prop.restitution = float(restitution)
    gym.set_actor_rigid_shape_properties(env, actor, props)


def _configure_robot_dofs(gym, env, robot, dof_names: list[str]) -> None:
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        if name.startswith("xhand_"):
            props["stiffness"][i] = 120.0
            props["damping"][i] = 12.0
            props["effort"][i] = max(float(props["effort"][i]), 35.0)
        else:
            props["stiffness"][i] = 560.0
            props["damping"][i] = 56.0
            props["effort"][i] = max(float(props["effort"][i]), 200.0)
    gym.set_actor_dof_properties(env, robot, props)


def _reset_box_to_source(gym, sim, root_states: torch.Tensor, box_index: int, actor_index: int, yaw: float) -> None:
    del box_index
    _set_actor_root_pose(gym, sim, root_states, actor_index, SOURCE_BOX_POSE, yaw, 0.0)


def _capture_box_pose(
    gym,
    env,
    actor,
    actor_index: int,
    sequence: int,
    target_center: tuple[float, float, float],
) -> PlacedBoxPose:
    position = _actor_center(gym, env, actor)
    yaw, pitch = _actor_yaw_pitch(gym, env, actor)
    return PlacedBoxPose(sequence, actor_index, target_center, position, yaw, pitch)


def _actor_position_from_root_state(root_states: torch.Tensor, actor_index: int) -> tuple[float, float, float]:
    state = root_states[actor_index]
    return (float(state[0]), float(state[1]), float(state[2]))


def _report_placed_box_motion(root_states: torch.Tensor, placed: list[PlacedBoxPose], context: str) -> None:
    for pose in placed:
        current = _actor_position_from_root_state(root_states, pose.actor_index)
        drift = _distance(current, pose.position)
        target_error = _distance(current, pose.target_center)
        if drift < 0.003 and target_error < 0.003:
            continue
        print(
            "stack_60_placed_motion "
            f"context={context} box={pose.sequence} "
            f"current=({current[0]:.3f},{current[1]:.3f},{current[2]:.3f}) "
            f"initial=({pose.position[0]:.3f},{pose.position[1]:.3f},{pose.position[2]:.3f}) "
            f"drift={drift:.4f} target_error={target_error:.4f}"
        )


def _run_box_timeline(
    gym,
    sim,
    env,
    robot,
    box,
    root_states: torch.Tensor,
    robot_index: int,
    box_index: int,
    plan: BoxPlan,
    timeline: list[tuple[str, Pose, torch.Tensor, tuple[float, float, float], float]],
    placed: list[PlacedBoxPose],
    viewer,
    viewer_render_every: int,
    render_viewer: bool,
    sync_viewer_frame_time: bool,
    max_frame_budget: int,
    attach_after_pick_frames: int,
    global_frame_start: int,
) -> tuple[int, bool, float]:
    _set_task_collision_filters(gym, env, robot, box)
    _reset_box_to_source(gym, sim, root_states, plan.cell.sequence - 1, box_index, plan.source_box_yaw)
    attached = False
    released = False
    hand_box_collision_enabled = True
    attach_offset_local: tuple[float, float, float] | None = None
    attach_yaw_offset = 0.0
    attach_theta_offset = 0.0
    placed_current = False
    pick_frame_count = 0
    pick_started = False
    final_error = float("inf")
    for local_frame, (phase, root_pose, q, box_center, box_theta) in enumerate(timeline):
        if max_frame_budget > 0 and local_frame >= max_frame_budget:
            return global_frame_start + local_frame, False, float("nan")
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            return global_frame_start + local_frame, False, float("nan")

        if phase == "pick" and not pick_started:
            _reset_box_to_source(gym, sim, root_states, plan.cell.sequence - 1, box_index, plan.source_box_yaw)
            pick_started = True

        if phase == "pick":
            pick_frame_count += 1

        if not attached and not released and phase == "pick" and pick_frame_count >= attach_after_pick_frames:
            attached = True
            actual_center = _actor_center(gym, env, box)
            actual_yaw, actual_pitch = _actor_yaw_pitch(gym, env, box)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root_pose.yaw)
            measured_offset = _inverse_rotate_yaw_pitch(_sub_vec(actual_center, grasp_center), root_pose.yaw, grasp_theta)
            attach_offset_local = tuple(float(v) for v in measured_offset)
            attach_yaw_offset = actual_yaw - root_pose.yaw
            attach_theta_offset = actual_pitch - grasp_theta
            _set_hand_box_collision_enabled(gym, env, robot, box, enabled=False)
            hand_box_collision_enabled = False
            print(
                "stack_60_attach "
                f"box={plan.cell.sequence} frame={global_frame_start + local_frame} phase={phase} "
                f"offset=({attach_offset_local[0]:.3f},{attach_offset_local[1]:.3f},{attach_offset_local[2]:.3f}) "
                f"measured=({measured_offset[0]:.3f},{measured_offset[1]:.3f},{measured_offset[2]:.3f})"
            )

        release_now = attached and not released and (phase == "release" or phase == "place:place_hold")
        if release_now:
            gym.refresh_actor_root_state_tensor(sim)
            released = True
            attached = False
            actual_release = _actor_center(gym, env, box)
            print(
                "stack_60_release "
                f"box={plan.cell.sequence} frame={global_frame_start + local_frame} phase={phase} "
                f"actual=({actual_release[0]:.3f},{actual_release[1]:.3f},{actual_release[2]:.3f})"
            )

        if (
            released
            and not hand_box_collision_enabled
            and (_uses_inner_edge_push_strategy(plan.cell) or _uses_second_layer_edge_corner_push_strategy(plan.cell))
            and (phase.startswith("post_release_outer_hand_contact") or phase.startswith("post_release_outer_hand_push"))
        ):
            outer_side = _outer_hand_side_for_cell(plan.cell)
            if outer_side is None:
                _set_hand_box_collision_enabled(gym, env, robot, box, enabled=True)
            else:
                _set_outer_hand_box_collision_enabled(gym, env, robot, box, outer_side)
            hand_box_collision_enabled = True

        if released and not placed_current and (phase.startswith("return_recover:") or phase.startswith("return:")):
            placed_pose = _capture_box_pose(gym, env, box, box_index, plan.cell.sequence, plan.target_center)
            _set_actor_collision_filter(gym, env, box, 0)
            placed.append(placed_pose)
            placed_current = True
            final_error = _distance(placed_pose.position, plan.target_center)
            print(
                "stack_60_record_actual "
                f"box={plan.cell.sequence} frame={global_frame_start + local_frame} "
                f"pose=({placed_pose.position[0]:.3f},{placed_pose.position[1]:.3f},{placed_pose.position[2]:.3f}) "
                f"yaw={math.degrees(placed_pose.yaw):.1f}deg error={final_error:.4f}"
            )

        gym.refresh_actor_root_state_tensor(sim)
        _set_actor_root_pose(gym, sim, root_states, robot_index, (root_pose.x, root_pose.y, root_pose.z), root_pose.yaw)
        _set_robot_dof_state(gym, env, robot, q)
        if phase == "pick":
            _lock_mobile_dof_state(gym, env, robot, plan.dof_names)
        gym.set_actor_dof_position_targets(env, robot, q.numpy())

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if attached and attach_offset_local is not None:
            gym.refresh_actor_root_state_tensor(sim)
            grasp_center = _grasp_center(gym, env, robot)
            grasp_theta = _grasp_theta(gym, env, robot, root_pose.yaw)
            attached_center = _add_vec(grasp_center, _rotate_yaw_pitch(attach_offset_local, root_pose.yaw, grasp_theta))
            attached_theta = grasp_theta + attach_theta_offset
            if plan.keep_box_world_yaw:
                attached_yaw = plan.box_world_yaw
            elif _uses_first_layer_y_row_strategy(plan.cell):
                attached_yaw = root_pose.yaw + attach_yaw_offset
            else:
                attached_yaw = root_pose.yaw
            _set_actor_root_pose(gym, sim, root_states, box_index, attached_center, attached_yaw, attached_theta)

        if render_viewer and viewer is not None and local_frame % viewer_render_every == 0:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if sync_viewer_frame_time:
                gym.sync_frame_time(sim)

        if local_frame % 240 == 0:
            actual_box = _actor_center(gym, env, box)
            q_err, q_err_joints = _dof_error_summary(gym, env, robot, q, plan.dof_names)
            print(
                f"frame={global_frame_start + local_frame} box={plan.cell.sequence} phase={phase} "
                f"root=({root_pose.x:.2f},{root_pose.y:.2f},{root_pose.z:.2f}) "
                f"actual_box=({actual_box[0]:.2f},{actual_box[1]:.2f},{actual_box[2]:.2f}) "
                f"q_err={q_err:.3f} top_q_err={q_err_joints} attached={attached} released={released}"
            )

    if not placed_current:
        placed_pose = _capture_box_pose(gym, env, box, box_index, plan.cell.sequence, plan.target_center)
        _set_actor_collision_filter(gym, env, box, 0)
        placed.append(placed_pose)
        placed_current = True
        final_error = _distance(placed_pose.position, plan.target_center)
    actual_box = _actor_center(gym, env, box)
    print(
        "stack_60_box_result "
        f"box={plan.cell.sequence} actual=({actual_box[0]:.4f},{actual_box[1]:.4f},{actual_box[2]:.4f}) "
        f"target=({plan.target_center[0]:.4f},{plan.target_center[1]:.4f},{plan.target_center[2]:.4f}) "
        f"error={final_error:.4f} released={released}"
    )
    return global_frame_start + len(timeline), True, final_error


def _reorder_box_plan(plan: BoxPlan, target_names: list[str]) -> BoxPlan:
    if plan.dof_names == target_names:
        return plan
    return BoxPlan(
        cell=plan.cell,
        dof_names=target_names,
        pick_targets=_reorder_targets(plan.pick_targets, plan.dof_names, target_names),
        pick_box_centers=plan.pick_box_centers,
        box_size=plan.box_size,
        attach_offset_local=plan.attach_offset_local,
        runtime_attach_offset_local=plan.runtime_attach_offset_local,
        move_approach_target=_reorder_targets(plan.move_approach_target, plan.dof_names, target_names),
        move_lift_target=_reorder_targets(plan.move_lift_target, plan.dof_names, target_names),
        move_approach_box_center_z=plan.move_approach_box_center_z,
        move_approach_box_theta=plan.move_approach_box_theta,
        move_preplace_box_center_z=plan.move_preplace_box_center_z,
        move_preplace_box_theta=plan.move_preplace_box_theta,
        place_targets=_reorder_targets(plan.place_targets, plan.dof_names, target_names),
        release_target=_reorder_targets(plan.release_target, plan.dof_names, target_names),
        pre_pick_route=plan.pre_pick_route,
        pick_root_pose=plan.pick_root_pose,
        source_box_yaw=plan.source_box_yaw,
        root_route=plan.root_route,
        return_route=plan.return_route,
        place_pose_path_world=plan.place_pose_path_world,
        place_handoff_center_world=plan.place_handoff_center_world,
        runtime_handoff_center_world=plan.runtime_handoff_center_world,
        target_center=plan.target_center,
        grip_axis=plan.grip_axis,
        place_box_yaw_local=plan.place_box_yaw_local,
        box_world_yaw=plan.box_world_yaw,
        keep_box_world_yaw=plan.keep_box_world_yaw,
        pick_max_error=plan.pick_max_error,
        place_max_error=plan.place_max_error,
    )


def _make_waist_stack_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        headless=args.headless,
        max_frames=args.max_frames,
        fast=False,
        render_every=None,
        fast_viewer=args.fast_viewer,
        viewer_render_every=args.viewer_render_every,
        viewer_start_box=args.viewer_start_box,
        frame_stride=args.waist_frame_stride,
        stack_sequence_count=0,
        stack_stand_off=waist_stack.DIRECT_OUTER_STAND_OFF,
        stack_body_dz=None,
        stack_release_height=waist_stack.STACK_RELEASE_HEIGHT,
        stack_lift_frames=140,
        stack_place_xy_frames=180,
        stack_place_descend_frames=180,
        stack_plan_ik_stride=args.waist_stack_plan_ik_stride,
        init_arm_iterations=120,
        lift_iterations=140,
        arm_iterations=45,
        arm_waypoints=35,
        lift_frames=120,
        frames_per_arm_waypoint=5,
        hold_frames=180,
    )


def _uses_waist_layer_corner_push_strategy(cell) -> bool:
    return (
        cell.layer in {2, 3}
        and waist_stack._stack_cell_layer_sequence(cell) in waist_stack.FIRST_LAYER_INNER_EDGE_PUSH_BOXES
    )


def _waist_layer_corner_push_world_delta(cell) -> tuple[float, float]:
    x_step = -waist_stack._stack_inner_hand_x_clearance_for_cell(cell)
    if abs(x_step) > 1e-6:
        x_step += math.copysign(FIRST_LAYER_OUTER_HAND_PUSH_EXTRA_DISTANCE, x_step)
    side = waist_stack._stack_direct_side_for_cell(cell)
    y_step = 0.0
    y_clearance = waist_stack._stack_y_release_clearance_for_cell(cell)
    if side == "-Y":
        y_step = y_clearance
    elif side == "+Y":
        y_step = -y_clearance
    return (x_step, y_step)


def _append_waist_layer_corner_push(cell, timeline: list[object], dof_names: list[str]) -> list[object]:
    if not _uses_waist_layer_corner_push_strategy(cell):
        return timeline

    try:
        release_index = next(index for index, frame in enumerate(timeline) if frame.phase == "release")
        settle_index = next(index for index, frame in enumerate(timeline) if index > release_index and frame.phase == "settle")
    except StopIteration:
        return timeline

    release_frame = timeline[release_index]
    final_root = release_frame.root
    release_center = release_frame.box_center
    q_open = timeline[settle_index - 1].q if settle_index > release_index else release_frame.q
    inner_side = waist_stack._stack_inner_hand_side_for_cell(cell)
    outer_side = waist_stack._stack_outer_hand_side_for_cell(cell)

    push_ready_q = q_open
    if inner_side is not None:
        _inner_pos, inner_palm_normal, inner_finger_dir = _hand_cartesian_feature(
            push_ready_q,
            dof_names,
            inner_side,
        )
        push_ready_q = _outer_hand_cartesian_move_q(
            push_ready_q,
            dof_names,
            inner_side,
            final_root,
            -math.cos(final_root.yaw) * WAIST_LAYER_INNER_HAND_HORIZONTAL_RETRACT,
            -math.sin(final_root.yaw) * WAIST_LAYER_INNER_HAND_HORIZONTAL_RETRACT,
            target_palm_normal=inner_palm_normal,
            target_finger_dir=inner_finger_dir,
        )
    if outer_side is not None:
        push_ready_q = _blend_side_dofs(
            push_ready_q,
            q_open,
            dof_names,
            outer_side,
            FIRST_LAYER_OUTER_PUSH_OPEN_ALPHA,
        )

    new_timeline = list(timeline[:settle_index])
    for i in range(1, FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES + 1):
        alpha = waist_stack._smoothstep(i / float(FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES))
        q = q_open * (1.0 - alpha) + push_ready_q * alpha
        new_timeline.append(waist_stack.StackFrame("post_release_waist_inner_hand_retract", final_root, q, release_center, False))

    push_dx, push_dy = _waist_layer_corner_push_world_delta(cell)
    push_length = max(math.hypot(push_dx, push_dy), 1e-6)
    push_unit_x = push_dx / push_length
    push_unit_y = push_dy / push_length
    push_distance_extra = WAIST_LAYER_CORNER_PUSH_DISTANCE_EXTRA
    if cell.layer == 2:
        push_distance_extra += WAIST_THIRD_LAYER_CORNER_PUSH_DISTANCE_EXTRA
    elif cell.layer == 3:
        push_distance_extra -= WAIST_FOURTH_LAYER_CORNER_PUSH_DISTANCE_REDUCTION
    push_distance = max(
        0.0,
        push_length - SECOND_LAYER_EDGE_CORNER_PUSH_DISTANCE_REDUCTION + push_distance_extra,
    )
    push_dx = push_unit_x * push_distance
    push_dy = push_unit_y * push_distance
    body_retract_distance = SECOND_LAYER_OUTER_HAND_BODY_RETRACT_DISTANCE
    if cell.layer == 3:
        body_retract_distance += WAIST_FOURTH_LAYER_OUTER_HAND_BACK_RETRACT_EXTRA
    body_retract_x = -math.cos(final_root.yaw) * body_retract_distance
    body_retract_y = -math.sin(final_root.yaw) * body_retract_distance
    hand_forward_x = math.cos(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_PUSH_HAND_FORWARD_OFFSET
    hand_forward_y = math.sin(final_root.yaw) * SECOND_LAYER_EDGE_CORNER_PUSH_HAND_FORWARD_OFFSET
    low_push_world_z = release_center[2] - SECOND_LAYER_OUTER_HAND_PUSH_BELOW_COM - WAIST_LAYER_CORNER_PUSH_Z_LOWER
    outer_retract_q = push_ready_q
    outer_low_q = push_ready_q
    outer_contact_q = push_ready_q
    push_target_q = push_ready_q
    if outer_side is not None:
        _outer_pos, outer_palm_normal, outer_finger_dir = _hand_cartesian_feature(q_open, dof_names, outer_side)
        outer_retract_q = _outer_hand_cartesian_move_q(
            push_ready_q,
            dof_names,
            outer_side,
            final_root,
            -push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            -push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        outer_low_q = _outer_hand_cartesian_move_q(
            outer_retract_q,
            dof_names,
            outer_side,
            final_root,
            body_retract_x + hand_forward_x,
            body_retract_y + hand_forward_y,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        outer_contact_q = _outer_hand_cartesian_move_q(
            outer_low_q,
            dof_names,
            outer_side,
            final_root,
            push_unit_x * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            push_unit_y * FIRST_LAYER_OUTER_HAND_RETRACT_DISTANCE,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )
        push_target_q = _outer_hand_cartesian_move_q(
            outer_contact_q,
            dof_names,
            outer_side,
            final_root,
            push_dx,
            push_dy,
            low_push_world_z,
            target_palm_normal=outer_palm_normal,
            target_finger_dir=outer_finger_dir,
        )

    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = waist_stack._smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = push_ready_q * (1.0 - alpha) + outer_retract_q * alpha
        new_timeline.append(waist_stack.StackFrame("post_release_waist_outer_hand_retract", final_root, q, release_center, False))
    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = waist_stack._smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = outer_retract_q * (1.0 - alpha) + outer_low_q * alpha
        new_timeline.append(waist_stack.StackFrame("post_release_waist_outer_hand_lower", final_root, q, release_center, False))
    for i in range(1, FIRST_LAYER_OUTER_HAND_PREP_FRAMES + 1):
        alpha = waist_stack._smoothstep(i / float(FIRST_LAYER_OUTER_HAND_PREP_FRAMES))
        q = outer_low_q * (1.0 - alpha) + outer_contact_q * alpha
        new_timeline.append(waist_stack.StackFrame("post_release_waist_outer_hand_contact", final_root, q, release_center, False))
    for i in range(1, FIRST_LAYER_INNER_EDGE_PUSH_FRAMES + 1):
        alpha = waist_stack._smoothstep(i / float(FIRST_LAYER_INNER_EDGE_PUSH_FRAMES))
        q = outer_contact_q * (1.0 - alpha) + push_target_q * alpha
        new_timeline.append(waist_stack.StackFrame("post_release_waist_outer_hand_push", final_root, q, release_center, False))
    for _ in range(FIRST_LAYER_INNER_EDGE_PUSH_SETTLE_FRAMES):
        new_timeline.append(waist_stack.StackFrame("post_release_waist_push_settle", final_root, push_target_q, release_center, False))

    return_start = next((index for index, frame in enumerate(timeline) if index > settle_index and frame.phase.startswith("return:")), len(timeline))
    recover_frames = [frame for frame in timeline[return_start:] if frame.phase == "return_recover_home"]
    for frame in timeline[return_start:]:
        if frame.phase.startswith("return:"):
            new_timeline.append(waist_stack.StackFrame(frame.phase, frame.root, push_target_q.clone(), release_center, False))
    if recover_frames:
        recover_goal = recover_frames[-1].q
        recover_root = recover_frames[-1].root
        for i in range(1, len(recover_frames) + 1):
            alpha = waist_stack._smoothstep(i / float(len(recover_frames)))
            q = push_target_q * (1.0 - alpha) + recover_goal * alpha
            new_timeline.append(waist_stack.StackFrame("return_recover_home", recover_root, q, release_center, False))
    return new_timeline


def _uses_waist_fourth_layer_x_neighbor_lift_adjustment(cell) -> bool:
    return cell.layer == 3 and waist_stack._stack_cell_layer_sequence(cell) in {2, 3}


def _waist_forward_lowered_center(frame, distance: float, z_reduction: float) -> tuple[float, float, float]:
    return (
        frame.box_center[0] + math.cos(frame.root.yaw) * float(distance),
        frame.box_center[1] + math.sin(frame.root.yaw) * float(distance),
        frame.box_center[2] - float(z_reduction),
    )


def _waist_arm_seed_with_previous(
    frame_q: torch.Tensor,
    previous_q: torch.Tensor | None,
    dof_names: list[str],
) -> torch.Tensor:
    if previous_q is None:
        return frame_q.clone()
    seed = frame_q.clone()
    name_to_index = {name: index for index, name in enumerate(dof_names)}
    for joint_name in tuple(LEFT_ARM_JOINTS) + tuple(RIGHT_ARM_JOINTS):
        index = name_to_index.get(joint_name)
        if index is not None:
            seed[index] = previous_q[index]
    return seed


def _extend_waist_fourth_layer_x_neighbor_lift(
    cell,
    timeline: list[object],
    metadata: dict[str, object],
    kin,
    dof_names: list[str],
    lower: torch.Tensor,
    upper: torch.Tensor,
    arm_iterations: int,
) -> tuple[list[object], dict[str, object]]:
    if not _uses_waist_fourth_layer_x_neighbor_lift_adjustment(cell):
        return timeline, metadata

    box_size = metadata.get("box_size", waist_stack._stack_box_size_for_cell(cell))
    contact_z = float(metadata.get("contact_z", waist_stack._box_grip_contact_z(box_size)))
    contact_forward = float(metadata.get("contact_forward", waist_stack.STACK_GRASP_CONTACT_X_OFFSET))
    lift_forward = waist_stack.FOURTH_LAYER_X_NEIGHBOR_LIFT_FORWARD
    lift_z_reduction = waist_stack.FOURTH_LAYER_X_NEIGHBOR_LIFT_Z_REDUCTION
    lift_total = sum(1 for frame in timeline if frame.phase == "lift_box_safe")
    place_total = sum(1 for frame in timeline if frame.phase == "place_move_over_target")
    place_goal = next((frame.box_center for frame in reversed(timeline) if frame.phase == "place_move_over_target"), None)
    descend_total = sum(1 for frame in timeline if frame.phase == "place_descend_to_release")
    release_goal = next((frame.box_center for frame in reversed(timeline) if frame.phase == "place_descend_to_release"), None)
    if lift_total <= 0:
        return timeline, metadata

    adjusted: list[object] = []
    previous_q: torch.Tensor | None = None
    lift_seen = 0
    place_seen = 0
    descend_seen = 0
    place_start: tuple[float, float, float] | None = None
    descend_start: tuple[float, float, float] | None = None
    max_extra_pos_error = 0.0
    for frame in timeline:
        shifted_center: tuple[float, float, float] | None = None
        reuse_previous_q = False
        if frame.phase == "lift_box_safe":
            lift_seen += 1
            alpha = waist_stack._smoothstep(lift_seen / float(lift_total))
            shifted_center = _waist_forward_lowered_center(frame, lift_forward * alpha, lift_z_reduction * alpha)
        elif frame.phase.startswith("move:"):
            shifted_center = _waist_forward_lowered_center(frame, lift_forward, lift_z_reduction)
            place_start = shifted_center
            reuse_previous_q = previous_q is not None
        elif frame.phase == "place_move_over_target" and place_total > 0 and place_goal is not None:
            place_seen += 1
            if place_start is None:
                place_start = _waist_forward_lowered_center(frame, lift_forward, lift_z_reduction)
            alpha = waist_stack._smoothstep(place_seen / float(place_total))
            no_lift_place_goal = (
                place_goal[0],
                place_goal[1],
                min(place_goal[2], place_start[2]),
            )
            shifted_center = (
                place_start[0] * (1.0 - alpha) + no_lift_place_goal[0] * alpha,
                place_start[1] * (1.0 - alpha) + no_lift_place_goal[1] * alpha,
                place_start[2] * (1.0 - alpha) + no_lift_place_goal[2] * alpha,
            )
            if place_seen == place_total:
                descend_start = shifted_center
        elif frame.phase == "place_descend_to_release" and descend_total > 0 and release_goal is not None:
            descend_seen += 1
            if descend_start is None:
                if place_start is not None and place_goal is not None:
                    descend_start = (place_goal[0], place_goal[1], min(place_goal[2], place_start[2]))
                else:
                    descend_start = frame.box_center
            alpha = waist_stack._smoothstep(descend_seen / float(descend_total))
            shifted_center = (
                descend_start[0] * (1.0 - alpha) + release_goal[0] * alpha,
                descend_start[1] * (1.0 - alpha) + release_goal[1] * alpha,
                descend_start[2] * (1.0 - alpha) + release_goal[2] * alpha,
            )

        if shifted_center is None:
            adjusted.append(frame)
            previous_q = frame.q if frame.attached else None
            continue

        if reuse_previous_q:
            q = previous_q.clone()
        else:
            seed_q = _waist_arm_seed_with_previous(frame.q, previous_q, dof_names)
            q, left_report, right_report = waist_stack.solve_box_grip(
                kin,
                dof_names,
                lower,
                upper,
                seed_q,
                frame.root,
                shifted_center,
                side_gap=0.0,
                contact_z=contact_z,
                contact_forward=contact_forward,
                iterations=arm_iterations,
                box_size=box_size,
            )
            max_extra_pos_error = max(max_extra_pos_error, left_report.pos_error, right_report.pos_error)
        adjusted.append(
            waist_stack.StackFrame(
                frame.phase,
                frame.root,
                q,
                shifted_center,
                frame.attached,
                frame.box_yaw,
                frame.box_pitch,
            )
        )
        previous_q = q if frame.attached else None

    updated_metadata = dict(metadata)
    updated_metadata["max_pos_error"] = max(float(updated_metadata.get("max_pos_error", 0.0)), max_extra_pos_error)
    updated_metadata["task1_2_lift_forward"] = lift_forward
    updated_metadata["task1_2_lift_z_reduction"] = lift_z_reduction
    updated_metadata["task1_2_no_prerelease_lift"] = True
    return adjusted, updated_metadata


def _build_waist_stack_plans(
    args: argparse.Namespace,
    asset_dof_names: list[str],
) -> tuple[SimpleNamespace, list["waist_stack.StackBoxPlan"]]:
    waist_args = _make_waist_stack_args(args)
    dof_names, lower, upper = waist_stack._parse_active_dofs(waist_stack.MOVE_URDF)
    kin = waist_stack.UrdfKinematics(waist_stack.MOVE_URDF)
    cells = list(waist_stack._stack_third_fourth_order())
    plans: list[waist_stack.StackBoxPlan] = []
    print(f"stack_60_waist_plan_build count={len(cells)}", flush=True)
    for index, cell in enumerate(cells, start=1):
        print(f"stack_60_waist_plan_build_start {index}/{len(cells)} {cell.label}", flush=True)
        timeline, metadata = waist_stack._build_stack_box_timeline(
            waist_args,
            kin,
            dof_names,
            lower,
            upper,
            cell,
            return_home=True,
        )
        timeline, metadata = _extend_waist_fourth_layer_x_neighbor_lift(
            cell,
            timeline,
            metadata,
            kin,
            dof_names,
            lower,
            upper,
            waist_args.arm_iterations,
        )
        timeline = _append_waist_layer_corner_push(cell, timeline, dof_names)
        plans.append(waist_stack.StackBoxPlan(cell, timeline, metadata))
        print(
            f"stack_60_waist_plan_build_done {index}/{len(cells)} {cell.label} "
            f"side={metadata['side']} stand_off={float(metadata['stand_off']):.3f} "
            f"frames={len(timeline)} arm_pos={float(metadata['max_pos_error']):.4f}",
            flush=True,
        )

    if asset_dof_names != dof_names:
        reorder = torch.tensor([dof_names.index(name) for name in asset_dof_names], dtype=torch.long)
        reordered: list[waist_stack.StackBoxPlan] = []
        for plan in plans:
            timeline = [
                waist_stack.StackFrame(
                    frame.phase,
                    frame.root,
                    frame.q[reorder].contiguous(),
                    frame.box_center,
                    frame.attached,
                    frame.box_yaw,
                    frame.box_pitch,
                )
                for frame in plan.timeline
            ]
            reordered.append(waist_stack.StackBoxPlan(plan.cell, timeline, plan.metadata))
        plans = reordered
    return waist_args, plans


def _configure_waist_stack_robot_dofs(gym, env, robot, dof_names: list[str]) -> None:
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    for i, name in enumerate(dof_names):
        if name.startswith("xhand_"):
            props["stiffness"][i] = 80.0
            props["damping"][i] = 8.0
            props["effort"][i] = max(float(props["effort"][i]), 20.0)
        elif name in waist_stack.LIFT_JOINTS:
            props["stiffness"][i] = 780.0
            props["damping"][i] = 78.0
            props["effort"][i] = max(float(props["effort"][i]), 260.0)
        else:
            props["stiffness"][i] = 560.0
            props["damping"][i] = 56.0
            props["effort"][i] = max(float(props["effort"][i]), 190.0)
    gym.set_actor_dof_properties(env, robot, props)


def _run_task1_2_waist_stack_box_timeline(
    gym,
    sim,
    env,
    robot,
    box,
    root_states: torch.Tensor,
    robot_index: int,
    box_index: int,
    plan,
    viewer,
    args: argparse.Namespace,
    global_frame_start: int,
    max_frames: int,
) -> tuple[int, bool]:
    robot_actor_indices = torch.tensor([robot_index], dtype=torch.int32)
    box_actor_indices = torch.tensor([box_index], dtype=torch.int32)
    waist_stack._reset_stack_box_to_source(gym, sim, root_states, box_actor_indices, box_index)
    waist_stack._set_box_hand_collision_enabled(gym, env, robot, box, enabled=True)

    released = False
    previous_attached = False
    attach_offset_local: tuple[float, float, float] | None = None
    attach_yaw_offset = 0.0
    attach_theta_offset = 0.0
    hand_box_collision_enabled = True
    local_frame = 0
    while local_frame < len(plan.timeline):
        frame = plan.timeline[local_frame]
        frame_index = global_frame_start + local_frame
        if max_frames > 0 and frame_index >= max_frames:
            return frame_index, False
        gym.refresh_actor_root_state_tensor(sim)
        waist_stack._set_actor_root_pose(
            gym,
            sim,
            root_states,
            robot_actor_indices,
            robot_index,
            (frame.root.x, frame.root.y, frame.root.z),
            frame.root.yaw,
        )
        if frame.attached:
            if not previous_attached:
                actual_center = waist_stack._actor_center_from_sim(gym, env, box)
                actual_yaw, actual_pitch = waist_stack._actor_yaw_pitch_from_sim(gym, env, box)
                grasp_center = waist_stack._grasp_center_from_sim(gym, env, robot)
                grasp_theta = waist_stack._grasp_theta_from_sim(gym, env, robot, frame.root.yaw)
                attach_offset_local = waist_stack._inverse_rotate_yaw_pitch(
                    waist_stack._sub_vec(actual_center, grasp_center),
                    frame.root.yaw,
                    grasp_theta,
                )
                attach_yaw_offset = actual_yaw - frame.root.yaw
                attach_theta_offset = actual_pitch - grasp_theta
                waist_stack._set_box_hand_collision_enabled(gym, env, robot, box, enabled=False)
                hand_box_collision_enabled = False
                print(
                    "waist_arm_stack_attach "
                    f"box={plan.cell.sequence} frame={frame_index} phase={frame.phase} "
                    f"offset=({attach_offset_local[0]:.3f},{attach_offset_local[1]:.3f},{attach_offset_local[2]:.3f})"
                )
        elif not released and (frame.phase == "release" or previous_attached):
            released = True
            actual_release = waist_stack._actor_center_from_sim(gym, env, box)
            print(
                "waist_arm_stack_release "
                f"box={plan.cell.sequence} frame={frame_index} phase={frame.phase} "
                f"actual=({actual_release[0]:.3f},{actual_release[1]:.3f},{actual_release[2]:.3f})"
            )

        if (
            released
            and not hand_box_collision_enabled
            and (
                frame.phase.startswith("post_release_waist_outer_hand_contact")
                or frame.phase.startswith("post_release_waist_outer_hand_push")
            )
        ):
            outer_side = waist_stack._stack_outer_hand_side_for_cell(plan.cell)
            if outer_side is None:
                waist_stack._set_box_hand_collision_enabled(gym, env, robot, box, enabled=True)
            else:
                _set_outer_hand_box_collision_enabled(gym, env, robot, box, outer_side)
            hand_box_collision_enabled = True

        waist_stack._set_robot_dof_state(gym, env, robot, frame.q)
        gym.set_actor_dof_position_targets(env, robot, frame.q.numpy())
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if frame.attached and attach_offset_local is not None:
            gym.refresh_actor_root_state_tensor(sim)
            grasp_center = waist_stack._grasp_center_from_sim(gym, env, robot)
            grasp_theta = waist_stack._grasp_theta_from_sim(gym, env, robot, frame.root.yaw)
            attached_center = waist_stack._add_vec(
                grasp_center,
                waist_stack._rotate_yaw_pitch(attach_offset_local, frame.root.yaw, grasp_theta),
            )
            waist_stack._set_actor_root_pose(
                gym,
                sim,
                root_states,
                box_actor_indices,
                box_index,
                attached_center,
                frame.root.yaw + attach_yaw_offset,
                grasp_theta + attach_theta_offset,
            )
        if plan.cell.sequence >= getattr(args, "viewer_start_box", 1):
            waist_stack._draw_viewer_frame(gym, sim, viewer, args, frame_index)

        if local_frame % 180 == 0 or local_frame == len(plan.timeline) - 1:
            actual_box = waist_stack._actor_center_from_sim(gym, env, box)
            body_z, body_pitch, _body_x = waist_stack._body_terms_from_sim(gym, env, robot)
            print(
                f"frame={frame_index} box={plan.cell.sequence} phase={frame.phase} attached={frame.attached} "
                f"planned=({frame.box_center[0]:.2f},{frame.box_center[1]:.2f},{frame.box_center[2]:.2f}) "
                f"actual=({actual_box[0]:.2f},{actual_box[1]:.2f},{actual_box[2]:.2f}) "
                f"body_z={body_z:.2f} pitch={math.degrees(body_pitch):.2f}deg"
            )

        previous_attached = frame.attached
        local_frame += waist_stack._frame_stride(args)
    if released:
        waist_stack._set_actor_collision_filter(gym, env, box, 0)
    return global_frame_start + len(plan.timeline), True


def _set_robot_initial_q(gym, env, robot, q: torch.Tensor) -> None:
    initial_state = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    initial_state["pos"] = q.numpy()
    initial_state["vel"].fill(0.0)
    gym.set_actor_dof_states(env, robot, initial_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot, q.numpy())


def main() -> None:
    # 运行模式说明：
    # - 默认：完整 60 箱；
    # - --second-layer-proxy：第一层代理块，只跑第 16..30 箱；
    # - --third-fourth-proxy：前两层代理块，只跑第 31..60 箱。
    parser = argparse.ArgumentParser(
        description="Run the complete 60-box stacking demo.",
        epilog=(
            "常用示例:\n"
            "  python -m move.tasks.task1_2\n"
            "  python -m move.tasks.task1_2 --second-layer-proxy\n"
            "  python -m move.tasks.task1_2 --third-fourth-proxy --fast-viewer\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs to task end.")
    parser.add_argument("--fast", action="store_true", help="Convenience mode: unthrottled viewer playback and reduced viewer rendering.")
    parser.add_argument("--fast-viewer", action="store_true", help="Do not throttle viewer playback to real time.")
    parser.add_argument("--viewer-render-every", type=int, default=None, help="Render the viewer once every N simulation frames. Default: 1, or 4 with --fast.")
    parser.add_argument("--viewer-start-box", type=int, default=1, help="Skip viewer rendering before this 1-based box sequence.")
    parser.add_argument(
        "--second-layer-proxy",
        action="store_true",
        help="Run only boxes 16..30, replacing layer 1 with one fixed 1.0 x 1.0 x 0.4 m support block.",
    )
    parser.add_argument(
        "--third-fourth-proxy",
        action="store_true",
        help="Run only boxes 31..60, replacing layers 1/2 with one fixed 1.0 x 1.0 x 0.8 m support block.",
    )
    parser.add_argument("--waist-frame-stride", type=int, default=None, help="Optional frame stride for the layer-3/4 waist-arm timeline.")
    parser.add_argument(
        "--waist-stack-plan-ik-stride",
        type=int,
        default=waist_stack.STACK_SEQUENCE_PLAN_IK_STRIDE,
        help="IK stride used while planning the layer-3/4 waist-arm sequence.",
    )
    args = parser.parse_args()

    if args.fast:
        args.fast_viewer = True
    if args.viewer_render_every is None:
        args.viewer_render_every = 4 if args.fast else 1
    if args.viewer_render_every < 1:
        raise ValueError("--viewer-render-every must be >= 1")
    if args.viewer_start_box < 1:
        raise ValueError("--viewer-start-box must be >= 1")
    if args.second_layer_proxy and args.third_fourth_proxy:
        raise ValueError("--second-layer-proxy and --third-fourth-proxy are mutually exclusive")
    if args.waist_frame_stride is not None and args.waist_frame_stride < 1:
        raise ValueError("--waist-frame-stride must be >= 1")
    if args.waist_stack_plan_ik_stride < 1:
        raise ValueError("--waist-stack-plan-ik-stride must be >= 1")

    order = build_stack_order()
    second_layer_start = GRID_X * GRID_Y + 1
    third_layer_start = GRID_X * GRID_Y * 2 + 1
    if args.third_fourth_proxy:
        native_cells = []
    elif args.second_layer_proxy:
        native_cells = list(order[second_layer_start - 1 : third_layer_start - 1])
    else:
        native_cells = list(order[: third_layer_start - 1])
    unsupported = [cell for cell in native_cells if not _supports_direct_box_plan(cell)]
    if unsupported:
        first = unsupported[0]
        raise NotImplementedError(f"{first.label} requires a strategy that is not available in task1.py")

    plans = [_delay_second_layer_release_until_hold_end(build_direct_box_plan(cell, DIRECT_STAND_OFF)) for cell in native_cells]
    for plan in plans:
        if plan.pick_max_error > PICK_ERROR_LIMIT or plan.place_max_error > PLACE_ERROR_LIMIT:
            raise RuntimeError(
                f"IK infeasible for {plan.cell.label}: pick_error={plan.pick_max_error:.4f}, "
                f"place_error={plan.place_max_error:.4f}"
            )

    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    robot_asset = load_robot_asset(gym, sim)
    asset_dof_names = list(gym.get_asset_dof_names(robot_asset))
    plans = [_reorder_box_plan(plan, asset_dof_names) for plan in plans]
    timelines = [build_task1_2_timeline(plan) for plan in plans]
    if args.second_layer_proxy:
        waist_args, waist_plans = None, []
    else:
        waist_args, waist_plans = _build_waist_stack_plans(args, asset_dof_names)
    box_sequences = [plan.cell.sequence for plan in plans] + [plan.cell.sequence for plan in waist_plans]

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.8, 6.8, 3.0), 1)
    proxy_layers = 2 if args.third_fourth_proxy else 1 if args.second_layer_proxy else 0
    robot, boxes = _create_static_scene(gym, sim, env, robot_asset, box_sequences, proxy_layers=proxy_layers)
    if timelines:
        _configure_robot_dofs(gym, env, robot, asset_dof_names)
        _set_robot_initial_q(gym, env, robot, timelines[0][0][2])
    else:
        _configure_waist_stack_robot_dofs(gym, env, robot, asset_dof_names)
        _set_robot_initial_q(gym, env, robot, waist_plans[0].timeline[0].q)

    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    robot_index = gym.get_actor_index(env, robot, gymapi.DOMAIN_SIM)
    box_indices = [gym.get_actor_index(env, box, gymapi.DOMAIN_SIM) for box in boxes]

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(PALLET_CENTER[0] + 1.8, PALLET_CENTER[1] - 2.2, 1.6),
            gymapi.Vec3(PALLET_CENTER[0], PALLET_CENTER[1], 0.55),
        )

    print("task1 complete 60-box stacking demo")
    print(f"  stack_size={STACK_SIZE} grid=({GRID_X},{GRID_Y},{GRID_Z}) box_size={STACK_BOX_SIZE}")
    if args.second_layer_proxy:
        print("  mode=second_layer_debug native_boxes=16..30 first_layer_proxy=(1.0,1.0,0.4)")
    elif args.third_fourth_proxy:
        print("  mode=third_fourth_debug waist_boxes=31..60 lower_proxy=(1.0,1.0,0.8)")
    else:
        print("  mode=complete_60 native_boxes=1..30 waist_boxes=31..60")
    print(
        f"  speed headless={args.headless} fast={args.fast} fast_viewer={args.fast_viewer} "
        f"viewer_render_every={args.viewer_render_every} viewer_start_box={args.viewer_start_box} "
        f"waist_frame_stride={args.waist_frame_stride} waist_ik_stride={args.waist_stack_plan_ik_stride}"
    )
    for plan, timeline in zip(plans, timelines):
        print(
            f"  {plan.cell.label} target=({plan.target_center[0]:.3f},{plan.target_center[1]:.3f},{plan.target_center[2]:.3f}) "
            f"ik_errors pick={plan.pick_max_error:.4f} place={plan.place_max_error:.4f} frames={len(timeline)}"
        )
    for plan in waist_plans:
        meta = plan.metadata
        print(
            f"  waist:{plan.cell.label} side={meta['side']} stand_off={float(meta['stand_off']):.3f} "
            f"target=({meta['target_center'][0]:.3f},{meta['target_center'][1]:.3f},{meta['target_center'][2]:.3f}) "
            f"release=({meta['release_center'][0]:.3f},{meta['release_center'][1]:.3f},{meta['release_center'][2]:.3f}) "
            f"arm_pos={float(meta['max_pos_error']):.4f} frames={len(plan.timeline)}"
        )

    global_frame = 0
    placed: list[PlacedBoxPose] = []
    completed_errors: list[float] = []
    completed_native = 0
    completed_waist = 0
    native_box_count = len(plans)
    native_boxes = boxes[:native_box_count]
    native_box_indices = box_indices[:native_box_count]
    waist_boxes = boxes[native_box_count:]
    waist_box_indices = box_indices[native_box_count:]
    keep_running = True
    try:
        for plan, timeline, box, box_index in zip(plans, timelines, native_boxes, native_box_indices):
            remaining_budget = 0 if args.max_frames == 0 else max(0, args.max_frames - global_frame)
            if args.max_frames > 0 and remaining_budget == 0:
                keep_running = False
                break
            print(f"task1_start_box box={plan.cell.sequence} label={plan.cell.label}")
            global_frame, keep_running, box_error = _run_box_timeline(
                gym,
                sim,
                env,
                robot,
                box,
                root_states,
                robot_index,
                box_index,
                plan,
                timeline,
                placed,
                viewer,
                args.viewer_render_every,
                plan.cell.sequence >= args.viewer_start_box,
                not args.fast_viewer,
                remaining_budget,
                ATTACH_AFTER_PICK_FRAMES,
                global_frame,
            )
            if not keep_running:
                break
            gym.refresh_actor_root_state_tensor(sim)
            _report_placed_box_motion(root_states, placed, f"after_box_{plan.cell.sequence}")
            completed_errors.append(box_error)
            completed_native += 1

        if keep_running and waist_plans:
            _configure_waist_stack_robot_dofs(gym, env, robot, asset_dof_names)
            _set_robot_initial_q(gym, env, robot, waist_plans[0].timeline[0].q)
            for plan, box, box_index in zip(waist_plans, waist_boxes, waist_box_indices):
                if args.max_frames > 0 and global_frame >= args.max_frames:
                    keep_running = False
                    break
                print(f"task1_waist_start_box box={plan.cell.sequence} label={plan.cell.label}")
                global_frame, keep_running = _run_task1_2_waist_stack_box_timeline(
                    gym,
                    sim,
                    env,
                    robot,
                    box,
                    root_states,
                    robot_index,
                    box_index,
                    plan,
                    viewer,
                    waist_args,
                    global_frame,
                    args.max_frames,
                )
                if not keep_running:
                    break
                completed_waist += 1
                gym.refresh_actor_root_state_tensor(sim)
                _report_placed_box_motion(root_states, placed, f"after_waist_box_{plan.cell.sequence}")
    finally:
        if "sim" in locals() and sim is not None and "root_states" in locals() and "placed" in locals():
            gym.refresh_actor_root_state_tensor(sim)
            _report_placed_box_motion(root_states, placed, "final")
        completed = completed_native + completed_waist
        total_plans = len(plans) + len(waist_plans)
        print(
            "task1_result "
            f"completed={completed}/{total_plans} "
            f"native_completed={completed_native}/{len(plans)} "
            f"waist_completed={completed_waist}/{len(waist_plans)} "
            f"max_release_error={max(completed_errors) if completed_errors else 0.0:.4f} "
            f"mean_release_error={sum(completed_errors) / len(completed_errors) if completed_errors else 0.0:.4f}"
        )
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
