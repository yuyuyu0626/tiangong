"""集成流水线 — 将 BPP 放置决策与机器人 IK 执行串联。

``MoveProIntegrator`` 是核心编排类，完成一次完整的智能码垛流程：

1. 对每个大小不同的箱子调用 ``BPPDecider.decide()`` 决定放置位置
2. 使用 move 的 IK 工具链生成抓取 → 移动 → 放置关节轨迹
3. 输出完整的运动计划供离线验证或仿真播放

使用示例::

    from move_pro.integrator import MoveProIntegrator

    integrator = MoveProIntegrator(method="LSAH")
    plan = integrator.build_plan(box_sequence)
    print(plan.summary())
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

# 确保 move 项目在 sys.path 中
_MOVE_ROOT = Path(__file__).resolve().parents[1] / "move"
if str(_MOVE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_MOVE_ROOT.parent))

from move.utils import (                                 # noqa: E402
    IKFlipBoxController,
    UrdfKinematics,
    CartesianKeyframe,
    APPROACH_GAP_READY,
    APPROACH_GAP_PRE,
    APPROACH_GAP_GRASP,
    PALM_BOX_CLEARANCE,
    PALM_CENTER_Z,
    PALM_SURFACE_X,
    LEFT_EE_LINK,
    RIGHT_EE_LINK,
)
from move.mobile_ik import MoveLiftIK, target_box_center_z  # noqa: E402
from move.planning import (                                # noqa: E402
    PALLET_SURFACE_Z,
    TABLE_POSE,
    TABLE_SIZE,
    PALLET_THICKNESS,
    SAFE_CORRIDOR_MARGIN,
    Pose,
    BoxPlacement,
    StackScene,
    pallet_center_near_table,
    sample_root_trajectory,
)
from move.grab_test import (                               # noqa: E402
    _parse_active_dofs,
    _make_keyframe,
    _solve_keyframes,
    _grasp_center_from_q,
    _box_theta_from_q,
    _world_to_root,
    _pose_path_key_indices,
    _expand_targets_for_pose_path,
    GRASP_CONTACT_X_OFFSET,
    MOVE_APPROACH_HEIGHT,
    MOVE_PREPLACE_HEIGHT,
    MOVE_URDF,
    PICK_LIFT_HEIGHT,
    PICK_READY_Z_OFFSET,
    PICK_SIDE_CONTACT_Z_RATIO,
    PICK_TOUCH_GAP,
    PICK_COMPRESS_GAP,
    PLACE_CONTACT_GAP,
    PLACE_HOLD_FRAMES,
    PLACE_UPRIGHT_FRAMES,
    PLACE_RELEASE_HEIGHT,
    PLACE_XY_SEGMENT_FRAMES,
    PLACE_DESCEND_FRAMES,
)

from move_pro.bpp_decider import BPPDecider, Placement     # noqa: E402
from move_pro.config import (                              # noqa: E402
    MOVE_STAND_OFF,
    STACK_TMP_RETREAT,
    TABLE_RETREAT,
    DEFAULT_ROOT_Z,
    DEFAULT_HEIGHT_CLEARANCE,
    pallet_center_world,
)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class BoxTask:
    """单个箱子的完整任务描述。"""
    index: int
    original_size: tuple         # 原始尺寸 (sx, sy, sz)
    pct_size: tuple              # 映射到 PCT 坐标系的尺寸
    placement: Placement         # BPP 决策结果
    world_target: tuple          # 世界坐标放置中心
    pick_q: Optional[torch.Tensor] = None      # 抓取姿态关节角
    move_q: Optional[torch.Tensor] = None      # 移动抬升关节角
    place_targets: Optional[torch.Tensor] = None  # 放置轨迹 [N, num_dofs]
    root_route: list = field(default_factory=list)  # root 路线


@dataclass
class MoveProPlan:
    """完整码垛计划。"""
    box_tasks: list[BoxTask] = field(default_factory=list)
    total_boxes: int = 0
    placed_boxes: int = 0
    utilization: float = 0.0
    ik_feasible_count: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "  MovePro 智能码垛计划",
            "=" * 72,
            f"  总箱数: {self.total_boxes}",
            f"  成功放置: {self.placed_boxes}",
            f"  最终利用率: {self.utilization:.2%}",
            f"  IK 可行: {self.ik_feasible_count}/{self.total_boxes}",
        ]
        for bt in self.box_tasks:
            status = "✓" if bt.placement.feasible else "✗"
            lines.append(
                f"  [{status}] 箱#{bt.index} {bt.original_size} "
                f"→ world({bt.world_target[0]:.3f}, {bt.world_target[1]:.3f}, "
                f"{bt.world_target[2]:.3f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 集成器
# ---------------------------------------------------------------------------

class MoveProIntegrator:
    """智能码垛集成器。

    将 Online-3D-BPP-PCT 的装箱决策与 move 的机器人 IK/运动规划
    串联为完整的码垛流水线。

    Parameters
    ----------
    method : str
        BPP 启发式方法，与 ``BPPDecider`` 相同。
    container_size : tuple
        PCT 容器尺寸。
    stand_off : float
        机器人站位距托盘的距离 (m)。
    urdf_path : Path
        URDF 模型路径，默认使用 move 的 move-state 模型。
    """
    def __init__(
        self,
        method: str = "LSAH",
        container_size: tuple = (10, 10, 16),
        stand_off: float = 0.55,
        urdf_path: Optional[Path] = None,
    ):
        self.method = method
        self.container_size = container_size
        self.stand_off = stand_off
        self.urdf_path = urdf_path or MOVE_URDF

        # BPP 决策器
        self.decider = BPPDecider(method=method, container_size=container_size)

        # URDF 与 DOF
        dof_names, lower, upper = _parse_active_dofs(self.urdf_path)
        self.dof_names = dof_names
        self.lower = lower
        self.upper = upper

        # IK 控制器（延迟初始化，每次 build_plan 重新创建）
        self._pick_controller: Optional[IKFlipBoxController] = None
        self._place_controller: Optional[IKFlipBoxController] = None
        self._lift_ik: Optional[MoveLiftIK] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build_plan(self, box_sequence: Sequence[tuple],
                   compute_ik: bool = True,
                   sizes_are_pct: bool = False) -> MoveProPlan:
        """为箱子序列构建完整码垛计划。

        Parameters
        ----------
        box_sequence : 尺寸序列 [(sx, sy, sz), ...]
            每个箱子的尺寸。
        compute_ik : bool
            是否计算 IK。False 时仅做 BPP 决策，快速预览。
        sizes_are_pct : bool
            True 表示尺寸已经是 PCT 坐标系，跳过单位转换。
        """
        self.decider.reset()
        plan = MoveProPlan()
        plan.total_boxes = len(box_sequence)

        # 初始化 IK 控制器
        if compute_ik:
            self._init_ik_controllers()

        for i, box_input_size in enumerate(box_sequence):
            # 1. 尺寸单位转换（如需要）
            if sizes_are_pct:
                pct_size = box_input_size
            else:
                pct_size = self._world_size_to_pct(box_input_size)

            # 2. BPP 决策
            placement = self.decider.decide(pct_size)
            if not placement.feasible:
                bt = BoxTask(
                    index=i, original_size=box_input_size,
                    pct_size=pct_size, placement=placement,
                    world_target=(0, 0, 0),
                )
                plan.box_tasks.append(bt)
                continue

            self.decider.commit(placement)
            world_target = (placement.world_x, placement.world_y,
                            placement.world_z)

            # 3. IK 求解
            bt = BoxTask(
                index=i, original_size=box_input_size,
                pct_size=pct_size, placement=placement,
                world_target=world_target,
            )

            if compute_ik and self._pick_controller is not None:
                bt = self._solve_ik_for_box(bt)

            plan.box_tasks.append(bt)
            plan.placed_boxes += 1

        plan.utilization = float(self.decider.utilization())
        if compute_ik:
            plan.ik_feasible_count = sum(
                1 for bt in plan.box_tasks if bt.pick_q is not None
            )
        return plan

    def reset(self) -> None:
        """重置所有内部状态。"""
        self.decider.reset()
        self._pick_controller = None
        self._place_controller = None
        self._lift_ik = None

    # ------------------------------------------------------------------
    # 坐标映射
    # ------------------------------------------------------------------

    def _world_size_to_pct(self, world_size: tuple) -> tuple:
        """世界尺寸 (m) → PCT 坐标系尺寸。"""
        cs_x, cs_y, cs_z = self.container_size
        from move_pro.config import BIN_TO_WORLD_SCALE
        sx, sy, sz = BIN_TO_WORLD_SCALE
        return (
            int(round(world_size[0] / sx * cs_x)),
            int(round(world_size[1] / sy * cs_y)),
            int(round(world_size[2] / sz * cs_z)),
        )

    # ------------------------------------------------------------------
    # IK 求解
    # ------------------------------------------------------------------

    def _init_ik_controllers(self) -> None:
        """初始化 IK 控制器。"""
        self._pick_controller = IKFlipBoxController(
            self.dof_names, self.lower, self.upper,
            self.urdf_path,
            (0.50, 0.0, 0.85),  # box_pose (will be overridden per box)
            (0.20, 0.20, 0.20),  # box_size placeholder
        )
        self._lift_ik = MoveLiftIK(
            self.urdf_path, self.dof_names, self.lower, self.upper
        )

    def _solve_ik_for_box(self, bt: BoxTask) -> BoxTask:
        """为单个箱子求解完整的抓取→移动→放置 IK 链。"""
        p = bt.placement
        box_world = bt.original_size

        # ---- 阶段 1: 抓取 IK ----
        # 世界坐标系中的源箱中心（桌面上的位置）
        source_center = np.array([0.50, 0.0, 0.73], dtype=np.float64)
        box_size_m = (float(box_world[0]), float(box_world[1]),
                      float(box_world[2]))

        # 构造抓取关键帧
        pick_contact_z = box_size_m[2] * PICK_SIDE_CONTACT_Z_RATIO
        pick_ready_z = source_center[2] + PICK_READY_Z_OFFSET
        pick_lift_z = source_center[2] + PICK_LIFT_HEIGHT

        pick_frames = [
            _make_keyframe("pick_ready", source_center + np.array([0, 0, PICK_READY_Z_OFFSET]),
                           box_size_m, 0.0, APPROACH_GAP_READY, pick_contact_z + 0.02,
                           GRASP_CONTACT_X_OFFSET),
            _make_keyframe("pick_grasp", source_center,
                           box_size_m, 0.0, PICK_TOUCH_GAP, pick_contact_z,
                           GRASP_CONTACT_X_OFFSET),
            _make_keyframe("pick_compress", source_center,
                           box_size_m, 0.0, PICK_COMPRESS_GAP, pick_contact_z,
                           GRASP_CONTACT_X_OFFSET),
            _make_keyframe("pick_lift", source_center + np.array([0, 0, PICK_LIFT_HEIGHT]),
                           box_size_m, 0.0, PICK_COMPRESS_GAP, pick_contact_z,
                           GRASP_CONTACT_X_OFFSET),
        ]

        try:
            self._pick_controller._solve_keyframe(
                pick_frames[0], iterations=80, regularize_wrist=True
            )
            pick_targets, _ = _solve_keyframes(
                self._pick_controller, pick_frames, iterations=16
            )
            bt.pick_q = pick_targets[-1]
        except Exception:
            return bt

        # ---- 阶段 2: 放置站位与抬升 IK ----
        world_target = bt.world_target
        # 机器人站在托盘左侧 (-X 方向)
        final_x = pallet_center_world()[0] - 0.5 - self.stand_off
        final_y = world_target[1]
        yaw = math.atan2(world_target[1] - final_y, world_target[0] - final_x)
        final_pose = Pose(float(final_x), float(final_y), 0.0, yaw)

        target_support_z = PALLET_SURFACE_Z + p.lz * 0.10  # approx
        desired_z = target_box_center_z(target_support_z, box_size_m[2])
        move_lift_q = self._lift_ik.solve_for_box_center_z(
            bt.pick_q, float(desired_z),
        )
        bt.move_q = move_lift_q

        # ---- 阶段 3: 放置 IK ----
        place_controller = IKFlipBoxController(
            self.dof_names, self.lower, self.upper,
            self.urdf_path, (0.0, 0.0, 0.0), box_size_m,
        )
        place_controller.q = move_lift_q.detach().cpu().numpy().astype(np.float64).copy()

        place_contact_z = box_size_m[2] * 0.18  # BOX_SIDE_CONTACT_Z_RATIO
        # 放置路径：当前位置 → release 上方 → release
        place_keyframes = [
            _make_keyframe("place_handoff",
                           np.array(world_target, dtype=np.float64),
                           box_size_m, 0.0, PLACE_CONTACT_GAP, place_contact_z,
                           GRASP_CONTACT_X_OFFSET),
            _make_keyframe("place_release",
                           np.array([world_target[0], world_target[1],
                                     world_target[2] + PLACE_RELEASE_HEIGHT],
                                    dtype=np.float64),
                           box_size_m, 0.0, APPROACH_GAP_READY,
                           place_contact_z + 0.02,
                           GRASP_CONTACT_X_OFFSET),
        ]

        try:
            for kf in place_keyframes:
                local_center = _world_to_root(
                    (float(kf.left_pos[0] + kf.right_pos[0]) * 0.5,
                     float(kf.left_pos[1] + kf.right_pos[1]) * 0.5,
                     float(kf.left_pos[2] + kf.right_pos[2]) * 0.5),
                    final_pose,
                )
                # re-make in root-local coords
                local_frame = _make_keyframe(
                    kf.name, local_center, box_size_m, 0.0,
                    PLACE_CONTACT_GAP if "handoff" in kf.name else APPROACH_GAP_READY,
                    place_contact_z if "handoff" in kf.name else place_contact_z + 0.02,
                    GRASP_CONTACT_X_OFFSET,
                )
                place_controller._solve_keyframe(local_frame, iterations=80)
            bt.place_targets = torch.tensor(place_controller.q, dtype=torch.float32)
        except Exception:
            pass

        # ---- 阶段 4: Root 路线 ----
        start = Pose(0.0, 0.0, 0.0, 0.0)
        # 从桌边退避
        dx = start.x - TABLE_POSE[0]
        dy = start.y - TABLE_POSE[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            dx, dy, length = -1.0, 0.0, 1.0
        retreat_pose = Pose(
            start.x + dx / length * TABLE_RETREAT,
            start.y + dy / length * TABLE_RETREAT,
            start.z, start.yaw,
        )
        # 安全角
        pc = pallet_center_world()
        safe_pose = Pose(
            pc[0] - 0.5 - SAFE_CORRIDOR_MARGIN,
            pc[1] - 0.5 - SAFE_CORRIDOR_MARGIN,
            0.0, 0.0,
        )
        # 临时站位
        tmp_pose = Pose(final_pose.x - STACK_TMP_RETREAT, final_pose.y, 0.0, final_pose.yaw)

        waypoints = [
            ("table_start", start),
            ("table_retreat", retreat_pose),
            ("safe_corner", safe_pose),
            ("tmp", tmp_pose),
            ("target", final_pose),
        ]
        route = []
        for (_, wp1), (_, wp2) in zip(waypoints, waypoints[1:]):
            for rp in sample_root_trajectory(wp1, wp2):
                route.append(rp)
        bt.root_route = route

        return bt


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

def demo() -> None:
    """离线快速演示：BPP 决策 + 放置位置预览。"""
    import random
    random.seed(123)

    from move_pro.config import DEFAULT_ITEM_SET
    boxes = [random.choice(DEFAULT_ITEM_SET) for _ in range(25)]

    integrator = MoveProIntegrator(method="LSAH")
    # DEFAULT_ITEM_SET 中的尺寸是 PCT 坐标系，无需单位转换
    plan = integrator.build_plan(boxes, compute_ik=False, sizes_are_pct=True)
    print(plan.summary())


if __name__ == "__main__":
    demo()
