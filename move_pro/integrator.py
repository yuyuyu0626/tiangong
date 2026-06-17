"""Thin integration layer between PCT decisions and the move task code.

This module intentionally contains no robot implementation. Motion planning,
IK, actor creation, timelines, and simulation remain in the files copied from
``move``. The only new responsibility here is converting incoming box sizes
into online PCT placements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from move_pro.bpp_decider import BPPDecider, Placement
from move_pro.config import BIN_TO_WORLD_SCALE, MOVE_URDF


@dataclass(frozen=True)
class BoxTask:
    index: int
    original_size: tuple[float, float, float]
    pct_size: tuple[int, int, int]
    world_size: tuple[float, float, float]
    placement: Placement

    @property
    def world_target(self) -> tuple[float, float, float]:
        return self.placement.world_center


@dataclass
class MoveProPlan:
    box_tasks: list[BoxTask] = field(default_factory=list)
    total_boxes: int = 0
    placed_boxes: int = 0
    utilization: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "MovePro online packing plan",
            "=" * 72,
            f"boxes: {self.placed_boxes}/{self.total_boxes}",
            f"utilization: {self.utilization:.2%}",
        ]
        for task in self.box_tasks:
            if not task.placement.feasible:
                lines.append(f"[failed] box {task.index}: {task.original_size}")
                continue
            x, y, z = task.world_target
            lines.append(
                f"[ok] box {task.index}: size={task.world_size} "
                f"target=({x:.3f}, {y:.3f}, {z:.3f}) "
                f"rotation={task.placement.orientation}"
            )
        return "\n".join(lines)


class MoveProIntegrator:
    """Build an online packing plan without duplicating move's robot code."""

    def __init__(
        self,
        method: str = "LASH",
        container_size: tuple[int, int, int] = (10, 10, 16),
        orientation: int = 2,
        stand_off: float = 0.55,
        urdf_path: Path | None = None,
        setting: int = 2,
    ) -> None:
        self.method = method
        self.container_size = container_size
        self.orientation = orientation
        self.stand_off = stand_off
        self.urdf_path = urdf_path or MOVE_URDF
        self.use_pct_model = method == "PCT"
        if self.use_pct_model:
            # PCT 模型路径：决策走训练好的 GAT 策略（pct_policy），容器/尺度由 config 的 PCT_* 决定。
            from move_pro.pct_policy import PctModelPlanner

            self.planner = PctModelPlanner()
            self.decider = None
        else:
            self.planner = None
            self.decider = BPPDecider(
                method=method,
                container_size=container_size,
                orientation=orientation,
                setting=setting,
            )

    def build_plan(
        self,
        box_sequence: Sequence[tuple[float, float, float]],
        compute_ik: bool = False,
        sizes_are_pct: bool = False,
    ) -> MoveProPlan:
        if compute_ik:
            raise ValueError(
                "IK is executed by the copied move task implementation, "
                "not by the PCT decision adapter"
            )
        if self.use_pct_model:
            return self._build_plan_pct(box_sequence)
        self.decider.reset()
        plan = MoveProPlan(total_boxes=len(box_sequence))
        for index, input_size in enumerate(box_sequence):
            pct_size = (
                self._validate_pct_size(input_size)
                if sizes_are_pct
                else self._world_size_to_pct(input_size)
            )
            placement = self.decider.decide_and_commit(pct_size)
            oriented_pct_size = (
                (placement.x, placement.y, placement.z)
                if placement.feasible
                else pct_size
            )
            world_size = self._pct_size_to_world(oriented_pct_size)
            plan.box_tasks.append(
                BoxTask(
                    index=index,
                    original_size=tuple(float(v) for v in input_size),
                    pct_size=pct_size,
                    world_size=world_size,
                    placement=placement,
                )
            )
            if placement.feasible:
                plan.placed_boxes += 1
        plan.utilization = self.decider.utilization()
        return plan

    def _build_plan_pct(self, box_sequence) -> MoveProPlan:
        """用 PCT 训练好的模型做决策，把每个放置包成 BoxTask。

        box_sequence 是整数 bin 尺寸序列（PCT 物品集 1..5）。模型按序放，放不下即停，
        放置数可能少于输入箱数（剩余箱视为未放）。
        """
        from move_pro.config import PCT_BIN_TO_WORLD_SCALE

        sequence = [tuple(int(v) for v in s) for s in box_sequence]
        placements = self.planner.plan(box_sequence=sequence)
        plan = MoveProPlan(total_boxes=len(sequence))
        for index, p in enumerate(placements):
            world_size = (
                p.x * PCT_BIN_TO_WORLD_SCALE[0],
                p.y * PCT_BIN_TO_WORLD_SCALE[1],
                p.z * PCT_BIN_TO_WORLD_SCALE[2],
            )
            plan.box_tasks.append(
                BoxTask(
                    index=index,
                    original_size=(float(p.x), float(p.y), float(p.z)),
                    pct_size=(p.x, p.y, p.z),
                    world_size=world_size,
                    placement=self.planner.as_placement(p),
                )
            )
            plan.placed_boxes += 1
        plan.utilization = self.planner.utilization()
        return plan

    def reset(self) -> None:
        if self.use_pct_model:
            self.planner.reset()
        else:
            self.decider.reset()

    def _validate_pct_size(self, size) -> tuple[int, int, int]:
        if len(size) != 3:
            raise ValueError("each box size must have three dimensions")
        result = tuple(int(value) for value in size)
        if any(value <= 0 for value in result):
            raise ValueError("box dimensions must be positive")
        if any(float(raw) != value for raw, value in zip(size, result)):
            raise ValueError("PCT dimensions must be integers")
        return result

    def _world_size_to_pct(self, size) -> tuple[int, int, int]:
        if len(size) != 3 or any(float(value) <= 0 for value in size):
            raise ValueError("world box dimensions must be three positive values")
        result = tuple(
            int(round(float(value) / scale * container))
            for value, scale, container in zip(
                size, BIN_TO_WORLD_SCALE, self.container_size
            )
        )
        if any(value <= 0 for value in result):
            raise ValueError("box is too small for the configured PCT resolution")
        return result

    def _pct_size_to_world(self, size) -> tuple[float, float, float]:
        return tuple(
            value / container * scale
            for value, container, scale in zip(
                size, self.container_size, BIN_TO_WORLD_SCALE
            )
        )


def demo() -> None:
    from move_pro.config import DEFAULT_ITEM_SET

    plan = MoveProIntegrator().build_plan(
        DEFAULT_ITEM_SET[:10], sizes_are_pct=True
    )
    print(plan.summary())


if __name__ == "__main__":
    demo()
