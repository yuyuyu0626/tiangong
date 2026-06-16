"""Online placement decisions backed by the original PCT packing space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from move_pro.config import (
    BIN_TO_WORLD_SCALE,
    DEFAULT_ITEM_SET,
    MAX_REACH_X_BIN,
    PALLET_SURFACE_Z,
    pallet_center_world,
)
from move_pro.pct_core import Space


@dataclass(frozen=True)
class Placement:
    """One PCT placement and its corresponding robot-world target."""

    lx: int
    ly: int
    lz: int
    x: int
    y: int
    z: int
    orientation: int
    world_x: float
    world_y: float
    world_z: float
    feasible: bool = True
    score: float = 0.0

    @property
    def world_center(self) -> tuple[float, float, float]:
        return (self.world_x, self.world_y, self.world_z)


class BPPDecider:
    """Stateful online 3D bin-packing decider.

    Candidate generation and feasibility checks use the original
    ``PctDiscrete0.space.Space`` implementation. The heuristic selection
    follows the LASH, OnlineBPH, DBL, and BR baselines in PCT's
    ``heuristic.py``.
    """

    _ORIENTATIONS = (
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 1),
        (2, 1, 0),
        (0, 2, 1),
        (1, 2, 0),
    )

    def __init__(
        self,
        method: str = "LASH",
        container_size: tuple[int, int, int] = (10, 10, 16),
        orientation: int = 2,
        continuous: bool = False,
        item_set: Optional[Iterable[tuple[int, int, int]]] = None,
        setting: int = 2,
        max_reach_x_bin: Optional[int] = MAX_REACH_X_BIN,
    ) -> None:
        method = "LASH" if method == "LSAH" else method
        if method not in {"LASH", "OnlineBPH", "DBL", "BR"}:
            raise ValueError(f"unknown heuristic method: {method}")
        if continuous:
            raise NotImplementedError("move_pro currently supports discrete PCT packing")
        if orientation not in (1, 2, 6):
            raise ValueError("orientation must be 1, 2, or 6")
        if setting not in (1, 2, 3):
            raise ValueError("setting must be 1, 2, or 3")

        self.container_size = self._positive_int_tuple(container_size, "container_size")
        self.item_set = tuple(
            self._positive_int_tuple(size, "item size")
            for size in (item_set or DEFAULT_ITEM_SET)
        )
        self.method = method
        self.orientation = orientation
        self.setting = setting
        self.max_reach_x_bin = max_reach_x_bin
        size_minimum = min(min(size) for size in self.item_set)
        self.space = Space(*self.container_size, size_minimum, holder=200)
        self.packed_count = 0
        self._packed_boxes: list[dict] = []
        self._reset_bounds()

    @staticmethod
    def _positive_int_tuple(values, name: str) -> tuple[int, int, int]:
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly three dimensions")
        result = tuple(int(value) for value in values)
        if any(value <= 0 for value in result):
            raise ValueError(f"{name} dimensions must be positive")
        if any(float(raw) != value for raw, value in zip(values, result)):
            raise ValueError(f"{name} dimensions must be integers")
        return result

    def _reset_bounds(self) -> None:
        self._max_xy = [0, 0]
        self._min_xy = [self.container_size[0], self.container_size[1]]

    def reset(self) -> None:
        self.space.reset()
        self.packed_count = 0
        self._packed_boxes.clear()
        self._reset_bounds()

    def decide(self, box_size: tuple[int, int, int]) -> Placement:
        box = self._positive_int_tuple(box_size, "box_size")
        if self.method == "LASH":
            return self._decide_lash(box)
        if self.method == "OnlineBPH":
            return self._decide_online_bph(box)
        if self.method == "DBL":
            return self._decide_dbl(box)
        return self._decide_br(box)

    def commit(self, placement: Placement) -> None:
        if not placement.feasible:
            raise ValueError("cannot commit an infeasible placement")
        size = (placement.x, placement.y, placement.z)
        position = (placement.lx, placement.ly)
        if not self.space.drop_box(size, position, False, 1.0, self.setting):
            raise RuntimeError("PCT rejected a placement that was previously feasible")
        self.space.GENEMS(
            [
                placement.lx,
                placement.ly,
                placement.lz,
                placement.lx + placement.x,
                placement.ly + placement.y,
                placement.lz + placement.z,
            ]
        )
        self._max_xy[0] = max(self._max_xy[0], placement.lx + placement.x)
        self._max_xy[1] = max(self._max_xy[1], placement.ly + placement.y)
        self._min_xy[0] = min(self._min_xy[0], placement.lx)
        self._min_xy[1] = min(self._min_xy[1], placement.ly)
        self.packed_count += 1
        self._packed_boxes.append(
            {
                "lx": placement.lx,
                "ly": placement.ly,
                "lz": placement.lz,
                "x": placement.x,
                "y": placement.y,
                "z": placement.z,
                "orientation": placement.orientation,
                "world": placement.world_center,
            }
        )

    def decide_and_commit(self, box_size: tuple[int, int, int]) -> Placement:
        placement = self.decide(box_size)
        if placement.feasible:
            self.commit(placement)
        return placement

    def to_world(
        self,
        lx: int,
        ly: int,
        lz: int,
        x: int,
        y: int,
        z: int,
    ) -> tuple[float, float, float]:
        center_x, center_y, _ = pallet_center_world()
        scale_x, scale_y, scale_z = BIN_TO_WORLD_SCALE
        bin_x, bin_y, bin_z = self.container_size
        return (
            center_x - scale_x / 2.0 + (lx + x / 2.0) / bin_x * scale_x,
            center_y - scale_y / 2.0 + (ly + y / 2.0) / bin_y * scale_y,
            PALLET_SURFACE_Z + (lz + z / 2.0) / bin_z * scale_z,
        )

    def utilization(self) -> float:
        return float(self.space.get_ratio())

    @property
    def packed_boxes(self) -> tuple[dict, ...]:
        return tuple(self._packed_boxes)

    @property
    def ems_list(self) -> tuple[np.ndarray, ...]:
        return tuple(np.array(ems, copy=True) for ems in self.space.EMS)

    def _oriented_sizes(self, box):
        seen = set()
        for index, permutation in enumerate(self._ORIENTATIONS[: self.orientation]):
            size = tuple(box[axis] for axis in permutation)
            if size in seen:
                continue
            seen.add(size)
            yield index, size

    def _virtual(self, size, lx: int, ly: int):
        feasible, height = self.space.drop_box_virtual(
            list(size), (int(lx), int(ly)), False, 1.0, self.setting, returnH=True
        )
        return bool(feasible), int(height)

    def _reachable(self, lx: int, x: int) -> bool:
        """箱子远端边缘是否在机器人可达深度内。

        机器人正对 +x 站在托盘近端，手臂前伸有限。lx+x 是箱子远端边缘的
        bin 坐标；超过 max_reach_x_bin 的格子机器人够不到（见 config 注释）。
        """
        if self.max_reach_x_bin is None:
            return True
        return lx + x <= self.max_reach_x_bin

    def _placement(self, candidate, score: float) -> Placement:
        lx, ly, lz, x, y, z, orientation = candidate
        world = self.to_world(lx, ly, lz, x, y, z)
        return Placement(lx, ly, lz, x, y, z, orientation, *world, score=score)

    def _empty(self, box) -> Placement:
        return Placement(0, 0, 0, *box, 0, 0.0, 0.0, 0.0, feasible=False, score=-1.0)

    def _ems_candidates(self, box):
        for ems in self.space.EMS:
            for orientation, (x, y, z) in self._oriented_sizes(box):
                if x > ems[3] - ems[0] or y > ems[4] - ems[1] or z > ems[5] - ems[2]:
                    continue
                lx, ly = int(ems[0]), int(ems[1])
                if not self._reachable(lx, x):
                    continue
                feasible, height = self._virtual((x, y, z), lx, ly)
                if feasible:
                    yield (lx, ly, height, x, y, z, orientation), ems

    def _decide_lash(self, box) -> Placement:
        best = None
        best_score = float("inf")
        best_fit = float("inf")
        for candidate, ems in self._ems_candidates(box):
            lx, ly, height, x, y, z, _ = candidate
            max_x = max(lx + x, self._max_xy[0])
            min_x = min(lx, self._min_xy[0])
            max_y = max(ly + y, self._max_xy[1])
            min_y = min(ly, self._min_xy[1])
            score = (
                (max_x - min_x) * (max_y - min_y)
                + (height + z) * (max_y - min_y)
                + (height + z) * (max_x - min_x)
            )
            fit = min(ems[3] - ems[0] - x, ems[4] - ems[1] - y, ems[5] - ems[2] - z)
            if score < best_score or (score == best_score and fit < best_fit):
                best, best_score, best_fit = candidate, score, fit
        return self._empty(box) if best is None else self._placement(best, -float(best_score))

    def _decide_online_bph(self, box) -> Placement:
        for ems in sorted(self.space.EMS, key=lambda value: (value[2], value[1], value[0])):
            for orientation, size in self._oriented_sizes(box):
                lx, ly = int(ems[0]), int(ems[1])
                if not self._reachable(lx, size[0]):
                    continue
                feasible, height = self._virtual(size, lx, ly)
                if feasible:
                    return self._placement((lx, ly, height, *size, orientation), 0.0)
        return self._empty(box)

    def _decide_dbl(self, box) -> Placement:
        best = None
        best_score = float("inf")
        bin_x, bin_y, _ = self.container_size
        for orientation, (x, y, z) in self._oriented_sizes(box):
            for lx in range(bin_x - x + 1):
                if not self._reachable(lx, x):
                    continue
                for ly in range(bin_y - y + 1):
                    feasible, height = self._virtual((x, y, z), lx, ly)
                    score = lx + ly + 100 * height
                    if feasible and score < best_score:
                        best = (lx, ly, height, x, y, z, orientation)
                        best_score = score
        return self._empty(box) if best is None else self._placement(best, -float(best_score))

    def _decide_br(self, box) -> Placement:
        def rank(ems) -> float:
            fits = sum(
                ems[3] - ems[0] >= x
                and ems[4] - ems[1] >= y
                and ems[5] - ems[2] >= z
                for x, y, z in self.item_set
            )
            volume = np.prod(np.asarray(ems[3:6]) - np.asarray(ems[0:3]))
            return float(volume + fits + (10 if fits == len(self.item_set) else 0))

        best = None
        best_score = -float("inf")
        for candidate, ems in self._ems_candidates(box):
            score = rank(ems)
            if score > best_score:
                best, best_score = candidate, score
        return self._empty(box) if best is None else self._placement(best, best_score)


def demo() -> None:
    import random

    random.seed(42)
    decider = BPPDecider()
    for index in range(30):
        box = random.choice(DEFAULT_ITEM_SET)
        placement = decider.decide_and_commit(box)
        if not placement.feasible:
            print(f"{index:02d}: no feasible placement for {box}")
            break
        print(f"{index:02d}: {box} -> {placement}")
    print(f"utilization={decider.utilization():.2%}")


if __name__ == "__main__":
    demo()
