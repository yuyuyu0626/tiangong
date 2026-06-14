"""BPP 放置决策层 — 封装 Online-3D-BPP-PCT 的启发式/DRL 决策能力。

提供统一的 ``BPPDecider`` 接口：
- 维护当前托盘状态（EMS 空间分解）
- 对每个待放置箱子，调用启发式算法（LSAH / OnlineBPH / DBL 等）
  或 DRL 策略网络，选择最优放置位置
- 将 PCT 坐标系下的放置位置映射到世界坐标系

使用示例::

    from move_pro.bpp_decider import BPPDecider

    decider = BPPDecider(method="LSAH", container_size=(10, 10, 16))
    for box_size in [(3, 3, 4), (2, 4, 2), ...]:
        placement = decider.decide(box_size)    # 在 PCT 坐标系中
        world_xyz = decider.to_world(placement) # 世界坐标
        # ... 交给机器人执行 ...
        decider.commit(placement)               # 确认放置，更新空间
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# 将 PCT 项目加入 sys.path，以便导入其模块
_PCT_ROOT = Path(__file__).resolve().parents[1] / "Online-3D-BPP-PCT"
if str(_PCT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PCT_ROOT))

# PCT 内部依赖 givenData，优先使用本地副本
import givenData                                          # noqa: E402
from pct_envs.PctDiscrete0.space import Space             # noqa: E402
from pct_envs.PctDiscrete0.binCreator import RandomBoxCreator, BoxCreator  # noqa: E402
from pct_envs.PctContinuous0.space import Space as SpaceContinuous      # noqa: E402

from move_pro.config import (                            # noqa: E402
    BIN_TO_WORLD_SCALE,
    PALLET_SURFACE_Z,
    PALLET_SIZE,
    STACK_HEIGHT_LIMIT,
    pallet_center_world,
)


@dataclass(frozen=True)
class Placement:
    """一次放置决策的结果。

    Attributes
    ----------
    lx, ly, lz : float
        PCT 坐标系中的放置位置（箱子左下角坐标）。
    x, y, z : float
        放置后箱子的实际尺寸（可能经过旋转）。
    orientation : int
        旋转模式 (0-5)，参见 PCT 的 6 种朝向。
    world_x, world_y, world_z : float
        世界坐标系中的箱体中心位置。
    feasible : bool
        此放置是否可行。
    score : float
        启发式评分（越大越好），仅用于调试。
    """
    lx: float
    ly: float
    lz: float
    x: float
    y: float
    z: float
    orientation: int
    world_x: float
    world_y: float
    world_z: float
    feasible: bool = True
    score: float = 0.0


class BPPDecider:
    """在线 3D 装箱放置决策器。

    Parameters
    ----------
    method : str
        启发式方法名称，可选：
        - ``"LSAH"`` (Least Surface Area Heuristic，推荐)
        - ``"OnlineBPH"`` (Online Bin Packing Heuristic)
        - ``"DBL"`` (Deepest Bottom Left)
        - ``"BR"`` (Best Rank)
        - ``"DBLF"`` (Deepest Bottom Left with Fill)
    container_size : tuple
        PCT 坐标系中的容器尺寸 (sx, sy, sz)，例如 (10, 10, 16)。
    orientation : int
        允许的旋转模式数：6 = 全旋转, 2 = 仅绕 Z 轴。
    continuous : bool
        是否使用连续环境（物品尺寸可为浮点数）。
    item_set : list
        可选，可放置物品的尺寸集合（用于 BR 启发式评估 EMS）。
    """

    _ORIENTATION_MAP = [
        (0, 1, 2),  # (x, y, z) 不变
        (1, 0, 2),  # (y, x, z)
        (2, 0, 1),  # (z, x, y)
        (2, 1, 0),  # (z, y, x)
        (0, 2, 1),  # (x, z, y)
        (1, 2, 0),  # (y, z, x)
    ]

    def __init__(
        self,
        method: str = "LSAH",
        container_size: tuple = (10, 10, 16),
        orientation: int = 6,
        continuous: bool = False,
        item_set: Optional[list] = None,
    ):
        if method not in ("LSAH", "OnlineBPH", "DBL", "BR", "MAC", "LASH"):
            raise ValueError(f"Unknown heuristic method: {method}")
        self.method = method
        self.container_size = container_size
        self.orientation = orientation
        self.continuous = continuous
        self.item_set = item_set or givenData.item_size_set

        # 初始化 PCT 空间
        size_min = np.min(np.array(self.item_set))
        if continuous:
            self.space = SpaceContinuous(
                *self.container_size, size_min, holder=200
            )
        else:
            self.space = Space(
                *self.container_size, size_min, holder=200
            )

        # 装箱统计
        self.packed_count = 0
        self._packed_boxes: list[dict] = []

        # 用于 LSAH 的包围盒跟踪
        bin_sx, bin_sy, _ = container_size
        self._max_xy = [0, 0]
        self._min_xy = [bin_sx, bin_sy]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def decide(self, box_size: tuple) -> Placement:
        """为给定尺寸的箱子决定放置位置。

        Parameters
        ----------
        box_size : (sx, sy, sz)
            当前箱子的原始尺寸（在 PCT 坐标系中）。

        Returns
        -------
        Placement
            包含 PCT 坐标、世界坐标和可行性标志的放置结果。
        """
        if self.method == "LSAH":
            return self._decide_lsah(box_size)
        elif self.method == "OnlineBPH":
            return self._decide_online_bph(box_size)
        elif self.method == "DBL":
            return self._decide_dbl(box_size)
        elif self.method == "BR":
            return self._decide_br(box_size)
        else:
            return self._decide_lsah(box_size)

    def commit(self, placement: Placement) -> None:
        """确认放置，更新内部 EMS 状态。

        调用此方法后，该位置被标记为占用，后续的 ``decide()``
        调用将不会返回与此冲突的位置。
        """
        box = (placement.x, placement.y, placement.z)
        pos = (int(placement.lx), int(placement.ly))
        self.space.drop_box(box, pos, placement.orientation, 1.0, 2)

        # 更新包围盒（LSAH 用）
        if placement.lx + placement.x > self._max_xy[0]:
            self._max_xy[0] = placement.lx + placement.x
        if placement.ly + placement.y > self._max_xy[1]:
            self._max_xy[1] = placement.ly + placement.y
        if placement.lx < self._min_xy[0]:
            self._min_xy[0] = placement.lx
        if placement.ly < self._min_xy[1]:
            self._min_xy[1] = placement.ly

        # 更新 EMS
        self.space.GENEMS([
            placement.lx, placement.ly, placement.lz,
            placement.lx + placement.x,
            placement.ly + placement.y,
            placement.lz + placement.z,
        ])

        self.packed_count += 1
        self._packed_boxes.append({
            "lx": placement.lx, "ly": placement.ly, "lz": placement.lz,
            "x": placement.x, "y": placement.y, "z": placement.z,
            "world": (placement.world_x, placement.world_y, placement.world_z),
        })

    def to_world(self, pct_lx: float, pct_ly: float, pct_lz: float,
                 box_x: float, box_y: float, box_z: float) -> tuple[float, float, float]:
        """将 PCT 坐标系中的放置位置映射到世界坐标系（箱体中心）。

        PCT 坐标: (lx, ly, lz) 是箱子左下角在 bin 中的位置。
        世界坐标: 托盘左下角 + 偏移 + 箱体半尺寸 = 箱体中心。
        """
        cx_w, cy_w, cz_w = pallet_center_world()
        scale_x, scale_y, scale_z = BIN_TO_WORLD_SCALE
        cs_x, cs_y, cs_z = self.container_size

        # PCT 坐标 → 世界偏移
        wx = (pct_lx / cs_x) * scale_x
        wy = (pct_ly / cs_y) * scale_y
        wz = (pct_lz / cs_z) * scale_z

        # 箱体半尺寸
        half_sx = (box_x / cs_x) * scale_x * 0.5
        half_sy = (box_y / cs_y) * scale_y * 0.5
        half_sz = (box_z / cs_z) * scale_z * 0.5

        # 托盘左下角 → 世界坐标
        world_x = cx_w - scale_x * 0.5 + wx + half_sx
        world_y = cy_w - scale_y * 0.5 + wy + half_sy
        world_z = PALLET_SURFACE_Z + wz + half_sz

        return (world_x, world_y, world_z)

    def reset(self) -> None:
        """重置为初始状态。"""
        self.space.reset()
        self.packed_count = 0
        self._packed_boxes.clear()
        bin_sx, bin_sy, _ = self.container_size
        self._max_xy = [0, 0]
        self._min_xy = [bin_sx, bin_sy]

    def utilization(self) -> float:
        """返回当前空间利用率 (0-1)。"""
        return float(self.space.get_ratio())

    @property
    def packed_boxes(self) -> list[dict]:
        """已放置箱子的列表（PCT 坐标 + 世界坐标）。"""
        return self._packed_boxes

    @property
    def ems_list(self) -> list:
        """当前所有 EMS 空位。"""
        return list(self.space.EMS)

    # ------------------------------------------------------------------
    # 启发式策略实现
    # ------------------------------------------------------------------

    def _gen_candidates(self, box: tuple) -> list[dict]:
        """生成所有可选放置候选（遍历 EMS × 旋转 × 角落）。"""
        candidates = []
        for ems in self.space.EMS:
            ems_sx = ems[3] - ems[0]
            ems_sy = ems[4] - ems[1]
            ems_sz = ems[5] - ems[2]
            if ems_sx <= 0 or ems_sy <= 0 or ems_sz <= 0:
                continue

            for rot in range(self.orientation):
                perm = self._ORIENTATION_MAP[rot]
                x, y, z = box[perm[0]], box[perm[1]], box[perm[2]]
                if x > ems_sx or y > ems_sy or z > ems_sz:
                    continue

                # 4 个角落放置
                corners = [
                    (ems[0], ems[1]),
                    (ems[3] - x, ems[1]),
                    (ems[0], ems[4] - y),
                    (ems[3] - x, ems[4] - y),
                ]
                seen = set()
                for lx, ly in corners:
                    key = (lx, ly, rot)
                    if key in seen:
                        continue
                    seen.add(key)

                    feasible, height = self.space.drop_box_virtual(
                        [x, y, z], (lx, ly), False, 1.0, 2, returnH=True
                    )
                    if feasible:
                        candidates.append({
                            "lx": lx, "ly": ly, "lz": height,
                            "x": x, "y": y, "z": z,
                            "orientation": rot,
                            "ems": ems,
                        })

        return candidates

    def _decide_lsah(self, box: tuple) -> Placement:
        """Least Surface Area Heuristic — 最小化包围盒表面积。"""
        cs_x, cs_y, cs_z = self.container_size
        candidates = self._gen_candidates(box)

        best_score = float("inf")
        best = None

        for c in candidates:
            lx, ly, lz = c["lx"], c["ly"], c["lz"]
            x, y, z = c["x"], c["y"], c["z"]

            # 表面积 = 累计包围盒 × 高度 的各面之和
            new_max_x = max(lx + x, self._max_xy[0])
            new_min_x = min(lx, self._min_xy[0])
            new_max_y = max(ly + y, self._max_xy[1])
            new_min_y = min(ly, self._min_xy[1])

            score = (
                (new_max_x - new_min_x) * (new_max_y - new_min_y)       # XY 面积
                + (lz + z) * (new_max_y - new_min_y)                    # YZ 面积
                + (lz + z) * (new_max_x - new_min_x)                    # XZ 面积
            )

            if score < best_score:
                best_score = score
                best = c
            elif score == best_score and best is not None:
                # tie-break: 选 EMS 尺寸更紧凑的
                ems_fit = min(c["ems"][3] - c["ems"][0] - x,
                              c["ems"][4] - c["ems"][1] - y,
                              c["ems"][5] - c["ems"][2] - z)
                best_fit = min(best["ems"][3] - best["ems"][0] - best["x"],
                               best["ems"][4] - best["ems"][1] - best["y"],
                               best["ems"][5] - best["ems"][2] - best["z"])
                if ems_fit < best_fit:
                    best = c

        if best is None:
            return self._empty_placement(box)

        wx, wy, wz = self.to_world(best["lx"], best["ly"], best["lz"],
                                   best["x"], best["y"], best["z"])
        return Placement(
            lx=best["lx"], ly=best["ly"], lz=best["lz"],
            x=best["x"], y=best["y"], z=best["z"],
            orientation=best["orientation"],
            world_x=wx, world_y=wy, world_z=wz,
            feasible=True, score=-best_score,
        )

    def _decide_online_bph(self, box: tuple) -> Placement:
        """OnlineBPH — 深底左优先 (deepest-bottom-left first)。"""
        # 按 (z, y, x) 排序 = 深底左
        sorted_ems = sorted(self.space.EMS,
                            key=lambda e: (e[2], e[1], e[0]),
                            reverse=False)

        for ems in sorted_ems:
            ems_sx = ems[3] - ems[0]
            ems_sy = ems[4] - ems[1]
            ems_sz = ems[5] - ems[2]
            if ems_sx <= 0 or ems_sy <= 0 or ems_sz <= 0:
                continue

            for rot in range(self.orientation):
                perm = self._ORIENTATION_MAP[rot]
                x, y, z = box[perm[0]], box[perm[1]], box[perm[2]]
                if x > ems_sx or y > ems_sy or z > ems_sz:
                    continue

                if self.space.drop_box_virtual([x, y, z], (ems[0], ems[1]),
                                                False, 1.0, 2):
                    wx, wy, wz = self.to_world(ems[0], ems[1], ems[2],
                                               x, y, z)
                    return Placement(
                        lx=ems[0], ly=ems[1], lz=ems[2],
                        x=x, y=y, z=z, orientation=rot,
                        world_x=wx, world_y=wy, world_z=wz,
                        feasible=True, score=0.0,
                    )

        return self._empty_placement(box)

    def _decide_dbl(self, box: tuple) -> Placement:
        """Deepest Bottom Left — 最底最左优先。"""
        cs_x, cs_y, _ = self.container_size
        candidates = self._gen_candidates(box)
        if not candidates:
            return self._empty_placement(box)

        best_score = float("inf")
        best = candidates[0]
        for c in candidates:
            score = c["lx"] + c["ly"] + 100 * c["lz"]
            if score < best_score:
                best_score = score
                best = c

        wx, wy, wz = self.to_world(best["lx"], best["ly"], best["lz"],
                                   best["x"], best["y"], best["z"])
        return Placement(
            lx=best["lx"], ly=best["ly"], lz=best["lz"],
            x=best["x"], y=best["y"], z=best["z"],
            orientation=best["orientation"],
            world_x=wx, world_y=wy, world_z=wz,
            feasible=True, score=-best_score,
        )

    def _decide_br(self, box: tuple) -> Placement:
        """Best Rank — 优先选择能容纳更多种类物品的 EMS。"""
        candidates = self._gen_candidates(box)
        if not candidates:
            return self._empty_placement(box)

        def _eval_ems(ems):
            s = 0
            s += (ems[3] - ems[0]) * (ems[4] - ems[1]) * (ems[5] - ems[2])
            valid_count = 0
            for bs in self.item_set:
                bx, by, bz = bs
                if (ems[3] - ems[0] >= bx and ems[4] - ems[1] >= by
                        and ems[5] - ems[2] >= bz):
                    valid_count += 1
            s += valid_count
            if valid_count == len(self.item_set):
                s += 10
            return s

        best_score = -float("inf")
        best = None
        for c in candidates:
            score = _eval_ems(c["ems"])
            if score > best_score:
                best_score = score
                best = c

        if best is None:
            return self._empty_placement(box)

        wx, wy, wz = self.to_world(best["lx"], best["ly"], best["lz"],
                                   best["x"], best["y"], best["z"])
        return Placement(
            lx=best["lx"], ly=best["ly"], lz=best["lz"],
            x=best["x"], y=best["y"], z=best["z"],
            orientation=best["orientation"],
            world_x=wx, world_y=wy, world_z=wz,
            feasible=True, score=best_score,
        )

    def _empty_placement(self, box: tuple) -> Placement:
        """无可放置位置时返回的占位结果。"""
        return Placement(
            lx=0, ly=0, lz=0,
            x=box[0], y=box[1], z=box[2],
            orientation=0,
            world_x=0, world_y=0, world_z=0,
            feasible=False, score=-1,
        )


def demo() -> None:
    """快速演示 BPP 决策器。"""
    import random
    random.seed(42)

    decider = BPPDecider(method="LSAH")
    boxes = [random.choice(givenData.item_size_set) for _ in range(30)]

    print(f"{'#':>3s}  {'box':>10s}  {'placement(PCT)':>20s}  {'world(x,y,z)':>28s}  {'score':>8s}")
    print("-" * 80)

    for i, box in enumerate(boxes):
        p = decider.decide(box)
        if not p.feasible:
            print(f"{i:3d}  {str(box):>10s}  NO FEASIBLE PLACEMENT")
            break
        decider.commit(p)
        print(f"{i:3d}  {str(box):>10s}  ({p.lx:5.0f},{p.ly:5.0f},{p.lz:5.0f}) "
              f"+ ({p.x:.0f},{p.y:.0f},{p.z:.0f})  "
              f"({p.world_x:6.3f},{p.world_y:6.3f},{p.world_z:6.3f})  "
              f"{p.score:8.1f}")

    print(f"\n最终利用率: {decider.utilization():.2%}, "
          f"已放置: {decider.packed_count} 箱")


if __name__ == "__main__":
    demo()
