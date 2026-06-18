"""把 PCT 在线决策接到 move 的 task1_2 放置执行层。

task1_2.build_direct_box_plan(cell, stand_off) 内部分两层：
- 决策层（grid 绑定）：cell_center_world / _box_size_for_sequence /
  _direct_side_for_cell —— 靠固定 3x5x4 垛型 + 等大箱的 sequence 查表。
  这正是 move_pro 要用 PCT 替换的部分。
- 执行层（通用）：pick keyframe、root 路线、place 路径、move IK、timeline、
  播放循环 —— 接收普通值，与垛型无关。

本模块用一个"中性" StackCell（layer=0, sequence=1, mode=direct，不触发任何
task1_2 的分层/推入/角块特例），并临时覆盖那 3 个决策函数，让 build_direct_box_plan
按 PCT 给定的 target/size/side 生成 BoxPlan，从而完整复用 task1_2 执行层。
"""

from __future__ import annotations

from contextlib import contextmanager

import move.tasks.task1_2 as t12


# layer=0 + layer_sequence=1 + mode="direct" 不命中 task1_2 任一特例策略：
#   _uses_first_layer_y_row_strategy: layer_sequence∈{4..15} → False
#   _uses_inner_edge_push_strategy / _uses_second_layer_inner_row_clearance: 需 layer==1 → False
#   _uses_third_layer_center_strategy: 需 layer==2 → False
# 因此所有 *_for_cell 调参函数都走通用默认；_stand_off_for_cell 直接返回 base_stand_off。
_NEUTRAL_SEQUENCE = 1
_NEUTRAL_LAYER = 0


@contextmanager
def _pct_decision(target_center, box_size, side, clearance_z=None, release_offset=None):
    """临时把 task1_2 的 3 个 grid 决策函数替换为 PCT 给定值。

    同时把 SOURCE_BOX_POSE 的 z 改为当前箱高对应的桌面贴合高度——pick IK 必须
    针对箱子实际出现的源位求解，否则规划抓取高度与运行时 reset 的源位不一致，
    放置时会把箱子拽偏（尤其矮箱）。x/y 沿用 task1_2 源位。

    clearance_z（顶降式避障）：若给定，把搬运/放置的三个抬升高度都改成"高过该世界
    高度 + 余量"，使搬运箱/手横移时不扫到更高的已放邻箱。默认 None 保持 task1_2 原值。

    release_offset（释放点偏外，米）：若 >0，把机器人松手的释放中心朝接近侧 side 偏外，
    使 double-hand 侧抓在松手瞬间手掌远离已放邻箱（消除手掌穿透）。释放后由 simulation
    把箱子从偏外点 kinematic 平移推入真实 target，保证垛形。默认 None 走 task1_2 原释放点。
    """
    from move.planning import TABLE_POSE, TABLE_SIZE
    from move_pro.config import PLACE_CARRY_SAFE_MARGIN

    src_z = TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + float(box_size[2]) * 0.5
    source_pose = (t12.SOURCE_BOX_POSE[0], t12.SOURCE_BOX_POSE[1], src_z)
    original = (
        t12.cell_center_world,
        t12._box_size_for_sequence,
        t12._direct_side_for_cell,
        t12.SOURCE_BOX_POSE,
    )
    t12.cell_center_world = lambda cell: tuple(float(v) for v in target_center)
    t12._box_size_for_sequence = lambda sequence: tuple(float(v) for v in box_size)
    t12._direct_side_for_cell = lambda cell: side
    t12.SOURCE_BOX_POSE = source_pose

    # 释放点偏外：朝接近侧把释放中心移出 release_offset，松手时手掌离邻箱有间隙。
    release_original = None
    if release_offset is not None and release_offset > 0.0:
        dx, dy = _outward_unit(side)
        d = float(release_offset)
        release_original = t12._direct_release_center_for_cell
        t12._direct_release_center_for_cell = (
            lambda cell, target: (target[0] + dx * d, target[1] + dy * d, target[2])
        )

    # 顶降式避障：把三个抬升高度抬到 clearance_z 之上（box-center 相对目标 z 的抬升量）。
    height_original = None
    if clearance_z is not None:
        target_z = float(target_center[2])
        # box-center 至少要到 clearance_z（已放箱顶 + 余量），换成相对目标的抬升量。
        # 下限保底 task1_2 原值，避免比原来还低。
        needed_lift = (float(clearance_z) + PLACE_CARRY_SAFE_MARGIN) - target_z
        approach_lift = max(t12.MOVE_APPROACH_HEIGHT, needed_lift)
        preplace_lift = max(t12.MOVE_PREPLACE_HEIGHT, needed_lift)
        place_clearance = max(t12.DIRECT_PREPLACE_CLEARANCE, needed_lift)
        height_original = (
            t12._move_approach_height_for_cell,
            t12._move_preplace_height_for_cell,
            t12.DIRECT_PREPLACE_CLEARANCE,
        )
        t12._move_approach_height_for_cell = lambda cell: approach_lift
        t12._move_preplace_height_for_cell = lambda cell: preplace_lift
        t12.DIRECT_PREPLACE_CLEARANCE = place_clearance

    try:
        yield
    finally:
        (
            t12.cell_center_world,
            t12._box_size_for_sequence,
            t12._direct_side_for_cell,
            t12.SOURCE_BOX_POSE,
        ) = original
        if release_original is not None:
            t12._direct_release_center_for_cell = release_original
        if height_original is not None:
            (
                t12._move_approach_height_for_cell,
                t12._move_preplace_height_for_cell,
                t12.DIRECT_PREPLACE_CLEARANCE,
            ) = height_original


def _outward_unit(side: str) -> tuple[float, float]:
    """接近侧 → 朝托盘外侧的单位向量（x,y）。释放点朝此方向偏出，远离内部邻箱。"""
    return {
        "-X": (-1.0, 0.0),
        "+X": (1.0, 0.0),
        "-Y": (0.0, -1.0),
        "+Y": (0.0, 1.0),
    }[side]


def build_pct_box_plan(target_center, box_size, side, stand_off, clearance_z=None,
                       release_offset=None):
    """按 PCT 决策生成一个 task1_2.BoxPlan（复用 task1_2 执行层）。

    target_center: 世界坐标放置中心 (x, y, z)
    box_size:      世界尺寸 (sx, sy, sz)
    side:          机器人接近侧 "-X"/"+X"/"-Y"/"+Y"（由 move_pro 四侧择优给出）
    stand_off:     站距(m)
    clearance_z:   顶降式避障——搬运/放置需清过的世界高度（已放箱顶+余量）。None=不抬升。
    release_offset: 释放点朝接近侧偏外量(m)。>0 时松手点远离邻箱，释放后推入 target。None=不偏。
    返回: t12.BoxPlan，可直接喂 t12.build_task1_2_timeline / t12._run_box_timeline。
    """
    cell = t12.StackCell(
        sequence=_NEUTRAL_SEQUENCE,
        layer=_NEUTRAL_LAYER,
        ix=0,
        iy=0,
        mode="direct",
    )
    with _pct_decision(target_center, box_size, side, clearance_z=clearance_z,
                       release_offset=release_offset):
        return t12.build_direct_box_plan(cell, stand_off)
