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
def _pct_decision(target_center, box_size, side):
    """临时把 task1_2 的 3 个 grid 决策函数替换为 PCT 给定值。

    同时把 SOURCE_BOX_POSE 的 z 改为当前箱高对应的桌面贴合高度——pick IK 必须
    针对箱子实际出现的源位求解，否则规划抓取高度与运行时 reset 的源位不一致，
    放置时会把箱子拽偏（尤其矮箱）。x/y 沿用 task1_2 源位。
    """
    from move.planning import TABLE_POSE, TABLE_SIZE

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
    try:
        yield
    finally:
        (
            t12.cell_center_world,
            t12._box_size_for_sequence,
            t12._direct_side_for_cell,
            t12.SOURCE_BOX_POSE,
        ) = original


def build_pct_box_plan(target_center, box_size, side, stand_off):
    """按 PCT 决策生成一个 task1_2.BoxPlan（复用 task1_2 执行层）。

    target_center: 世界坐标放置中心 (x, y, z)
    box_size:      世界尺寸 (sx, sy, sz)
    side:          机器人接近侧 "-X"/"+X"/"-Y"/"+Y"（由 move_pro 四侧择优给出）
    stand_off:     站距(m)
    返回: t12.BoxPlan，可直接喂 t12.build_task1_2_timeline / t12._run_box_timeline。
    """
    cell = t12.StackCell(
        sequence=_NEUTRAL_SEQUENCE,
        layer=_NEUTRAL_LAYER,
        ix=0,
        iy=0,
        mode="direct",
    )
    with _pct_decision(target_center, box_size, side):
        return t12.build_direct_box_plan(cell, stand_off)
