#!/usr/bin/env python3
"""task2_1：9 箱/层垛型的四层堆叠脚本。

脚本内容：
1. 整体垛型仍为 1.0 x 1.0 x 1.6 m，共 4 层，每层 0.4 m；
2. 每层由 9 个不同尺寸箱子组成：中心块、十字块和四个角块；
3. 第 1、2 层复用 task1 的直接放置仿真路径，并替换为 task2 的箱体尺寸和目标点；
4. 第 3、4 层复用 task1 的 waist-stack 控制路径，并替换为 task2 的顺序、尺寸和目标点。

运行方式：
    python -m move.tasks.task2_1
    python -m move.tasks.task2_1 --until-box 9
    python -m move.tasks.task2_1 --second-layer-proxy
    python -m move.tasks.task2_1 --third-fourth-proxy

模式说明：
    默认模式              跑 task2 四层 36 箱。
    --until-box N          只跑到第 N 个 task2 箱。
    --second-layer-proxy   只调试第 2 层，第 1 层用代理块。
    --third-fourth-proxy   只调试第 3、4 层，前两层用代理块。
    --fast-viewer          用于 viewer 加速播放。
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
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("PyTorch is required for task2_1.") from exc


MOVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOVE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from move_pre.tasks import task1 as base  # noqa: E402


_BASE_INNER_HAND_X_CLEARANCE_FOR_CELL = base._inner_hand_x_clearance_for_cell
_WAIST_ORIGINAL_SOLVE_BOX_GRIP = base.waist_stack.solve_box_grip
_TASK2_WAIST_ACTIVE_CELL = None

STACK_SIZE = base.STACK_SIZE
LAYER_HEIGHT = base.STACK_BOX_SIZE[2]
PALLET_CENTER = base.PALLET_CENTER

CENTER_BOX_SIZE = base.STACK_BOX_SIZE
Y_CROSS_BOX_SIZE = (0.40, 1.0 / 3.0, LAYER_HEIGHT)
X_CROSS_BOX_SIZE = (1.0 / 3.0, 0.30, LAYER_HEIGHT)
CORNER_BOX_SIZE = (0.35, 1.0 / 3.0, LAYER_HEIGHT)

TASK2_Y_ROW_STRATEGY_BOXES = {2, 3, 6, 7, 8, 9}
TASK2_INNER_EDGE_PUSH_BOXES = {6, 7, 8, 9}
TASK2_X_CROSS_BOXES = {4, 5}
TASK2_LAYER_BOX_COUNT = 9
TASK2_NATIVE_LAYERS = 2
TASK2_TOTAL_LAYERS = 4
TASK2_WAIST_START_BOX = TASK2_LAYER_BOX_COUNT * TASK2_NATIVE_LAYERS + 1
TASK2_Y_CROSS_FORWARD_RELEASE_COMPENSATION = 0.09
TASK2_X_CROSS_BACK_RELEASE_COMPENSATION = 0.02
TASK2_SECOND_LAYER_Y_CROSS_STAND_OFF_EXTRA = 0.20
TASK2_SECOND_LAYER_Y_CROSS_RELEASE_BACK_EXTRA = 0.02
TASK2_SECOND_LAYER_Y_CROSS_GRASP_FORWARD_OFFSET = 0.05
TASK2_SECOND_LAYER_Y_CROSS_TARGET_FORWARD_OFFSET = 0.02
TASK2_UPPER_LAYER_Y_CROSS_TARGET_BACK_OFFSET = TASK2_SECOND_LAYER_Y_CROSS_TARGET_FORWARD_OFFSET
TASK2_UPPER_LAYER_Y_CROSS_RELEASE_BACK_EXTRA = TASK2_Y_CROSS_FORWARD_RELEASE_COMPENSATION
TASK2_THIRD_LAYER_CROSS_TARGET_FORWARD_OFFSET = 0.01
TASK2_THIRD_LAYER_CROSS_STAND_OFF_FORWARD_DELTA = 0.02
TASK2_FOURTH_LAYER_X_CROSS_OUTWARD_TARGET_OFFSET = 0.02
TASK2_WAIST_Y_CROSS_GRASP_FORWARD_OFFSET = TASK2_SECOND_LAYER_Y_CROSS_GRASP_FORWARD_OFFSET
TASK2_THIRD_LAYER_Y_CROSS_GRASP_BACK_DELTA = 0.10
TASK2_WAIST_Y_CROSS_PHYSICAL_CLAMP_HOLD_FRAMES = 12
TASK2_FOURTH_LAYER_CROSS_LIFT_FORWARD = 0.10
TASK2_UPPER_LAYER_CORNER_PREVIOUS_BOX_CLEARANCE = 0.02
TASK2_UPPER_LAYER_CORNER_STAND_OFF_REDUCTION = 0.20
TASK2_CORNER_INNER_HAND_X_CLEARANCE = 0.08
TASK2_UPPER_LAYER_CORNER_INNER_HAND_CLEARANCE = 0.05


@dataclass(frozen=True)
class Task2CellSpec:
    sequence: int
    ix: int
    iy: int
    mode: str
    size: tuple[float, float, float]
    local_xy: tuple[float, float]
    description: str


def _task1_grid_local_xy(ix: int, iy: int) -> tuple[float, float]:
    sx = STACK_SIZE[0] / base.GRID_X
    sy = STACK_SIZE[1] / base.GRID_Y
    return (
        -STACK_SIZE[0] * 0.5 + (ix + 0.5) * sx,
        -STACK_SIZE[1] * 0.5 + (iy + 0.5) * sy,
    )


def _corner_local_xy(sign_x: float, sign_y: float) -> tuple[float, float]:
    return (
        sign_x * (STACK_SIZE[0] * 0.5 - CORNER_BOX_SIZE[0] * 0.5),
        sign_y * (STACK_SIZE[1] * 0.5 - CORNER_BOX_SIZE[1] * 0.5),
    )


TASK2_CELL_SPECS: dict[int, Task2CellSpec] = {
    1: Task2CellSpec(1, 1, 2, "direct", CENTER_BOX_SIZE, _task1_grid_local_xy(1, 2), "center"),
    # These two use task1 box 4/5 positions and y-row direct strategy.
    2: Task2CellSpec(2, 1, 1, "direct", Y_CROSS_BOX_SIZE, _task1_grid_local_xy(1, 1), "y_cross_minus"),
    3: Task2CellSpec(3, 1, 3, "direct", Y_CROSS_BOX_SIZE, _task1_grid_local_xy(1, 3), "y_cross_plus"),
    # These two use task1 box 2/3 positions and x-neighbor direct strategy.
    4: Task2CellSpec(4, 0, 2, "direct", X_CROSS_BOX_SIZE, _task1_grid_local_xy(0, 2), "x_cross_minus"),
    5: Task2CellSpec(5, 2, 2, "direct", X_CROSS_BOX_SIZE, _task1_grid_local_xy(2, 2), "x_cross_plus"),
    # Corner targets are flush to the 1.0 x 1.0 m footprint while retaining
    # task1 corner placement/push strategy through ix/iy and mode.
    6: Task2CellSpec(6, 0, 0, "offset_push", CORNER_BOX_SIZE, _corner_local_xy(-1.0, -1.0), "corner_minus_minus"),
    7: Task2CellSpec(7, 2, 0, "offset_push", CORNER_BOX_SIZE, _corner_local_xy(1.0, -1.0), "corner_plus_minus"),
    8: Task2CellSpec(8, 0, 4, "offset_push", CORNER_BOX_SIZE, _corner_local_xy(-1.0, 1.0), "corner_minus_plus"),
    9: Task2CellSpec(9, 2, 4, "offset_push", CORNER_BOX_SIZE, _corner_local_xy(1.0, 1.0), "corner_plus_plus"),
}


def build_task2_layer_order(layer: int) -> tuple[base.StackCell, ...]:
    return tuple(
        base.StackCell(layer * TASK2_LAYER_BOX_COUNT + spec.sequence, layer, spec.ix, spec.iy, spec.mode)
        for spec in TASK2_CELL_SPECS.values()
    )


def build_task2_order(layers: int = TASK2_NATIVE_LAYERS) -> tuple[base.StackCell, ...]:
    cells: list[base.StackCell] = []
    for layer in range(layers):
        cells.extend(build_task2_layer_order(layer))
    return tuple(cells)


def build_task2_first_layer_order() -> tuple[base.StackCell, ...]:
    return build_task2_layer_order(0)


def _task2_cell_spec(cell: base.StackCell) -> Task2CellSpec:
    layer_sequence = _task2_cell_layer_sequence(cell)
    try:
        return TASK2_CELL_SPECS[layer_sequence]
    except KeyError as exc:
        raise ValueError(f"Unsupported task2_1 box sequence: {cell.sequence}") from exc


def _task2_layer_sequence(sequence: int) -> int:
    return ((sequence - 1) % TASK2_LAYER_BOX_COUNT) + 1


def _task2_cell_layer_sequence(cell: base.StackCell) -> int:
    return _task2_layer_sequence(cell.sequence)


def _task2_uses_first_layer_y_row_strategy(cell: base.StackCell) -> bool:
    return _task2_cell_layer_sequence(cell) in TASK2_Y_ROW_STRATEGY_BOXES


def _task2_supports_direct_box_plan(cell: base.StackCell) -> bool:
    return cell.mode == "direct" or _task2_uses_first_layer_y_row_strategy(cell)


def _task2_uses_inner_edge_push_strategy(cell: base.StackCell) -> bool:
    if cell.layer == 1:
        return False
    return _task2_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES


def _task2_uses_second_layer_direct_edge_strategy(cell: base.StackCell) -> bool:
    return cell.layer == 1 and _task2_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES


def _task2_uses_second_layer_inner_row_clearance_strategy(cell: base.StackCell) -> bool:
    return cell.layer == 1 and _task2_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES


def _task2_box_size_for_sequence(sequence: int) -> tuple[float, float, float]:
    return TASK2_CELL_SPECS[_task2_layer_sequence(sequence)].size


def _task2_box_size_for_cell(cell: base.StackCell) -> tuple[float, float, float]:
    return _task2_cell_spec(cell).size


def _task2_cell_center_world(cell: base.StackCell) -> tuple[float, float, float]:
    spec = _task2_cell_spec(cell)
    layer_sequence = _task2_cell_layer_sequence(cell)
    local_x, local_y = spec.local_xy
    if cell.layer == 1 and layer_sequence == 2:
        local_y += TASK2_SECOND_LAYER_Y_CROSS_TARGET_FORWARD_OFFSET
    elif cell.layer == 1 and layer_sequence == 3:
        local_y -= TASK2_SECOND_LAYER_Y_CROSS_TARGET_FORWARD_OFFSET
    return (
        PALLET_CENTER[0] + local_x,
        PALLET_CENTER[1] + local_y,
        base.PALLET_SURFACE_Z + (cell.layer + 0.5) * LAYER_HEIGHT,
    )


def _task2_direct_release_center_for_cell(
    cell: base.StackCell,
    target_center: tuple[float, float, float],
) -> tuple[float, float, float]:
    layer_sequence = _task2_cell_layer_sequence(cell)
    if layer_sequence == 4:
        return (
            target_center[0] - base.FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE + TASK2_X_CROSS_BACK_RELEASE_COMPENSATION,
            target_center[1],
            target_center[2],
        )
    if layer_sequence == 5:
        return (
            target_center[0] + base.FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE - TASK2_X_CROSS_BACK_RELEASE_COMPENSATION,
            target_center[1],
            target_center[2],
        )
    if _task2_uses_first_layer_y_row_strategy(cell):
        side = base._direct_side_for_cell(cell)
        y_clearance = base._y_release_clearance_for_cell(cell)
        if layer_sequence in {2, 3}:
            y_clearance += TASK2_Y_CROSS_FORWARD_RELEASE_COMPENSATION
            if cell.layer == 1:
                y_clearance += TASK2_SECOND_LAYER_Y_CROSS_RELEASE_BACK_EXTRA
        x_clearance = base._inner_hand_x_clearance_for_cell(cell)
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


def _task2_base_stand_off_for_cell(cell: base.StackCell, base_stand_off: float) -> float:
    layer_sequence = _task2_cell_layer_sequence(cell)
    if layer_sequence == 1:
        return base_stand_off
    if cell.layer == 1 and layer_sequence in TASK2_X_CROSS_BOXES:
        return max(base_stand_off, base.SECOND_LAYER_X_NEIGHBOR_STAND_OFF)
    if cell.layer == 1 and _task2_uses_first_layer_y_row_strategy(cell) and cell.iy in {0, base.GRID_Y - 1}:
        return max(base_stand_off, base.SECOND_LAYER_Y_EXTENSION_STAND_OFF)
    if cell.layer == 1 and _task2_uses_first_layer_y_row_strategy(cell):
        return max(base_stand_off, base.SECOND_LAYER_Y_SIDE_STAND_OFF)
    if _task2_uses_first_layer_y_row_strategy(cell) and cell.iy in {0, base.GRID_Y - 1}:
        return max(base_stand_off, base.DIRECT_Y_EXTENSION_STAND_OFF)
    if base._direct_side_for_cell(cell) in {"-Y", "+Y"}:
        return max(base_stand_off, base.DIRECT_Y_SIDE_STAND_OFF)
    return max(base_stand_off, base.DIRECT_OUTER_STAND_OFF)


def _task2_stand_off_for_cell(cell: base.StackCell, base_stand_off: float) -> float:
    layer_sequence = _task2_cell_layer_sequence(cell)
    stand_off = _task2_base_stand_off_for_cell(cell, base_stand_off)
    if cell.layer == 1 and layer_sequence in {2, 3}:
        return stand_off + TASK2_SECOND_LAYER_Y_CROSS_STAND_OFF_EXTRA
    if layer_sequence in {2, 3}:
        return stand_off + 0.06
    if layer_sequence in TASK2_X_CROSS_BOXES:
        return stand_off + 0.05
    if layer_sequence == 1:
        return max(0.0, stand_off - 0.05)
    return stand_off


def _task2_contact_forward_for_cell(cell: base.StackCell) -> float:
    if _task2_uses_first_layer_y_row_strategy(cell):
        contact_forward = base.FIRST_LAYER_Y_SWAPPED_CONTACT_FORWARD
    else:
        contact_forward = base.GRASP_CONTACT_X_OFFSET
    if cell.layer == 1 and _task2_cell_layer_sequence(cell) in {2, 3}:
        contact_forward += TASK2_SECOND_LAYER_Y_CROSS_GRASP_FORWARD_OFFSET
    return contact_forward


def _task2_inner_hand_x_clearance_for_cell(cell: base.StackCell) -> float:
    if cell.layer == 1 and _task2_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES:
        if cell.ix == 0:
            return -TASK2_CORNER_INNER_HAND_X_CLEARANCE
        if cell.ix == base.GRID_X - 1:
            return TASK2_CORNER_INNER_HAND_X_CLEARANCE
        return 0.0
    return _BASE_INNER_HAND_X_CLEARANCE_FOR_CELL(cell)


def _install_task2_first_layer_overrides() -> None:
    base.build_first_layer_order = build_task2_first_layer_order
    base._layer_sequence = _task2_layer_sequence
    base._cell_layer_sequence = _task2_cell_layer_sequence
    base._uses_first_layer_y_row_strategy = _task2_uses_first_layer_y_row_strategy
    base._supports_direct_box_plan = _task2_supports_direct_box_plan
    base._uses_inner_edge_push_strategy = _task2_uses_inner_edge_push_strategy
    base._uses_second_layer_direct_edge_strategy = _task2_uses_second_layer_direct_edge_strategy
    base._uses_second_layer_inner_row_clearance_strategy = _task2_uses_second_layer_inner_row_clearance_strategy
    base._uses_third_layer_center_strategy = lambda _cell: False
    base._box_size_for_sequence = _task2_box_size_for_sequence
    base._box_size_for_cell = _task2_box_size_for_cell
    base.cell_center_world = _task2_cell_center_world
    base._direct_release_center_for_cell = _task2_direct_release_center_for_cell
    base._stand_off_for_cell = _task2_stand_off_for_cell
    base._contact_forward_for_cell = _task2_contact_forward_for_cell
    base._inner_hand_x_clearance_for_cell = _task2_inner_hand_x_clearance_for_cell


def _parked_box_pose(index: int, box_size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        base.BOX_PARK_X - base.BOX_PARK_SPACING * index,
        -1.2,
        base.TABLE_POSE[2] + base.TABLE_SIZE[2] * 0.5 + box_size[2] * 0.5,
    )


def _source_box_pose_for_size(box_size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        base.TABLE_POSE[0] - 0.10,
        base.TABLE_POSE[1],
        base.TABLE_POSE[2] + base.TABLE_SIZE[2] * 0.5 + box_size[2] * 0.5,
    )


def _density_for_box_size(box_size: tuple[float, float, float]) -> float:
    return base.BOX_MASS / (box_size[0] * box_size[1] * box_size[2])


def _create_task2_static_scene(gym, sim, env, robot_asset, cells: list[object], proxy_layers: int = 0):
    robot = gym.create_actor(env, robot_asset, base._make_transform(0.0, 0.0, 0.0), "task2_1_robot", 0, 1)

    fixed_opts = gymapi.AssetOptions()
    fixed_opts.fix_base_link = True
    table_asset = gym.create_box(sim, base.TABLE_SIZE[0], base.TABLE_SIZE[1], base.TABLE_SIZE[2], fixed_opts)
    pallet_asset = gym.create_box(sim, STACK_SIZE[0], STACK_SIZE[1], base.PALLET_THICKNESS, fixed_opts)
    proxy_asset = None
    proxy_layers = max(0, min(int(proxy_layers), TASK2_TOTAL_LAYERS))
    if proxy_layers > 0:
        proxy_asset = gym.create_box(sim, STACK_SIZE[0], STACK_SIZE[1], LAYER_HEIGHT * proxy_layers, fixed_opts)

    table = gym.create_actor(env, table_asset, base._make_transform(*base.TABLE_POSE), "table", 0, 0)
    pallet_z = base.PALLET_SURFACE_Z - base.PALLET_THICKNESS * 0.5
    pallet = gym.create_actor(
        env,
        pallet_asset,
        base._make_transform(PALLET_CENTER[0], PALLET_CENTER[1], pallet_z),
        "pallet",
        0,
        0,
    )
    lower_layer_proxy_actor = None
    if proxy_asset is not None:
        lower_layer_proxy_actor = gym.create_actor(
            env,
            proxy_asset,
            base._make_transform(
                PALLET_CENTER[0],
                PALLET_CENTER[1],
                base.PALLET_SURFACE_Z + LAYER_HEIGHT * proxy_layers * 0.5,
            ),
            f"task2_lower_{proxy_layers}_layer_proxy",
            0,
            0,
        )

    box_assets: dict[tuple[float, float, float], object] = {}
    boxes = []
    for index, cell in enumerate(cells):
        box_size = _task2_box_size_for_cell(cell)
        if box_size not in box_assets:
            box_opts = gymapi.AssetOptions()
            box_opts.density = _density_for_box_size(box_size)
            box_assets[box_size] = gym.create_box(sim, box_size[0], box_size[1], box_size[2], box_opts)
        pose = _source_box_pose_for_size(box_size) if index == 0 else _parked_box_pose(index, box_size)
        boxes.append(
            gym.create_actor(
                env,
                box_assets[box_size],
                base._make_transform(*pose),
                f"task2_box_{cell.sequence:02d}",
                0,
                0,
            )
        )

    base._set_color(gym, env, table, (1.0, 1.0, 1.0))
    base._set_color(gym, env, pallet, (0.15, 0.55, 0.45))
    if lower_layer_proxy_actor is not None:
        base._set_color(gym, env, lower_layer_proxy_actor, (0.62, 0.62, 0.58))
    for index, box in enumerate(boxes):
        layer_alpha = 0.15 + 0.10 * (index % 5)
        base._set_color(gym, env, box, (0.9, layer_alpha, 0.02))

    base._set_shape_friction(gym, env, robot, 6.0, rolling_friction=0.05, torsion_friction=0.05)
    base._set_shape_friction(gym, env, table, 2.5)
    base._set_shape_friction(gym, env, pallet, base.PALLET_CONTACT_FRICTION)
    if lower_layer_proxy_actor is not None:
        base._set_shape_friction(gym, env, lower_layer_proxy_actor, base.PALLET_CONTACT_FRICTION)
    for box in boxes:
        base._set_contact_material(
            gym,
            env,
            box,
            base.BOX_CONTACT_FRICTION,
            base.BOX_CONTACT_ROLLING_FRICTION,
            base.BOX_CONTACT_TORSION_FRICTION,
            base.BOX_CONTACT_RESTITUTION,
        )
    if boxes:
        base._set_task_collision_filters(gym, env, robot, boxes[0])
    return robot, boxes


def _reorder_plans(plans: list[base.BoxPlan], asset_dof_names: list[str]) -> list[base.BoxPlan]:
    return [base._reorder_box_plan(plan, asset_dof_names) for plan in plans]


def _delay_second_layer_release_until_hold_end(plan: base.BoxPlan) -> base.BoxPlan:
    if plan.cell.layer != 1:
        return plan
    place_path = [
        ("place_attached_hold", center, theta) if label == "place_hold" else (label, center, theta)
        for label, center, theta in plan.place_pose_path_world
    ]
    return replace(plan, place_pose_path_world=place_path)


def _uses_task2_second_layer_corner_strategy(cell: base.StackCell) -> bool:
    return cell.layer == 1 and _task2_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES


def _replace_second_layer_corner_post_release(
    plan: base.BoxPlan,
    timeline: list[tuple[str, base.Pose, torch.Tensor, tuple[float, float, float], float]],
) -> list[tuple[str, base.Pose, torch.Tensor, tuple[float, float, float], float]]:
    if not _uses_task2_second_layer_corner_strategy(plan.cell):
        return timeline

    try:
        release_index = next(index for index, frame in enumerate(timeline) if frame[0] == "release")
        open_start = next(index for index, frame in enumerate(timeline) if frame[0] == "post_release_upper_layer_inner_hand_lift")
    except StopIteration:
        return timeline

    release_phase, final_root, place_hold_q, release_center, release_theta = timeline[release_index]
    del release_phase, release_theta
    outer_side = base._outer_hand_side_for_cell(plan.cell)
    outer_open_q = place_hold_q
    if outer_side is not None:
        outer_open_q = base._blend_side_dofs(place_hold_q, plan.release_target, plan.dof_names, outer_side, 1.0)

    new_timeline = list(timeline[:open_start])
    open_frames = max(1, base.FIRST_LAYER_INNER_EDGE_HAND_LIFT_FRAMES)
    for i in range(1, open_frames + 1):
        alpha = base._smoothstep(i / float(open_frames))
        q = place_hold_q * (1.0 - alpha) + outer_open_q * alpha
        new_timeline.append(("post_release_upper_layer_outer_hand_open", final_root, q, release_center, 0.0))

    for label, root_pose in plan.return_route:
        new_timeline.append((f"return:{label}", root_pose, outer_open_q, release_center, 0.0))

    recover_target = plan.pick_targets[0]
    recover_root = plan.return_route[-1][1] if plan.return_route else final_root
    for i in range(1, base.RETURN_RECOVER_FRAMES + 1):
        alpha = base._smoothstep(i / float(base.RETURN_RECOVER_FRAMES))
        q = outer_open_q * (1.0 - alpha) + recover_target * alpha
        new_timeline.append(("return_recover:home", recover_root, q, release_center, 0.0))
    return base._smooth_timeline_joint_steps(new_timeline, max_joint_step=0.02)


def build_task2_timeline(plan: base.BoxPlan) -> list[tuple[str, base.Pose, torch.Tensor, tuple[float, float, float], float]]:
    return _replace_second_layer_corner_post_release(plan, base.build_timeline(plan))


def build_task2_waist_order() -> tuple[object, ...]:
    waist = base.waist_stack
    cells: list[object] = []
    cells.append(waist.StackCell(19, 2, TASK2_CELL_SPECS[1].ix, TASK2_CELL_SPECS[1].iy))
    for layer_sequence in (2, 3):
        spec = TASK2_CELL_SPECS[layer_sequence]
        cells.append(waist.StackCell(18 + layer_sequence, 2, spec.ix, spec.iy))
    cells.append(waist.StackCell(28, 3, TASK2_CELL_SPECS[1].ix, TASK2_CELL_SPECS[1].iy))
    for layer_sequence in (4, 5):
        spec = TASK2_CELL_SPECS[layer_sequence]
        cells.append(waist.StackCell(18 + layer_sequence, 2, spec.ix, spec.iy))
    for layer_sequence in (2, 3, 4, 5):
        spec = TASK2_CELL_SPECS[layer_sequence]
        cells.append(waist.StackCell(27 + layer_sequence, 3, spec.ix, spec.iy))
    for layer, offset in ((2, 18), (3, 27)):
        for layer_sequence in (6, 7, 8, 9):
            spec = TASK2_CELL_SPECS[layer_sequence]
            cells.append(waist.StackCell(offset + layer_sequence, layer, spec.ix, spec.iy))
    return tuple(cells)


def _task2_waist_cell_layer_sequence(cell) -> int:
    return _task2_layer_sequence(cell.sequence)


def _task2_waist_uses_upper_layer_y_cross(cell) -> bool:
    return cell.layer in {2, 3} and _task2_waist_cell_layer_sequence(cell) in {2, 3}


def _task2_waist_uses_third_layer_y_cross(cell) -> bool:
    return cell.layer == 2 and _task2_waist_cell_layer_sequence(cell) in {2, 3}


def _task2_waist_uses_fourth_layer_y_cross(cell) -> bool:
    return cell.layer == 3 and _task2_waist_cell_layer_sequence(cell) in {2, 3}


def _task2_waist_uses_fourth_layer_cross(cell) -> bool:
    return cell.layer == 3 and _task2_waist_cell_layer_sequence(cell) in {2, 3, 4, 5}


def _task2_waist_uses_fourth_layer_cross_third_layer_two_grasp(cell) -> bool:
    return cell.layer == 3 and _task2_waist_cell_layer_sequence(cell) in {2, 3}


def _task2_waist_uses_third_layer_cross_forward_target(cell) -> bool:
    return cell.layer == 2 and _task2_waist_cell_layer_sequence(cell) in {2, 5}


def _task2_waist_cell_target(cell) -> tuple[float, float, float]:
    spec = TASK2_CELL_SPECS[_task2_waist_cell_layer_sequence(cell)]
    layer_sequence = _task2_waist_cell_layer_sequence(cell)
    local_x, local_y = spec.local_xy
    if _task2_waist_uses_upper_layer_y_cross(cell) and layer_sequence == 2:
        local_y -= TASK2_UPPER_LAYER_Y_CROSS_TARGET_BACK_OFFSET
    elif _task2_waist_uses_upper_layer_y_cross(cell) and layer_sequence == 3:
        local_y += TASK2_UPPER_LAYER_Y_CROSS_TARGET_BACK_OFFSET
    if _task2_waist_uses_third_layer_cross_forward_target(cell):
        if layer_sequence == 2:
            local_y += TASK2_THIRD_LAYER_CROSS_TARGET_FORWARD_OFFSET
        elif layer_sequence == 3:
            local_y -= TASK2_THIRD_LAYER_CROSS_TARGET_FORWARD_OFFSET
        elif layer_sequence == 4:
            local_x += TASK2_THIRD_LAYER_CROSS_TARGET_FORWARD_OFFSET
        elif layer_sequence == 5:
            local_x -= TASK2_THIRD_LAYER_CROSS_TARGET_FORWARD_OFFSET
    if cell.layer == 3 and layer_sequence == 4:
        local_x -= TASK2_FOURTH_LAYER_X_CROSS_OUTWARD_TARGET_OFFSET
    elif cell.layer == 3 and layer_sequence == 5:
        local_x += TASK2_FOURTH_LAYER_X_CROSS_OUTWARD_TARGET_OFFSET
    return (
        PALLET_CENTER[0] + local_x,
        PALLET_CENTER[1] + local_y,
        base.PALLET_SURFACE_Z + (cell.layer + 0.5) * LAYER_HEIGHT,
    )


def _task2_waist_box_size_for_cell(cell) -> tuple[float, float, float]:
    return TASK2_CELL_SPECS[_task2_waist_cell_layer_sequence(cell)].size


def _task2_waist_uses_y_row_strategy(cell) -> bool:
    return _task2_waist_cell_layer_sequence(cell) in TASK2_Y_ROW_STRATEGY_BOXES


def _task2_waist_direct_side_for_cell(cell) -> str:
    if _task2_waist_uses_y_row_strategy(cell):
        if cell.iy <= 1:
            return "-Y"
        if cell.iy >= base.GRID_Y - 2:
            return "+Y"
    if cell.ix == 0:
        return "-X"
    if cell.ix == base.GRID_X - 1:
        return "+X"
    if cell.iy <= 1:
        return "-Y"
    if cell.iy >= base.GRID_Y - 2:
        return "+Y"
    return "-X"


def _task2_waist_uses_corner(cell) -> bool:
    return _task2_waist_cell_layer_sequence(cell) in TASK2_INNER_EDGE_PUSH_BOXES


def _task2_waist_inner_hand_side_for_cell(cell) -> str | None:
    side = _task2_waist_direct_side_for_cell(cell)
    if cell.ix == 0:
        return "right" if side == "-Y" else "left"
    if cell.ix == base.GRID_X - 1:
        return "left" if side == "-Y" else "right"
    return None


def _task2_waist_outer_hand_side_for_cell(cell) -> str | None:
    inner_side = _task2_waist_inner_hand_side_for_cell(cell)
    if inner_side == "left":
        return "right"
    if inner_side == "right":
        return "left"
    return None


def _task2_waist_stand_off_for_cell(cell, base_stand_off: float) -> float:
    waist = base.waist_stack
    layer_sequence = _task2_waist_cell_layer_sequence(cell)
    if layer_sequence == 1:
        return float(base_stand_off)
    if layer_sequence in {2, 3, 4, 5}:
        stand_off = max(float(base_stand_off), waist.STACK_CROSS_NEIGHBOR_STAND_OFF)
        if cell.layer == 2:
            return max(float(base_stand_off), stand_off - TASK2_THIRD_LAYER_CROSS_STAND_OFF_FORWARD_DELTA)
        return stand_off
    if cell.layer == 2 and _task2_waist_uses_corner(cell):
        return max(float(base_stand_off), waist.THIRD_LAYER_OUTER_CORNER_STAND_OFF - TASK2_UPPER_LAYER_CORNER_STAND_OFF_REDUCTION)
    if cell.layer == 3 and _task2_waist_uses_corner(cell):
        return max(float(base_stand_off), waist.FOURTH_LAYER_OUTER_CORNER_STAND_OFF - TASK2_UPPER_LAYER_CORNER_STAND_OFF_REDUCTION)
    if _task2_waist_uses_y_row_strategy(cell) and cell.iy in {0, base.GRID_Y - 1}:
        return max(float(base_stand_off), waist.STACK60_DIRECT_Y_EXTENSION_STAND_OFF)
    if _task2_waist_direct_side_for_cell(cell) in {"-Y", "+Y"}:
        return max(float(base_stand_off), waist.STACK60_DIRECT_Y_SIDE_STAND_OFF)
    return max(float(base_stand_off), waist.STACK60_DIRECT_OUTER_STAND_OFF)


def _task2_waist_y_release_clearance_for_cell(cell) -> float:
    waist = base.waist_stack
    if _task2_waist_uses_upper_layer_y_cross(cell):
        return waist.FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE + TASK2_UPPER_LAYER_Y_CROSS_RELEASE_BACK_EXTRA
    if _task2_waist_uses_corner(cell):
        return waist.THIRD_LAYER_CROSS_CORNER_RELEASE_CLEARANCE + TASK2_UPPER_LAYER_CORNER_PREVIOUS_BOX_CLEARANCE
    if cell.iy in {0, base.GRID_Y - 1}:
        return waist.FIRST_LAYER_Y_EXTENSION_RELEASE_CLEARANCE
    return waist.FIRST_LAYER_Y_NEIGHBOR_RELEASE_CLEARANCE


def _task2_waist_inner_hand_x_clearance_for_cell(cell) -> float:
    if not _task2_waist_uses_y_row_strategy(cell):
        return 0.0
    if _task2_waist_uses_corner(cell):
        hand_clearance = TASK2_UPPER_LAYER_CORNER_INNER_HAND_CLEARANCE
        if cell.ix == 0:
            return -hand_clearance
        if cell.ix == base.GRID_X - 1:
            return hand_clearance
    return 0.0


def _task2_waist_cross_center_release_offset(cell) -> tuple[float, float]:
    waist = base.waist_stack
    layer_sequence = _task2_waist_cell_layer_sequence(cell)
    clearance = waist.STACK_CROSS_CENTER_RELEASE_CLEARANCE
    if cell.layer == 2 and layer_sequence in TASK2_X_CROSS_BOXES:
        clearance += waist.THIRD_LAYER_X_NEIGHBOR_CENTER_RELEASE_EXTRA
    if layer_sequence == 2:
        return (0.0, -clearance)
    if layer_sequence == 3:
        return (0.0, clearance)
    if layer_sequence == 4:
        return (-clearance, 0.0)
    if layer_sequence == 5:
        return (clearance, 0.0)
    return (0.0, 0.0)


def _task2_waist_release_center_for_cell(cell, target_center: tuple[float, float, float]) -> tuple[float, float, float]:
    waist = base.waist_stack
    layer_sequence = _task2_waist_cell_layer_sequence(cell)
    cross_dx, cross_dy = _task2_waist_cross_center_release_offset(cell)
    target_center = (target_center[0] + cross_dx, target_center[1] + cross_dy, target_center[2])
    if layer_sequence == 4:
        return (target_center[0] - waist.FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE, target_center[1], target_center[2])
    if layer_sequence == 5:
        return (target_center[0] + waist.FIRST_LAYER_X_NEIGHBOR_RELEASE_CLEARANCE, target_center[1], target_center[2])
    if _task2_waist_uses_y_row_strategy(cell):
        side = _task2_waist_direct_side_for_cell(cell)
        y_clearance = _task2_waist_y_release_clearance_for_cell(cell)
        x_clearance = _task2_waist_inner_hand_x_clearance_for_cell(cell)
        if side == "-Y":
            return (target_center[0] + x_clearance, target_center[1] - y_clearance, target_center[2])
        if side == "+Y":
            return (target_center[0] + x_clearance, target_center[1] + y_clearance, target_center[2])
    return target_center


def _task2_waist_stance_center_for_cell(cell, target_center: tuple[float, float, float]) -> tuple[float, float, float]:
    if _task2_waist_uses_corner(cell):
        return (
            target_center[0] + _task2_waist_inner_hand_x_clearance_for_cell(cell),
            target_center[1],
            target_center[2],
        )
    return target_center


def _task2_waist_direct_root_pose(cell, target_center: tuple[float, float, float], stand_off: float):
    waist = base.waist_stack
    side = _task2_waist_direct_side_for_cell(cell)
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
        raise ValueError(f"Unsupported task2 waist side: {side}")
    yaw = math.atan2(target_center[1] - final_y, target_center[0] - final_x)
    final_root = waist.RootPose(final_x, final_y, 0.0, yaw)
    tmp_root = waist.RootPose(final_x + normal[0] * waist.DIRECT_TMP_RETREAT, final_y + normal[1] * waist.DIRECT_TMP_RETREAT, 0.0, yaw)
    return final_root, tmp_root


def _task2_waist_uses_inner_hand_hold_after_release(cell) -> bool:
    return _task2_waist_uses_corner(cell)


def _task2_is_y_cross_box_size(box_size: tuple[float, float, float]) -> bool:
    return all(abs(float(actual) - expected) < 1e-6 for actual, expected in zip(box_size, Y_CROSS_BOX_SIZE))


def _task2_waist_solve_box_grip(
    kin,
    dof_names: list[str],
    lower: torch.Tensor,
    upper: torch.Tensor,
    q_start: torch.Tensor,
    root,
    box_center_world: tuple[float, float, float],
    side_gap: float,
    contact_z: float,
    iterations: int,
    box_size: tuple[float, float, float] = base.waist_stack.STACK_BOX_SIZE,
    contact_forward: float = base.waist_stack.STACK_GRASP_CONTACT_X_OFFSET,
):
    active_cell = _TASK2_WAIST_ACTIVE_CELL
    if active_cell is not None and _task2_waist_uses_fourth_layer_cross_third_layer_two_grasp(active_cell):
        contact_forward += TASK2_WAIST_Y_CROSS_GRASP_FORWARD_OFFSET
        contact_forward -= TASK2_THIRD_LAYER_Y_CROSS_GRASP_BACK_DELTA
    elif _task2_is_y_cross_box_size(box_size):
        if active_cell is None or not _task2_waist_uses_fourth_layer_y_cross(active_cell):
            contact_forward += TASK2_WAIST_Y_CROSS_GRASP_FORWARD_OFFSET
            if active_cell is not None and _task2_waist_uses_upper_layer_y_cross(active_cell):
                contact_forward -= TASK2_THIRD_LAYER_Y_CROSS_GRASP_BACK_DELTA
    return _WAIST_ORIGINAL_SOLVE_BOX_GRIP(
        kin,
        dof_names,
        lower,
        upper,
        q_start,
        root,
        box_center_world,
        side_gap,
        contact_z,
        iterations,
        box_size=box_size,
        contact_forward=contact_forward,
    )


def _uses_task2_waist_physical_clamp_before_attach(cell) -> bool:
    return _task2_waist_uses_third_layer_y_cross(cell) or _task2_waist_uses_fourth_layer_cross(cell)


def _delay_task2_waist_source_attach(cell, timeline: list[object]) -> list[object]:
    if not _uses_task2_waist_physical_clamp_before_attach(cell):
        return timeline
    waist = base.waist_stack
    delayed: list[object] = []
    remaining = TASK2_WAIST_Y_CROSS_PHYSICAL_CLAMP_HOLD_FRAMES
    for frame in timeline:
        if remaining > 0 and frame.phase == "attach_hold" and frame.attached:
            delayed.append(
                waist.StackFrame(
                    "pick_physical_clamp_hold",
                    frame.root,
                    frame.q,
                    frame.box_center,
                    False,
                    frame.box_yaw,
                    frame.box_pitch,
                )
            )
            remaining -= 1
        else:
            delayed.append(frame)
    return delayed


def _task2_forward_shifted_center(frame, distance: float) -> tuple[float, float, float]:
    return (
        frame.box_center[0] + math.cos(frame.root.yaw) * float(distance),
        frame.box_center[1] + math.sin(frame.root.yaw) * float(distance),
        frame.box_center[2],
    )


def _task2_arm_seed_with_previous(
    frame_q: torch.Tensor,
    previous_q: torch.Tensor | None,
    dof_names: list[str],
) -> torch.Tensor:
    if previous_q is None:
        return frame_q.clone()
    seed = frame_q.clone()
    name_to_index = {name: index for index, name in enumerate(dof_names)}
    for joint_name in tuple(base.waist_stack.LEFT_ARM_JOINTS) + tuple(base.waist_stack.RIGHT_ARM_JOINTS):
        index = name_to_index.get(joint_name)
        if index is not None:
            seed[index] = previous_q[index]
    return seed


def _extend_fourth_layer_cross_lift_forward(
    cell,
    timeline: list[object],
    metadata: dict[str, object],
    kin,
    dof_names: list[str],
    lower: torch.Tensor,
    upper: torch.Tensor,
    arm_iterations: int,
) -> tuple[list[object], dict[str, object]]:
    if not _task2_waist_uses_fourth_layer_cross(cell):
        return timeline, metadata

    waist = base.waist_stack
    box_size = _task2_waist_box_size_for_cell(cell)
    contact_z = waist._box_grip_contact_z(box_size)
    lift_total = sum(1 for frame in timeline if frame.phase == "lift_box_safe")
    place_total = sum(1 for frame in timeline if frame.phase == "place_move_over_target")
    place_goal = next((frame.box_center for frame in reversed(timeline) if frame.phase == "place_move_over_target"), None)
    if lift_total <= 0:
        return timeline, metadata

    global _TASK2_WAIST_ACTIVE_CELL
    previous_active_cell = _TASK2_WAIST_ACTIVE_CELL
    _TASK2_WAIST_ACTIVE_CELL = cell
    try:
        adjusted: list[object] = []
        previous_q: torch.Tensor | None = None
        lift_seen = 0
        place_seen = 0
        place_start: tuple[float, float, float] | None = None
        max_extra_pos_error = 0.0
        for frame in timeline:
            shifted_center: tuple[float, float, float] | None = None
            reuse_previous_q = False
            if frame.phase == "lift_box_safe":
                lift_seen += 1
                distance = TASK2_FOURTH_LAYER_CROSS_LIFT_FORWARD * waist._smoothstep(lift_seen / float(lift_total))
                shifted_center = _task2_forward_shifted_center(frame, distance)
            elif frame.phase.startswith("move:"):
                shifted_center = _task2_forward_shifted_center(frame, TASK2_FOURTH_LAYER_CROSS_LIFT_FORWARD)
                place_start = shifted_center
                reuse_previous_q = previous_q is not None
            elif frame.phase == "place_move_over_target" and place_total > 0 and place_goal is not None:
                place_seen += 1
                if place_start is None:
                    place_start = _task2_forward_shifted_center(frame, TASK2_FOURTH_LAYER_CROSS_LIFT_FORWARD)
                alpha = waist._smoothstep(place_seen / float(place_total))
                shifted_center = (
                    place_start[0] * (1.0 - alpha) + place_goal[0] * alpha,
                    place_start[1] * (1.0 - alpha) + place_goal[1] * alpha,
                    place_start[2] * (1.0 - alpha) + place_goal[2] * alpha,
                )

            if shifted_center is None:
                adjusted.append(frame)
                previous_q = frame.q if frame.attached else None
                continue

            if reuse_previous_q:
                q = previous_q.clone()
            else:
                seed_q = _task2_arm_seed_with_previous(frame.q, previous_q, dof_names)
                q, left_report, right_report = _task2_waist_solve_box_grip(
                    kin,
                    dof_names,
                    lower,
                    upper,
                    seed_q,
                    frame.root,
                    shifted_center,
                    side_gap=0.0,
                    contact_z=contact_z,
                    iterations=arm_iterations,
                    box_size=box_size,
                )
                max_extra_pos_error = max(max_extra_pos_error, left_report.pos_error, right_report.pos_error)
            adjusted.append(
                waist.StackFrame(
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
    finally:
        _TASK2_WAIST_ACTIVE_CELL = previous_active_cell

    updated_metadata = dict(metadata)
    updated_metadata["max_pos_error"] = max(float(updated_metadata.get("max_pos_error", 0.0)), max_extra_pos_error)
    updated_metadata["task2_lift_forward"] = TASK2_FOURTH_LAYER_CROSS_LIFT_FORWARD
    return adjusted, updated_metadata


def _patch_function_closure(func, name: str, value) -> None:
    if func.__closure__ is None:
        raise RuntimeError(f"{func.__name__} has no closure")
    freevars = func.__code__.co_freevars
    try:
        index = freevars.index(name)
    except ValueError as exc:
        raise RuntimeError(f"{func.__name__} does not close over {name}") from exc
    func.__closure__[index].cell_contents = value


def _patch_waist_build_closure(name: str, value) -> None:
    _patch_function_closure(base.waist_stack._build_stack_box_timeline, name, value)


def _install_task2_waist_overrides() -> None:
    _patch_waist_build_closure("_stack_box_size_for_cell", _task2_waist_box_size_for_cell)
    _patch_waist_build_closure("_stack_cell_target", _task2_waist_cell_target)
    _patch_waist_build_closure("_stack_direct_root_pose", _task2_waist_direct_root_pose)
    _patch_waist_build_closure("_stack_direct_side_for_cell", _task2_waist_direct_side_for_cell)
    _patch_waist_build_closure("_stack_outer_hand_side_for_cell", _task2_waist_outer_hand_side_for_cell)
    _patch_waist_build_closure("_stack_release_center_for_cell", _task2_waist_release_center_for_cell)
    _patch_waist_build_closure("_stack_stance_center_for_cell", _task2_waist_stance_center_for_cell)
    _patch_waist_build_closure("_stack_stand_off_for_cell", _task2_waist_stand_off_for_cell)
    _patch_waist_build_closure("_uses_stack_inner_hand_hold_after_release", _task2_waist_uses_inner_hand_hold_after_release)
    _patch_waist_build_closure("solve_box_grip", _task2_waist_solve_box_grip)
    _patch_function_closure(base.waist_stack._append_box_grip_cartesian_segment, "solve_box_grip", _task2_waist_solve_box_grip)


def _make_task2_waist_args(args: argparse.Namespace) -> SimpleNamespace:
    waist = base.waist_stack
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
        stack_stand_off=waist.DIRECT_OUTER_STAND_OFF,
        stack_body_dz=None,
        stack_release_height=waist.STACK_RELEASE_HEIGHT,
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


def _build_task2_waist_plans(
    args: argparse.Namespace,
    asset_dof_names: list[str],
    cells: list[object],
) -> tuple[SimpleNamespace, list[object]]:
    global _TASK2_WAIST_ACTIVE_CELL

    _install_task2_waist_overrides()
    waist = base.waist_stack
    waist_args = _make_task2_waist_args(args)
    dof_names, lower, upper = waist._parse_active_dofs(waist.MOVE_URDF)
    kin = waist.UrdfKinematics(waist.MOVE_URDF)
    plans: list[object] = []
    print(f"task2_1_waist_plan_build count={len(cells)}", flush=True)
    for index, cell in enumerate(cells, start=1):
        print(f"task2_1_waist_plan_build_start {index}/{len(cells)} {cell.label}", flush=True)
        _TASK2_WAIST_ACTIVE_CELL = cell
        try:
            timeline, metadata = waist._build_stack_box_timeline(
                waist_args,
                kin,
                dof_names,
                lower,
                upper,
                cell,
                return_home=True,
            )
        finally:
            _TASK2_WAIST_ACTIVE_CELL = None
        timeline = _delay_task2_waist_source_attach(cell, timeline)
        timeline, metadata = _extend_fourth_layer_cross_lift_forward(
            cell,
            timeline,
            metadata,
            kin,
            dof_names,
            lower,
            upper,
            waist_args.arm_iterations,
        )
        plans.append(waist.StackBoxPlan(cell, timeline, metadata))
        print(
            f"task2_1_waist_plan_build_done {index}/{len(cells)} {cell.label} "
            f"side={metadata['side']} stand_off={float(metadata['stand_off']):.3f} "
            f"frames={len(timeline)} arm_pos={float(metadata['max_pos_error']):.4f} "
            f"pick_hold_delay={TASK2_WAIST_Y_CROSS_PHYSICAL_CLAMP_HOLD_FRAMES if _uses_task2_waist_physical_clamp_before_attach(cell) else 0}",
            flush=True,
        )

    if asset_dof_names != dof_names:
        reorder = torch.tensor([dof_names.index(name) for name in asset_dof_names], dtype=torch.long)
        reordered: list[object] = []
        for plan in plans:
            timeline = [
                waist.StackFrame(
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
            reordered.append(waist.StackBoxPlan(plan.cell, timeline, plan.metadata))
        plans = reordered
    return waist_args, plans


def main() -> None:
    _install_task2_first_layer_overrides()

    # task2_1 是新垛型的基线脚本：保留分层代理调试模式，便于单独验证第二层或三四层。
    parser = argparse.ArgumentParser(
        description="Run task2 36-box layer-1..4 stacking demo.",
        epilog=(
            "常用示例:\n"
            "  python -m move.tasks.task2_1\n"
            "  python -m move.tasks.task2_1 --until-box 9\n"
            "  python -m move.tasks.task2_1 --second-layer-proxy\n"
            "  python -m move.tasks.task2_1 --third-fourth-proxy --viewer-render-every 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--headless", action="store_true", help="Run without creating an Isaac Gym viewer.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many simulation frames; 0 runs to task end.")
    parser.add_argument("--until-box", type=int, default=0, help="Highest 1-based task2 box sequence number to run; 0 uses the mode default.")
    parser.add_argument("--stand-off", type=float, default=base.DIRECT_STAND_OFF, help="Base direct placement stand-off.")
    parser.add_argument("--attach-after-pick-frames", type=int, default=base.ATTACH_AFTER_PICK_FRAMES)
    parser.add_argument("--fast-viewer", action="store_true", help="Do not throttle viewer playback to real time.")
    parser.add_argument("--viewer-render-every", type=int, default=1, help="Render the viewer once every N simulation frames.")
    parser.add_argument("--viewer-start-box", type=int, default=1, help="Skip viewer rendering before this 1-based box sequence.")
    parser.add_argument("--waist-frame-stride", type=int, default=None, help="Optional frame stride for the layer-3/4 waist-arm timeline.")
    parser.add_argument(
        "--waist-stack-plan-ik-stride",
        type=int,
        default=base.waist_stack.STACK_SEQUENCE_PLAN_IK_STRIDE,
        help="IK stride used while planning the layer-3/4 waist-arm sequence.",
    )
    parser.add_argument(
        "--second-layer-proxy",
        action="store_true",
        help="Run only the second layer, replacing the first layer with one fixed 1.0 x 1.0 x 0.4 m support block.",
    )
    parser.add_argument(
        "--third-fourth-proxy",
        action="store_true",
        help="Run only layers 3/4 with one fixed 1.0 x 1.0 x 0.8 m support block replacing layers 1/2.",
    )
    args = parser.parse_args()

    if args.viewer_render_every < 1:
        raise ValueError("--viewer-render-every must be >= 1")
    if args.viewer_start_box < 1:
        raise ValueError("--viewer-start-box must be >= 1")
    if args.waist_frame_stride is not None and args.waist_frame_stride < 1:
        raise ValueError("--waist-frame-stride must be >= 1")
    if args.waist_stack_plan_ik_stride < 1:
        raise ValueError("--waist-stack-plan-ik-stride must be >= 1")
    if args.second_layer_proxy and args.third_fourth_proxy:
        raise ValueError("--second-layer-proxy and --third-fourth-proxy are mutually exclusive")
    max_box = TASK2_LAYER_BOX_COUNT * TASK2_TOTAL_LAYERS
    if args.until_box == 0:
        args.until_box = TASK2_LAYER_BOX_COUNT * TASK2_NATIVE_LAYERS if args.second_layer_proxy else max_box
    if args.until_box < 1 or args.until_box > max_box:
        raise ValueError(f"--until-box must be in [1, {max_box}]")
    if args.second_layer_proxy and args.until_box > TASK2_LAYER_BOX_COUNT * TASK2_NATIVE_LAYERS:
        raise ValueError("--second-layer-proxy only supports debugging boxes 10..18")

    native_order = list(build_task2_order(TASK2_NATIVE_LAYERS))
    waist_order = list(build_task2_waist_order())
    first_requested_box = 1
    if args.second_layer_proxy:
        first_requested_box = TASK2_LAYER_BOX_COUNT + 1
        if args.until_box < first_requested_box:
            args.until_box = first_requested_box
        cells = list(native_order[first_requested_box - 1 : args.until_box])
    elif args.third_fourth_proxy:
        first_requested_box = TASK2_WAIST_START_BOX
        if args.until_box < first_requested_box:
            args.until_box = first_requested_box
        cells = []
    else:
        cells = [cell for cell in native_order if cell.sequence <= min(args.until_box, TASK2_LAYER_BOX_COUNT * TASK2_NATIVE_LAYERS)]
    if args.second_layer_proxy:
        waist_cells = []
    elif args.third_fourth_proxy:
        waist_cells = [cell for cell in waist_order if first_requested_box <= cell.sequence <= args.until_box]
    else:
        waist_cells = [cell for cell in waist_order if cell.sequence <= args.until_box]
    if not cells and not waist_cells:
        raise RuntimeError("No boxes selected to run")
    unsupported = [cell for cell in cells if not base._supports_direct_box_plan(cell)]
    if unsupported:
        first = unsupported[0]
        raise NotImplementedError(f"{first.label} is not supported by task2_1 strategy")

    plans = [_delay_second_layer_release_until_hold_end(base.build_direct_box_plan(cell, args.stand_off)) for cell in cells]
    for plan in plans:
        if plan.pick_max_error > base.PICK_ERROR_LIMIT or plan.place_max_error > base.PLACE_ERROR_LIMIT:
            raise RuntimeError(
                f"IK infeasible for {plan.cell.label}: pick_error={plan.pick_max_error:.4f}, "
                f"place_error={plan.place_max_error:.4f}"
            )

    gym = gymapi.acquire_gym()
    sim = base.create_sim(gym)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    robot_asset = base.load_robot_asset(gym, sim)
    asset_dof_names = list(gym.get_asset_dof_names(robot_asset))
    plans = _reorder_plans(plans, asset_dof_names)
    timelines = [build_task2_timeline(plan) for plan in plans]
    waist_args, waist_plans = _build_task2_waist_plans(args, asset_dof_names, waist_cells) if waist_cells else (None, [])

    scene_cells = cells + waist_cells
    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(6.8, 6.8, 3.0), 1)
    proxy_layers = 2 if args.third_fourth_proxy else 1 if args.second_layer_proxy else 0
    robot, boxes = _create_task2_static_scene(gym, sim, env, robot_asset, scene_cells, proxy_layers=proxy_layers)
    if timelines:
        base._configure_robot_dofs(gym, env, robot, asset_dof_names)
    else:
        base._configure_waist_stack_robot_dofs(gym, env, robot, asset_dof_names)

    if timelines:
        initial_q = timelines[0][0][2]
    else:
        initial_q = waist_plans[0].timeline[0].q
    base._set_robot_initial_q(gym, env, robot, initial_q)

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

    first_running = scene_cells[0].sequence
    print("task2_1 layer-1..4 stacking demo")
    print(f"  stack_size={STACK_SIZE} layer_height={LAYER_HEIGHT:.3f} running_boxes={first_running}..{args.until_box}")
    if args.third_fourth_proxy:
        print("  mode=waist_stack_layers_3_4_debug lower_proxy=(1.0,1.0,0.8)")
    else:
        print("  mode=native_layers_1_2 waist_stack_layers_3_4")
    if args.second_layer_proxy:
        print("  first_layer_proxy=(1.0,1.0,0.4)")
    for plan, timeline in zip(plans, timelines):
        spec = TASK2_CELL_SPECS[_task2_cell_layer_sequence(plan.cell)]
        print(
            f"  {plan.cell.label} {spec.description} size={spec.size} "
            f"target=({plan.target_center[0]:.3f},{plan.target_center[1]:.3f},{plan.target_center[2]:.3f}) "
            f"ik_errors pick={plan.pick_max_error:.4f} place={plan.place_max_error:.4f} frames={len(timeline)}"
        )
    for plan in waist_plans:
        spec = TASK2_CELL_SPECS[_task2_waist_cell_layer_sequence(plan.cell)]
        meta = plan.metadata
        print(
            f"  waist:{plan.cell.label} {spec.description} size={spec.size} "
            f"side={meta['side']} stand_off={float(meta['stand_off']):.3f} "
            f"target=({meta['target_center'][0]:.3f},{meta['target_center'][1]:.3f},{meta['target_center'][2]:.3f}) "
            f"release=({meta['release_center'][0]:.3f},{meta['release_center'][1]:.3f},{meta['release_center'][2]:.3f}) "
            f"arm_pos={float(meta['max_pos_error']):.4f} frames={len(plan.timeline)}"
        )

    global_frame = 0
    placed: list[base.PlacedBoxPose] = []
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
            print(f"task2_1_start_box box={plan.cell.sequence} label={plan.cell.label}")
            global_frame, keep_running, box_error = base._run_box_timeline(
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
                args.attach_after_pick_frames,
                global_frame,
            )
            if not keep_running:
                break
            gym.refresh_actor_root_state_tensor(sim)
            base._report_placed_box_motion(root_states, placed, f"after_box_{plan.cell.sequence}")
            completed_errors.append(box_error)
            completed_native += 1
        if keep_running and waist_plans:
            base._configure_waist_stack_robot_dofs(gym, env, robot, asset_dof_names)
            base._set_robot_initial_q(gym, env, robot, waist_plans[0].timeline[0].q)
            for plan, box, box_index in zip(waist_plans, waist_boxes, waist_box_indices):
                if args.max_frames > 0 and global_frame >= args.max_frames:
                    keep_running = False
                    break
                print(f"task2_1_waist_start_box box={plan.cell.sequence} label={plan.cell.label}")
                global_frame, keep_running = base.waist_stack._run_stack_box_timeline(
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
    finally:
        if "sim" in locals() and sim is not None and "root_states" in locals() and "placed" in locals():
            gym.refresh_actor_root_state_tensor(sim)
            base._report_placed_box_motion(root_states, placed, "final")
        print(
            "task2_1_result "
            f"completed_native={completed_native}/{len(plans)} "
            f"completed_waist={completed_waist}/{len(waist_plans)} "
            f"max_release_error={max(completed_errors) if completed_errors else 0.0:.4f} "
            f"mean_release_error={sum(completed_errors) / len(completed_errors) if completed_errors else 0.0:.4f}"
        )
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
