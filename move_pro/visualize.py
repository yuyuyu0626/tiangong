"""move_pro 可视化模块。

提供托盘 2D/3D 布局的可视化，无需 Isaac Gym。
支持：俯视图分层着色、3D 散点图。
"""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: F811
from typing import Optional

# 路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from move_pro.integrator import MoveProIntegrator, MoveProPlan  # noqa: E402
from move_pro.config import pallet_center_world, PALLET_SIZE  # noqa: E402


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("TkAgg")  # 支持交互式窗口
        import matplotlib.pyplot as plt  # noqa: F811
        return plt
    except ImportError:
        raise SystemExit(
            "matplotlib 未安装。请执行: pip install matplotlib"
        )


def plot_topdown(plan: MoveProPlan, title: str = "move_pro 智能码垛布局"):
    """托盘俯视图：按高度分层着色。

    Parameters
    ----------
    plan : MoveProPlan
        由 ``MoveProIntegrator.build_plan()`` 生成的计划。
    title : str
        图表标题。
    """
    plt = _ensure_matplotlib()

    pc = pallet_center_world()
    sx, sy = PALLET_SIZE
    x0, y0 = pc[0] - sx / 2, pc[1] - sy / 2

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(x0 - 0.05, x0 + sx + 0.05)
    ax.set_ylim(y0 - 0.05, y0 + sy + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    # 托盘边框
    rect = plt.Rectangle((x0, y0), sx, sy, fill=False,
                          edgecolor="black", linewidth=2, linestyle="--")
    ax.add_patch(rect)

    # 按高度排序
    sorted_boxes = sorted(
        [bt for bt in plan.box_tasks if bt.placement.feasible],
        key=lambda bt: bt.world_target[2],
    )

    max_z = max((bt.world_target[2] for bt in sorted_boxes), default=1.0)
    cmap = plt.cm.viridis

    for bt in sorted_boxes:
        wx, wy, wz = bt.world_target
        # 原始尺寸映射到世界坐标
        # PCT 尺寸 → 世界尺寸
        from move_pro.config import BIN_TO_WORLD_SCALE
        pct_x, pct_y, pct_z = bt.pct_size
        cs_x, cs_y, cs_z = (10, 10, 16)
        bw = pct_x / cs_x * BIN_TO_WORLD_SCALE[0]
        bh = pct_y / cs_y * BIN_TO_WORLD_SCALE[1]

        # 左下角
        lx = wx - bw / 2
        ly = wy - bh / 2

        color = cmap(min(wz / max(max_z, 1e-6), 1.0))
        rect = plt.Rectangle((lx, ly), bw, bh, fill=True,
                              facecolor=color, edgecolor="black",
                              linewidth=0.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(wx, wy, str(bt.index), ha="center", va="center",
                fontsize=6, fontweight="bold", color="white")

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_z))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("高度 (m)")

    plt.tight_layout()
    plt.show()


def plot_3d(plan: MoveProPlan, title: str = "move_pro 3D 布局"):
    """3D 散点 + 线框图。"""
    plt = _ensure_matplotlib()

    pc = pallet_center_world()
    sx, sy = PALLET_SIZE
    from move_pro.config import STACK_HEIGHT_LIMIT

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # 托盘底面
    x0, y0 = pc[0] - sx / 2, pc[1] - sy / 2
    xx = [x0, x0 + sx, x0 + sx, x0, x0]
    yy = [y0, y0, y0 + sy, y0 + sy, y0]
    zz = [0, 0, 0, 0, 0]
    ax.plot(xx, yy, zz, "k--", linewidth=1, alpha=0.5)

    from move_pro.config import BIN_TO_WORLD_SCALE
    cmap = plt.cm.viridis

    for bt in plan.box_tasks:
        if not bt.placement.feasible:
            continue
        wx, wy, wz = bt.world_target
        pct_x, pct_y, pct_z = bt.pct_size
        bw = pct_x / 10 * BIN_TO_WORLD_SCALE[0]
        bh = pct_y / 10 * BIN_TO_WORLD_SCALE[1]
        bd = pct_z / 16 * BIN_TO_WORLD_SCALE[2]

        color = cmap(min(wz / STACK_HEIGHT_LIMIT, 1.0))

        # 绘制箱体线框
        xc, yc, zc = wx - bw / 2, wy - bh / 2, wz - bd / 2
        _draw_box_wireframe(ax, xc, yc, zc, bw, bh, bd, color=color, alpha=0.7)

    ax.set_xlim(x0 - 0.1, x0 + sx + 0.1)
    ax.set_ylim(y0 - 0.1, y0 + sy + 0.1)
    ax.set_zlim(0, STACK_HEIGHT_LIMIT)
    plt.tight_layout()
    plt.show()


def _draw_box_wireframe(ax, x, y, z, w, h, d, color="blue", alpha=0.5):
    """绘制 3D 箱体线框。"""
    verts = [
        [x, y, z], [x + w, y, z], [x + w, y + h, z], [x, y + h, z],
        [x, y, z + d], [x + w, y, z + d],
        [x + w, y + h, z + d], [x, y + h, z + d],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
    ]
    for e in edges:
        ax.plot3D(
            [verts[e[0]][0], verts[e[1]][0]],
            [verts[e[0]][1], verts[e[1]][1]],
            [verts[e[0]][2], verts[e[1]][2]],
            color=color, alpha=alpha, linewidth=0.8,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(
        description="move_pro 可视化工具"
    )
    parser.add_argument("--method", default="LSAH",
                        choices=("LSAH", "OnlineBPH", "DBL", "BR"))
    parser.add_argument("--num-boxes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="topdown",
                        choices=("topdown", "3d", "both"),
                        help="可视化模式")
    args = parser.parse_args()

    from move_pro.config import DEFAULT_ITEM_SET

    random.seed(args.seed)
    boxes = [random.choice(DEFAULT_ITEM_SET) for _ in range(args.num_boxes)]

    print(f"计算 {args.num_boxes} 箱 {args.method} 放置计划 ...")
    integrator = MoveProIntegrator(method=args.method)
    plan = integrator.build_plan(boxes, compute_ik=False, sizes_are_pct=True)
    print(plan.summary())

    if args.mode in ("topdown", "both"):
        plot_topdown(plan, f"move_pro — {args.method} ({args.num_boxes} 箱, "
                     f"利用率 {plan.utilization:.1%})")
    if args.mode in ("3d", "both"):
        plot_3d(plan, f"move_pro 3D — {args.method}")


if __name__ == "__main__":
    main()
