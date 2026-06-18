#!/usr/bin/env python3
"""move_pro — 智能码垛系统 CLI 入口。

使用方法::

    # 离线 BPP 决策预览（无 Isaac Gym 依赖）
    python -m move_pro.run --mode plan --method LSAH --num-boxes 30

    # Isaac Gym 仿真
    python -m move_pro.run --mode sim --method LSAH --num-boxes 20

    # 指定不同方法
    python -m move_pro.run --mode plan --method OnlineBPH
    python -m move_pro.run --mode plan --method DBL
    python -m move_pro.run --mode plan --method BR

模式说明:
    plan    仅 BPP 决策 + 离线 IK 规划，可脱离 Isaac Gym 运行
    sim     完整 Isaac Gym 仿真（需要 GPU + Isaac Gym）
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from move_pro.config import DEFAULT_ITEM_SET, pallet_center_world, PALLET_SIZE
# 注意：BPPDecider 在 cmd_plan 内部延迟导入，
# 以避免在 sim 模式中早于 isaacgym 导入 torch。


def _build_box_sequence(args):
    """按方法构造箱子尺寸序列（整数 bin 尺寸）。

    - PCT 方法：从 PCT 物品集 (1..5) 采样，或从 --dataset 指定的 PCT 数据集读取一条轨迹；
    - 启发式方法：从 DEFAULT_ITEM_SET (1..4) 采样。
    """
    random.seed(args.seed)
    if args.method == "PCT":
        from move_pro.config import PCT_SAMPLE_ITEM_SET

        if getattr(args, "dataset", None):
            return _load_dataset_sequence(args.dataset, args.num_boxes, args.seed)
        # 采样走收窄的 2..4 区间（抓取友好），模型推理 item_set 仍是 1..5（见 config 注释）。
        return [random.choice(PCT_SAMPLE_ITEM_SET) for _ in range(args.num_boxes)]
    return [random.choice(DEFAULT_ITEM_SET) for _ in range(args.num_boxes)]


def _load_dataset_sequence(dataset_path, num_boxes, seed):
    """从 PCT 数据集 (dataset_setting123_discrete.pt) 取一条箱子轨迹。

    数据集每条轨迹是 [(x,y,z,density), ...]；阶段一只用整数尺寸 (x,y,z)。
    """
    import torch

    trajs = torch.load(dataset_path, map_location="cpu")
    idx = seed % len(trajs)
    boxes = []
    for item in trajs[idx][:num_boxes]:
        boxes.append((int(item[0]), int(item[1]), int(item[2])))
    return boxes


def cmd_plan(args):
    """离线 BPP 决策 + 放置预览。"""
    from move_pro.bpp_decider import BPPDecider
    from move_pro.integrator import MoveProIntegrator

    boxes = _build_box_sequence(args)

    print(f"\n{'='*60}")
    print(f"  move_pro 智能码垛 — 离线决策模式")
    print(f"  方法: {args.method}, 箱数: {args.num_boxes}, "
          f"容器: 10×10×16 (PCT) = {PALLET_SIZE[0]}×{PALLET_SIZE[1]}×1.6m")
    print(f"{'='*60}\n")

    integrator = MoveProIntegrator(method=args.method)
    plan = integrator.build_plan(boxes, compute_ik=False, sizes_are_pct=True)
    print(plan.summary())

    # 世界坐标可视化（文本）
    if args.verbose:
        print(f"\n{'='*60}")
        print("  世界坐标放置图 (俯视图, 单位: m)")
        print(f"{'='*60}")
        pc = pallet_center_world()
        sx, sy = PALLET_SIZE
        print(f"  托盘: ({pc[0]-sx/2:.1f}, {pc[1]-sy/2:.1f}) → "
              f"({pc[0]+sx/2:.1f}, {pc[1]+sy/2:.1f})")
        for bt in plan.box_tasks:
            if bt.placement.feasible:
                print(f"  箱#{bt.index:02d} {bt.original_size} → "
                      f"({bt.world_target[0]:.2f}, {bt.world_target[1]:.2f}, "
                      f"{bt.world_target[2]:.2f})")
    print()


def cmd_sim(args):
    """完整 Isaac Gym 仿真。"""
    from move_pro.simulation import MoveProSimulator

    boxes = _build_box_sequence(args)

    print(f"\n{'='*60}")
    print(f"  move_pro 智能码垛 — Isaac Gym 仿真模式")
    print(f"  方法: {args.method}, 箱数: {args.num_boxes}")
    print(f"{'='*60}\n")

    sim = MoveProSimulator(method=args.method)
    sim.run(boxes, sizes_are_pct=True,
            headless=args.headless, fast=args.fast,
            max_frames=args.max_frames)


def main():
    parser = argparse.ArgumentParser(
        description="move_pro — 智能码垛系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m move_pro.run --mode plan --method LSAH --num-boxes 30
  python -m move_pro.run --mode sim --method LSAH --num-boxes 20
  python -m move_pro.run --mode plan --method OnlineBPH --verbose
        """,
    )

    parser.add_argument(
        "--mode", type=str, default="plan",
        choices=("plan", "sim"),
        help="运行模式: plan=离线计划, sim=Isaac Gym 仿真",
    )
    parser.add_argument(
        "--method", type=str, default="LSAH",
        choices=("LSAH", "OnlineBPH", "DBL", "BR", "PCT"),
        help="BPP 决策方法：启发式(LSAH/OnlineBPH/DBL/BR) 或 PCT 可学习模型",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="[PCT] 从 PCT 数据集(.pt)读取箱子序列，不指定则随机采样 PCT 物品集(1..5)",
    )
    parser.add_argument(
        "--num-boxes", type=int, default=25,
        help="箱子数量",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出（世界坐标列表）",
    )

    # 仿真专用参数
    parser.add_argument(
        "--headless", action="store_true",
        help="[sim] 不弹出 viewer 窗口",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="[sim] 最大仿真帧数，0=不限",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="[sim] 加速播放（不等真实时间，减少插值帧）",
    )

    args = parser.parse_args()

    if args.mode == "plan":
        cmd_plan(args)
    elif args.mode == "sim":
        cmd_sim(args)


if __name__ == "__main__":
    main()
