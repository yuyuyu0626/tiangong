#!/usr/bin/env python3
"""Build per-item dual-arm IK plans for PCT palletizing cases."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from move.grab_test import build_custom_stack_scene, run
from move.robot_placement import classify_orientation
from move.planning import BoxPlacement, TABLE_POSE, TABLE_SIZE

STAND_OFF_CANDIDATES = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45)


def source_center(size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (TABLE_POSE[0] - 0.10, TABLE_POSE[1], TABLE_POSE[2] + TABLE_SIZE[2] * 0.5 + size[2] * 0.5)


def build_case(case_path: Path, out_path: Path, max_items: int = 0) -> None:
    case = json.loads(case_path.read_text())
    placements = case["placements"][: max_items or None]
    placed: list[BoxPlacement] = []
    item_plans = []
    for idx, placement in enumerate(placements):
        original_size = tuple(float(v) for v in placement["original_size_m"])
        placed_size = tuple(float(v) for v in placement["placed_size_m"])
        orientation = classify_orientation(original_size, placed_size)
        if not orientation.supported:
            print(
                f"case={case['case_id']:02d} item={idx:02d} original_size={original_size} placed_size={placed_size} "
                f"orientation={orientation.type} skipped=unsupported_flip",
                flush=True,
            )
            break
        target_yaw = float(orientation.target_yaw_options[0])
        target = BoxPlacement(f"item_{idx:02d}", tuple(float(v) for v in placement["min_corner_m"]), placed_size)
        source = source_center(original_size)
        best = None
        best_payload = None
        for stand_off in STAND_OFF_CANDIDATES:
            scene = build_custom_stack_scene(f"case_{case['case_id']:02d}_item_{idx:02d}", tuple(placed), target, stand_off=stand_off, final_yaw=target_yaw)
            source_yaw = target_yaw - scene.final_pose.yaw
            report, payload = run(
                place_mode="move",
                stand_off=stand_off,
                source_pose=source,
                source_yaw=source_yaw,
                box_size=original_size,
                target_aabb_size=placed_size,
                target_yaw=target_yaw,
                scene=scene,
            )
            score = report.place_max_error + report.pick_max_error
            if best is None or score < best[0]:
                best = (score, stand_off, report)
                best_payload = payload
            if report.pick_feasible and report.place_feasible:
                break
        assert best is not None and best_payload is not None
        score, stand_off, report = best
        item_plans.append(
            {
                "item_index": idx,
                "source_center": source,
                "source_yaw": source_yaw,
                "target_yaw": target_yaw,
                "orientation_type": orientation.type,
                "target_box": target,
                "stand_off": stand_off,
                "report": asdict(report),
                "plan": best_payload,
            }
        )
        placed.append(target)
        print(
            f"case={case['case_id']:02d} item={idx:02d} original_size={original_size} placed_size={placed_size} "
            f"stand_off={stand_off:.2f} pick_err={report.pick_max_error:.4f} "
            f"place_err={report.place_max_error:.4f} feasible={report.pick_feasible and report.place_feasible}",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"case": case, "item_plans": item_plans}, out_path)
    print(f"wrote {out_path} items={len(item_plans)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=None)
    parser.add_argument("--case-dir", type=Path, default=Path("/2024233240/move/outputs/palletizing_cases"))
    parser.add_argument("--out-dir", type=Path, default=Path("/2024233240/move/outputs/palletizing_ik_plans"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args()
    cases = [args.case] if args.case else sorted(args.case_dir.glob("case_*.json"))
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]
    for case_path in cases:
        build_case(case_path, args.out_dir / f"{case_path.stem}_ik.pt", args.max_items)


if __name__ == "__main__":
    main()
