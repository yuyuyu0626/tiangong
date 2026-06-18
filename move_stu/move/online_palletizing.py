#!/usr/bin/env python3
"""Generate online 3D palletizing plans for Tianyi Isaac Gym execution."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from move.pct_policy_bridge import PctPlacement, run_pct_policy
from move.planning import PALLET_SIZE, STACK_HEIGHT_LIMIT


DISCRETE_SIZES_M = (0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass(frozen=True)
class PalletizingCase:
    case_id: int
    seed: int
    pallet_size_m: tuple[float, float, float]
    item_sizes_m: tuple[tuple[float, float, float], ...]
    placements: tuple[PctPlacement, ...]




@dataclass(frozen=True)
class BoxSpec:
    item_index: int
    original_size_m: tuple[float, float, float]


def iter_online_items(seed: int, max_items: int):
    rng = random.Random(seed)
    item_range = range(max_items) if max_items > 0 else itertools.count()
    for item_idx in item_range:
        yield BoxSpec(
            item_index=item_idx,
            original_size_m=tuple(rng.choice(DISCRETE_SIZES_M) for _ in range(3)),
        )


def sample_items(seed: int, count: int) -> tuple[tuple[float, float, float], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.choice(DISCRETE_SIZES_M) for _ in range(3)) for _ in range(count))  # type: ignore[return-value]


def build_cases(
    model_path: Path,
    pct_root: Path,
    out_dir: Path,
    cases: int = 5,
    items_per_case: int = 12,
    seed: int = 20260614,
    device: str | None = None,
) -> list[PalletizingCase]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built: list[PalletizingCase] = []
    for case_idx in range(cases):
        case_seed = seed + case_idx
        items = sample_items(case_seed, items_per_case)
        placements = tuple(run_pct_policy(items, model_path=model_path, pct_root=pct_root, device=device))
        case = PalletizingCase(
            case_id=case_idx,
            seed=case_seed,
            pallet_size_m=(PALLET_SIZE[0], PALLET_SIZE[1], STACK_HEIGHT_LIMIT),
            item_sizes_m=items,
            placements=placements,
        )
        path = out_dir / f"case_{case_idx:02d}.json"
        path.write_text(json.dumps(asdict(case), indent=2), encoding="utf-8")
        built.append(case)
    summary = [
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "items": len(case.item_sizes_m),
            "final_utilization": case.placements[-1].utilization if case.placements else 0.0,
            "plan_file": str((out_dir / f"case_{case.case_id:02d}.json").resolve()),
        }
        for case in built
    ]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5+ online PCT palletizing cases.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pct-root", type=Path, default=Path("/2024233240/external/Online-3D-BPP-PCT"))
    parser.add_argument("--out-dir", type=Path, default=Path("/2024233240/move/outputs/palletizing_cases"))
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--items-per-case", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    built = build_cases(args.model_path, args.pct_root, args.out_dir, args.cases, args.items_per_case, args.seed, args.device)
    for case in built:
        util = case.placements[-1].utilization if case.placements else 0.0
        print(f"case={case.case_id:02d} seed={case.seed} items={len(case.item_sizes_m)} utilization={util:.3f}")


if __name__ == "__main__":
    main()
