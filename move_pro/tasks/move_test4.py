#!/usr/bin/env python3
"""Move test 4: pallet-ground EMS leaf on the right side of the first 0.3m box."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from move.planning import BOX_SIZE, PALLET_SURFACE_Z, BoxPlacement
from move.tasks import move_test3 as base

base.TEST_TITLE = "Move test 4: pallet-ground right-side EMS leaf beside the first 0.3m box"
base.TEST_DESCRIPTION = "Run the pallet-ground right-side EMS leaf lift/IK scene beside the first 0.3m box."


def build_height_test_scenes() -> tuple[base.HeightTestScene, ...]:
    stack = base._make_stack_scene(
        (
            BoxPlacement("scene2_box_1_0p30_base", (0.0, 0.0, 0.0), (0.30, 0.30, 0.30)),
            BoxPlacement("scene2_box_2_0p20_on_top", (0.05, 0.05, 0.30), (0.20, 0.20, 0.20)),
        ),
        BoxPlacement("scene2_box_3_target_ground_leaf_right_of_0p30", (0.0, 0.30, 0.0), BOX_SIZE),
    )
    support_z = PALLET_SURFACE_Z
    final, tmp = base.solve_root_pose_for_target(stack, support_z)
    return (
        base.HeightTestScene(
            "scene2_ground_leaf_right_of_0p30",
            "third 0.2m cube target on the pallet-ground EMS leaf at the robot-right side of the first 0.3m cube",
            stack,
            support_z,
            final,
            tmp,
            base.build_waypoints(stack, final, tmp),
            False,
        ),
    )


base.build_height_test_scenes = build_height_test_scenes


if __name__ == "__main__":
    base.main()
