from __future__ import annotations

import unittest

from move_pro.bpp_decider import BPPDecider
from move_pro.integrator import MoveProIntegrator


def _overlap_1d(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


class BPPDeciderTests(unittest.TestCase):
    def test_all_heuristics_pack_inside_container_without_overlap(self):
        boxes = [(3, 2, 2), (2, 4, 2), (2, 2, 3), (1, 3, 2), (3, 3, 1)]
        for method in ("LASH", "OnlineBPH", "DBL", "BR"):
            with self.subTest(method=method):
                decider = BPPDecider(method=method)
                placements = [decider.decide_and_commit(box) for box in boxes]
                self.assertTrue(all(item.feasible for item in placements))
                for item in placements:
                    self.assertGreaterEqual(item.lx, 0)
                    self.assertGreaterEqual(item.ly, 0)
                    self.assertGreaterEqual(item.lz, 0)
                    self.assertLessEqual(item.lx + item.x, 10)
                    self.assertLessEqual(item.ly + item.y, 10)
                    self.assertLessEqual(item.lz + item.z, 16)
                for index, left in enumerate(placements):
                    for right in placements[index + 1 :]:
                        overlaps = (
                            _overlap_1d(left.lx, left.lx + left.x, right.lx, right.lx + right.x)
                            and _overlap_1d(left.ly, left.ly + left.y, right.ly, right.ly + right.y)
                            and _overlap_1d(left.lz, left.lz + left.z, right.lz, right.lz + right.z)
                        )
                        self.assertFalse(overlaps)

    def test_failed_placement_does_not_mutate_space(self):
        decider = BPPDecider(container_size=(2, 2, 2))
        placement = decider.decide((3, 1, 1))
        self.assertFalse(placement.feasible)
        self.assertEqual(decider.packed_count, 0)
        self.assertEqual(decider.utilization(), 0.0)

    def test_default_rotation_keeps_height_upright(self):
        decider = BPPDecider()
        placement = decider.decide((2, 3, 4))
        self.assertEqual(placement.z, 4)
        self.assertIn((placement.x, placement.y), {(2, 3), (3, 2)})

    def test_world_mapping_and_plan_are_consistent(self):
        plan = MoveProIntegrator().build_plan(
            [(0.2, 0.3, 0.4), (0.1, 0.2, 0.2)]
        )
        self.assertEqual(plan.placed_boxes, 2)
        self.assertGreater(plan.utilization, 0.0)
        for task in plan.box_tasks:
            self.assertTrue(task.placement.feasible)
            self.assertEqual(task.world_target, task.placement.world_center)
            self.assertEqual(task.world_size[2], task.original_size[2])

    def test_commit_rejects_infeasible_placement(self):
        decider = BPPDecider(container_size=(1, 1, 1))
        placement = decider.decide((2, 1, 1))
        with self.assertRaises(ValueError):
            decider.commit(placement)


if __name__ == "__main__":
    unittest.main()
