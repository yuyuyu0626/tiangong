from __future__ import annotations

import unittest

from move_pro.config import PCT_MODEL_PATH
from move_pro.pct_policy import PctModelPlanner
from move_pro.integrator import MoveProIntegrator


def _overlap_1d(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


@unittest.skipUnless(PCT_MODEL_PATH.exists(), f"PCT model not found at {PCT_MODEL_PATH}")
class PctModelPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 模型加载较慢，整个测试类共用一个 planner。
        cls.planner = PctModelPlanner()

    def test_model_loads(self):
        self.assertIsNotNone(self.planner.policy)

    def test_plan_places_inside_container_without_overlap(self):
        placements = self.planner.plan(num_boxes=40, seed=7)
        self.assertGreater(len(placements), 0)
        bx, by, bz = self.planner.container_size
        for p in placements:
            self.assertGreaterEqual(p.lx, 0)
            self.assertGreaterEqual(p.ly, 0)
            self.assertGreaterEqual(p.lz, 0)
            self.assertLessEqual(p.lx + p.x, bx)
            self.assertLessEqual(p.ly + p.y, by)
            self.assertLessEqual(p.lz + p.z, bz)
        for i, left in enumerate(placements):
            for right in placements[i + 1:]:
                overlaps = (
                    _overlap_1d(left.lx, left.lx + left.x, right.lx, right.lx + right.x)
                    and _overlap_1d(left.ly, left.ly + left.y, right.ly, right.ly + right.y)
                    and _overlap_1d(left.lz, left.lz + left.z, right.lz, right.lz + right.z)
                )
                self.assertFalse(overlaps, f"overlap between {left} and {right}")

    def test_utilization_reasonable(self):
        self.planner.plan(num_boxes=60, seed=42)
        ratio = self.planner.utilization()
        # PCT setting1 单容器利用率通常 0.6~0.8。
        self.assertGreater(ratio, 0.4)
        self.assertLessEqual(ratio, 1.0)

    def test_deterministic_for_fixed_sequence(self):
        seq = [(2, 3, 4), (2, 2, 2), (3, 3, 1), (1, 4, 2), (2, 2, 3)]
        a = self.planner.plan(box_sequence=seq)
        b = self.planner.plan(box_sequence=seq)
        self.assertEqual(
            [(p.lx, p.ly, p.lz, p.x, p.y, p.z) for p in a],
            [(p.lx, p.ly, p.lz, p.x, p.y, p.z) for p in b],
        )

    def test_stops_when_container_full(self):
        # 喂远超容量的箱子，应在放不下时停止而非报错。
        placements = self.planner.plan(num_boxes=500, seed=1)
        self.assertLess(len(placements), 500)
        self.assertGreater(len(placements), 0)

    def test_integrator_pct_method(self):
        integ = MoveProIntegrator(method="PCT")
        seq = [(2, 3, 4), (2, 2, 2), (3, 3, 1), (1, 4, 2)]
        plan = integ.build_plan(seq)
        self.assertGreater(plan.placed_boxes, 0)
        self.assertLessEqual(plan.placed_boxes, len(seq))
        for task in plan.box_tasks:
            self.assertTrue(task.placement.feasible)
            self.assertEqual(task.world_target, task.placement.world_center)
            # 世界尺寸等比 0.1：z 世界 = bin z * 0.1
            self.assertAlmostEqual(task.world_size[2], task.pct_size[2] * 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
