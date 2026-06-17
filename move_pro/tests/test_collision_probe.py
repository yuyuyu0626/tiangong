from __future__ import annotations

import unittest

from move_pro.collision_probe import (
    AABB,
    box_aabb,
    overlap_extent,
    overlap_volume,
    penetration_depth,
    is_overlapping,
    CollisionStats,
    CollisionReport,
)


class AabbGeometryTests(unittest.TestCase):
    def test_box_aabb_center_size(self):
        a = box_aabb((1.0, 2.0, 3.0), (0.2, 0.4, 0.6))
        self.assertEqual(a.lo, (0.9, 1.8, 2.7))
        self.assertEqual(a.hi, (1.1, 2.2, 3.3))

    def test_separated_boxes_no_overlap(self):
        a = box_aabb((0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
        b = box_aabb((1.0, 0.0, 0.0), (0.2, 0.2, 0.2))
        self.assertFalse(is_overlapping(a, b))
        self.assertEqual(overlap_volume(a, b), 0.0)
        self.assertEqual(penetration_depth(a, b), 0.0)

    def test_touching_boxes_not_counted_as_overlap(self):
        # 相切（面贴面）不算穿透。
        a = box_aabb((0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
        b = box_aabb((0.2, 0.0, 0.0), (0.2, 0.2, 0.2))  # 共享 x=0.1 面
        self.assertFalse(is_overlapping(a, b))
        self.assertEqual(penetration_depth(a, b), 0.0)

    def test_overlapping_boxes_penetration(self):
        a = box_aabb((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))   # [-0.2,0.2]^3
        b = box_aabb((0.3, 0.0, 0.0), (0.4, 0.4, 0.4))   # x in [0.1,0.5]
        # x 重叠 [0.1,0.2]=0.1, y/z 重叠 0.4 → 最小穿透在 x = 0.1
        self.assertTrue(is_overlapping(a, b))
        self.assertAlmostEqual(penetration_depth(a, b), 0.1, places=6)
        self.assertAlmostEqual(overlap_volume(a, b), 0.1 * 0.4 * 0.4, places=6)

    def test_overlap_extent_signs(self):
        a = box_aabb((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
        b = box_aabb((0.3, 1.0, 0.0), (0.4, 0.4, 0.4))   # y 分离
        ox, oy, oz = overlap_extent(a, b)
        self.assertGreater(ox, 0)
        self.assertLess(oy, 0)   # y 轴分离 → 负
        self.assertGreater(oz, 0)
        self.assertFalse(is_overlapping(a, b))

    def test_fully_contained(self):
        outer = box_aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        inner = box_aabb((0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
        self.assertTrue(is_overlapping(outer, inner))
        # 完全包含：穿透深度 = 内盒最小边长
        self.assertAlmostEqual(penetration_depth(outer, inner), 0.2, places=6)


class CollisionStatsTests(unittest.TestCase):
    def test_ratio_empty(self):
        s = CollisionStats(box_index=0)
        self.assertEqual(s.collision_ratio, 0.0)

    def test_ratio_partial(self):
        s = CollisionStats(box_index=1, frames_checked=10, frames_in_collision=3)
        self.assertAlmostEqual(s.collision_ratio, 0.3)

    def test_report_summary_none(self):
        report = CollisionReport()
        report.add(CollisionStats(box_index=0, frames_checked=5, frames_in_collision=0))
        line = report.summary_line()
        self.assertIn("boxes_with_collision=0/1", line)
        self.assertIn("none", line)

    def test_report_summary_worst(self):
        report = CollisionReport()
        report.add(CollisionStats(box_index=0, frames_checked=5, frames_in_collision=0))
        report.add(CollisionStats(
            box_index=2, frames_checked=5, frames_in_collision=4,
            max_penetration=0.05, max_pen_source="carried_box", max_pen_victim=1,
        ))
        line = report.summary_line()
        self.assertIn("boxes_with_collision=1/2", line)
        self.assertIn("box=2", line)
        self.assertIn("0.05", line)


if __name__ == "__main__":
    unittest.main()
