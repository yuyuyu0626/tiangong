from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from move_pro.corner_cases import (
    record_corner_case,
    load_corner_cases,
    summarize_corner_cases,
)


class CornerCaseLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = Path(self._tmp.name) / "corner_cases.log"

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_when_missing(self):
        self.assertEqual(load_corner_cases(self.log), [])
        self.assertEqual(summarize_corner_cases(self.log), {})

    def test_record_and_load(self):
        rec = record_corner_case(
            "unreachable_high_stack",
            {"box_index": 5, "world_size": [0.2, 0.4, 0.4], "clearance_z": 0.6},
            log_path=self.log,
        )
        self.assertEqual(rec["kind"], "unreachable_high_stack")
        self.assertIn("time", rec)
        loaded = load_corner_cases(self.log)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["detail"]["box_index"], 5)

    def test_append_multiple(self):
        record_corner_case("unreachable_high_stack", {"box_index": 1}, log_path=self.log)
        record_corner_case("residual_collision", {"box_index": 2}, log_path=self.log)
        record_corner_case("residual_collision", {"box_index": 3}, log_path=self.log)
        loaded = load_corner_cases(self.log)
        self.assertEqual(len(loaded), 3)
        counts = summarize_corner_cases(self.log)
        self.assertEqual(counts["unreachable_high_stack"], 1)
        self.assertEqual(counts["residual_collision"], 2)

    def test_non_serializable_detail_falls_back(self):
        # numpy-like 标量（带 .item）与任意对象都应能落盘（_json_default 兜底）。
        class FakeScalar:
            def item(self):
                return 0.42

        rec = record_corner_case(
            "residual_collision",
            {"pen": FakeScalar(), "obj": object()},
            log_path=self.log,
        )
        loaded = load_corner_cases(self.log)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["detail"]["pen"], 0.42)


if __name__ == "__main__":
    unittest.main()
