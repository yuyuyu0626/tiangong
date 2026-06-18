#!/usr/bin/env python3
"""Headless online palletizing execution trace task.

This entry point intentionally does not create a camera unless the caller uses
online_robot_palletizing_task.py directly with --record-video.  It runs the same
continuous Isaac Gym execution path and writes JSONL trace plus metrics for the
offline trace renderer.
"""
from __future__ import annotations

import sys

from move.tasks.online_robot_palletizing_task import main


if __name__ == "__main__":
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")
    main()
