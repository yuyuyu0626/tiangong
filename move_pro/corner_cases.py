"""Corner case 记录：把放置策略遇到的异常情况落盘，指导后续研究/调参/是否上 RL。

阶段二用 corner-case 规则（顶降式避障）尽量消除碰撞。无法规则解决的情况记录为
corner case，按类归档到 JSONL 日志，供分析。两类典型：
- unreachable_high_stack：抬高到邻箱顶之上后 IK 够不到，箱子只能悬空。
- residual_collision：顶降后仍残留的碰撞（如手肘横扫），穿透超阈值。

记录是 append-only JSONL（每行一个 JSON），便于增量累积与离线分析。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from move_pro.config import MOVE_PRO_ROOT

CORNER_CASE_LOG = MOVE_PRO_ROOT / "corner_cases.log"


def record_corner_case(kind: str, detail: dict, log_path: Path | None = None) -> dict:
    """追加一条 corner case 到 JSONL 日志，返回写入的记录。

    kind:   归类标签，如 "unreachable_high_stack" / "residual_collision"。
    detail: 该情况的结构化信息（箱尺寸、目标、已放箱高度、IK误差、碰撞源/穿透等）。
    """
    path = Path(log_path) if log_path is not None else CORNER_CASE_LOG
    record = {
        "kind": str(kind),
        "time": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return record


def _json_default(obj):
    # numpy 标量/数组等转成原生类型，保证可序列化。
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def load_corner_cases(log_path: Path | None = None) -> list[dict]:
    """读回所有 corner case 记录（JSONL → list[dict]）。文件不存在返回空列表。"""
    path = Path(log_path) if log_path is not None else CORNER_CASE_LOG
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_corner_cases(log_path: Path | None = None) -> dict[str, int]:
    """按 kind 统计 corner case 数量，便于快速了解分布。"""
    counts: dict[str, int] = {}
    for rec in load_corner_cases(log_path):
        kind = rec.get("kind", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
