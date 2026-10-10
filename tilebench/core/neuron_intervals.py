"""Aggregation of device execution intervals.

A run() may issue several device executions, and one execution may appear on
several physical cores (LNC=2). Three different numbers can be derived, and they
are kept apart:

    busy_sum_ms  sum over executions of each execution's duration, where the
                 per-core copies of one execution are merged (union) first
    union_ms     length of the union of all intervals (overlap counted once)
    span_ms      last end minus first start (includes idle gaps)

``union_ms <= busy_sum_ms`` and ``union_ms <= span_ms`` always hold; the
difference between busy_sum and union is overlap between distinct executions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def _union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if e < s:
            raise ValueError(f"interval ends before it starts: {(s, e)}")
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def aggregate(executions: Iterable[Mapping]) -> dict:
    """``executions``: dicts with ``start_ns``, ``end_ns`` and an ``execution_id``
    (the same id for the per-core copies of one execution; defaults to a unique id).
    """
    per_exec: dict = defaultdict(list)
    all_iv: list[tuple[int, int]] = []
    for i, e in enumerate(executions):
        s, t = int(e["start_ns"]), int(e["end_ns"])
        per_exec[e.get("execution_id", ("__unique__", i))].append((s, t))
        all_iv.append((s, t))
    if not all_iv:
        return {"n_executions": 0, "busy_sum_ms": None, "union_ms": None, "span_ms": None,
                "overlap_between_executions": False}
    busy = sum(_union_length(v) for v in per_exec.values())
    union = _union_length(all_iv)
    span = max(t for _, t in all_iv) - min(s for s, _ in all_iv)
    return {
        "n_executions": len(per_exec),
        "busy_sum_ms": busy / 1e6,
        "union_ms": union / 1e6,
        "span_ms": span / 1e6,
        "overlap_between_executions": busy > union,
    }
