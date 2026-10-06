"""Pure statistics and selection rules of the calibration protocol.

point time   = median over batches of (median of the batch's samples)
throughput   = work / point time
mode value   = max throughput over the registered points that are valid;
               for bandwidth, only points whose working set is at least
               `hbm_min_working_set_over_llc` x LLC are eligible (cache-scale
               points are reported as diagnostics and never enter the maximum),
               and the eligible valid points must form a plateau.
Nothing here selects the fastest single sample.
"""
from __future__ import annotations

import math
import statistics


def batch_medians(batches: list[list[float]]) -> list[float]:
    if not batches or any(not b for b in batches):
        raise ValueError("every batch needs at least one sample")
    return [statistics.median(b) for b in batches]


def point_time(batches: list[list[float]]) -> float:
    return statistics.median(batch_medians(batches))


def batch_spread(batches: list[list[float]]) -> float:
    meds = batch_medians(batches)
    return (max(meds) - min(meds)) / statistics.median(meds)


def throughput(work: float, seconds: float) -> float:
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("time must be finite positive")
    return work / seconds


def hbm_eligible(working_set_bytes: int, llc_bytes: int | None, factor: float) -> bool:
    if not llc_bytes:
        return False                     # unknown LLC: no point is eligible (fail closed)
    return working_set_bytes >= factor * llc_bytes


def select_mode_value(points: list[dict], *, bandwidth: bool = False, plateau_tolerance: float = 0.05,
                      saturation_tolerance: float = 0.02) -> dict:
    """points: dicts with `throughput`, `valid` (bool) and, for bandwidth,
    `hbm_eligible` (bool) and `probe`/`primary` (bool). Returns
    {status, value, selected, reason, plateau}."""
    cand = [p for p in points if p.get("valid")]
    if bandwidth:
        cand = [p for p in cand if p.get("hbm_eligible") and p.get("primary")]
    if not cand:
        return {"status": "failed", "value": None, "selected": None,
                "reason": "no valid point" + (" with an HBM-scale working set" if bandwidth else "")}
    best = max(cand, key=lambda p: p["throughput"])
    out = {"status": "calibrated", "value": best["throughput"], "selected": best.get("id"), "reason": None, "flags": []}
    if not bandwidth and len(cand) > 1:
        # registration order = increasing size/occupancy; a best point at the edge of the registered range that
        # still exceeds the runner-up by more than the tolerance does not demonstrate saturation
        ordered = [p for p in points if p.get("valid")]
        runner_up = max(p["throughput"] for p in cand if p is not best)
        if ordered and ordered[-1] is best and best["throughput"] > runner_up * (1 + saturation_tolerance):
            out["flags"].append(f"saturation_not_demonstrated: best point is the last registered one and exceeds the "
                                f"runner-up by {best['throughput'] / runner_up - 1:.1%}")
    if bandwidth:
        by_probe: dict[str, list[float]] = {}
        for p in cand:
            by_probe.setdefault(p["probe"], []).append(p["throughput"])
        plateau = {}
        for probe, vals in by_probe.items():
            spread = (max(vals) - min(vals)) / max(vals) if len(vals) > 1 else None
            plateau[probe] = {"points": len(vals), "spread": spread,
                              "ok": spread is not None and spread <= plateau_tolerance}
        out["plateau"] = plateau
        if not plateau.get(best["probe"], {}).get("ok"):
            out.update(status="failed", value=None,
                       reason=f"probe {best['probe']} has no plateau across HBM-scale sizes (tolerance {plateau_tolerance})")
    return out
