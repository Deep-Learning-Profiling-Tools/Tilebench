"""Roofline computation for LLM-generated kernel evaluation.

Given an op's flops_expr / bytes_expr (from its config.yaml), the
case params, dtype, and a measured latency, compute what fraction of
the achievable roofline (at this case's arithmetic intensity) the
kernel reached.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from tilebench.core.dtypes import dtype_size
from tilebench.paths import PEAK_PERFORMANCE_ROOT



def load_peak(gpu_name: str = "B200") -> dict:
    """Load peak compute + bandwidth from tilebench/data/peak_performance/<gpu>.json."""
    path = PEAK_PERFORMANCE_ROOT / f"{gpu_name}.json"
    return json.loads(path.read_text())


def _eval_expr(expr: str | None, ctx: dict) -> float | None:
    if not expr:
        return None
    return float(eval(str(expr), {"__builtins__": {}, "math": math}, ctx))


def _eval_ctx(params: dict, dtype_str: str) -> dict:
    """Build the eval context the same way core/metrics.py does."""
    n = int(params.get("n", params.get("problem_size", 1)))
    ds = dtype_size(dtype_str)
    ctx = {"n": n, "dtype_size": ds, "math": math}
    ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float))})
    return ctx


def roofline_pct(
    flops_expr: str | None,
    bytes_expr: str | None,
    params: dict,
    dtype_str: str,
    latency_s: float,
    peak: dict,
) -> dict:
    """Compute roofline metrics for a single (case, dtype, latency) point.

    Returns a dict with:
        flops, bytes, AI, measured_perf, ceiling, roofline_pct, bound_by

    Convention: when flops_expr is 0 or absent (pure-data-movement ops),
    roofline_pct is computed as bandwidth utilization: measured_bw / peak_bw.
    """
    ctx = _eval_ctx(params, dtype_str)
    flops = _eval_expr(flops_expr, ctx) or 0.0
    bytes_ = _eval_expr(bytes_expr, ctx) or 0.0

    if latency_s <= 0:
        return {"error": "non-positive latency"}

    peak_bw = float(peak["peak_bw_GBs"]) * 1e9   # B/s
    peak_compute_map = peak.get("peak_tflops", {})

    # Look up dtype-specific peak compute. Fall back to fp16 if missing.
    peak_compute = peak_compute_map.get(dtype_str) or peak_compute_map.get("fp16")
    peak_compute = float(peak_compute) * 1e12 if peak_compute is not None else None  # FLOPS/s

    measured_bw = bytes_ / latency_s
    measured_perf = flops / latency_s if flops > 0 else 0.0

    if flops == 0:
        # Pure-data-movement: roofline_pct = bandwidth utilization.
        pct = (measured_bw / peak_bw) if peak_bw > 0 else 0.0
        return {
            "flops": 0,
            "bytes": bytes_,
            "AI": 0.0,
            "measured_bw_GBs": measured_bw / 1e9,
            "measured_perf_TFLOPS": 0.0,
            "ceiling_TFLOPS": None,
            "ceiling_bw_GBs": peak_bw / 1e9,
            "roofline_pct": pct,
            "bound_by": "bandwidth (no compute)",
        }

    ai = flops / bytes_ if bytes_ > 0 else float("inf")
    bw_bound_perf = peak_bw * ai                       # FLOPS/s at this AI
    if peak_compute is not None:
        ceiling = min(peak_compute, bw_bound_perf)
        bound_by = "compute" if peak_compute < bw_bound_perf else "bandwidth"
    else:
        ceiling = bw_bound_perf
        bound_by = "bandwidth (no peak compute for dtype)"

    pct = (measured_perf / ceiling) if ceiling > 0 else 0.0
    return {
        "flops": flops,
        "bytes": bytes_,
        "AI": ai,
        "measured_bw_GBs": measured_bw / 1e9,
        "measured_perf_TFLOPS": measured_perf / 1e12,
        "ceiling_TFLOPS": ceiling / 1e12,
        "ceiling_bw_GBs": peak_bw / 1e9,
        "roofline_pct": pct,
        "bound_by": bound_by,
    }


def geo_mean(values: list[float]) -> float:
    """Geometric mean of strictly positive values. Returns 0 if any value <= 0."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return statistics.geometric_mean(values)
