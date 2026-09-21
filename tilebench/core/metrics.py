"""Compute derived performance metrics from raw benchmark results."""
from __future__ import annotations

import json
import math
import os

from tilebench.core.dtypes import dtype_size
from tilebench.paths import PEAK_PERFORMANCE_ROOT


def load_peak_config(gpu_name: str) -> dict:
    """Load peak performance summary from tilebench/data/peak_performance/<gpu_name>.json.

    Returns the parsed dict, or an empty dict if the file does not exist.
    """
    path = os.path.join(PEAK_PERFORMANCE_ROOT, f"{gpu_name}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _eval_expr(expr: str | None, ctx: dict) -> float | None:
    if not expr:
        return None
    try:
        return float(eval(str(expr), {"__builtins__": {}, "math": math}, ctx))  # noqa: S307
    except Exception:
        return None


BACKENDS = ("torch", "triton", "cutile", "tilelang", "nki")

# Backends that do NOT run on the GPU whose peak_* values are configured here.
# Their measured throughput (bandwidth_GBs, tflops, speedup) is still valid and
# hardware-agnostic, but %-of-peak and the GPU roofline must not be applied —
# that would divide Trainium timings by GPU peaks (see PR #102 review).
NON_GPU_BACKENDS = ("nki",)


def applicable_backends(peak_cfg: dict | None) -> tuple[str, ...]:
    """Which backends' measured throughput may be compared against peak_cfg.

    peak_cfg may declare "backends": [...] explicitly (e.g. Trainium2.json ->
    ["torch", "nki"], since only those actually ran on the Trainium hardware
    that peak_cfg describes). Peak files without this field (e.g. legacy
    NVIDIA GPU datasheets) fall back to every backend except NON_GPU_BACKENDS,
    preserving prior behavior.
    """
    declared = (peak_cfg or {}).get("backends")
    if declared:
        return tuple(declared)
    return tuple(b for b in BACKENDS if b not in NON_GPU_BACKENDS)


def compute_derived(result: dict, metrics_cfg: dict, peak_cfg: dict | None = None) -> dict[str, dict[str, float]]:
    """Return per-backend derived metrics dict."""
    params = result.get("params", {})
    dtype_str = result.get("dtype", "fp32")
    n = int(params.get("n", result.get("problem_size", 1)))
    ds = dtype_size(dtype_str)

    eval_ctx = {"n": n, "dtype_size": ds}
    eval_ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float))})

    flops = _eval_expr(metrics_cfg.get("flops_expr"), eval_ctx)
    bytes_transferred = _eval_expr(metrics_cfg.get("bytes_expr"), eval_ctx)

    # Peak values: prefer GPU-specific peak_cfg, fall back to operator metrics_cfg
    _peak = peak_cfg or {}
    peak_bw_raw = _peak.get("peak_bw_GBs") or metrics_cfg.get("peak_bw_GBs")
    peak_bw = float(peak_bw_raw) if peak_bw_raw is not None else None

    peak_tflops_map = _peak.get("peak_tflops") or metrics_cfg.get("peak_tflops", {})
    raw = peak_tflops_map.get(dtype_str) if isinstance(peak_tflops_map, dict) else None
    peak_tflops = float(raw) if raw is not None else None

    torch_ms_raw = result.get("torch_ms")
    torch_ms = float(torch_ms_raw) if torch_ms_raw is not None else None
    if torch_ms is not None and (torch_ms <= 0 or math.isnan(torch_ms)):
        torch_ms = None

    out: dict[str, dict[str, float]] = {}
    applicable = applicable_backends(_peak)

    for backend in BACKENDS:
        raw_ms = result.get(f"{backend}_ms")
        if raw_ms is None:
            continue
        ms = float(raw_ms)
        if ms <= 0 or math.isnan(ms):
            continue

        d: dict[str, float] = {"latency_ms": ms}

        # %-of-peak compares against the configured peak_cfg's hardware, so it
        # is only meaningful for backends that actually ran on that hardware.
        gpu_peak_applies = backend in applicable

        if bytes_transferred and bytes_transferred > 0:
            bw = bytes_transferred / (ms * 1e-3) / 1e9
            d["bandwidth_GBs"] = bw
            if gpu_peak_applies and peak_bw and peak_bw > 0:
                d["pct_peak_bw"] = bw / peak_bw * 100.0

        if flops and flops > 0:
            tf = flops / (ms * 1e-3) / 1e12
            d["tflops"] = tf
            if gpu_peak_applies and peak_tflops and peak_tflops > 0:
                d["pct_peak_tflops"] = tf / peak_tflops * 100.0

        if flops and bytes_transferred and bytes_transferred > 0:
            d["arithmetic_intensity"] = flops / bytes_transferred

        if backend != "torch" and torch_ms is not None:
            d["speedup"] = torch_ms / ms

        out[backend] = d

    return out
