"""Compute derived performance metrics from raw benchmark results.

Usage (from visualize.py):
    from core.metrics import compute_derived

    # result  – one entry from the timing JSON produced by run_bench.py
    # metrics_cfg – the 'metrics' section of the operator's config.yaml
    derived = compute_derived(result, metrics_cfg)
    # derived == {
    #   "torch":  {"latency_ms": 0.12, "bandwidth_GBs": 450.0, ...},
    #   "triton": {...},
    #   "cutile": {...},
    # }
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_SIZE_MAP: dict[str, int] = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2,
    "bfloat16": 2, "bf16": 2,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "fp8": 1, "fp8_e5m2": 1, "fp8_e4m3fn": 1,
    "float8_e5m2": 1, "float8_e4m3fn": 1,
}


def _dtype_bytes(dtype_str: str) -> int:
    return _DTYPE_SIZE_MAP.get(dtype_str.lower(), 4)


# ---------------------------------------------------------------------------
# safe eval helper
# ---------------------------------------------------------------------------

def _eval_expr(expr: str | None, ctx: dict) -> float | None:
    """Evaluate a metric expression string; return None on any error."""
    if not expr or str(expr).strip().lower() in ("null", "none", ""):
        return None
    try:
        result = eval(str(expr), {"__builtins__": {}, "math": math}, ctx)  # noqa: S307
        return float(result)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

BACKENDS = ("torch", "triton", "cutile")


def compute_derived(result: dict, metrics_cfg: dict) -> dict[str, dict[str, float]]:
    """Return per-backend derived metrics dict.

    Keys in each per-backend dict (only present when value is computable):
        latency_ms          – raw mean latency
        bandwidth_GBs       – memory bandwidth
        pct_peak_bw         – % of peak bandwidth
        tflops              – arithmetic throughput (if flops_expr != null)
        pct_peak_tflops     – % of peak TFLOPS (if both defined)
        arithmetic_intensity – FLOP / Byte (if both defined)
        speedup             – vs PyTorch (only for triton / cutile)
    """
    params   = result.get("params", {})
    dtype_str = result.get("dtype", "fp32")
    n        = int(params.get("n", result.get("problem_size", 1)))
    ds       = _dtype_bytes(dtype_str)

    eval_ctx = {"n": n, "dtype_size": ds}

    flops            = _eval_expr(metrics_cfg.get("flops_expr"), eval_ctx)
    bytes_transferred = _eval_expr(metrics_cfg.get("bytes_expr"), eval_ctx)

    peak_bw = metrics_cfg.get("peak_bw_GBs")
    try:
        peak_bw = float(peak_bw) if peak_bw is not None else None
    except (TypeError, ValueError):
        peak_bw = None

    peak_tflops_map = metrics_cfg.get("peak_tflops", {})
    peak_tflops = None
    if isinstance(peak_tflops_map, dict):
        raw = peak_tflops_map.get(dtype_str)
        if raw is not None:
            try:
                peak_tflops = float(raw)
            except (TypeError, ValueError):
                pass

    torch_ms_raw = result.get("torch_ms")
    torch_ms: float | None = None
    try:
        v = float(torch_ms_raw)  # type: ignore[arg-type]
        if v > 0 and not math.isnan(v):
            torch_ms = v
    except (TypeError, ValueError):
        pass

    out: dict[str, dict[str, float]] = {}

    for backend in BACKENDS:
        raw_ms = result.get(f"{backend}_ms")
        if raw_ms is None:
            continue
        try:
            ms = float(raw_ms)
        except (TypeError, ValueError):
            continue
        if ms <= 0 or math.isnan(ms):
            continue

        d: dict[str, float] = {"latency_ms": ms}

        # Memory bandwidth
        if bytes_transferred and bytes_transferred > 0:
            bw = bytes_transferred / (ms * 1e-3) / 1e9
            d["bandwidth_GBs"] = bw
            if peak_bw and peak_bw > 0:
                d["pct_peak_bw"] = bw / peak_bw * 100.0

        # Arithmetic throughput
        if flops and flops > 0:
            tf = flops / (ms * 1e-3) / 1e12
            d["tflops"] = tf
            if peak_tflops and peak_tflops > 0:
                d["pct_peak_tflops"] = tf / peak_tflops * 100.0

        # Arithmetic intensity
        if flops and bytes_transferred and bytes_transferred > 0:
            d["arithmetic_intensity"] = flops / bytes_transferred

        # Speedup over torch (only meaningful for non-torch backends)
        if backend != "torch" and torch_ms is not None:
            d["speedup"] = torch_ms / ms

        out[backend] = d

    return out
