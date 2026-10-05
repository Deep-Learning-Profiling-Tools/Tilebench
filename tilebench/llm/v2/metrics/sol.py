"""Analytical SOL target.

    T_SOL = max(F / P_peak, Q / BW_peak)

F and Q come from the operator's frozen flops_expr / bytes_expr evaluated on
the task's selected case (the same evaluation context as
tilebench.core.metrics.compute_derived). P_peak is the device peak for the
task's DECLARED arithmetic mode (manifests/arithmetic_modes.yaml); when the
peak table has no value for that mode the result is `peak_missing` and no
efficiency is computed. Nothing is substituted."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from tilebench.core.dtypes import dtype_size
from tilebench.core.metrics import _eval_expr, load_peak_config

# How the existing peak_performance/<device>.json keys map onto arithmetic modes.
# B200.json declares that its "fp32" entry IS the TF32 tensor-core value, so it
# is mapped to tf32 and NOT to fp32_vector.
_PEAK_KEY_BY_MODE = {
    "tc_fp16": "fp16", "tc_bf16": "bf16", "tc_fp8": "fp8_e4m3fn", "tc_int8": "int8",
    "tf32": "tf32", "fp32_vector": "fp32_vector", "int_vector": "int_vector", "fp64": "fp64",
}


@dataclass
class SolRecord:
    device: str
    operator: str
    dtype: str
    arithmetic_mode: str
    F: float | None
    Q: float | None
    p_peak_tflops: float | None
    bw_peak_gbs: float | None
    t_sol_ms: float | None
    compute_term_ms: float | None
    memory_term_ms: float | None
    status: str                 # ok | peak_missing | expr_missing | bandwidth_missing | memory_only
    audit_flags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def declared_mode(modes: dict, operator: str, dtype: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    entry = modes.get("operators", {}).get(operator, {}) or {}
    if entry.get("_audit"):
        flags.append(str(entry["_audit"]))
    mode = entry.get(dtype) or modes["default_by_dtype"].get(dtype)
    if mode is None:
        raise ValueError(f"no arithmetic mode for {operator}/{dtype}")
    return mode, flags


def eval_workload(metrics_cfg: dict, params: dict, dtype: str, problem_size: int) -> tuple[float | None, float | None]:
    ctx = {"n": int(params.get("n", problem_size)), "dtype_size": dtype_size(dtype)}
    ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return _eval_expr(metrics_cfg.get("flops_expr"), ctx), _eval_expr(metrics_cfg.get("bytes_expr"), ctx)


def peak_for_mode(peak_cfg: dict, mode: str) -> float | None:
    key = _PEAK_KEY_BY_MODE.get(mode, mode)
    table = peak_cfg.get("peak_tflops") or {}
    if mode == "tf32" and "tf32" not in table and peak_cfg.get("fp32_is_tf32", True) and "fp32" in table:
        return float(table["fp32"])
    v = table.get(key)
    return float(v) if v is not None else None


def t_sol(device: str, operator: str, dtype: str, params: dict, problem_size: int,
          metrics_cfg: dict, modes: dict, peak_cfg: dict | None = None) -> SolRecord:
    peak_cfg = peak_cfg if peak_cfg is not None else load_peak_config(device)
    mode, flags = declared_mode(modes, operator, dtype)
    F, Q = eval_workload(metrics_cfg, params, dtype, problem_size)
    bw = peak_cfg.get("peak_bw_GBs")
    bw = float(bw) if bw else None
    p = peak_for_mode(peak_cfg, mode) if mode != "memory_only" else None
    rec = SolRecord(device=device, operator=operator, dtype=dtype, arithmetic_mode=mode, F=F, Q=Q,
                    p_peak_tflops=p, bw_peak_gbs=bw, t_sol_ms=None, compute_term_ms=None,
                    memory_term_ms=None, status="ok", audit_flags=flags)
    if F is None or Q is None:
        rec.status = "expr_missing"
        return rec
    if bw is None:
        rec.status = "bandwidth_missing"
        return rec
    rec.memory_term_ms = Q / (bw * 1e9) * 1e3
    if mode == "memory_only":
        rec.status = "memory_only"
        rec.t_sol_ms = rec.memory_term_ms
        rec.audit_flags.append("compute_term_declared_negligible")
        return rec
    if p is None:
        rec.status = "peak_missing"
        rec.audit_flags.append(f"no peak value for mode {mode} on {device}")
        return rec
    rec.compute_term_ms = F / (p * 1e12) * 1e3
    rec.t_sol_ms = max(rec.compute_term_ms, rec.memory_term_ms)
    if not math.isfinite(rec.t_sol_ms) or rec.t_sol_ms <= 0:
        rec.status = "expr_missing"
        rec.t_sol_ms = None
    return rec
