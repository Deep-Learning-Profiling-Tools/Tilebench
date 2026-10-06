"""Legacy / datasheet-reference target (NOT the scoring ceiling).

    T_SOL = max(F / P_peak, Q / BW_peak)   with P_peak from the legacy peak
    table tilebench/data/peak_performance/<device>.json

Scoring uses tilebench.llm.v2.metrics.empirical (ceiling_basis=empirical,
measured profile). This module remains for the offline sensitivity
reference against the datasheet table. The task's mode is the v2
declaration (manifests/arithmetic_modes.yaml, per operator, no dtype-wide
defaults); the legacy table is read by exact key only: no fp32->tf32 or
any other alias unless the table itself declares `fp32_is_tf32: true`."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from tilebench.core.dtypes import dtype_size
from tilebench.core.metrics import _eval_expr, load_peak_config

# How the existing peak_performance/<device>.json keys map onto arithmetic modes.
# B200.json declares that its "fp32" entry IS the TF32 tensor-core value, so it
# is mapped to tf32 and NOT to fp32_vector.
CEILING_BASIS = "datasheet_reference"
_PEAK_KEY_BY_MODE = {  # v2 mode name -> legacy table key (exact; a missing key is peak_missing)
    "mma_fp16_f32acc": "fp16", "mma_bf16_f32acc": "bf16", "mma_tf32_f32acc": "tf32",
    "mma_fp8_e4m3_f32acc": "fp8_e4m3fn", "mma_fp8_e5m2_f32acc": "fp8_e5m2", "mma_int8_i32acc": "int8",
    "fp32_fma_vector": "fp32_vector", "fp16x2_fma_vector": "fp16_vector", "bf16x2_fma_vector": "bf16_vector",
    "int32_vector": "int_vector", "gemm_fp32_ieee": "fp32_ieee",
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
    """v2 declaration: operators.<op>.modes.<dtype>; no dtype-wide default."""
    entry = modes.get("operators", {}).get(operator) or {}
    mode = (entry.get("modes") or {}).get(dtype)
    if mode is None:
        raise ValueError(f"no declared arithmetic mode for {operator}/{dtype}")
    flags = [f"f_kind={entry.get('f_kind')}", f"q_kind={entry.get('q_kind')}", f"ceiling_basis={CEILING_BASIS}"]
    if mode == "no_compute_term":
        mode = "memory_only"
    return mode, flags


def eval_workload(metrics_cfg: dict, params: dict, dtype: str, problem_size: int) -> tuple[float | None, float | None]:
    ctx = {"n": int(params.get("n", problem_size)), "dtype_size": dtype_size(dtype)}
    ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return _eval_expr(metrics_cfg.get("flops_expr"), ctx), _eval_expr(metrics_cfg.get("bytes_expr"), ctx)


def peak_for_mode(peak_cfg: dict, mode: str) -> float | None:
    """Exact key lookup in the legacy table. The only alias is an explicit
    `fp32_is_tf32: true` declared by the table itself (none of the tracked
    tables declares it); nothing is inferred from a nearby dtype."""
    key = _PEAK_KEY_BY_MODE.get(mode)
    if key is None:
        return None
    table = peak_cfg.get("peak_tflops") or {}
    if key == "tf32" and "tf32" not in table and peak_cfg.get("fp32_is_tf32") is True and "fp32" in table:
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
