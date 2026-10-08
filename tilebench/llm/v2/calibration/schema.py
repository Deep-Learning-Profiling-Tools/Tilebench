"""Mode registry and empirical-profile schema.

Every rate is registered under an explicit arithmetic mode whose meaning is
fixed here: input format, product/accumulation format, unit. Names are
compared exactly; there are no aliases (an IEEE fp32 GEMM is not TF32, an
int8 matrix dot is not an int32 vector rate, fp8 e4m3 is not e5m2 or fnuz).

Units: FLOP/s for floating-point work, OP/s for integer operation counts,
byte/s for bandwidth (decimal: 1 GB/s = 1e9 byte/s; buffer sizes are MiB).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

PROFILE_SCHEMA = "tilebench-empirical-profile/1"
CEILING_BASIS = "empirical"

UNITS = ("FLOP/s", "OP/s", "byte/s")
MODE_STATUSES = ("calibrated", "failed", "unsupported", "not_measured")
SOURCE_KINDS = ("measured", "datasheet", "measured_legacy")

# name -> fixed meaning. `role`: scoring (may be a task's declared mode),
# control (measured for comparison only; never a scoring ceiling).
MODES: dict[str, dict] = {
    "hbm_stream_bw": {"unit": "byte/s", "kind": "bandwidth", "role": "scoring",
                      "meaning": "sustained streaming bandwidth of device memory (read+write copy probes), "
                                 "working sets far above the last-level cache"},
    "mma_fp16_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                        "input": "fp16", "product": "fp16 x fp16", "accumulate": "fp32", "output": "fp16"},
    "mma_bf16_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                        "input": "bf16", "product": "bf16 x bf16", "accumulate": "fp32", "output": "bf16"},
    "mma_tf32_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                        "input": "fp32 operands rounded to TF32", "product": "tf32 x tf32", "accumulate": "fp32",
                        "output": "fp32"},
    "mma_xf32_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                        "input": "fp32 operands in the ROCm XF32 matrix path", "accumulate": "fp32", "output": "fp32"},
    "mma_fp8_e4m3_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                            "input": "float8_e4m3fn", "accumulate": "fp32", "output": "bf16"},
    "mma_fp8_e5m2_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                            "input": "float8_e5m2", "accumulate": "fp32", "output": "bf16"},
    "mma_fp8_e4m3fnuz_f32acc": {"unit": "FLOP/s", "kind": "matrix", "role": "scoring",
                                "input": "float8_e4m3fnuz (AMD)", "accumulate": "fp32", "output": "bf16"},
    "mma_int8_i32acc": {"unit": "OP/s", "kind": "matrix", "role": "scoring",
                        "input": "int8", "product": "int8 x int8", "accumulate": "int32", "output": "int32"},
    "gemm_fp32_ieee": {"unit": "FLOP/s", "kind": "matrix", "role": "control",
                       "input": "fp32 (IEEE, TF32 disabled)", "accumulate": "fp32", "output": "fp32",
                       "note": "library SGEMM; an empirical control, not a vector FMA rate"},
    "fp32_fma_vector": {"unit": "FLOP/s", "kind": "vector", "role": "scoring",
                        "input": "fp32", "operation": "fused multiply-add, 2 FLOP each, non-matrix units"},
    "fp16x2_fma_vector": {"unit": "FLOP/s", "kind": "vector", "role": "scoring",
                          "input": "fp16 (packed pairs)", "operation": "fused multiply-add, 2 FLOP per lane"},
    "bf16x2_fma_vector": {"unit": "FLOP/s", "kind": "vector", "role": "scoring",
                          "input": "bf16 (packed pairs)", "operation": "fused multiply-add, 2 FLOP per lane"},
    "int32_vector": {"unit": "OP/s", "kind": "vector", "role": "scoring",
                     "input": "int32", "operation": "integer add/compare/shift (no probe registered; never inferred "
                                                    "from an int8 matrix rate)"},
}

# Declarations a task may use instead of a calibrated compute mode.
NON_CALIBRATED_DECLARATIONS = {
    "memory_only": "T = Q / BW: the compute term is declared negligible or not modelled (research-model choice; "
                   "requires the owner's confirmation)",
    "no_compute_term": "the frozen F is 0, so T = Q / BW by the existing definition",
}


class ProfileError(ValueError):
    pass


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def profile_hash(profile: dict) -> str:
    body = copy.deepcopy(profile)
    body.pop("profile_sha256", None)
    return sha256_json(body)


def seal(profile: dict) -> dict:
    profile["profile_sha256"] = profile_hash(profile)
    return profile


def validate_profile(p: dict, *, require_sealed: bool = True) -> list[str]:
    errs: list[str] = []
    if p.get("schema") != PROFILE_SCHEMA:
        errs.append(f"schema must be {PROFILE_SCHEMA}")
    if p.get("ceiling_basis") != CEILING_BASIS:
        errs.append("ceiling_basis must be 'empirical'")
    for key in ("calibration_id", "device", "scope", "protocol", "code", "environment", "modes"):
        if key not in p:
            errs.append(f"missing {key}")
    if require_sealed and p.get("profile_sha256") != profile_hash(p):
        errs.append("profile_sha256 does not match the profile content")
    for name, m in (p.get("modes") or {}).items():
        reg = MODES.get(name)
        if reg is None:
            errs.append(f"mode {name!r} is not in the registry (no aliases)")
            continue
        if m.get("unit") != reg["unit"]:
            errs.append(f"mode {name}: unit {m.get('unit')!r} != registered {reg['unit']!r}")
        if m.get("status") not in MODE_STATUSES:
            errs.append(f"mode {name}: status {m.get('status')!r}")
        if m.get("source_kind") != "measured":
            errs.append(f"mode {name}: source_kind must be 'measured' inside an empirical profile")
        if m.get("status") == "calibrated":
            v = m.get("value")
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                errs.append(f"mode {name}: calibrated without a finite positive value")
        elif m.get("value") is not None:
            errs.append(f"mode {name}: status {m.get('status')} must not carry a value")
    for ref in ("datasheet_reference", "legacy_reference"):
        r = p.get(ref)
        if r is not None and r.get("source_kind") not in ("datasheet", "measured_legacy"):
            errs.append(f"{ref}: source_kind must be datasheet / measured_legacy")
    return errs
