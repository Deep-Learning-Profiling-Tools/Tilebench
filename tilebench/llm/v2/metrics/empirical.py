"""Empirical Roofline target per task (ceiling_basis = "empirical").

    T_emp = max(F / P_emp[mode], Q / BW_emp)      (compute term only when the
                                                  declared model has one)
    E_emp(B) = max({T_emp / T_i : C_i <= B and valid_i} union {0})

P_emp and BW_emp are measured rates of the device's active empirical profile
(manifests/calibration.yaml -> calibration directory). The declared mode of
each (operator, dtype) comes from manifests/arithmetic_modes.yaml (revision 2)
and is looked up by its exact name: there is no alias and no fallback (an
IEEE fp32 GEMM rate is never used as TF32, an int8 matrix rate never as an
int32 vector rate, e4m3 never as e5m2). The F unit (FLOP / OP) must equal the
mode's unit. Missing or invalid values give a status, never a number.

T_emp is an empirically calibrated modelled target, not a proof of the
shortest achievable time; ratios above 1 are kept (audit flag), never clipped.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from tilebench.core.dtypes import dtype_size
from tilebench.core.metrics import _eval_expr
from tilebench.llm.v2.calibration import schema as cs
from tilebench.paths import REPO_ROOT

MANIFESTS = Path(__file__).resolve().parent.parent / "manifests"
CALIBRATION_MANIFEST = MANIFESTS / "calibration.yaml"
MODES_FILE = MANIFESTS / "arithmetic_modes.yaml"
MODES_SCHEMA = "tilebench-arithmetic-modes/2"
NON_CALIBRATED = tuple(cs.NON_CALIBRATED_DECLARATIONS)
STATUSES = ("ok", "definition_pending", "peak_unavailable", "profile_missing", "declaration_missing", "expr_missing")


class PeakUnavailable(LookupError):
    pass


class ProfileUnavailable(LookupError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- declarations


def load_modes(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or MODES_FILE).read_text())


def modes_sha256(path: Path | None = None) -> str:
    return sha256_file(Path(path or MODES_FILE))


def validate_modes(doc: dict, operators_dtypes: dict[str, list[str]] | None = None) -> list[str]:
    errs = []
    if doc.get("schema") != MODES_SCHEMA:
        errs.append(f"schema must be {MODES_SCHEMA}")
    if "default_by_dtype" in doc:
        errs.append("dtype-wide defaults are not allowed (declare per operator)")
    kinds = doc.get("f_kinds", {})
    decisions = doc.get("decisions", {})
    for op, e in (doc.get("operators") or {}).items():
        if e.get("f_kind") not in kinds:
            errs.append(f"{op}: unknown f_kind {e.get('f_kind')!r}")
            continue
        unit = kinds[e["f_kind"]].get("unit")
        for dt, mode in (e.get("modes") or {}).items():
            if mode in NON_CALIBRATED:
                if mode == "no_compute_term" and e["f_kind"] != "zero":
                    errs.append(f"{op}/{dt}: no_compute_term requires f_kind zero")
                continue
            reg = cs.MODES.get(mode)
            if reg is None:
                errs.append(f"{op}/{dt}: mode {mode!r} not in the calibration registry")
            elif reg["role"] != "scoring":
                errs.append(f"{op}/{dt}: mode {mode!r} is a {reg['role']} mode, never a scoring ceiling")
            elif unit is None or reg["unit"] != f"{unit}/s":
                errs.append(f"{op}/{dt}: f_kind {e['f_kind']} (unit {unit}) does not match mode {mode} ({reg['unit']})")
        for d in list(e.get("decisions", [])) + [x for v in (e.get("decisions_by_dtype") or {}).values() for x in v]:
            if d not in decisions:
                errs.append(f"{op}: unknown decision {d!r}")
    if operators_dtypes:
        for op, dts in operators_dtypes.items():
            e = (doc.get("operators") or {}).get(op)
            if e is None:
                errs.append(f"{op}: no declaration")
                continue
            for dt in dts:
                if dt not in (e.get("modes") or {}):
                    errs.append(f"{op}/{dt}: no declared mode")
    return errs


def declaration(doc: dict, operator: str, dtype: str) -> dict | None:
    e = (doc.get("operators") or {}).get(operator)
    if e is None or dtype not in (e.get("modes") or {}):
        return None
    mode = e["modes"][dtype]
    kind = doc["f_kinds"][e["f_kind"]]
    pending = [d for d in e.get("decisions", []) + (e.get("decisions_by_dtype") or {}).get(dtype, [])]
    if mode == "memory_only":
        pending.append("M2_memory_only_for_operation_counts")
    pending.append("M1_declaration_revision_2")
    status = {d: (doc.get("decisions", {}).get(d) or {}).get("status") for d in dict.fromkeys(pending)}
    return {"mode": mode, "f_kind": e["f_kind"], "f_unit": kind.get("unit"), "q_kind": e["q_kind"],
            "decisions": status, "note": e.get("note")}


def approved_overrides(doc: dict, operator: str) -> list[dict]:
    out = []
    for name, d in (doc.get("decisions") or {}).items():
        ov = d.get("override")
        if ov and ov.get("operator") == operator and d.get("status") == "approved":
            out.append({"decision": name, **ov})
    return out


# --------------------------------------------------------------------------- profiles


def load_calibration_manifest(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CALIBRATION_MANIFEST).read_text())


def device_entry(device: str, manifest: dict | None = None) -> dict:
    m = manifest if manifest is not None else load_calibration_manifest()
    return dict((m.get("devices") or {}).get(device) or {"profile": None, "sha256": None, "status": "none"})


def load_profile(entry: dict, *, root: Path = REPO_ROOT) -> dict:
    """Load and verify the profile an entry names: file sha256 must equal the
    entry's sha256 and the profile must validate and be sealed."""
    if not entry.get("profile"):
        raise ProfileUnavailable("no empirical profile registered for this device")
    path = (root / entry["profile"]) if not Path(entry["profile"]).is_absolute() else Path(entry["profile"])
    if not path.is_file():
        raise ProfileUnavailable(f"profile file {path} not found")
    sha = sha256_file(path)
    if entry.get("sha256") != sha:
        raise ProfileUnavailable(f"profile {path} sha256 {sha[:12]} != registered {str(entry.get('sha256'))[:12]}")
    p = json.loads(path.read_text())
    errs = cs.validate_profile(p)
    if errs:
        raise ProfileUnavailable(f"profile {path} is invalid: {errs}")
    if p.get("protocol", {}).get("quick"):
        raise ProfileUnavailable(f"profile {path} was produced by the quick smoke-test protocol")
    return p


def peak_for_mode(profile: dict, mode: str, *, unit: str) -> float:
    """Exact lookup; fails closed (PeakUnavailable) for an unknown/uncalibrated
    mode, a control mode, a unit mismatch or a non-finite value."""
    reg = cs.MODES.get(mode)
    if reg is None:
        raise PeakUnavailable(f"mode {mode!r} is not registered (no aliases)")
    if reg["role"] != "scoring":
        raise PeakUnavailable(f"mode {mode!r} is a {reg['role']} measurement, never a scoring ceiling")
    if reg["unit"] != unit:
        raise PeakUnavailable(f"mode {mode!r} has unit {reg['unit']}, the workload needs {unit}")
    m = (profile.get("modes") or {}).get(mode)
    if not m or m.get("status") != "calibrated":
        raise PeakUnavailable(f"mode {mode!r} is {(m or {}).get('status', 'absent')} in profile {profile.get('calibration_id')}")
    v = m.get("value")
    if m.get("unit") != unit or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise PeakUnavailable(f"mode {mode!r}: invalid value/unit in profile")
    return float(v)


def bandwidth(profile: dict) -> float:
    return peak_for_mode(profile, "hbm_stream_bw", unit="byte/s")


# --------------------------------------------------------------------------- T_emp


@dataclass
class EmpiricalTarget:
    device: str
    operator: str
    dtype: str
    status: str
    mode: str | None = None
    f_kind: str | None = None
    f_unit: str | None = None
    q_kind: str | None = None
    F: float | None = None
    Q: float | None = None
    F_frozen: float | None = None
    Q_frozen: float | None = None
    overrides_applied: list = field(default_factory=list)
    p_emp: float | None = None
    p_unit: str | None = None
    bw_emp_bytes_per_s: float | None = None
    compute_term_ms: float | None = None
    memory_term_ms: float | None = None
    t_emp_ms: float | None = None                 # only when status == "ok"
    provisional_t_emp_ms: float | None = None     # computed with the proposed declaration, for review
    pending_decisions: list = field(default_factory=list)
    ceiling_basis: str = cs.CEILING_BASIS
    calibration_id: str | None = None
    profile_sha256: str | None = None
    reason: str | None = None
    audit_flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def eval_workload(metrics_cfg: dict, params: dict, dtype: str, problem_size: int, f_expr: str | None = None):
    ctx = {"n": int(params.get("n", problem_size)), "dtype_size": dtype_size(dtype)}
    ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    F = _eval_expr(f_expr if f_expr is not None else metrics_cfg.get("flops_expr"), ctx)
    Q = _eval_expr(metrics_cfg.get("bytes_expr"), ctx)
    return F, Q, ctx


def t_emp(device: str, operator: str, dtype: str, params: dict, problem_size: int, metrics_cfg: dict,
          modes_doc: dict, profile: dict | None) -> EmpiricalTarget:
    rec = EmpiricalTarget(device=device, operator=operator, dtype=dtype, status="ok")
    F0, Q0, ctx = eval_workload(metrics_cfg, params, dtype, problem_size)
    rec.F_frozen, rec.Q_frozen = F0, Q0
    decl = declaration(modes_doc, operator, dtype)
    if decl is None:
        rec.status, rec.reason = "declaration_missing", f"no declared mode for {operator}/{dtype}"
        return rec
    rec.mode, rec.f_kind, rec.f_unit, rec.q_kind = decl["mode"], decl["f_kind"], decl["f_unit"], decl["q_kind"]
    rec.pending_decisions = [d for d, st in decl["decisions"].items() if st != "approved"]
    F, Q = F0, Q0
    for ov in approved_overrides(modes_doc, operator):
        if "F" in ov:
            F = _eval_expr(ov["F"], ctx)
            rec.f_kind = ov.get("f_kind", rec.f_kind)
        if "Q" in ov:
            Q = _eval_expr(ov["Q"], ctx)
            rec.q_kind = ov.get("q_kind", rec.q_kind)
        rec.overrides_applied.append(ov["decision"])
    rec.F, rec.Q = F, Q
    if Q is None or (F is None and decl["mode"] not in NON_CALIBRATED):
        rec.status, rec.reason = "expr_missing", "F or Q expression did not evaluate"
        return rec
    if profile is None:
        rec.status, rec.reason = "profile_missing", f"no active empirical profile for {device}"
        return rec
    rec.calibration_id, rec.profile_sha256 = profile.get("calibration_id"), profile.get("profile_sha256")
    try:
        bw = bandwidth(profile)
    except PeakUnavailable as e:
        rec.status, rec.reason = "peak_unavailable", str(e)
        return rec
    rec.bw_emp_bytes_per_s = bw
    rec.memory_term_ms = Q / bw * 1e3
    terms = [rec.memory_term_ms]
    if decl["mode"] == "memory_only":
        rec.audit_flags.append("compute term not modelled (memory_only); F kept as a diagnostic")
    elif decl["mode"] == "no_compute_term":
        rec.audit_flags.append("frozen F is 0")
    else:
        try:
            p = peak_for_mode(profile, decl["mode"], unit=f"{decl['f_unit']}/s")
        except PeakUnavailable as e:
            rec.status, rec.reason = "peak_unavailable", str(e)
            return rec
        rec.p_emp, rec.p_unit = p, f"{decl['f_unit']}/s"
        rec.compute_term_ms = F / p * 1e3
        terms.append(rec.compute_term_ms)
    t = max(terms)
    if not math.isfinite(t) or t <= 0:
        rec.status, rec.reason = "expr_missing", "non-positive target"
        return rec
    rec.provisional_t_emp_ms = t
    if rec.pending_decisions:
        rec.status = "definition_pending"
        rec.reason = "pending: " + ", ".join(rec.pending_decisions)
        return rec
    rec.t_emp_ms = t
    return rec


# --------------------------------------------------------------------------- scoring table / binding


def scoring_binding(device: str, *, manifest: dict | None = None) -> dict:
    """What a campaign pins at creation: its device's calibration entry and the
    declaration file. Other devices' entries are deliberately not included."""
    e = device_entry(device, manifest)
    return {"device": device, "profile": e.get("profile"), "profile_sha256": e.get("sha256"),
            "profile_status": e.get("status"), "arithmetic_modes_sha256": modes_sha256(),
            "ceiling_basis": cs.CEILING_BASIS}


def scoring_table(device: str, tasks: list[dict], *, modes_doc: dict | None = None, profile: dict | None = None,
                  config_loader=None) -> dict:
    """One row per (operator, dtype) of `tasks` (eligibility rows of this device)."""
    from tilebench.llm.v2.tasks.case_selection import load_operator_config, operator_config
    modes_doc = modes_doc or load_modes()
    config_loader = config_loader or load_operator_config
    rows, seen = [], set()
    for t in tasks:
        k = t["key"]
        if k["device"] != device or t.get("status") != "eligible" or (k["operator"], k["dtype"]) in seen:
            continue
        seen.add((k["operator"], k["dtype"]))
        cfg = config_loader(k["operator"])
        rec = t_emp(device, k["operator"], k["dtype"], t["params"], t.get("problem_size", 1),
                    cfg.get("metrics", {}), modes_doc, profile).to_dict()
        rec["params"] = t["params"]
        rec["case_id"] = k.get("case_id")
        rec["flops_expr"] = cfg.get("metrics", {}).get("flops_expr")
        rec["bytes_expr"] = cfg.get("metrics", {}).get("bytes_expr")
        try:
            rec["config_sha256"] = sha256_file(operator_config(k["operator"]))
        except Exception:  # noqa: BLE001
            rec["config_sha256"] = None
        rows.append(rec)
    rows.sort(key=lambda r: (r["operator"], r["dtype"]))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    body = {"device": device, "ceiling_basis": cs.CEILING_BASIS, "declaration_status": modes_doc.get("status"),
            "arithmetic_modes_sha256": modes_sha256(), "profile_sha256": (profile or {}).get("profile_sha256"),
            "calibration_id": (profile or {}).get("calibration_id"), "rows": rows, "status_counts": counts}
    body["scoring_sha256"] = cs.sha256_json({k: v for k, v in body.items() if k != "status_counts"})
    return body
