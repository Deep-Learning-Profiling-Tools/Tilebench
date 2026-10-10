"""Algorithm-aware empirical SOL target and proximity per input case (RQ1, RQ2, A1, A2).

    T_SOL[o,d,c] = max(F / P_peak[mode(o, dtype), d], Q / BW_peak[d])      (compute term omitted for memory-only targets)
    R_SOL[o,b,d,c] = T_SOL[o,d,c] / T_k[o,b,d,c]                            ("proximity to modeled SOL"; never clipped)

mode(o, dtype) is the frozen algorithm-level decision of sol_modes.py; P_peak and BW_peak are the PR #323 empirical
profiles; F and Q are the frozen config.yaml expressions (plus the approved M3/M4 overrides), evaluated with the
engine's own convention (n = params['n'] or infer_problem_size, dtype_size, every numeric case parameter); T_k is the
formal autotuned latency from combined/benchmark_cases_normalized.csv.gz. T_SOL depends on (operator, dtype, case,
device) only, so every DSL of a device shares one denominator.

Aggregation (comparison protocol): GM over an operator's valid cases -> GM over a category's operators -> GM over the
45 operators. Cross-device change: both R over the case_id_v2 valid on both devices.

  python scripts/paper_figures/sol_data.py      -> artifacts/paper_figures/sol/{sol_cases.csv.gz, ...}
"""
import csv
import gzip
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_data as FD  # noqa: E402
import sol_modes as SM  # noqa: E402

sys.path.insert(0, str(FD.REPO))
from tilebench.core.dtypes import dtype_size  # noqa: E402
from tilebench.core.metrics import _eval_expr  # noqa: E402
from tilebench.data.tensors import infer_problem_size  # noqa: E402

SOL = FD.REPO / "artifacts" / "paper_figures" / "sol"
COLUMNS = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"),
           ("GH200", "tilelang"), ("MI300X", "triton")]
CONDITIONAL = SM.DECISIONS["D2_mi300x_bf16_vector"]["status_label"]
UNIT_OF_SECTION = {"peak_tflops": "FLOP", "peak_tops": "OP"}
CASE_FIELDS = ["device", "dsl", "operator", "category", "dtype", "case_id_v2", "params_full_json", "compute_mode", "peak_key",
               "peak_value", "peak_unit", "bw_GBs", "F", "F_unit", "Q_bytes", "compute_term_ms", "memory_term_ms", "T_SOL_ms",
               "bound", "critical_throughput", "target_status", "dsl_ms", "R_SOL", "above_one"]


def _ctx(op, dtype, params):
    ctx = {"n": int(params.get("n", infer_problem_size(op, params))), "dtype_size": dtype_size(dtype)}
    ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return ctx


def target(row, device, params, peaks):
    """T_SOL of one (operator, dtype, case) on one device from its manifest row."""
    op, dt, mode = row["operator"], row["dtype"], row["compute_mode"]
    ctx = _ctx(op, dt, params)
    F = _eval_expr(row["F_expression_frozen"] if row["F_expression"] is None else row["F_expression"], ctx)
    Q = _eval_expr(row["Q_expression"], ctx)
    assert Q is not None and Q > 0 and F is not None, (op, dt, params)
    bw = peaks[device]["peak_bw_GBs"]                               # GB/s
    mem_s = Q / (bw * 1e9)
    out = {"compute_mode": mode, "peak_key": row["device_specific_peak_key"][device], "peak_value": None, "peak_unit": None,
           "bw_GBs": bw, "F": F, "F_unit": "FLOP" if row["f_kind"] != "int_op_count" else "OP", "Q_bytes": Q,
           "compute_term_ms": None, "memory_term_ms": mem_s * 1e3, "critical_throughput": None,
           "target_status": row["target_status"][device]}
    if mode == "memory_only":
        out["bound"] = "memory_only"
        t = mem_s
    else:
        sec = out["peak_key"].split(".")[0]
        assert UNIT_OF_SECTION[sec] == out["F_unit"], (op, dt, out["peak_key"], row["f_kind"])
        out["critical_throughput"] = F / mem_s / 1e12               # TFLOP/s or TOP/s at which compute = memory term
        p = row["peak_value"][device]
        if p is None:
            # D2: the only approved case without a calibrated peak. No peak is substituted; the memory term is the
            # target only after this case's critical throughput is verified below the approved threshold.
            assert out["target_status"] == CONDITIONAL, (op, dt, device, out["target_status"])
            assert out["critical_throughput"] <= SM.DECISIONS["D2_mi300x_bf16_vector"]["max_critical_throughput_tflops"], \
                (op, dt, device, out["critical_throughput"])
            out["bound"] = "conditional_memory_dominance"
            t = mem_s
        else:
            out["peak_value"], out["peak_unit"] = p, row["peak_unit"][device]
            comp_s = F / (p * 1e12)
            out["compute_term_ms"] = comp_s * 1e3
            out["bound"] = "compute" if comp_s > mem_s else "memory"
            t = max(comp_s, mem_s)
    out["T_SOL_ms"] = t * 1e3
    assert math.isfinite(out["T_SOL_ms"]) and out["T_SOL_ms"] > 0
    return out


def build():
    """Case rows (one per device x DSL x valid autotuned case) and the mode manifest."""
    dts = defaultdict(set)
    rows_in = [r for r in FD.read_csv(FD.COMBINED / "benchmark_cases_normalized.csv.gz") if r["mode"] == "autotune"]
    for r in rows_in:
        dts[r["operator"]].add(r["dtype"])
    mrows, problems, peaks_raw, _ = SM.build({o: sorted(v) for o, v in dts.items()})
    if problems:
        raise SystemExit("mode audit failed: " + "; ".join(problems))
    peaks = {d: peaks_raw[d][0] for d in SM.DEVICES}
    man = {(r["operator"], r["dtype"]): r for r in mrows}
    cache, out, invalid = {}, [], []
    for r in rows_in:
        if r["validity"] != "valid":
            invalid.append({k: r[k] for k in ("device", "dsl", "operator", "dtype", "case_id_v2", "validity")})
            continue
        key = (r["device"], r["operator"], r["dtype"], r["case_id_v2"])
        if key not in cache:
            cache[key] = target(man[(r["operator"], r["dtype"])], r["device"], json.loads(r["params_full_json"]), peaks)
        t = cache[key]
        dsl_ms = float(r["dsl_ms"])
        R = t["T_SOL_ms"] / dsl_ms
        out.append({"device": r["device"], "dsl": r["dsl"], "operator": r["operator"], "category": r["category"],
                    "dtype": r["dtype"], "case_id_v2": r["case_id_v2"], "params_full_json": r["params_full_json"], **t,
                    "dsl_ms": dsl_ms, "R_SOL": R, "above_one": R > 1.0})
    out.sort(key=lambda x: (x["device"], x["dsl"], x["operator"], x["dtype"], x["case_id_v2"]))
    return out, mrows, peaks_raw, invalid


class Sol:
    """Read access to sol_cases.csv.gz for the figure scripts."""

    def __init__(self, sol_dir=SOL):
        self.rows = FD.read_csv(Path(sol_dir) / "sol_cases.csv.gz")
        self.cases = defaultdict(dict)        # (device, dsl, op) -> {case_id_v2: row}
        for r in self.rows:
            for k in ("R_SOL", "T_SOL_ms", "dsl_ms", "memory_term_ms"):
                r[k] = float(r[k])
            self.cases[(r["device"], r["dsl"], r["operator"])][r["case_id_v2"]] = r
        cm = FD.read_csv(FD.COMBINED / "category_mapping.csv")
        self.categories = {r["operator"]: r["category"] for r in cm}
        self.cat_order = [c for _, c in sorted({(int(r["category_order"]), r["category"]) for r in cm})]
        self.manifest = json.load(open(Path(sol_dir) / "sol_mode_manifest.json"))
        self.memory_only_ops = sorted({r["operator"] for r in self.manifest["rows"]} -
                                      {r["operator"] for r in self.manifest["rows"] if r["compute_mode"] != "memory_only"})

    def operators(self):
        return sorted(self.categories, key=lambda o: (self.cat_order.index(self.categories[o]), o))

    def r_op(self, dev, dsl, op, case_ids=None, exclude=None):
        c = self.cases.get((dev, dsl, op), {})
        ids = c.keys() if case_ids is None else [i for i in case_ids if i in c]
        vals = [c[i]["R_SOL"] for i in ids if not (exclude and exclude(c[i]))]
        return FD.gm(vals), len(vals)

    def r_group(self, dev, dsl, ops, exclude=None):
        per = [self.r_op(dev, dsl, o, exclude=exclude)[0] for o in ops]
        return FD.gm(per), sum(1 for p in per if p is not None)

    def matched_ids(self, op, pairs):
        sets = [set(self.cases.get((d, s, op), {})) for d, s in pairs]
        return sorted(set.intersection(*sets)) if sets else []

    def delta_op(self, dsl, dev1, dev2, op):
        """log2(R_dev2 / R_dev1), both over the case_id_v2 valid on both devices."""
        ids = self.matched_ids(op, [(dev1, dsl), (dev2, dsl)])
        if not ids:
            return None, 0
        return math.log2(self.r_op(dev2, dsl, op, ids)[0] / self.r_op(dev1, dsl, op, ids)[0]), len(ids)

    def case(self, dev, dsl, op, case_id):
        return self.cases.get((dev, dsl, op), {}).get(case_id)


SOL_FILES = ("sol_mode_manifest.json", "sol_cases.csv.gz", "sol_provenance.json", "sol_operator_summary.csv",
             "sol_sensitivity_mi300x_bf16.csv", "sol_above_one_cases.csv", "sol_above_one_audit.json")

# Audit of R_SOL > 1 (values are kept, never clipped). Each case is assigned to one documented cause; the statistic that
# supports it is recomputed per group. L2 capacities: B200 126.5 MB (repository notes), GH200 50 MB (H100 L2).
L2_BYTES = {"B200": 126.5e6, "GH200": 50e6}
ABOVE_ONE_CAUSES = {
    "q_counts_second_read": {
        "operators": ("rmsnorm", "layernorm"),
        "kind": "performance model",
        "explanation": "the frozen Q = 3 * n * dtype_size counts two traversals of the input; the contract allows a kernel to keep "
                       "the row on chip and read it once, so at the compulsory 2/3 Q every case falls below 1"},
    "output_resident_in_l2": {
        "operators": ("rope",),
        "kind": "timing boundary",
        "explanation": "short launches (at most 37 us) whose written output is largely L2-resident: the write-back to HBM can finish "
                       "after the timed launch, so less than Q reaches HBM inside the measured interval; Q (one read of q, one write) "
                       "is otherwise complete. Cases whose output exceeds L2 (n_cases - n_output_within_L2, GH200 only) are only "
                       "partly explained by this and remain flagged"},
    "at_stream_copy_rate": {
        "operators": ("swiglu", "weight_dequant"),
        "kind": "measurement",
        "explanation": "working sets far above L2 within 2.2% of the calibrated stream-copy bandwidth, which is a sustained "
                       "measured rate rather than a hardware bound"},
    "above_library_gemm_rate": {
        "operators": ("matmul_fp32_fp16_fp8",),
        "kind": "calibration / measurement",
        "explanation": "compute-bound GEMMs whose achieved rate exceeds the calibrated sustained library-GEMM rate "
                       "(torch.matmul / torch._scaled_mm at M = 4096-16384, telemetry with SW power-cap clock events); the "
                       "numerical contract is met (TF32-class, native FP16 and e4m3 operands, fp32 accumulation)"},
}


def sol_input_hashes():
    return {f: FD.sha256(SOL / f) for f in SOL_FILES}


def sol_code_hashes():
    return {f"scripts/paper_figures/{f}": FD.sha256(HERE / f) for f in ("sol_modes.py", "sol_data.py")}


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_gz_csv(path, fields, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(r.get(k)) for k in fields})
    with open(path, "wb") as f:                       # mtime 0 and no file name: byte-reproducible
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            g.write(buf.getvalue().encode())


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def write_above_one_audit(out, above):
    cause_of = {o: k for k, v in ABOVE_ONE_CAUSES.items() for o in v["operators"]}
    groups = defaultdict(list)
    for r in above:
        if r["operator"] not in cause_of:
            raise SystemExit(f"R_SOL > 1 without an audited cause: {r['operator']}/{r['dtype']}/{r['device']}/{r['dsl']}")
        groups[(cause_of[r["operator"]], r["operator"], r["dtype"], r["device"])].append(r)
    rows = []
    for (cause, op, dt, dev), rs in sorted(groups.items()):
        g = {"cause": cause, "kind": ABOVE_ONE_CAUSES[cause]["kind"], "operator": op, "dtype": dt, "device": dev,
             "n_cases": len(rs), "dsls": sorted({r["dsl"] for r in rs}), "max_R_SOL": max(r["R_SOL"] for r in rs)}
        if cause == "q_counts_second_read":
            g["max_R_SOL_at_two_thirds_Q"] = max(r["R_SOL"] * 2 / 3 for r in rs)
        if cause == "output_resident_in_l2":
            g["n_output_within_L2"] = sum(1 for r in rs if r["Q_bytes"] / 2 <= L2_BYTES[dev])
            g["max_T_k_us"] = max(r["dsl_ms"] for r in rs) * 1e3
        if cause == "at_stream_copy_rate":
            g["min_Q_over_L2"] = min(r["Q_bytes"] / L2_BYTES[dev] for r in rs)
        if cause == "above_library_gemm_rate":
            g["bound"] = sorted({r["bound"] for r in rs})
            g["max_achieved_over_calibrated"] = max(r["F"] / (r["dsl_ms"] * 1e-3) / (r["peak_value"] * 1e12) for r in rs)
        rows.append(g)
    doc = {"schema": "tilearena-sol-above-one-audit/1", "n_cases": len(above), "causes": ABOVE_ONE_CAUSES,
           "l2_bytes": L2_BYTES, "groups": rows,
           "note": "values above 1 are kept in every table and figure; this audit explains them, it does not change them"}
    json.dump(doc, open(out / "sol_above_one_audit.json", "w"), indent=1)
    return doc


def main(out=SOL):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, mrows, peaks_raw, invalid = build()
    write_gz_csv(out / "sol_cases.csv.gz", CASE_FIELDS, rows)
    S = Sol(out)
    ops = S.operators()
    is_cond = (lambda r: r["target_status"] == CONDITIONAL)
    # operator-level table
    op_rows = []
    for o in ops:
        for d, s in COLUMNS:
            v, n = S.r_op(d, s, o)
            c = S.cases.get((d, s, o), {})
            op_rows.append({"operator": o, "category": S.categories[o], "device": d, "dsl": s, "R_SOL_gm": v, "n_cases": n,
                            "n_above_one": sum(1 for x in c.values() if x["R_SOL"] > 1.0),
                            "n_conditional_memory_dominance": sum(1 for x in c.values() if is_cond(x)),
                            "bounds": ";".join(f"{b}:{sum(1 for x in c.values() if x['bound'] == b)}"
                                               for b in sorted({x["bound"] for x in c.values()}))})
    write_csv(out / "sol_operator_summary.csv", list(op_rows[0]), op_rows)
    # above-one audit list (kept, never clipped)
    above = [r for r in rows if r["above_one"]]
    write_csv(out / "sol_above_one_cases.csv", ["device", "dsl", "operator", "dtype", "case_id_v2", "params_full_json", "compute_mode",
                                                "bound", "F", "Q_bytes", "compute_term_ms", "memory_term_ms", "T_SOL_ms", "dsl_ms", "R_SOL"], above)
    write_above_one_audit(out, above)
    # RQ1 aggregates with and without the D2 conditional cases (sensitivity, appendix)
    groups = [(c, [o for o in ops if S.categories[o] == c]) for c in S.cat_order] + [("Overall", ops),
              ("Memory-only targets", S.memory_only_ops), ("Compute+memory targets", [o for o in ops if o not in S.memory_only_ops])]
    sens = []
    for g, gops in groups:
        for d, s in COLUMNS:
            v, n = S.r_group(d, s, gops)
            v2, n2 = S.r_group(d, s, gops, exclude=is_cond)
            sens.append({"row": g, "device": d, "dsl": s, "R_SOL_with": v, "n_operators_with": n, "R_SOL_without": v2,
                         "n_operators_without": n2, "relative_change": v2 / v - 1.0})
    write_csv(out / "sol_sensitivity_mi300x_bf16.csv", list(sens[0]), sens)
    # coverage + provenance
    cov = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cov[f"{r['device']}:{r['dsl']}"][r["target_status"]] += 1
    prov = {
        "schema": "tilearena-sol-provenance/1",
        "formula": {"T_SOL": "max(F / P_peak[mode], Q / BW_peak); memory-only targets: Q / BW_peak",
                    "R_SOL": "T_SOL / T_k (T_k = formal autotuned dsl_ms); not clipped",
                    "units": "F in FLOP (OP for int8_mma); P in TFLOP/s (TOP/s) -> x1e12; BW in GB/s -> x1e9; Q in bytes; "
                             "terms in s -> x1e3 ms; dsl_ms in ms",
                    "aggregation": ["GM of case-level R_SOL within an operator", "GM of operator values within a category",
                                    "GM of the 45 operator values (Overall)"],
                    "cross_device": "log2(R_dev2 / R_dev1), both GM over the case_id_v2 valid on both devices"},
        "peaks": {d: {"file": peaks_raw[d][3], "commit": SM.PR323_MERGE, "sha256": peaks_raw[d][1], "git_blob": peaks_raw[d][2],
                      "calibration_id": peaks_raw[d][0]["calibration_id"]} for d in SM.DEVICES},
        "legacy_peak_files_used": False,
        "inputs": {"benchmark_cases_normalized.csv.gz": FD.sha256(FD.COMBINED / "benchmark_cases_normalized.csv.gz"),
                   "sol_mode_manifest.json": FD.sha256(out / "sol_mode_manifest.json")},
        "frozen_mode_manifest_rows": len(mrows),
        "coverage_cases": {k: dict(v) for k, v in sorted(cov.items())},
        "coverage_operators": {f"{d}:{s}": sum(1 for o in ops if S.r_op(d, s, o)[1]) for d, s in COLUMNS},
        "invalid_or_missing_rows": len(invalid),
        "missing_peak_mode_mappings": [],
        "memory_only_operators": S.memory_only_ops,
        "above_one": {"n_cases": len(above), "by_operator": {f"{o}|{d}|{s}": n for (o, d, s), n in sorted(
            {(r["operator"], r["device"], r["dsl"]): sum(1 for x in above if (x["operator"], x["device"], x["dsl"]) ==
                                                             (r["operator"], r["device"], r["dsl"])) for r in above}.items())}},
        "conditional_memory_dominance": {
            "status_label": CONDITIONAL, "n_cases": sum(1 for r in rows if is_cond(r)),
            "operators": sorted({f"{r['operator']}/{r['dtype']}" for r in rows if is_cond(r)}),
            "max_critical_throughput_tflops": max((float(r["critical_throughput"]) for r in rows if is_cond(r)), default=None),
            "approved_threshold_tflops": SM.DECISIONS["D2_mi300x_bf16_vector"]["max_critical_throughput_tflops"]},
    }
    json.dump(prov, open(out / "sol_provenance.json", "w"), indent=1)
    print(f"{len(rows)} case rows; {len(above)} above 1; conditional {prov['conditional_memory_dominance']['n_cases']}")
    return prov


if __name__ == "__main__":
    main()
