"""Validate the MI300X paper-figure data package and write qa_summary.json.

    python3 scripts/paper_figures/validate_mi300x.py [--outputs-root /root/Tilebench/outputs] [--no-raw]

Checks that only need tracked files always run (the package itself, results/MI300X/csv, the frozen
diagnosis JSON). Checks that re-read the git-ignored profiling reports (SHA256 recomputation,
value-by-value comparison with kernel_metric.csv) run when --outputs-root exists, unless --no-raw.
Exit status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_identity import case_id, parse_params_cell  # noqa: E402
from mi300x_common import (CATEGORY, DEVICE, DIAGNOSIS_JSON, EXPECTED_OPERATORS, EXPECTED_PAIRS, KNOWN_EXCLUSION, OUT_DIR,  # noqa: E402
                           REPO, read_csv, sha256_file, tree_sha256, write_json)

REQUIRED = {
    "benchmark_cases.csv": ["device", "architecture", "dsl", "operator", "category", "dtype", "case_id", "params_json", "mode",
                            "torch_ms", "dsl_ms", "validity", "source_csv", "source_git_sha", "notes"],
    "profile_index.csv": ["profile_id", "device", "dsl", "operator", "dtype", "case_id", "params_json", "collection_level",
                          "replay_mode", "report_path", "report_sha256", "report_origin", "report_revision",
                          "benchmark_case_match_status", "code_match_status", "launch_count", "notes"],
    "kernel_metrics_long.csv.gz": ["profile_id", "launch_id", "kernel_name", "stage", "raw_metric_name", "normalized_group", "value",
                                   "unit", "scope", "counter_kind", "extraction_method", "status", "raw_source", "notes"],
    "instruction_mix.csv": ["profile_id", "launch_id", "instruction_family", "instruction_name", "count", "count_kind", "scope",
                            "evidence_path", "notes"],
    "pc_hotspots.csv.gz": ["profile_id", "launch_id", "pc", "instruction_family", "instruction_text", "sample_count",
                           "executed_instruction_count", "stall_reason", "source_line", "evidence_path", "notes"],
    "execution_paths.csv": ["profile_id", "stage", "access_path", "matrix_path", "buffer_location", "layout_operations", "atomic_path",
                            "resource_summary", "evidence_path", "evidence_confidence", "notes"],
    "diagnosis_evidence.csv": ["device", "dsl", "operator", "dtype", "comparison_scope", "mechanism_id", "observation",
                               "mechanism_hypothesis", "supporting_metric_names", "supporting_profile_ids", "alternative_explanation",
                               "confounders", "evidence_quality", "review_status", "source_paths"],
    "diagnostic_experiments.csv": ["experiment_id", "operator", "dtype", "variant", "changed_factor", "other_configuration_changes",
                                   "latency_ms", "reference_variant", "measurement_protocol", "correctness_status", "evidence_path", "notes"],
}
OTHER_FILES = ["environment.json", "README.md"]
KEY_METRICS = {  # metric-id -> label, coverage reported per triton rocprof-compute profile
    "2.1.9": "VALU Util", "2.1.10": "MFMA Util", "2.1.15": "Wavefront Occupancy", "4.1.9": "HBM Bandwidth",
    "7.1.5": "VGPRs", "7.1.6": "AGPRs", "7.1.7": "SGPRs", "7.1.8": "LDS Allocation", "7.1.9": "Scratch Allocation",
    "7.2.5": "Issue Wait Cycles", "10.1.0": "VALU instr", "10.1.1": "VMEM instr", "10.1.3": "MFMA instr", "10.2.0": "INT32 VALU",
    "11.2.9": "VMEM latency", "16.1.3": "vL1D coalescing", "17.1.2": "L2 hit", "17.3.4": "L2 atomic req", "17.6.11": "L2-Fabric atomic",
}
STATIC_KINDS = {"static_isa_opcode_count"}
DYNAMIC_KINDS = {"dynamic_hw_counter_total_summed_over_dispatches", "dynamic_att_hitcount_traced_waves"}


class QA:
    def __init__(self):
        self.checks = collections.OrderedDict()

    def add(self, name, ok, **detail):
        self.checks[name] = {"pass": bool(ok), **detail}
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {json.dumps(detail, default=str)[:600]}"))


def run(out: Path, outputs_root: Path | None):
    qa = QA()
    T = {}
    for f, cols in REQUIRED.items():
        p = out / f
        if not p.exists():
            qa.add(f"file_present:{f}", False)
            continue
        T[f] = read_csv(p)
        hdr = list(T[f][0].keys()) if T[f] else []
        qa.add(f"schema:{f}", hdr[:len(cols)] == cols, missing=[c for c in cols if c not in hdr], header=hdr[:len(cols)])
    for f in OTHER_FILES:
        qa.add(f"file_present:{f}", (out / f).exists())
    B, P, KM, IM, PC, EP, D, X = (T.get(k, []) for k in REQUIRED)
    env = json.load(open(out / "environment.json"))

    # 1. coverage --------------------------------------------------------------------------------------
    ops_b = sorted({r["operator"] for r in B})
    pairs_b = sorted({(r["operator"], r["dtype"]) for r in B if r["mode"] == "autotune"})
    pairs_p = sorted({(r["operator"], r["dtype"]) for r in P if r["profile_id"].endswith(".rocprof_compute")})
    qa.add("1.operators_accounted", len(ops_b) == EXPECTED_OPERATORS and set(ops_b) == set(CATEGORY),
           n_operators=len(ops_b), missing=sorted(set(CATEGORY) - set(ops_b)), extra=sorted(set(ops_b) - set(CATEGORY)))
    qa.add("1.pairs_formal_and_profiled", len(pairs_b) == EXPECTED_PAIRS and pairs_b == pairs_p,
           formal_pairs=len(pairs_b), profiled_pairs=len(pairs_p),
           formal_not_profiled=sorted(set(pairs_b) - set(pairs_p)), profiled_not_formal=sorted(set(pairs_p) - set(pairs_b)))
    per_pair_kinds = collections.defaultdict(set)
    for r in P:
        per_pair_kinds[(r["operator"], r["dtype"])].add(r["profile_id"].rsplit(".", 1)[-1])
    need = {"rocprof_compute", "pc_sampling", "static_isa", "kernel_trace"}
    incomplete = {f"{o}/{d}": sorted(need - k) for (o, d), k in per_pair_kinds.items() if need - k}
    qa.add("1.each_pair_has_counter_pcs_static_trace_profiles", not incomplete, incomplete=incomplete)

    # 2./3. profile identity and provenance ---------------------------------------------------------------
    ids = [r["profile_id"] for r in P]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    qa.add("2.profile_id_unique", not dup, duplicates=dup)
    nosrc = [r["profile_id"] for r in P if not r["report_path"] or len(r["report_sha256"]) != 64]
    qa.add("3.profile_path_and_sha256_recorded", not nosrc, missing=nosrc)

    # 4. profile -> benchmark case --------------------------------------------------------------------------
    bidx = {(r["case_id"], r["mode"]): r for r in B}
    bad = []
    for r in P:
        b = bidx.get((r["case_id"], "autotune"))
        if r["benchmark_case_match_status"] != "matched" or b is None or (b["operator"], b["dtype"], b["params_json"]) != (r["operator"], r["dtype"], r["params_json"]) \
                or r["device"] != DEVICE:
            bad.append(r["profile_id"])
        elif r["dsl"] == "triton" and r["profile_id"].endswith((".rocprof_compute", ".pc_sampling")):
            seg = f"rocprof_compute/MI300X/{r['operator']}/triton_{r['dtype']}/"
            if seg not in r["report_path"]:
                bad.append(r["profile_id"] + " (report_path)")
    qa.add("4.profiles_matched_to_device_dsl_operator_dtype_case", not bad, unmatched_or_wrong=bad,
           status_counts=dict(collections.Counter(r["benchmark_case_match_status"] for r in P)))
    full_ok = []
    for r in P:
        if r["profile_id"].endswith(".rocprof_compute"):
            fp = json.loads(r["profiled_params_full_json"])
            full_ok.append(all(fp.get(k) == v for k, v in json.loads(r["params_json"]).items()))
    qa.add("4.profiled_full_params_contain_case_params", all(full_ok), n=len(full_ok))

    # 5./6. formal latency ------------------------------------------------------------------------------------
    mism, n = [], 0
    for r in B:
        with open(REPO / r["source_csv"], newline="") as f:
            rows = list(csv.DictReader(f))
        src = rows[int(r["csv_row_number"]) - 2]
        n += 1
        p = parse_params_cell(src["params"])
        if (src["torch_ms"], src["triton_ms"], src["dtype"].lower()) != (r["torch_ms"], r["dsl_ms"], r["dtype"]) or \
                case_id(r["operator"], src["dtype"], p) != r["case_id"] or sha256_file(REPO / r["source_csv"]) != r["source_csv_sha256"]:
            mism.append((r["source_csv"], r["csv_row_number"]))
    csv_rows_total = 0
    for f in sorted((REPO / "results" / "MI300X" / "csv").glob("*.csv")):
        with open(f, newline="") as fh:
            csv_rows_total += sum(1 for _ in csv.DictReader(fh))
    qa.add("5.formal_latency_equals_source_csv", not mism and n == csv_rows_total, rows_checked=n, csv_rows_total=csv_rows_total,
           mismatches=mism[:20])
    dupcase = [k for k, v in collections.Counter((r["case_id"], r["mode"]) for r in B).items() if v > 1]
    qa.add("5.case_id_unique_per_mode", not dupcase, duplicates=dupcase[:10])
    prof_cols = [c for c in (B[0].keys() if B else []) if any(x in c.lower() for x in ("dur", "prof", "rocprof", "kernel_time"))]
    qa.add("6.no_profiler_duration_in_benchmark_cases", not prof_cols and not mism, suspicious_columns=prof_cols)
    dur_rows = [r for r in KM if r["raw_metric_name"] in ("kernel.csv | duration_ns_sum", "trace.dur_ns")]
    qa.add("6.profiler_durations_flagged_diagnostic_only", all(r["counter_kind"] == "profiler_duration_diagnostic_only" for r in dur_rows),
           n=len(dur_rows))

    # 7. no silent zero fill ------------------------------------------------------------------------------------
    bad7 = [r for r in KM if (r["status"].startswith("unavailable") and r["value"] != "") or (r["status"] in ("collected", "derived") and r["value"] == "")]
    qa.add("7.no_absent_metric_filled", not bad7, violations=len(bad7))
    raw_cmp = None
    if outputs_root and outputs_root.exists():
        src_vals = {}
        for r in KM:
            if r["counter_kind"] == "rocprof_compute_derived_metric":
                src_vals.setdefault(r["raw_source"], []).append(r)
        diffs, compared = [], 0
        for rel, rows in src_vals.items():
            path = outputs_root / rel[len("outputs/"):]
            idx = {}
            with open(path, newline="") as f:
                for s in csv.DictReader(f):
                    idx[(s["kernel_name"], f"{s['metric_id']} | {s['metric_name']} | {s['value_name']}", s["table_name"])] = s["value"]
            for r in rows:
                table = r["scope"].split("table '", 1)[1][:-1]
                v = idx.get((r["kernel_name"], r["raw_metric_name"], table))
                compared += 1
                if v != r["value"]:
                    diffs.append((rel, r["raw_metric_name"], v, r["value"]))
        raw_cmp = {"compared": compared, "differences": len(diffs)}
        qa.add("7.derived_metrics_equal_source_kernel_metric_csv", not diffs, compared=compared, differences=diffs[:10])

    # 8. static vs dynamic ---------------------------------------------------------------------------------------
    kinds = collections.Counter(r["count_kind"] for r in IM)
    bad8 = [r for r in IM if (r["count_kind"] in STATIC_KINDS and not (r["profile_id"].endswith(".static_isa") or r["launch_id"] == "att:code_object"))
            or (r["count_kind"] in DYNAMIC_KINDS and r["profile_id"].endswith(".static_isa"))
            or r["count_kind"] not in STATIC_KINDS | DYNAMIC_KINDS]
    qa.add("8.static_dynamic_instruction_counts_separated", not bad8, count_kinds=dict(kinds), violations=len(bad8))
    bad8b = [r for r in PC if r["profile_id"].endswith(".pc_sampling") and r["executed_instruction_count"] != ""]
    qa.add("8.pc_samples_not_presented_as_executions", not bad8b, violations=len(bad8b))

    # 9. units and scope of resource metrics ------------------------------------------------------------------
    res = [r for r in KM if r["normalized_group"] in ("resources", "occupancy")]
    bad9 = [r for r in res if not r["unit"] and r["counter_kind"] not in ("rocprof_compute_derived_metric",) or not r["scope"]]
    no_unit_derived = sorted({r["raw_metric_name"] for r in res if not r["unit"]})
    qa.add("9.resource_metrics_have_unit_and_scope", not bad9, n=len(res), violations=len(bad9),
           derived_metrics_without_unit_in_source=no_unit_derived[:20])

    # 10. diagnosis traceability -------------------------------------------------------------------------------
    pids = set(ids)
    dj = json.load(open(DIAGNOSIS_JSON))
    bad10 = [(r["operator"], r["mechanism_id"]) for r in D if not set(r["supporting_profile_ids"].split(";")) <= pids or not r["supporting_profile_ids"]]
    trace = collections.Counter((r["mechanism_id"], r["trace_check_status"]) for r in D)
    untraced = [{"operator": r["operator"], "mechanism": r["mechanism_id"], "role": r["mechanism_role"], "status": r["trace_check_status"],
                 "detail": r["trace_check_detail"][:500]}
                for r in D if r["mechanism_id"] != "C" and r["trace_check_status"] not in ("traced", "traced_by_metric")]
    qa.add("10.diagnosis_records_link_existing_profiles", not bad10 and len({r["operator"] for r in D}) == EXPECTED_OPERATORS, broken=bad10)
    m1_7 = [r for r in D if r["mechanism_id"] != "C"]
    hard_fail = [u for u in untraced if u["status"] == "not_traced" and u["role"] == "primary"]
    qa.add("10.primary_M1_M7_records_traced_to_raw_evidence", not hard_fail, records=len(m1_7), trace_status=dict(sorted((f"{a}:{b}", c) for (a, b), c in trace.items())),
           not_fully_traced=untraced)
    same = all(dj["operators"][r["operator"]]["evidence"] == r["observation"] for r in D)
    qa.add("10.diagnosis_text_verbatim_from_frozen_source", same)

    # 11. experiments separate -----------------------------------------------------------------------------------
    exp_ms = {r["latency_ms"] for r in X if r["latency_ms"]}
    overlap = [r["case_id"] for r in B if r["dsl_ms"] in exp_ms or r["torch_ms"] in exp_ms]
    qa.add("11.diagnostic_experiments_separate", all("DIAGNOSTIC" in r["measurement_protocol"] for r in X) and not overlap,
           experiments=sorted({r["experiment_id"] for r in X}), rows=len(X), formal_values_equal_to_experiment_values=overlap[:10])

    # 12. FP8 omission ------------------------------------------------------------------------------------------------
    fp8_rows = [r for r in B + P if "fp8" in r["dtype"]]
    qa.add("12.fp8_e4m3fn_omission_documented_not_fabricated",
           not fp8_rows and env["formal_benchmark"]["known_exclusion"]["dtype"] == "fp8_e4m3fn"
           and any(e.get("dtype") == "fp8_e4m3fn" for e in env["profiling"]["coverage"]["excluded"]),
           fp8_rows=len(fp8_rows))

    # raw provenance re-verification ------------------------------------------------------------------------------
    if outputs_root and outputs_root.exists():
        bad_sha = []
        for r in P:
            path = r["report_path"].split("#")[0]
            if r["profile_id"].endswith(".static_isa"):
                continue
            ap = outputs_root / path[len("outputs/"):]
            got = tree_sha256(ap) if ap.is_dir() else sha256_file(ap)
            if got != r["report_sha256"]:
                bad_sha.append(r["profile_id"])
        qa.add("3.report_sha256_recomputed_from_local_reports", not bad_sha, mismatches=bad_sha,
               checked=sum(1 for r in P if not r["profile_id"].endswith(".static_isa")))

    # cross-check against the prior 2026-10-05 analysis JSON --------------------------------------------------------
    prior = None
    an = outputs_root / "profiling" / "MI300X" / "analysis_2026-10-05" if outputs_root else None
    if an and an.exists():
        prior = crosscheck_prior(an, KM, IM, B)
        qa.add("prior_analysis_values_reproduced", all(v["mismatches"] == 0 for v in prior.values()),
               **{k: {"compared": v["compared"], "mismatches": v["mismatches"]} for k, v in prior.items()})

    # coverage summaries ---------------------------------------------------------------------------------------------
    cov = {}
    trip = [r for r in P if r["profile_id"].endswith(".rocprof_compute")]
    have = collections.defaultdict(set)
    for r in KM:
        if r["counter_kind"] == "rocprof_compute_derived_metric" and r["status"] == "collected":
            have[r["profile_id"]].add(r["raw_metric_name"].split(" | ")[0])
    for mid, lab in KEY_METRICS.items():
        cov[f"{mid} {lab}"] = sum(1 for r in trip if mid in have[r["profile_id"]])
    status = collections.Counter((r["counter_kind"], r["status"]) for r in KM)
    log = json.load(open(out / "extraction_log.json"))
    summary = {
        "schema": "tilearena-paper-figures-qa/1", "device": DEVICE,
        "all_checks_pass": all(c["pass"] for c in qa.checks.values()),
        "checks": qa.checks,
        "counts": {
            "operators": len(ops_b), "operator_dtype_pairs_formal": len(pairs_b), "operator_dtype_pairs_profiled": len(pairs_p),
            "benchmark_case_rows": len(B), "benchmark_case_rows_by_mode": dict(collections.Counter(r["mode"] for r in B)),
            "benchmark_validity": dict(collections.Counter(r["validity"] for r in B)),
            "profiles": len(P), "profiles_by_kind": dict(collections.Counter(r["profile_id"].rsplit(".", 1)[-1] for r in P)),
            "kernel_launches_rocprof_compute": sum(int(r["launch_count"]) for r in trip),
            "kernel_launches_pytorch_trace": sum(int(r["launch_count"] or 0) for r in P if r["profile_id"].endswith(".kernel_trace")),
            "kernel_metric_rows": len(KM), "instruction_mix_rows": len(IM), "pc_hotspot_rows": len(PC),
            "execution_path_rows": len(EP), "diagnosis_rows": len(D), "diagnostic_experiment_rows": len(X),
        },
        "missing_entries": {
            "known_exclusion": KNOWN_EXCLUSION,
            "pytorch_raw_rocprofv3_traces": "not retained (derived per-kernel table only)",
            "pytorch_counter_profiles": "never collected",
            "triton_att_traces": "never collected",
            "aten_att_decodes": "9 kernels only",
            "profiles_at_non_max_inputs": "none: every profile is at one input case per pair (the largest CSV row)",
            "cutile_tilelang_on_mi300x": "not applicable (no MI300X DSL comparison exists)",
        },
        "parse_failures_and_issues": log["issues"],
        "extraction_stats": log["stats"],
        "metric_coverage_triton_profiles_of_109": cov,
        "metric_status_counts": {f"{a}|{b}": c for (a, b), c in sorted(status.items())},
        "metric_6_2_all_zero_across_suite": log["metric_6_2_all_zero_across_suite"],
        "unresolved_mappings": {
            "profiles_not_matched": [r["profile_id"] for r in P if r["benchmark_case_match_status"] != "matched"],
            "diagnosis_records_not_fully_traced": untraced,
        },
        "raw_value_comparison": raw_cmp,
        "prior_analysis_crosscheck": prior,
    }
    write_json(out / "qa_summary.json", summary)
    print("ALL PASS" if summary["all_checks_pass"] else "SOME CHECKS FAILED")
    return summary


def irtag(path_text):
    """'outputs/.../triton_ir/<op>__<dt>__<kernel>__<CACHE8>/<k>.amdgcn ...' -> '<CACHE8>' (Triton cache-dir prefix)."""
    d = path_text.split(" ")[0].split("/")[-2]
    return d.rsplit("__", 1)[-1]


def crosscheck_prior(an: Path, KM, IM, B):
    """Re-derive values reported by the 2026-10-05 analysis (data/counters.json, isa.json, pcs.json, perf.json)
    from this package and count disagreements."""
    import re
    out = {}
    # counters.json: per-kernel rocprof-compute metrics (Avg/Value rows)
    sel = dict(re.findall(r'"(\d+\.\d+\.\d+)":"(\w+)"', (an / "scripts" / "counters.py").read_text()))
    km = {}
    for r in KM:
        if r["counter_kind"] == "rocprof_compute_derived_metric":
            mid, _, vn = r["raw_metric_name"].split(" | ")
            if vn in ("Avg", "Value", "Mean", "") and r["value"] != "":
                km[(r["profile_id"], r["kernel_name"], sel.get(mid))] = float(r["value"])
    c = json.load(open(an / "data" / "counters.json"))
    n = bad = 0; ex = []
    for key, v in c.items():
        op, dt = key.split("/")
        pid = f"{DEVICE}.triton.{op}.{dt}.rocprof_compute"
        for k, mets in v["kernels"].items():
            for name, val in mets.items():
                if name in ("dispatch_count", "prof_dur_ns_sum", "prof_dur_frac") or val is None:
                    continue
                n += 1
                got = km.get((pid, k, name))
                if got is None or abs(got - val) > 1e-9 * max(1.0, abs(val)):
                    bad += 1; ex.append((key, k, name, val, got))
    out["counters.json"] = {"compared": n, "mismatches": bad, "examples": ex[:10]}
    # isa.json: AMDGCN metadata and memory-access widths per code object
    stat = collections.defaultdict(dict)
    for r in KM:
        if r["counter_kind"] == "static_compiler_metadata":
            stat[(r["profile_id"].replace(".static_isa", ""), r["kernel_name"], irtag(r["raw_source"]))][r["raw_metric_name"]] = r["value"]
    widths = collections.defaultdict(collections.Counter)
    for r in IM:
        if r["count_kind"] == "static_isa_opcode_count" and r["profile_id"].endswith(".static_isa"):
            m = re.match(r"^(?:global|buffer)_(?:load|store)_(\w+)", r["instruction_name"])
            if m:
                widths[(r["profile_id"].replace(".static_isa", ""), irtag(r["evidence_path"]))][m.group(1)] += int(r["count"])
    isa = json.load(open(an / "data" / "isa.json"))
    keymap = {"vgpr": "amdgcn NumVgprs", "agpr": "amdgcn NumAgprs", "sgpr": "amdgcn TotalNumSgprs", "scratch": "amdgcn ScratchSize",
              "occupancy": "amdgcn Occupancy", "code_bytes": "amdgcn codeLenInByte"}
    n = bad = 0; ex = []
    for key, ks in isa.items():
        op, dt = key.split("/")
        base = f"{DEVICE}.triton.{op}.{dt}"
        for k in ks:
            h = Path(k["amdgcn"]).parent.name[:8] if k.get("amdgcn") else None
            cand = [v for (p, kn, hh), v in stat.items() if p == base and kn == k["kernel"]]
            hits = [v for (p, kn, hh), v in stat.items() if p == base and kn == k["kernel"] and h and hh == h]
            s = hits[0] if hits else (cand[0] if len(cand) == 1 else None)
            for a, b in keymap.items():
                n += 1
                if s is None or str(k["isa"]["meta"][a]) != s.get(b):
                    bad += 1; ex.append((key, k["kernel"], a, k["isa"]["meta"][a], None if s is None else s.get(b)))
            w = [v for (p, hh), v in widths.items() if p == base and h and hh == h]
            n += 1
            if w and dict(w[0]) != {kk: vv for kk, vv in k["isa"]["mem_width"].items()}:
                bad += 1; ex.append((key, k["kernel"], "mem_width", k["isa"]["mem_width"], dict(w[0])))
    out["isa.json"] = {"compared": n, "mismatches": bad, "examples": ex[:10]}
    # pcs.json: operator-kernel sample totals
    pcs = json.load(open(an / "data" / "pcs.json"))
    tot = collections.Counter()
    for r in KM:
        if r["raw_metric_name"] == "pcs.samples" and r["value"] != "":
            tot[r["profile_id"]] += int(r["value"])
    n = bad = 0; ex = []
    for key, v in pcs.items():
        op, dt = key.split("/")
        n += 1
        if tot[f"{DEVICE}.triton.{op}.{dt}.pc_sampling"] != v["samples"]:
            bad += 1; ex.append((key, v["samples"], tot[f"{DEVICE}.triton.{op}.{dt}.pc_sampling"]))
    out["pcs.json"] = {"compared": n, "mismatches": bad, "examples": ex[:10]}
    # perf.json: full-precision timing-log values used by the prior analysis vs the formal CSV (4-decimal) values
    perf = json.load(open(an / "data" / "perf.json"))
    bidx = {(r["operator"], r["dtype"], r["params_json"], r["mode"]): r for r in B}
    n = bad = 0; ex = []
    for op, v in perf.items():
        for dt, d in v["dtypes"].items():
            mc = d.get("max_case")
            if not mc:
                continue
            for r in [r for (o, t, pj, m), r in bidx.items() if o == op and t == dt and m == "autotune"]:
                if all(mc["params"].get(kk) == vv for kk, vv in json.loads(r["params_json"]).items()):
                    for col, pk in (("torch_ms", "mi_torch"), ("dsl_ms", "mi_triton")):
                        n += 1
                        if f"{mc[pk]:.4f}" != r[col]:
                            bad += 1; ex.append((op, dt, col, mc[pk], r[col]))
    out["perf.json(max_case)"] = {"compared": n, "mismatches": bad, "examples": ex[:10],
                                  "note": "prior analysis used full-precision timing logs; the package uses the 4-decimal CSV values"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--outputs-root", type=Path, default=Path("/root/Tilebench/outputs"))
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()
    s = run(a.out, None if a.no_raw else a.outputs_root)
    sys.exit(0 if s["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
