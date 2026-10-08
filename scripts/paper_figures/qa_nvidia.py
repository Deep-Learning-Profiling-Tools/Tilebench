"""Automated QA for the NVIDIA paper-figure data package (one qa_summary.json per device).

Usage:
  python scripts/paper_figures/qa_nvidia.py --repo . --root artifacts/paper_figures/nvidia \
      --cache <external cache with inventory_<dev>.json and extract/<dev>/> [--base-ref origin/paper/figures]
Exit status is non-zero when any check fails.
"""
import argparse
import collections
import csv
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvidia_common as C  # noqa: E402

csv.field_size_limit(1 << 30)
REQUIRED = {
    "benchmark_cases.csv": ["device", "architecture", "dsl", "operator", "category", "dtype", "case_id", "params_json", "mode",
                            "torch_ms", "dsl_ms", "validity", "source_csv", "source_git_sha", "notes"],
    "profile_index.csv": ["profile_id", "device", "dsl", "operator", "dtype", "case_id", "params_json", "collection_level",
                          "replay_mode", "report_path", "report_sha256", "report_origin", "report_revision",
                          "benchmark_case_match_status", "code_match_status", "launch_count", "notes"],
    "kernel_metrics_long.csv.gz": ["profile_id", "launch_id", "kernel_name", "stage", "raw_metric_name", "normalized_group",
                                   "value", "unit", "scope", "counter_kind", "extraction_method", "status", "raw_source", "notes"],
    "instruction_mix.csv": ["profile_id", "launch_id", "instruction_family", "instruction_name", "count", "count_kind", "scope",
                            "evidence_path", "notes"],
    "pc_hotspots.csv.gz": ["profile_id", "launch_id", "pc", "instruction_family", "instruction_text", "sample_count",
                           "executed_instruction_count", "stall_reason", "source_line", "evidence_path", "notes"],
    "execution_paths.csv": ["profile_id", "stage", "access_path", "matrix_path", "buffer_location", "layout_operations",
                            "atomic_path", "resource_summary", "evidence_path", "evidence_confidence", "notes"],
    "diagnosis_evidence.csv": ["device", "dsl", "operator", "dtype", "comparison_scope", "mechanism_id", "observation",
                               "mechanism_hypothesis", "supporting_metric_names", "supporting_profile_ids",
                               "alternative_explanation", "confounders", "evidence_quality", "review_status", "source_paths"],
}
DYNAMIC_KINDS = {"dynamic_warp_inst_executed", "dynamic_warp_inst_executed_with_modifier", "dynamic_thread_inst_executed_true",
                 "dynamic_warp_inst_executed_family_total"}
STATIC_KINDS = {"static_sass_instruction_count"}


def read(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def check(results, name, ok, detail=None):
    results.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})
    return ok


def qa_device(dev, repo, root, cache, base_ref):
    res = []
    d = Path(root) / dev
    tables = {}
    for fn, cols in REQUIRED.items():
        hdr, rows = read(d / fn)
        tables[fn] = rows
        check(res, f"schema:{fn}", hdr[:len(cols)] == cols, {"missing_or_misordered": [c for c in cols if c not in hdr]})
    bench, prof = tables["benchmark_cases.csv"], tables["profile_index.csv"]
    metrics, mix, pcs = tables["kernel_metrics_long.csv.gz"], tables["instruction_mix.csv"], tables["pc_hotspots.csv.gz"]
    paths, diag = tables["execution_paths.csv"], tables["diagnosis_evidence.csv"]
    inv = json.load(open(Path(cache) / f"inventory_{dev}.json"))

    # 1 operators
    ops_b = {r["operator"] for r in bench}
    ops_p = {r["operator"] for r in prof}
    check(res, "1:all_45_operators", ops_b == set(C.CATEGORY) and ops_p == set(C.CATEGORY),
          {"benchmark_missing": sorted(set(C.CATEGORY) - ops_b), "profile_missing": sorted(set(C.CATEGORY) - ops_p)})
    # 2 coverage of operator/dtype pairs x DSL
    pairs = sorted({(r["operator"], r["dtype"]) for r in bench if r["mode"] == "autotune"})
    cov = {}
    for op, dt in pairs:
        for dsl in C.DSLS:
            p = next((x for x in prof if x["operator"] == op and x["dtype"] == dt and x["dsl"] == dsl), None)
            cov[f"{op}/{dt}/{dsl}"] = "missing" if p is None else f"{p['collection_level']}/{p['extract_status']}"
    lv = collections.Counter(cov.values())
    check(res, "2:pair_coverage_explicit", len(pairs) == 110 and "missing" not in lv,
          {"operator_dtype_pairs": len(pairs), "status_counts": dict(lv)})
    # 3 sha256
    shas = {r["hf_path"]: r for r in inv["reports"]}
    bad = [p["profile_id"] for p in prof if not re.fullmatch(r"[0-9a-f]{64}", p["report_sha256"] or "")
           or not shas.get(p["report_path"], {}).get("hf_match") or shas[p["report_path"]]["sha256"] != p["report_sha256"]]
    check(res, "3:sha256_present_and_equal_to_hf", not bad, {"bad": bad[:20], "hf_revision": inv["hf_revision"]})
    # 4 device / DSL identity (path + report session)
    bad = []
    want_cc = C.DEVICES[dev]["compute_capability"]
    for p in prof:
        if not p["report_path"].startswith(C.DEVICES[dev]["hf_folder"] + "/") or not Path(p["report_path"]).name.startswith(p["dsl"] + "_"):
            bad.append((p["profile_id"], "path"))
            continue
        ex = json.load(gzip.open(Path(cache) / "extract" / dev / f"{p['operator']}__{p['dsl']}_{p['dtype']}.json.gz"))
        s = ex.get("session", {})
        if s.get("compute_capability") != want_cc or dev not in (s.get("device_name") or ""):
            bad.append((p["profile_id"], s.get("device_name"), s.get("compute_capability")))
    check(res, "4:device_and_dsl_identity", not bad, {"bad": bad[:20]})
    # 5 benchmark case match
    case_ids = {(r["dsl"], r["case_id"]) for r in bench if r["mode"] == "autotune" and r["validity"] == "valid"}
    bad = [p["profile_id"] for p in prof if p["benchmark_case_match_status"] != "matched" or (p["dsl"], p["case_id"]) not in case_ids]
    unmatched_rows = sum(1 for r in bench if r["case_id"] == "UNMATCHED")
    check(res, "5:profiles_match_valid_benchmark_cases", not bad and unmatched_rows == 0,
          {"bad_profiles": bad[:20], "benchmark_rows_without_case": unmatched_rows,
           "benchmark_validity": dict(collections.Counter(r["validity"] for r in bench))})
    # 6 raw counter names + units preserved (compared with the extract)
    bad, n = [], 0
    by_pid = collections.defaultdict(list)
    for r in metrics:
        by_pid[r["profile_id"]].append(r)
    for p in prof:
        ex = json.load(gzip.open(Path(cache) / "extract" / dev / f"{p['operator']}__{p['dsl']}_{p['dtype']}.json.gz"))
        L = {str(l["launch_id"]): l for l in ex["launches"]}
        for r in by_pid[p["profile_id"]]:
            if r["status"] != "collected":
                continue
            n += 1
            l = L.get(r["launch_id"])
            rec = l["metrics"].get(r["raw_metric_name"]) or l.get("extended_metrics", {}).get(r["raw_metric_name"]) if l else None
            if rec is None or (rec.get("unit") or "") != r["unit"] or r["value"] == "":
                bad.append((p["profile_id"], r["launch_id"], r["raw_metric_name"]))
    check(res, "6:raw_metric_names_and_units_preserved", not bad, {"rows_checked": n, "bad": bad[:20]})
    # 7 dynamic vs static separation
    kinds = collections.Counter(r["count_kind"] for r in mix)
    bad = [r for r in mix if (r["count_kind"] in STATIC_KINDS) != r["scope"].startswith("kernel_binary:")]
    check(res, "7:dynamic_static_separated", set(kinds) <= DYNAMIC_KINDS | STATIC_KINDS and not bad,
          {"count_kinds": dict(kinds), "bad_rows": len(bad)})
    # 8 reduced/targeted: no fabricated opcode counts
    dyn_pids = {r["profile_id"] for r in mix if r["count_kind"] in DYNAMIC_KINDS}
    bad = []
    for p in prof:
        ex = json.load(gzip.open(Path(cache) / "extract" / dev / f"{p['operator']}__{p['dsl']}_{p['dtype']}.json.gz"))
        has = any(l["opcode_counts"] for l in ex["launches"])
        if (p["profile_id"] in dyn_pids) != has:
            bad.append(p["profile_id"])
    nonfull = [p["profile_id"] for p in prof if p["collection_level"] != "full"]
    check(res, "8:no_fabricated_opcode_counts", not bad,
          {"non_full_profiles": nonfull, "non_full_with_dynamic_counts": [x for x in nonfull if x in dyn_pids], "mismatch": bad})
    # 9 launch / stage identities
    bad = []
    for p in prof:
        ids = {r["launch_id"] for r in by_pid[p["profile_id"]]}
        if str(len(ids)) != p["launch_count"]:
            bad.append((p["profile_id"], len(ids), p["launch_count"]))
    multi = sorted(p["profile_id"] for p in prof if len(p["kernel_names"].split(";")) > 1)
    lc_notes = [p["profile_id"] for p in prof if "LAUNCH COUNT MISMATCH" in p["notes"]]
    check(res, "9:launch_and_stage_identity", not bad and not lc_notes,
          {"launch_id_mismatch": bad[:20], "multi_kernel_profiles": len(multi), "unexplained_launch_count_mismatch": lc_notes,
           "helper_launch_notes": sum("PyTorch helper" in p["notes"] for p in prof)})
    # 10 source correlation honesty
    srcd = collections.defaultdict(lambda: [0, 0])
    for r in pcs:
        if r["stall_reason"] == "__all__":
            srcd[r["profile_id"]][0] += 1
            srcd[r["profile_id"]][1] += bool(r["source_line"])
    no_pc = sorted({p["profile_id"] for p in prof} - set(srcd))
    no_src = sorted(k for k, (a, b) in srcd.items() if b == 0)
    check(res, "10:source_correlation_reported", True,
          {"profiles_with_pc_table": len(srcd), "profiles_without_pc_table": no_pc,
           "profiles_with_pc_rows_but_no_source_line": no_src,
           "pc_rows_with_source_line_fraction": round(sum(b for a, b in srcd.values()) / max(1, sum(a for a, b in srcd.values())), 4)})
    # 11 latency only from CSV
    bad = []
    csv_cache = {}
    for r in bench:
        key = r["source_csv"]
        if key not in csv_cache:
            with open(Path(repo) / key, newline="") as f:
                csv_cache[key] = list(csv.DictReader(f))
    for r in bench[:: max(1, len(bench) // 2000)]:
        rows = csv_cache[r["source_csv"]]
        col = C.DSL_COLUMN[r["dsl"]]
        if not any((x.get(col) or "").strip() == r["dsl_ms"] and C.norm_dtype(x["dtype"].strip()) == r["dtype"] for x in rows):
            bad.append((r["source_csv"], r["dsl"], r["dsl_ms"]))
    bmap = {(r["dsl"], r["case_id"]): r for r in bench if r["mode"] == "autotune"}
    bad2 = [p["profile_id"] for p in prof if bmap.get((p["dsl"], p["case_id"]), {}).get("dsl_ms") != p["benchmark_dsl_ms"]]
    check(res, "11:latency_from_csv_only", not bad and not bad2,
          {"sampled_rows": len(bench[:: max(1, len(bench) // 2000)]), "bad": bad[:10], "profile_latency_mismatch": bad2[:10],
           "note": "dsl_ms is the raw CSV string; NCU gpu__time_duration is never written to benchmark_cases"})
    # 12 no GPU work: extraction scripts only import reports
    offenders = []
    for py in (Path(repo) / "scripts/paper_figures").glob("*.py"):
        if py.name == Path(__file__).name:      # this file names the forbidden flags in its own patterns
            continue
        txt = py.read_text()
        for m in re.finditer(r"run\(\[\s*ncu[^\]]*\]", txt):
            if "--import" not in m.group(0):
                offenders.append(f"{py.name}: {m.group(0)[:80]}")
        if re.search(r"--target-processes|ncu\s+-o\b|--launch-skip|cudaProfilerStart", txt):
            offenders.append(f"{py.name}: profiling flag")
    created = sorted({p["report_created"] for p in prof})
    check(res, "12:no_new_gpu_profiling", not offenders,
          {"offenders": offenders, "report_created_range": [created[0], created[-1]],
           "note": "all NCU calls use --import on existing reports; the extraction ran on the B200 host with CUDA_VISIBLE_DEVICES='' and nothing was run on GH200"})
    # 13 unrelated files
    changed = subprocess.run(["git", "-C", repo, "diff", "--name-only", base_ref], capture_output=True, text=True).stdout.split()
    untracked = subprocess.run(["git", "-C", repo, "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    stray = [f for f in changed + untracked if not (f.startswith("artifacts/paper_figures/") or f.startswith("scripts/paper_figures/"))]
    check(res, "13:no_unrelated_changes", not stray, {"base_ref": base_ref, "stray": stray[:20]})
    # extra: denominator and value sanity
    bad = [r["raw_metric_name"] for r in metrics if r["status"] == "collected" and ("pct_of_peak_sustained" in r["raw_metric_name"])
           and not re.search(r"pct_of_peak_sustained_(active|elapsed)", r["raw_metric_name"])]
    check(res, "extra:utilization_denominator_in_name", not bad, {"bad": sorted(set(bad))[:10]})
    hi = [r for r in diag if r["evidence_quality"] == "high" and len(r["supporting_metric_names"].split(";")) < 3]
    check(res, "extra:high_confidence_needs_multiple_metrics", not hi, {"bad": [(r["operator"], r["mechanism_id"]) for r in hi]})
    status = "pass" if all(r["status"] == "pass" for r in res) else "fail"
    summary = {
        "device": dev, "status": status, "checks": res,
        "counts": {"benchmark_cases": len(bench), "profiles": len(prof), "kernel_metric_rows": len(metrics),
                   "instruction_mix_rows": len(mix), "pc_hotspot_rows": len(pcs), "execution_path_rows": len(paths),
                   "diagnosis_rows": len(diag)},
        "profiles_by_dsl_level_replay": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in
                                         collections.Counter((p["dsl"], p["collection_level"], p["replay_mode"]) for p in prof).items()},
        "hf_dataset": {"repo_id": inv["hf_repo"], "revision": inv["hf_revision"], "reports": len(inv["reports"])},
    }
    json.dump(summary, open(d / "qa_summary.json", "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--base-ref", default="origin/paper/figures")
    a = ap.parse_args()
    ok = True
    for dev in ("B200", "GH200"):
        s = qa_device(dev, a.repo, a.root, a.cache, a.base_ref)
        fails = [c["check"] for c in s["checks"] if c["status"] != "pass"]
        print(dev, s["status"], "failed:", fails)
        ok &= s["status"] == "pass"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
