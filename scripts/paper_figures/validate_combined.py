"""Mandatory gate before plotting: validate the cross-device normalization layer (artifacts/paper_figures/combined).

Uses only committed artifacts, the formal CSVs and git history. Device QA that writes files is run on COPIES so the
source packages are never modified. Raw-report-dependent MI300X checks are reported as not rerun locally.

Usage:
  PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/validate_combined.py --repo . \
      [--nvidia-cache /projects/.../ncu_report_cache] [--skip-external]
Writes artifacts/paper_figures/combined/qa_combined.json; exit status 1 when a gating check fails.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvidia_common as NC  # noqa: E402
from build_combined import (CANONICAL_CATEGORIES, DSL_SUPPORT, KNOWN_UNSUPPORTED, PKG, case_id_v2, canon,  # noqa: E402
                            read_csv, sha256_file)

csv.field_size_limit(1 << 30)
EXPECTED_S = {"B200|triton": 2.02, "B200|cutile": 1.58, "B200|tilelang": 1.71, "GH200|triton": 1.85, "GH200|cutile": 1.53,
              "GH200|tilelang": 1.57, "MI300X|triton": 1.27}
EXPECTED_WINNERS = {"B200": {"triton": 22, "tilelang": 18, "cutile": 5}, "GH200": {"triton": 26, "tilelang": 17, "cutile": 2}}
EXTRACTION_COMMITS = {"nvidia": "3992d1ce6f639436f59ad9b9d0ff00e344fff2ae", "amd": "25bf1581558e82c3792c417dfa9ddf06b656ae75"}
REQUIRED_LIMITATION_TERMS = {
    "B200": ["cuda-tile", "later campaign", "auxiliary PyTorch", "source commit", "13 TileLang"],
    "GH200": ["cuda-tile", "warmup 1 / repeat 3", "fragment-accumulator", "transposed B"],
    "MI300X": ["warmup 20 / repeat 100", "eager", "E4M3FN", "not fully traced"],
}


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", default="artifacts/paper_figures")
    ap.add_argument("--nvidia-cache", default=None, help="external NCU extract cache (enables the NVIDIA QA rerun)")
    ap.add_argument("--skip-external", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo, root = a.repo, Path(a.repo) / a.root
    comb = root / "combined"
    res = []

    def check(name, ok, detail=None, gating=True):
        res.append({"check": name, "status": "pass" if ok else "fail", "gating": gating, "detail": detail})

    bench = read_csv(comb / "benchmark_cases_normalized.csv.gz")
    xw = read_csv(comb / "case_id_crosswalk.csv")
    prof = read_csv(comb / "profile_index_normalized.csv")
    cov = read_csv(comb / "coverage_summary.csv")
    cats = read_csv(comb / "category_mapping.csv")
    man = json.load(open(comb / "comparison_manifest.json"))
    src = {d: read_csv(root / PKG[d] / "benchmark_cases.csv") for d in PKG}

    # 1 operators on every device
    ops = {d: {b["operator"] for b in bench if b["device"] == d} for d in PKG}
    check("1:all_45_operators_on_all_devices", all(v == set(NC.CATEGORY) for v in ops.values()),
          {d: sorted(set(NC.CATEGORY) - v) for d, v in ops.items()})
    # 2 DSL support matrix
    dsls = {d: sorted({b["dsl"] for b in bench if b["device"] == d}) for d in PKG}
    check("2:dsl_support_matrix", all(dsls[d] == sorted(DSL_SUPPORT[d]) for d in PKG) and not any(b["dsl"] == "nki" for b in bench), dsls)
    # 3 / 4 pair counts
    pairs = {(d, s): {(b["operator"], b["dtype"]) for b in bench if b["device"] == d and b["dsl"] == s and b["mode"] == m}
             for d in PKG for s in DSL_SUPPORT[d] for m in ("autotune",)}
    pairs_def = {(d, s): {(b["operator"], b["dtype"]) for b in bench if b["device"] == d and b["dsl"] == s and b["mode"] == "default"}
                 for d in PKG for s in DSL_SUPPORT[d]}
    nv = {f"{d}:{s}": (len(pairs[(d, s)]), len(pairs_def[(d, s)])) for d in ("B200", "GH200") for s in DSL_SUPPORT[d]}
    check("3:nvidia_110_pairs_per_dsl", all(v == (110, 110) for v in nv.values()), nv)
    check("4:mi300x_109_triton_pairs", len(pairs[("MI300X", "triton")]) == 109 and len(pairs_def[("MI300X", "triton")]) == 109,
          {"autotune": len(pairs[("MI300X", "triton")]), "default": len(pairs_def[("MI300X", "triton")])})
    # 5 FP8 omission explicit
    fp8 = [c for c in cov if (c["device"], c["dsl"], c["operator"], c["dtype"]) == ("MI300X", "triton", "matmul_fp32_fp16_fp8", "fp8_e4m3fn")]
    fp8_rows = [b for b in bench if (b["device"], b["operator"], b["dtype"]) == ("MI300X", "matmul_fp32_fp16_fp8", "fp8_e4m3fn")]
    check("5:mi300x_fp8_omission_explicit", len(fp8) == 1 and fp8[0]["status"] == "unsupported_dtype" and not fp8_rows
          and "MI300X/triton/matmul_fp32_fp16_fp8/fp8_e4m3fn" in man["known_unsupported"],
          {"coverage_row": fp8, "benchmark_rows": len(fp8_rows)})
    # 6 each row -> exactly one full case; no loss; no duplicates
    dup = Counter((b["device"], b["dsl"], b["mode"], b["case_id_v2"]) for b in bench)
    dups = [k for k, v in dup.items() if v > 1]
    recomputed_bad = [b for b in bench if case_id_v2(b["operator"], b["dtype"], json.loads(b["params_full_json"])) != b["case_id_v2"]]
    lost = {d: (len(src[d]), sum(1 for b in bench if b["device"] == d)) for d in PKG}
    check("6:rows_map_to_exactly_one_full_case", not dups and not recomputed_bad and all(x == y for x, y in lost.values())
          and not man["problems"], {"duplicates": dups[:10], "recompute_mismatch": len(recomputed_bad), "rows_source_vs_normalized": lost,
                                    "build_problems": man["problems"][:10]})
    # 7 cross-device joins use full params
    by_id = defaultdict(set)
    for x in xw:
        by_id[x["case_id_v2"]].add(x["params_full_json"])
    subset_bad = []
    for x in xw:
        full = json.loads(x["params_full_json"])
        sw = json.loads(x["source_params_json"])
        if any(str(full.get(k)) != str(v) and not _num_eq(full.get(k), v) for k, v in sw.items() if k in full) or \
                any(k not in full for k in sw if k != "n"):
            subset_bad.append((x["device"], x["source_case_id"][:12]))
    nv_eq = all(x["source_equals_v2"] == "True" for x in xw if x["device"] != "MI300X")
    full_keys = {}
    for b in bench:
        full_keys.setdefault(b["operator"], set()).add(tuple(sorted(json.loads(b["params_full_json"]))))
    multi_keysets = {o: len(v) for o, v in full_keys.items() if len(v) > 1}
    check("7:joins_use_canonical_full_params", all(len(v) == 1 for v in by_id.values()) and not subset_bad and nv_eq and not multi_keysets,
          {"case_ids_with_conflicting_params": sum(len(v) > 1 for v in by_id.values()), "swept_not_subset_of_full": subset_bad[:10],
           "nvidia_source_id_equals_v2": nv_eq, "operators_with_multiple_param_keysets": multi_keysets,
           "mi300x_source_ids_differ_from_v2": sum(x["device"] == "MI300X" and x["source_equals_v2"] == "False" for x in xw)})
    # 8 categories
    catmap = {c["operator"]: c["category"] for c in cats}
    per_op = defaultdict(set)
    for b in bench:
        per_op[b["operator"]].add(b["category"])
    cnt = Counter(catmap.values())
    check("8:categories_identical_across_devices", len(catmap) == 45 and set(catmap.values()) == set(CANONICAL_CATEGORIES)
          and all(v == {catmap[o]} for o, v in per_op.items())
          and [cnt[c] for c in CANONICAL_CATEGORIES] == [12, 11, 8, 6, 8],
          {"counts": {c: cnt[c] for c in CANONICAL_CATEGORIES}, "mi300x_short_labels_consistent": all(c["mi300x_package_label"] for c in cats)})
    # 9 original CSV latency strings unchanged
    bad, files = [], {}
    for b in bench:
        if b["source_csv"] not in files:
            p = Path(repo) / b["source_csv"]
            files[b["source_csv"]] = (sha256_file(p), list(csv.DictReader(open(p, newline=""))))
        sh, rows = files[b["source_csv"]]
        if sh != b["source_csv_sha256"] or man["source_csv_sha256"].get(b["source_csv"]) != sh:
            bad.append(("sha", b["source_csv"]))
            continue
        col = f"{b['dsl']}_ms"
        if not any(r["params"].strip() == b["csv_params_label"] and NC.norm_dtype(r["dtype"].strip()) == b["dtype"]
                   and (r.get(col) or "").strip() == b["dsl_ms"] and (r.get("torch_ms") or "").strip() == b["torch_ms"] for r in rows):
            bad.append((b["device"], b["source_csv"], b["csv_params_label"], b["dtype"], b["dsl"]))
    check("9:csv_latency_strings_unchanged", not bad, {"rows_checked": len(bench), "csv_files": len(files), "bad": bad[:10]})
    # 10 no profiler duration in formal metrics
    cols = list(bench[0].keys())
    sus = [c for c in cols if any(t in c.lower() for t in ("duration", "gpu__time", "cycles", "kernel_time"))]
    check("10:no_profiler_duration_in_formal_metrics", not sus and not bad, {"benchmark_columns": cols, "suspicious": sus})
    # 11 profiled cases linked via crosswalk
    ids = {(b["device"], b["dsl"], b["case_id_v2"]) for b in bench if b["mode"] == "autotune" and b["validity"] == "valid"}
    unlinked = [p["profile_id"] for p in prof
                if (p["device"], "triton" if p["dsl"] == "pytorch" else p["dsl"], p["case_id_v2"]) not in ids]
    check("11:profiles_linked_to_benchmark_cases", not unlinked, {"profiles": len(prof), "unlinked": unlinked[:10]})
    # 12 static vs dynamic distinguishable
    kinds = {}
    for d in PKG:
        kinds[d] = sorted({r["count_kind"] for r in read_csv(root / PKG[d] / "instruction_mix.csv")})
    nv_static = {"static_sass_instruction_count"}
    amd_static = {"static_isa_opcode_count"}
    ok12 = all(set(kinds[d]) & nv_static for d in ("B200", "GH200")) and set(kinds["MI300X"]) & amd_static \
        and not (set(kinds["MI300X"]) & set(kinds["B200"])) and all(p["instruction_count_semantics"] for p in prof)
    check("12:static_dynamic_counts_distinguishable", ok12, kinds)
    # 13 missing metrics never zero
    viol = Counter()
    for d in PKG:
        with gzip.open(root / PKG[d] / "kernel_metrics_long.csv.gz", "rt", newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] not in ("collected", "derived") and r["value"] not in ("", None):
                    viol[d] += 1
    zero_lat = [b for b in bench if b["validity"] == "valid" and (float(b["dsl_ms"]) <= 0 or float(b["torch_ms"]) <= 0)]
    nonvalid_numeric = [b for b in bench if b["validity"] != "valid" and b["dsl_ms"] not in ("",) and b["dsl_ms"] == "0"]
    check("13:no_missing_metric_as_zero", not viol and not zero_lat and not nonvalid_numeric,
          {"non_collected_with_value": dict(viol), "valid_rows_with_nonpositive_latency": len(zero_lat),
           "validity_counts": {"|".join(k): v for k, v in Counter((b["device"], b["validity"], b["validity_source"]) for b in bench).items()}})
    # 14 documented historical differences
    lim = man["device_limitations"]
    miss = {d: [t for t in terms if not any(t.lower() in x.lower() for x in lim.get(d, []))] for d, terms in REQUIRED_LIMITATION_TERMS.items()}
    check("14:historical_differences_documented", not any(miss.values()), {"missing_terms": miss})
    # 15 source packages preserved
    pres = {}
    for vend, sha in EXTRACTION_COMMITS.items():
        path = f"{a.root}/{vend}"
        r1 = git(repo, "diff", "--name-only", sha, "--", path).stdout.split()
        r2 = git(repo, "status", "--porcelain", "--", path).stdout.split("\n")
        pres[vend] = {"changed_vs_extraction_commit": r1, "worktree_changes": [x for x in r2 if x.strip()]}
    pkg_hash_ok = all(sha256_file(root / PKG[d] / f) == h for d in PKG for f, h in man["source_packages"][d].items())
    check("15:source_packages_preserved", pkg_hash_ok and not any(v["changed_vs_extraction_commit"] or v["worktree_changes"] for v in pres.values()), pres)

    # intersections reported (not a pass/fail on a fixed number, but must be explicit and non-empty)
    inter = man["intersections_case_id_v2"]
    check("report:case_intersections_explicit", all(inter[k] > 0 for k in ("triton_B200_GH200_MI300X", "cutile_B200_GH200", "tilelang_B200_GH200")),
          {k: v for k, v in inter.items() if k != "triton_only_in"}, gating=True)
    # aggregate sanity vs previously observed values (rounded)
    agg = man["sanity_aggregates"]
    diffs = {k: (round(v["S"], 2), EXPECTED_S[k]) for k, v in agg["overall"].items() if round(v["S"], 2) != EXPECTED_S[k]}
    check("sanity:overall_speedups_match_previous", not diffs, {"recomputed": {k: round(v["S"], 4) for k, v in agg["overall"].items()}, "differences": diffs})
    wd = {d: (agg["winners"][d]["counts"], EXPECTED_WINNERS[d]) for d in EXPECTED_WINNERS if agg["winners"][d]["counts"] != EXPECTED_WINNERS[d]}
    check("sanity:winner_counts_match_previous", not wd, {"recomputed": agg["winners"], "differences": wd})

    # ---------------- external QA (on copies)
    ext = {}
    if not a.skip_external:
        tmp = Path(tempfile.mkdtemp(prefix="pf_qa_"))
        try:
            # MI300X raw reports live on another machine: point the AMD tests at an absent directory so that the
            # raw-dependent tests self-skip (the default /root/Tilebench/outputs is not readable on this host)
            env = dict(os.environ, PYTHONPATH=f"{repo}:{repo}/scripts/paper_figures", CUDA_VISIBLE_DEVICES="",
                       TILEBENCH_OUTPUTS_ROOT=str(tmp / "mi300x_raw_reports_not_available_here"))
            # AMD unit tests (raw-dependent tests self-skip)
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_paper_figures_mi300x.py", "-rs"],
                               cwd=repo, capture_output=True, text=True, env=env)
            ext["amd_unit_tests"] = {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-6:],
                                     "skipped_raw_dependent": [l for l in r.stdout.splitlines() if l.startswith("SKIPPED")],
                                     "note": "with the raw-report directory absent the package tests run validate_mi300x with raw=None, so the raw-dependent validator checks are NOT executed (listed under amd_validator_no_raw.raw_dependent_checks_not_rerun_locally); these tests do not certify them"}
            # AMD validator without raw reports, on a copy of the package
            cp = tmp / "MI300X"
            shutil.copytree(root / PKG["MI300X"], cp)
            r = subprocess.run([sys.executable, "scripts/paper_figures/validate_mi300x.py", "--out", str(cp), "--no-raw"],
                               cwd=repo, capture_output=True, text=True, env=env)
            q = json.load(open(cp / "qa_summary.json"))
            committed = json.load(open(root / PKG["MI300X"] / "qa_summary.json"))
            not_rerun = sorted(set(committed["checks"]) - set(q["checks"]))
            ext["amd_validator_no_raw"] = {"returncode": r.returncode, "all_checks_pass": q["all_checks_pass"],
                                           "checks_rerun": len(q["checks"]),
                                           "failed": [k for k, v in q["checks"].items() if not v.get("pass")],
                                           "raw_dependent_checks_not_rerun_locally": not_rerun or
                                           [k for k, v in q["checks"].items() if v.get("skipped")],
                                           "note": "checks that re-read MI300X reports are reported as not rerun locally; their committed results are not re-asserted here"}
            # NVIDIA QA on a copy (needs the external extract cache)
            if a.nvidia_cache and Path(a.nvidia_cache).exists():
                cpn = tmp / "nvidia"
                shutil.copytree(root / "nvidia", cpn)
                r = subprocess.run([sys.executable, "scripts/paper_figures/qa_nvidia.py", "--repo", repo, "--root", str(cpn),
                                    "--cache", a.nvidia_cache, "--base-ref", "origin/paper/figures-mi300x-data"],
                                   cwd=repo, capture_output=True, text=True, env=env)
                ext["nvidia_qa_rerun"] = {"returncode": r.returncode, "stdout": r.stdout.strip().splitlines()[-4:],
                                          "note": "run on a copy; base ref = MI300X branch so that only paper_figures paths differ"}
            else:
                ext["nvidia_qa_rerun"] = {"status": "not_rerun_locally", "reason": "external NCU extract cache not available"}
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_artifacts.py", "tests/test_profiling_metadata.py"],
                               cwd=repo, capture_output=True, text=True, env=env)
            ext["repo_cpu_tests"] = {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-2:]}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ok_ext = ext["amd_unit_tests"]["returncode"] == 0 and ext["amd_validator_no_raw"]["returncode"] == 0 and \
            ext["repo_cpu_tests"]["returncode"] == 0 and ext["nvidia_qa_rerun"].get("returncode", 0) == 0
        check("external:device_qa_and_cpu_tests", ok_ext, ext)

    gate = all(r["status"] == "pass" for r in res if r["gating"])
    out = {"status": "pass" if gate else "fail", "repo_head": git(repo, "rev-parse", "HEAD").stdout.strip(), "checks": res}
    json.dump(out, open(comb / "qa_combined.json", "w"), indent=1)
    for r in res:
        print(f"{r['status']:4s} {r['check']}")
    print("GATE", out["status"])
    sys.exit(0 if gate else 1)


def _num_eq(x, y):
    try:
        return float(x) == float(y)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
