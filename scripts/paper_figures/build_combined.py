"""Cross-device normalization layer for the TileArena paper figures (B200, GH200, MI300X).

Reads ONLY the committed device packages (artifacts/paper_figures/{nvidia/B200,nvidia/GH200,amd/MI300X}),
the formal CSVs and git history; never rewrites the device packages. Writes artifacts/paper_figures/combined/.

Canonical case identity (case_id_v2):
    canonical = {"operator": op, "dtype": dtype, "params": full_input_parameter_dict}
    case_id_v2 = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
full_input_parameter_dict = engine params of the case (`expand_cases`, dtype/block_size removed), resolved from the
formal-CSV row with the engine's matching rule against the operator config of the MEASUREMENT SOURCE:
    GH200  : c882fe50 (89 CSVs), 3c5eccbf (batched_matmul_autotune.csv)          [developer_guide, PROVENANCE.md]
    MI300X : 4d08985a (autotune), 005ab63b (default)                              [MI300X environment.json]
    B200   : not recorded -> for every CSV column, the config in the tree of the commit that introduced the column's
             current values (git history of the CSV); the TileLang column is additionally checked at every commit since
             the column first appeared. All resolutions must agree with each other.
Usage:
    PYTHONPATH=.:scripts/paper_figures CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_combined.py --repo .
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvidia_common as NC  # noqa: E402

csv.field_size_limit(1 << 30)

PKG = {"B200": "nvidia/B200", "GH200": "nvidia/GH200", "MI300X": "amd/MI300X"}
VENDOR = {"B200": "NVIDIA", "GH200": "NVIDIA", "MI300X": "AMD"}
ARCH = {"B200": "blackwell/sm_100", "GH200": "hopper/sm_90", "MI300X": "cdna3/gfx942"}
DSL_SUPPORT = {"B200": ("triton", "cutile", "tilelang"), "GH200": ("triton", "cutile", "tilelang"), "MI300X": ("triton",)}
MEASUREMENT_CONFIG = {
    "GH200": lambda op, mode: "3c5eccbf" if (op, mode) == ("batched_matmul", "autotune") else "c882fe50",
    "MI300X": lambda op, mode: "4d08985a" if mode == "autotune" else "005ab63b",
}
CANONICAL_CATEGORIES = ["Point-wise", "Reduction/Normalization", "Matrix Multiplication/Attention", "Stencil/Convolution", "Data Layout"]
SHORT_CATEGORY = {"Point-wise": "Point-wise", "Reduction/Normalization": "Reduction/Norm.",
                  "Matrix Multiplication/Attention": "Matrix Mult./Attn.", "Stencil/Convolution": "Stencil/Conv.",
                  "Data Layout": "Data Layout"}
KNOWN_UNSUPPORTED = {("MI300X", "triton", "matmul_fp32_fp16_fp8", "fp8_e4m3fn"):
                     "UNSUPPORTED_DTYPE on gfx942 (FP8 there is E4M3FNUZ; the PyTorch reference rejects e4m3fn); no CSV row, no winner, no profile"}

LIMITATIONS = {
    "B200": [
        "Formal Triton/cuTile columns come from the paper campaign with a historical cuda-tile version (1.3.0 per developer_guide); benchmark source commit not recorded.",
        "Frozen columns used a fixed 64 MB L2 eviction for 41 operators (< 126.5 MB L2) and config warmup/repeat 20/100.",
        "TileLang column measured in a later campaign (direct runtime, PR #319) with a different environment.",
        "9 Triton/cuTile NCU reports exclude auxiliary PyTorch launches (fill/copy) that the formal run() timing includes.",
        "No B200 NCU report records its source commit.",
        "13 TileLang reports are reduced (8, kernel replay) or targeted (5) collections without per-opcode or PC-sampling data.",
    ],
    "GH200": [
        "CUDA 13.1 and cuda-tile 1.5.0 (tileiras 13.4.92) differ from B200; NCU 2025.4.0.",
        "Formal protocol warmup 1 / repeat 3 with a 120 MiB eviction (B200 frozen: 20/100, 64 MB for 41 operators).",
        "TileLang uses fragment-accumulator kernel bodies on sm_90 for 9 operators whose Blackwell path uses TMEM.",
        "Triton matmul_fp32_fp16_fp8, matmul_int8, batched_matmul, streamk_matmul cache a transposed B operand outside the timed region.",
    ],
    "MI300X": [
        "Formal CSV timing uses warmup 20 / repeat 100 (config at 4d08985a/005ab63b), not the later 1/3 configuration.",
        "Formal run uses ROCm eager timing (HIP-graph request falls back to eager) with a 512 MiB flush.",
        "matmul_fp32_fp16_fp8 FP8 E4M3FN is unsupported and remains missing (109 operator/dtype pairs).",
        "Three diagnosis records are not fully traced (batch_normalization secondary M6; batch_normalization and softmax primary M3; destindex secondary M2).",
        "Only Triton (and the PyTorch reference) exist on MI300X.",
    ],
}


def git(repo, *a, check=False):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def case_id_v2(op, dtype, params):
    canonical = {"operator": op, "dtype": dtype, "params": params}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def canon(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def norm_op(o):
    return o.strip().lower().replace("-", "_")


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read_csv(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


class Resolver:
    """Expand operator configs at arbitrary commits and resolve CSV labels with the engine rule."""

    def __init__(self, repo):
        self.repo = repo
        self._cfg, self._cases = {}, {}
        from tilebench.data.tensors import expand_cases, infer_problem_size
        self.expand_cases, self.infer_problem_size = expand_cases, infer_problem_size

    def cfg(self, commit, op):
        key = (commit, op)
        if key not in self._cfg:
            import yaml
            self._cfg[key] = None
            for p in (f"tilebench/benchmarks/operators/{op}/config.yaml", f"benchmarks/operators/{op}/config.yaml"):
                r = git(self.repo, "show", f"{commit}:{p}")
                if r.returncode == 0:
                    self._cfg[key] = yaml.safe_load(r.stdout)
                    break
        return self._cfg[key]

    def cases(self, commit, op):
        key = (commit, op)
        if key not in self._cases:
            cfg = self.cfg(commit, op)
            out = []
            if cfg is not None:
                for c in self.expand_cases(op, cfg):
                    params = {k: v for k, v in c.items() if k not in ("dtype", "block_size")}
                    out.append((NC.norm_dtype(str(c.get("dtype", "fp32"))), params, self.infer_problem_size(op, params)))
            self._cases[key] = out
        return self._cases[key]

    def resolve(self, commit, op, label, dtype):
        import build_nvidia_tables as B
        return B.match_label(label, self.cases(commit, op), dtype)


def csv_history(repo, path):
    """[(commit, path_at_commit)] newest first, following renames."""
    out = git(repo, "log", "--follow", "--format=%H", "--name-only", "HEAD", "--", path).stdout.split("\n")
    hist, c = [], None
    for line in out:
        line = line.strip()
        if len(line) == 40 and all(ch in "0123456789abcdef" for ch in line):
            c = line
        elif line and c:
            hist.append((c, line))
            c = None
    return hist


def csv_rows_at(repo, commit, path):
    r = git(repo, "show", f"{commit}:{path}")
    return list(csv.DictReader(io.StringIO(r.stdout))) if r.returncode == 0 else None


def b200_config_commits(repo, op, mode):
    """Commits whose operator config must resolve the B200 CSV: value-introducing commit of each column, plus every
    commit since the TileLang column first appeared."""
    path = f"results/B200/csv/{op}_{mode}.csv"
    hist = csv_history(repo, path)
    with open(Path(repo) / path, newline="") as f:
        cur = list(csv.DictReader(f))
    commits, detail = set(), {}
    for col in ("triton_ms", "cutile_ms", "tilelang_ms"):
        def colmap(rows):
            return None if rows is None else {(r["params"].strip(), NC.norm_dtype(r["dtype"].strip())): (r.get(col) or "").strip() for r in rows}
        target, intro = colmap(cur), None
        tl_window = []
        for c, p in hist:
            rows = csv_rows_at(repo, c, p)
            if col == "tilelang_ms" and rows and "tilelang_ms" in rows[0]:
                tl_window.append(c)
            if colmap(rows) == target and intro is not None or (intro is None and colmap(rows) == target):
                intro = c
                continue
            break
        detail[col] = intro
        if intro:
            commits.add(intro)
        if col == "tilelang_ms":
            # every commit of the CSV history that already carries a tilelang column
            for c, p in hist:
                rows = csv_rows_at(repo, c, p)
                if rows and "tilelang_ms" in rows[0]:
                    commits.add(c)
    return sorted(commits), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", default="artifacts/paper_figures")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, a.repo)
    repo, root = a.repo, Path(a.repo) / a.root
    out = root / "combined"
    out.mkdir(parents=True, exist_ok=True)
    R = Resolver(repo)
    problems = []
    head = git(repo, "rev-parse", "HEAD").stdout.strip()

    # ------------------------------------------------------------------ category mapping
    mi_cat = {r["operator"]: r["category"] for r in read_csv(root / PKG["MI300X"] / "benchmark_cases.csv")}
    cat_rows = []
    for op in sorted(NC.CATEGORY):
        c = NC.CATEGORY[op]
        cat_rows.append({"operator": op, "category": c, "category_short": SHORT_CATEGORY[c],
                         "category_order": CANONICAL_CATEGORIES.index(c),
                         "mi300x_package_label": mi_cat.get(op, ""),
                         "source": "TileBench paper Table 'Operator benchmark suite' (table_tex/benchmark_list.tex)"})
        if mi_cat.get(op) != SHORT_CATEGORY[c]:
            problems.append(f"category mismatch {op}: MI300X '{mi_cat.get(op)}' vs canonical '{c}'")
    write(out / "category_mapping.csv", cat_rows)

    # ------------------------------------------------------------------ benchmark cases
    bench, cross = [], []
    csv_sha = {}
    for dev in ("B200", "GH200", "MI300X"):
        rows = read_csv(root / PKG[dev] / "benchmark_cases.csv")
        # per (op, mode) resolution configs
        cfg_commits = {}
        for op in sorted(NC.CATEGORY):
            for mode in ("autotune", "default"):
                if dev == "B200":
                    cfg_commits[(op, mode)], _ = b200_config_commits(repo, op, mode)
                else:
                    cfg_commits[(op, mode)] = [MEASUREMENT_CONFIG[dev](op, mode)]
        raw_cache = {}
        for r in rows:
            op, dt, mode = norm_op(r["operator"]), NC.norm_dtype(r["dtype"].strip().lower()), r["mode"]
            src = r["source_csv"]
            if src not in raw_cache:
                with open(Path(repo) / src, newline="") as f:
                    raw_cache[src] = list(csv.reader(f))
                csv_sha[src] = sha256_file(Path(repo) / src)
            raw = raw_cache[src]
            hdr = raw[0]
            # locate the formal CSV row
            if dev == "MI300X":
                rr = raw[int(r["csv_row_number"]) - 1]
                rowd = dict(zip(hdr, rr))
            else:
                cands = [dict(zip(hdr, x)) for x in raw[1:]]
                cands = [x for x in cands if NC.norm_dtype(x["dtype"].strip()) == dt
                         and (x.get(f"{r['dsl']}_ms") or "").strip() == r["dsl_ms"]
                         and (x.get("torch_ms") or "").strip() == r["torch_ms"]]
                full_pkg = json.loads(r["params_json"])
                cands = [x for x in cands if any(canon(h) == canon(full_pkg)
                                                 for h in R.resolve("HEAD", op, x["params"].strip(), dt))]
                if len(cands) != 1:
                    problems.append(f"{dev} {src}: cannot locate CSV row for {r['case_id'][:12]} ({len(cands)})")
                    continue
                rowd = cands[0]
            label = rowd["params"].strip()
            col = f"{r['dsl']}_ms"
            if (rowd.get(col) or "").strip() != r["dsl_ms"] or (rowd.get("torch_ms") or "").strip() != r["torch_ms"]:
                problems.append(f"{dev} {src} {label}: package latency string differs from CSV")
            res = set()
            for c in cfg_commits[(op, mode)]:
                hits = R.resolve(c, op, label, dt)
                res.add(canon(hits[0]) if len(hits) == 1 else f"AMBIGUOUS:{len(hits)}@{c[:8]}")
            if len(res) != 1 or next(iter(res)).startswith("AMBIGUOUS"):
                problems.append(f"{dev} {op} {mode} '{label}' {dt}: unresolved full params {sorted(res)}")
                continue
            full = json.loads(next(iter(res)))
            cid = case_id_v2(op, dt, full)
            if dev != "MI300X" and canon(full) != canon(json.loads(r["params_json"])):
                problems.append(f"{dev} {op} {label}: package full params differ from measurement-source resolution")
            v_src = r["validity"]
            t, d = fnum(r["torch_ms"]), fnum(r["dsl_ms"])
            ok_src = v_src in ("valid", "verified_ok")
            if ok_src and t is not None and t > 0 and d is not None and d > 0:
                v = "valid"
            elif (r["dsl_ms"] or "").strip() == "":
                v = "missing"
            else:
                v = "invalid"
            bench.append({
                "device": dev, "vendor": VENDOR[dev], "architecture": ARCH[dev], "dsl": r["dsl"], "operator": op,
                "category": NC.CATEGORY[op], "dtype": dt, "mode": mode, "case_id_v2": cid, "params_full_json": canon(full),
                "source_case_id": r["case_id"], "source_params_json": r["params_json"], "csv_params_label": label,
                "torch_ms": r["torch_ms"], "dsl_ms": r["dsl_ms"], "validity": v, "validity_source": v_src,
                "source_csv": src, "source_csv_sha256": csv_sha[src], "source_git_sha": r["source_git_sha"],
                "params_config_commits": ";".join(c[:12] for c in cfg_commits[(op, mode)]),
                "notes": r.get("notes", "")})
            cross.append((dev, mode, op, dt, r["case_id"], r["params_json"], cid, canon(full),
                          ";".join(c[:12] for c in cfg_commits[(op, mode)])))
    # explicit unsupported rows
    for (dev, dsl, op, dt), why in KNOWN_UNSUPPORTED.items():
        if any(b["device"] == dev and b["dsl"] == dsl and b["operator"] == op and b["dtype"] == dt for b in bench):
            problems.append(f"known-unsupported pair has benchmark rows: {dev} {dsl} {op} {dt}")
    write(out / "benchmark_cases_normalized.csv.gz", bench, gz=True)

    xw, seen = [], set()
    for d, m, o, t, s, sp, c, pf, cc in cross:          # one row per device x mode x source case (DSL rows share it)
        if (d, m, s) in seen:
            continue
        seen.add((d, m, s))
        xw.append({"device": d, "mode": m, "operator": o, "dtype": t, "source_case_id": s, "source_params_json": sp,
                   "case_id_v2": c, "params_full_json": pf, "params_config_commits": cc,
                   "source_rule": ("sha256(canonical{dtype,operator,params_full})" if d != "MI300X" else "sha256('op|dtype|swept_params_json')"),
                   "source_equals_v2": s == c})
    write(out / "case_id_crosswalk.csv", xw)

    # ------------------------------------------------------------------ profile index
    bench_ids = {(b["device"], b["dsl"], b["case_id_v2"]) for b in bench if b["mode"] == "autotune" and b["validity"] == "valid"}
    prof = []
    for dev in ("B200", "GH200"):
        for p in read_csv(root / PKG[dev] / "profile_index.csv"):
            full = json.loads(p["params_json"])
            cid = case_id_v2(p["operator"], p["dtype"], full)
            prof.append({
                "device": dev, "vendor": "NVIDIA", "dsl": p["dsl"], "operator": p["operator"], "dtype": p["dtype"],
                "category": NC.CATEGORY[p["operator"]], "profile_id": p["profile_id"], "report_kind": f"ncu_{p['collection_level']}",
                "profiler": f"Nsight Compute {p['ncu_version'].split(' ')[0]}", "collection_level_source": p["collection_level"],
                "replay_mode": p["replay_mode"], "source_case_id": p["case_id"], "case_id_v2": cid, "params_full_json": canon(full),
                "benchmark_case_v2_match": (dev, p["dsl"], cid) in bench_ids, "benchmark_case_match_status_source": p["benchmark_case_match_status"],
                "code_match_status": p["code_match_status"],
                "code_match_confidence": "documented" if p["code_match_status"].startswith("documented") else "not_recorded",
                "winner_config_json": p["winner_config_json"], "launch_count": p["launch_count"],
                "report_path": p["report_path"], "report_sha256": p["report_sha256"], "report_origin": p["report_origin"],
                "report_revision": p["report_revision"], "instruction_count_semantics":
                    ("dynamic SASS per-opcode (sass__inst_executed_per_opcode)" if p["collection_level"] == "full" else "static SASS only"),
                "notes": p["notes"]})
    for p in read_csv(root / PKG["MI300X"] / "profile_index.csv"):
        full = json.loads(p["profiled_params_full_json"])
        op, dt = norm_op(p["operator"]), NC.norm_dtype(p["dtype"])
        full = {k: v for k, v in full.items() if k != "dtype"}   # two captures record a dtype key (README)
        cid = case_id_v2(op, dt, full)
        kind = p["profile_id"].split(".")[-1]
        dsl = "triton" if p["dsl"] == "triton" else "pytorch"
        prof.append({
            "device": "MI300X", "vendor": "AMD", "dsl": dsl, "operator": op, "dtype": dt, "category": NC.CATEGORY[op],
            "profile_id": p["profile_id"], "report_kind": kind, "profiler": p["collection_level"][:120],
            "collection_level_source": p["collection_level"], "replay_mode": p["replay_mode"], "source_case_id": p["case_id"],
            "case_id_v2": cid, "params_full_json": canon(full),
            "benchmark_case_v2_match": ("MI300X", "triton", cid) in bench_ids, "benchmark_case_match_status_source": p["benchmark_case_match_status"],
            "code_match_status": p["code_match_status"],
            "code_match_confidence": ("verified" if p["code_match_status"].startswith(("verified", "matched", "ok")) or "tree hash" in p["code_match_status"]
                                      else ("not_applicable" if p["code_match_status"].startswith("not_applicable") else "partial")),
            "winner_config_json": p.get("kernel_config_json", ""), "launch_count": p["launch_count"], "report_path": p["report_path"],
            "report_sha256": p["report_sha256"], "report_origin": p["report_origin"], "report_revision": p["report_revision"],
            "instruction_count_semantics": {"static_isa": "static AMDGCN opcode counts", "rocprof_compute": "dynamic SQ_INSTS_* hardware counters (summed over dispatches)",
                                            "pc_sampling": "stochastic PC samples (not executions)", "att": "ATT hit counts of traced waves on one CU",
                                            "kernel_trace": "kernel trace only"}.get(kind, "see package README"),
            "notes": p["notes"]})
    write(out / "profile_index_normalized.csv", prof)

    # ------------------------------------------------------------------ coverage summary
    cov = []
    pairs_all = sorted({(b["operator"], b["dtype"]) for b in bench if b["device"] == "B200"} | {(o, t) for (_, _, o, t) in KNOWN_UNSUPPORTED})
    for dev in ("B200", "GH200", "MI300X"):
        for dsl in ("triton", "cutile", "tilelang"):
            for op, dt in pairs_all:
                rows = [b for b in bench if b["device"] == dev and b["dsl"] == dsl and b["operator"] == op and b["dtype"] == dt and b["mode"] == "autotune"]
                pp = [p for p in prof if p["device"] == dev and p["dsl"] == dsl and p["operator"] == op and p["dtype"] == dt]
                if dsl not in DSL_SUPPORT[dev]:
                    status = "dsl_not_available_on_device"
                elif (dev, dsl, op, dt) in KNOWN_UNSUPPORTED:
                    status = "unsupported_dtype"
                elif rows:
                    status = "measured"
                else:
                    status = "missing"
                cov.append({"device": dev, "dsl": dsl, "operator": op, "dtype": dt, "category": NC.CATEGORY[op], "status": status,
                            "n_cases_autotune": len(rows), "n_valid_autotune": sum(b["validity"] == "valid" for b in rows),
                            "n_profiles": len(pp), "profile_kinds": ";".join(sorted({p["report_kind"] for p in pp})),
                            "note": KNOWN_UNSUPPORTED.get((dev, dsl, op, dt), "")})
    write(out / "coverage_summary.csv", cov)

    # ------------------------------------------------------------------ intersections + sanity aggregates
    def ids(dev, dsl, mode="autotune"):
        return {b["case_id_v2"] for b in bench if b["device"] == dev and b["dsl"] == dsl and b["mode"] == mode and b["validity"] == "valid"}
    inter = {
        "triton_B200_GH200_MI300X": len(ids("B200", "triton") & ids("GH200", "triton") & ids("MI300X", "triton")),
        "triton_B200_GH200": len(ids("B200", "triton") & ids("GH200", "triton")),
        "triton_B200_MI300X": len(ids("B200", "triton") & ids("MI300X", "triton")),
        "cutile_B200_GH200": len(ids("B200", "cutile") & ids("GH200", "cutile")),
        "tilelang_B200_GH200": len(ids("B200", "tilelang") & ids("GH200", "tilelang")),
        "per_device_valid_autotune": {f"{d}:{s}": len(ids(d, s)) for d in DSL_SUPPORT for s in DSL_SUPPORT[d]},
        "triton_only_in": {d: sorted({b["operator"] + "/" + b["dtype"] for b in bench if b["device"] == d and b["dsl"] == "triton" and b["mode"] == "autotune"
                                      and b["case_id_v2"] in (ids(d, "triton") - (ids("B200", "triton") & ids("GH200", "triton") & ids("MI300X", "triton")))})
                           for d in ("B200", "GH200", "MI300X")},
    }
    agg = aggregates(bench)
    manifest = {
        "schema": "combined/1", "repo_head": head, "case_id_rule": "sha256(json.dumps({operator,dtype,params_full}, sort_keys=True, separators=(',',':'), ensure_ascii=True))",
        "params_resolution": {"GH200": "config at c882fe50 (batched_matmul_autotune: 3c5eccbf)", "MI300X": "config at 4d08985a (autotune) / 005ab63b (default)",
                              "B200": "config at the value-introducing commit of every CSV column (+ every commit carrying the TileLang column); all resolutions agree"},
        "dsl_support": {d: list(v) for d, v in DSL_SUPPORT.items()}, "known_unsupported": {"/".join(k): v for k, v in KNOWN_UNSUPPORTED.items()},
        "intersections_case_id_v2": inter,
        "aggregation_protocol": {
            "speedup": "S[o,b,d] = GM over valid autotune cases of torch_ms/dsl_ms; S[b,d] = GM over operators of S[o,b,d]",
            "dsl_ratio": "R[X/Triton][o,d] = GM over cases valid for both of X_ms/triton_ms (>1: X slower)",
            "cross_device_delta": "delta = log2(S_dev2/S_dev1) over matched case_id_v2 (relative to each device's local PyTorch; not an absolute hardware speedup)",
            "winner": "per operator and device, the DSL with the lowest GM latency over cases valid for all DSLs of that device; near parity = within 5% of the winner"},
        "sanity_aggregates": agg,
        "source_packages": {d: {f: sha256_file(root / PKG[d] / f) for f in sorted(os.listdir(root / PKG[d]))} for d in PKG},
        "source_csv_sha256": csv_sha,
        "device_limitations": LIMITATIONS,
        "metric_semantics_note": "normalized/semantic groups are labels; NVIDIA dynamic SASS counts, AMD static AMDGCN counts, AMD PC samples and AMD ATT hit counts are different measurements and are never placed on one numeric axis",
        "problems": problems,
    }
    json.dump(manifest, open(out / "comparison_manifest.json", "w"), indent=1, sort_keys=False)
    write(out / "metric_semantic_groups.csv", semantic_groups(root))
    print("bench", len(bench), "crosswalk", len(xw), "profiles", len(prof), "coverage", len(cov), "problems", len(problems))
    print(json.dumps(inter, indent=0)[:1500])
    print(json.dumps(agg["overall"], indent=0))
    for p in problems[:30]:
        print("PROBLEM", p)


def gm(xs):
    xs = [x for x in xs if x is not None and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def aggregates(bench):
    by = defaultdict(list)
    for b in bench:
        if b["mode"] == "autotune" and b["validity"] == "valid":
            by[(b["device"], b["dsl"], b["operator"])].append(fnum(b["torch_ms"]) / fnum(b["dsl_ms"]))
    per_op = {f"{d}|{s}|{o}": gm(v) for (d, s, o), v in by.items()}
    overall = {}
    for d in DSL_SUPPORT:
        for s in DSL_SUPPORT[d]:
            vals = [v for k, v in per_op.items() if k.startswith(f"{d}|{s}|")]
            overall[f"{d}|{s}"] = {"S": gm(vals), "n_operators": len(vals)}
    # winners over cases valid for all DSLs of the device
    lat = defaultdict(dict)
    for b in bench:
        if b["mode"] == "autotune" and b["validity"] == "valid":
            lat[(b["device"], b["operator"], b["case_id_v2"])][b["dsl"]] = fnum(b["dsl_ms"])
    winners = {}
    for d in ("B200", "GH200"):
        cnt, near = Counter(), Counter()
        for o in NC.CATEGORY:
            keys = [k for k in lat if k[0] == d and k[1] == o and all(s in lat[k] for s in DSL_SUPPORT[d])]
            g = {s: gm([lat[k][s] for k in keys]) for s in DSL_SUPPORT[d]}
            w = min(g, key=g.get)
            cnt[w] += 1
            second = sorted(g.values())[1]
            if second / g[w] <= 1.05:
                near[w] += 1
        winners[d] = {"counts": dict(cnt), "winner_within_5pct_of_runner_up": dict(near)}
    return {"overall": overall, "winners": winners}


def semantic_groups(root):
    m = {"integer_address": "indexing", "predicate_select": "indexing", "global_memory": "memory_access", "global_load_store": "memory_access",
         "shared_memory": "memory_access", "local_memory": "resource_pressure", "cache": "cache_behavior", "matrix_pipeline": "matrix_execution",
         "tcgen05": "matrix_execution", "wgmma": "matrix_execution", "legacy_mma": "matrix_execution", "tma": "memory_access",
         "synchronization": "synchronization", "occupancy": "occupancy", "resources": "resource_pressure", "stalls": "latency_hiding",
         "atomics": "memory_access", "execution": "execution", "instruction_mix": "instruction_mix"}
    rows = []
    for dev in PKG:
        seen = Counter()
        p = root / PKG[dev] / "kernel_metrics_long.csv.gz"
        with gzip.open(p, "rt", newline="") as f:
            for r in csv.DictReader(f):
                seen[(r["normalized_group"], r["counter_kind"].split("|")[0])] += 1
        for (g, k), n in sorted(seen.items()):
            rows.append({"device": dev, "vendor": VENDOR[dev], "source_normalized_group": g, "source_counter_kind": k,
                         "semantic_group": m.get(g, "other"), "rows": n,
                         "comparability": "label only; values comparable only within the same vendor, profiler, metric name and scope"})
    return rows


def write(p, rows, gz=False):
    cols = list(rows[0].keys()) if rows else []
    opener = (lambda q: gzip.GzipFile(q, "wb", mtime=0)) if gz else None
    if gz:
        with gzip.GzipFile(p, "wb", mtime=0) as g:
            t = io.TextIOWrapper(g, encoding="utf-8", newline="")
            w = csv.DictWriter(t, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            t.flush()
            t.detach()
    else:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
