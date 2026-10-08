"""Build the normalized NVIDIA profiling tables for one device from offline extracts.

Inputs (nothing here touches a GPU):
  --extract-dir   per-report JSON produced by ncu_extract.py (external cache)
  --inventory     inventory_<device>.json produced by nvidia_inventory.py (sha256 + HF revision)
  --repo          a TileBench checkout (formal CSVs results/<device>/csv, operator configs, archive refs)
Outputs (artifacts/paper_figures/nvidia/<device>/):
  benchmark_cases.csv, profile_index.csv, kernel_metrics_long.csv.gz, instruction_mix.csv,
  pc_hotspots.csv.gz, execution_paths.csv, environment.json
Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES= python scripts/paper_figures/build_nvidia_tables.py --device B200 \
      --repo . --extract-dir <cache>/extract/B200 --inventory <cache>/inventory_B200.json \
      --out artifacts/paper_figures/nvidia/B200
"""
import argparse
import collections
import csv
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvidia_common as C  # noqa: E402

TYPE_NAMES = {0: "other", 1: "counter", 2: "ratio", 3: "throughput"}
SUBTYPE_NAMES = {0: "none", 1: "peak_sustained", 2: "peak_sustained_active", 3: "peak_sustained_active_per_second",
                 4: "peak_sustained_elapsed", 5: "peak_sustained_elapsed_per_second", 10: "per_cycle_active",
                 11: "per_cycle_elapsed", 14: "per_second", 15: "pct_of_peak_sustained_active",
                 16: "pct_of_peak_sustained_elapsed", 19: "max_rate", 20: "pct", 21: "ratio"}
ROLLUP_NAMES = {0: "none", 1: "avg", 2: "max", 3: "min", 4: "sum"}

# Export filter for the EXTENDED metric set (the external extract keeps everything)
_EXPORT_DROP = re.compile(r"\.(per_second|peak_sustained|per_cycle_elapsed|pct_of_peak_sustained_region|pct_of_peak_sustained_frame)$|per_second\.|peak_sustained_(active|elapsed)_per_second")
_ZERO_KEEP = re.compile(r"tma|tensor|_tc|tmem|ldgsts|local|spill|atom|_red|bank_conflict|dram__bytes|sectors|requests|shared|gmma|hmma|imma|stalled|inst_executed")
PER_LAUNCH_LIMIT = 16   # profiles with more launches aggregate instruction mix per kernel stage
PC_TOP_SAMPLES, PC_TOP_INST = 40, 10

csv.field_size_limit(1 << 30)


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def stage_of(kernel_name):
    """Kernel stage label: kernel name with the cuTile type-mangling suffix (_Kt<digit>...) removed."""
    return re.sub(r"_Kt\d.*$", "", kernel_name)


# ------------------------------------------------------------------ benchmark cases
def load_cases(repo, op):
    import yaml
    from tilebench.data.tensors import expand_cases, infer_problem_size
    cfg = yaml.safe_load(open(Path(repo) / "tilebench/benchmarks/operators" / op / "config.yaml"))
    out = []
    for c in expand_cases(op, cfg):
        params = {k: v for k, v in c.items() if k not in ("dtype", "block_size")}
        out.append((C.norm_dtype(str(c.get("dtype", "fp32"))), params, infer_problem_size(op, params)))  # engine default dtype
    return out


def _value_matches(actual, expected):
    if str(actual) == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def match_label(label, cases, dtype):
    """Same matching rule as scripts/run_bench.py::_label_matches_result (label = varying keys only)."""
    parsed = {}
    for part in label.split(","):
        if "=" not in part:
            return []
        k, v = part.split("=", 1)
        parsed[k.strip()] = v.strip()
    hits = []
    for dt, params, psize in cases:
        if dt != dtype:
            continue
        ok = True
        for k, v in parsed.items():
            if k == "n" and k not in params:
                actual = psize
            elif k in params:
                actual = params[k]
            else:
                ok = False
                break
            if not _value_matches(actual, v):
                ok = False
                break
        if ok:
            hits.append(params)
    return hits


def build_benchmark_cases(device, repo, out_rows, problems):
    dev = C.DEVICES[device]
    csv_dir = Path(repo) / "results" / device / "csv"
    head_sha = C.git_rev(repo, "HEAD")
    for op in sorted(C.CATEGORY):
        cases = load_cases(repo, op)
        for mode in ("autotune", "default"):
            rel = f"results/{device}/csv/{op}_{mode}.csv"
            path = Path(repo) / rel
            if not path.exists():
                problems.append(f"missing CSV {rel}")
                continue
            src_sha = C.git_last_commit(repo, rel)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                raw_dt = r["dtype"].strip()
                dt = C.norm_dtype(raw_dt)
                hits = match_label(r["params"].strip(), cases, dt)
                note = []
                if raw_dt != dt:
                    note.append(f"csv dtype label '{raw_dt}' normalized to '{dt}'")
                if len(hits) != 1:
                    problems.append(f"{device} {rel} params='{r['params']}' dtype={dt}: {len(hits)} case matches")
                    params, cid = {"__unmatched_label__": r["params"].strip()}, "UNMATCHED"
                else:
                    params = hits[0]
                    cid = C.case_id(op, dt, params)
                for dsl in C.DSLS:
                    col = C.DSL_COLUMN[dsl]
                    if col not in r:
                        continue
                    raw = (r[col] or "").strip()
                    v = fnum(raw)
                    validity = "valid" if (v is not None and v > 0) else ("missing" if raw == "" else f"invalid:{raw}")
                    out_rows.append({
                        "device": device, "architecture": dev["architecture"], "dsl": dsl, "operator": op,
                        "category": C.CATEGORY[op], "dtype": dt, "case_id": cid, "params_json": C.canonical_params(params),
                        "mode": mode, "torch_ms": (r.get("torch_ms") or "").strip(), "dsl_ms": raw, "validity": validity,
                        "source_csv": rel, "source_git_sha": src_sha, "notes": "; ".join(note)})
    return head_sha


# ------------------------------------------------------------------ profiles
B200_TL_HF_PR2 = {   # collection limitations stated in HF bcui2/NCU_report PR #2 (commit 21037737)
    "targeted": {("linear_self_attention", "fp32"), ("bitonic_sort", "fp16"), ("bitonic_sort", "fp32"),
                 ("radix_sort", "int32"), ("top_k_selection", "fp32")},
    "reduced": {("batched_matmul", "bf16"), ("batched_matmul", "fp16"), ("block_sparse_attention", "fp16"),
                ("matmul_fp32_fp16_fp8", "fp16"), ("matmul_fp32_fp16_fp8", "fp8_e4m3fn"), ("matmul_int8", "int8"),
                ("streamk_matmul", "bf16"), ("streamk_matmul", "fp16")},
}


def winner_and_params(device, repo, catalogue, op, dtype, dsl, tl_cache):
    entry = catalogue.get(op, {}).get("autotune_winner_per_dtype", {}).get(dtype)
    if not entry:
        return None, None, "no catalogue entry"
    params = entry.get("params")
    win = entry.get(dsl)
    note = "params+winner from NCU catalogue"
    if win is None and device == "B200" and dsl == "tilelang":
        ref, pat = C.DEVICES["B200"]["tilelang_autotune_ref"]
        key = op
        if key not in tl_cache:
            tl_cache[key] = C.git_show_json(repo, ref, pat.format(op=op)) or []
        hits = [x for x in tl_cache[key] if C.norm_dtype(x.get("dtype", "")) == dtype and x.get("params") == params]
        win = hits[0].get("tilelang_autotune_cfg") if len(hits) == 1 else None
        note = ("params from the B200 NCU catalogue (HF PR #2: TileLang reports use the catalogued maximum input); "
                f"winner from {ref}:{pat.format(op=op)}" + ("" if win is not None else " (winner not found)"))
    return params, win, note


def code_match(device, dsl, op, dtype, gh200_prov):
    if device == "GH200":
        if dsl == "triton" and op == "bitonic_sort":
            return ("documented:reprofiled_from_5610f18f",
                    "Triton bitonic_sort re-profiled from 5610f18f (pad BLOCK fix); formal CSV from c882fe50; launch sequence equal to the formal autotune path (PROVENANCE.md)")
        if dsl == "tilelang" and op == "destindex":
            return ("documented:default_config_split",
                    "profiling source d6ddb622 splits the TileLang destindex default config into nope/rope (same values); kernel and search space unchanged (PROVENANCE.md)")
        src = "d6ddb622" if dsl == "tilelang" else "638ea849"
        return ("documented:profiling_infra_only",
                f"profiling source {src}; formal CSV source c882fe50 (batched_matmul_autotune 3c5eccbf); only profiling infrastructure changed (PROVENANCE.md)")
    if dsl == "tilelang":
        return ("not_recorded", "B200 TileLang reports (HF PR #2): source commit not stated; paper-era packages, recorded autotune winner, catalogued max input")
    return ("not_recorded", "B200 Triton/cuTile profiling source not recorded (developer_guide: Profiling inventory)")


def build_profiles(device, repo, inv, extract_dir, bench_rows, problems):
    dev = C.DEVICES[device]
    catalogue = {x["op"]: x for x in (C.git_show_json(repo, *dev["catalogue_ref"]) or [])}
    kcounts = {(x["op"], x["dtype"], x["backend"]): x for x in (C.git_show_json(repo, *dev["kernel_counts_ref"]) or [])}
    manifest = C.git_show_json(repo, *dev["manifest_ref"]) if "manifest_ref" in dev else None
    bench_ok = {(r["dsl"], r["case_id"]): r for r in bench_rows if r["mode"] == "autotune"}
    tl_cache, profiles, extracts = {}, [], {}
    for rep in inv["reports"]:
        op, dsl, dtype = rep["operator"], rep["dsl"], rep["dtype"]
        pid = f"{device}/{dsl}/{op}/{dtype}"
        ex_path = Path(extract_dir) / f"{op}__{dsl}_{dtype}.json.gz"
        ex = json.load(gzip.open(ex_path)) if ex_path.exists() else None
        extracts[pid] = ex
        params, win, wnote = winner_and_params(device, repo, catalogue, op, dtype, dsl, tl_cache)
        cid = C.case_id(op, dtype, params) if params is not None else ""
        b = bench_ok.get((dsl, cid))
        if params is None:
            bstat = "no_profile_params"
        elif b is None:
            bstat = "case_not_in_benchmark_csv"
        elif b["validity"] != "valid":
            bstat = f"matched_{b['validity']}"
        else:
            bstat = "matched"
        notes = [wnote]
        if ex is None:
            level, replay, nlaunch, status = "unknown", "unknown", "", "not_extracted"
        elif ex.get("status") != "ok":
            level, replay, nlaunch, status = "unknown", "unknown", "", ex.get("status")
            notes.append("import failed: " + "; ".join(ex.get("errors", []))[:300])
        else:
            level = ex["collection"]["level_from_command_line"]
            replay = ex["collection"]["replay_mode"]
            nlaunch = len(ex["launches"])
            status = "ok"
            if device == "B200" and dsl == "tilelang":
                stated = next((k for k, v in B200_TL_HF_PR2.items() if (op, dtype) in v), "full")
                if stated != level:
                    problems.append(f"{pid}: command-line collection level {level} != HF PR #2 statement {stated}")
                    notes.append(f"HF PR #2 states {stated}")
        exp = kcounts.get((op, dtype, dsl))
        exp_compute = None
        if exp is not None:
            names = exp.get("names") or []
            helpers = [n for n in names if n.startswith("void at::") or n.startswith("at::")]
            exp_compute = exp.get("count") - len(helpers) if names else exp.get("count")
            if nlaunch != "" and nlaunch != exp.get("count"):
                if nlaunch == exp_compute:
                    notes.append(f"report holds the {nlaunch} operator kernel launches; kernel_counts.json lists {exp.get('count')} launches including {len(helpers)} PyTorch helper launch(es) (fill/copy) that the profile does not capture")
                else:
                    notes.append(f"LAUNCH COUNT MISMATCH: report {nlaunch}, kernel_counts.json {exp.get('count')} ({exp_compute} non-helper)")
        if manifest is not None:
            mrec = manifest.get(f"{op}/{dsl}_{dtype}") if isinstance(manifest, dict) else None
            if mrec is None and isinstance(manifest, dict):
                mrec = next((v for k, v in manifest.items() if k.endswith(f"{op}/{dsl}_{dtype}.ncu-rep")), None)
        cm_status, cm_note = code_match(device, dsl, op, dtype, None)
        notes.append(cm_note)
        sess = (ex or {}).get("session", {})
        profiles.append({
            "profile_id": pid, "device": device, "dsl": dsl, "operator": op, "dtype": dtype, "case_id": cid,
            "params_json": C.canonical_params(params) if params is not None else "",
            "collection_level": level, "replay_mode": replay, "report_path": rep["hf_path"],
            "report_sha256": rep["sha256"] or "", "report_origin": ("hf_download" if device == "GH200" else "local_copy_sha256_identical_to_hf"),
            "report_revision": f"{inv['hf_repo']}@{inv['hf_revision']}", "benchmark_case_match_status": bstat,
            "code_match_status": cm_status, "launch_count": nlaunch, "notes": " | ".join(n for n in notes if n),
            # optional columns
            "extract_status": status, "kernel_names": ";".join(sorted({l["kernel_name"] for l in (ex or {}).get("launches", [])})),
            "expected_launch_count": exp.get("count") if exp else "", "expected_operator_launch_count": exp_compute if exp_compute is not None else "",
            "winner_config_json": json.dumps(win, sort_keys=True) if win is not None else "",
            "ncu_version": sess.get("ncu_target_version") or "", "report_created": sess.get("created") or "",
            "profiler_cuda_version": sess.get("cuda_version") or "", "profiler_command_line": sess.get("command_line") or "",
            "benchmark_dsl_ms": b["dsl_ms"] if b else "", "benchmark_torch_ms": b["torch_ms"] if b else "",
        })
    return profiles, extracts


# ------------------------------------------------------------------ long metrics
def counter_kind(rec):
    return "|".join([TYPE_NAMES.get(rec.get("type"), str(rec.get("type"))),
                     SUBTYPE_NAMES.get(rec.get("subtype"), str(rec.get("subtype"))),
                     ROLLUP_NAMES.get(rec.get("rollup"), str(rec.get("rollup")))])


def metric_rows(pid, ex, hf_path):
    from ncu_extract import CORE_METRICS, group_of
    rows = []
    for l in ex["launches"]:
        src = f"{hf_path}#range{l['range']}/action{l['action']}"
        base = {"profile_id": pid, "launch_id": l["launch_id"], "kernel_name": l["kernel_name"], "stage": stage_of(l["kernel_name"])}
        rep = "extended_metrics" in l
        for n, rec in l["metrics"].items():
            rows.append({**base, "raw_metric_name": n, "normalized_group": group_of(n), "value": repr(rec["value"]) if isinstance(rec["value"], float) else rec["value"],
                         "unit": rec.get("unit") or "", "scope": "launch", "counter_kind": counter_kind(rec),
                         "extraction_method": "ncu_report:core", "status": "collected", "raw_source": src, "notes": ""})
        if not rep:
            continue
        ext = l["extended_metrics"]
        for n, rec in ext.items():
            if _EXPORT_DROP.search(n):
                continue
            if n.replace(".sum.pct_of_peak", ".avg.pct_of_peak") != n and n.replace(".sum.pct_of_peak", ".avg.pct_of_peak") in ext:
                continue
            v = rec["value"]
            if isinstance(v, (int, float)) and v == 0 and not _ZERO_KEEP.search(n):
                continue
            if n.startswith("sm__ops_path") and not v:
                continue
            rows.append({**base, "raw_metric_name": n, "normalized_group": group_of(n), "value": repr(v) if isinstance(v, float) else v,
                         "unit": rec.get("unit") or "", "scope": "launch", "counter_kind": counter_kind(rec),
                         "extraction_method": "ncu_report:extended(shape_representative_launch)", "status": "collected",
                         "raw_source": src, "notes": ""})
        for n in CORE_METRICS:
            if n not in l["metrics"]:
                rows.append({**base, "raw_metric_name": n, "normalized_group": group_of(n), "value": "", "unit": "",
                             "scope": "launch", "counter_kind": "", "extraction_method": "ncu_report:core",
                             "status": "not_collected", "raw_source": src, "notes": "metric absent from this report"})
    return rows


# ------------------------------------------------------------------ instruction mix
OPC_KIND = {"sass__inst_executed_per_opcode": "dynamic_warp_inst_executed",
            "sass__inst_executed_per_opcode_with_modifier_all": "dynamic_warp_inst_executed_with_modifier",
            "sass__thread_inst_executed_true_per_opcode": "dynamic_thread_inst_executed_true"}


def launch_groups(ex):
    """Per-launch groups when the profile is small, else one group per kernel name (sum over launches)."""
    L = ex["launches"]
    if len(L) <= PER_LAUNCH_LIMIT:
        return [(str(l["launch_id"]), [l], "launch") for l in L]
    by = collections.OrderedDict()
    for l in L:
        by.setdefault(l["kernel_name"], []).append(l)
    out = []
    for k, ls in by.items():
        ids = [l["launch_id"] for l in ls]
        out.append((f"sum:{ids[0]}-{ids[-1]}", ls, f"sum_over_launches(n={len(ls)})"))
    return out


def mix_rows(pid, ex, hf_path):
    rows = []
    for gid, ls, scope in launch_groups(ex):
        for metric, kind in OPC_KIND.items():
            agg, have = collections.Counter(), 0
            for l in ls:
                d = l["opcode_counts"].get(metric)
                if d:
                    have += 1
                    agg.update(d)
            if not have:
                continue
            note = "" if have == len(ls) else f"per-opcode data present for {have}/{len(ls)} launches (sum over those only)"
            fam = collections.Counter()
            for opc, v in sorted(agg.items()):
                f = C.instruction_family(opc)
                fam[f] += v
                rows.append({"profile_id": pid, "launch_id": gid, "instruction_family": f, "instruction_name": opc,
                             "count": int(v), "count_kind": kind, "scope": scope,
                             "evidence_path": f"ncu:{metric}", "notes": note})
            if metric == "sass__inst_executed_per_opcode":
                for f, v in sorted(fam.items()):
                    rows.append({"profile_id": pid, "launch_id": gid, "instruction_family": f, "instruction_name": "__family_total__",
                                 "count": int(v), "count_kind": "dynamic_warp_inst_executed_family_total", "scope": scope,
                                 "evidence_path": f"ncu:{metric}", "notes": note})
    for kname, st in ex.get("static_sass", {}).items():
        for opc, v in sorted(st["opcodes"].items()):
            rows.append({"profile_id": pid, "launch_id": "static" if st.get("launch_id") is None else str(st["launch_id"]),
                         "instruction_family": C.instruction_family(opc), "instruction_name": opc, "count": int(v),
                         "count_kind": "static_sass_instruction_count", "scope": f"kernel_binary:{stage_of(kname)}",
                         "evidence_path": "ncu:" + ("per-PC SASS table" if st.get("launch_id") is not None else "--page source --print-source sass"), "notes": ""})
    return rows


# ------------------------------------------------------------------ PC hotspots
def pc_rows(pid, ex, hf_path):
    rows = []
    for lid, t in ex.get("pc_tables", {}).items():
        R = t["rows"]
        top = sorted(R, key=lambda r: -(r.get("samples") or 0))[:PC_TOP_SAMPLES]
        top += sorted(R, key=lambda r: -(r.get("inst_executed") or 0))[:PC_TOP_INST]
        seen = set()
        for r in top:
            if r["pc"] in seen or not ((r.get("samples") or 0) or (r.get("inst_executed") or 0)):
                continue
            seen.add(r["pc"])
            sass = re.sub(r"\s+", " ", (r.get("sass") or "").strip())
            op = re.sub(r"^@!?U?P\w+\s+", "", sass).split(" ")[0] if sass else ""
            base = {"profile_id": pid, "launch_id": lid, "pc": hex(r["pc"]), "instruction_family": C.instruction_family(op),
                    "instruction_text": sass, "executed_instruction_count": r.get("inst_executed", ""),
                    "source_line": r.get("src", ""), "evidence_path": "ncu:inst_executed+smsp__pcsamp_*"}
            note = "" if t.get("pc_sampling_collected") else "PC sampling not collected in this report"
            rows.append({**base, "sample_count": r.get("samples", 0) or 0, "stall_reason": "__all__", "notes": note})
            for s, v in sorted((r.get("stall") or {}).items(), key=lambda x: -x[1]):
                rows.append({**base, "sample_count": v, "stall_reason": s, "notes": note})
    return rows


# ------------------------------------------------------------------ execution paths
def _sum_opcodes(ls, metric):
    agg, have = collections.Counter(), 0
    for l in ls:
        d = l["opcode_counts"].get(metric)
        if d:
            have += 1
            agg.update(d)
    return agg, have


def path_rows(pid, ex, hf_path):
    rows = []
    by = collections.OrderedDict()
    for l in ex["launches"]:
        by.setdefault(l["kernel_name"], []).append(l)
    for kname, ls in by.items():
        mods, have = _sum_opcodes(ls, "sass__inst_executed_per_opcode_with_modifier_all")
        static_marker = ""
        if have:
            src, conf = f"dynamic: sass__inst_executed_per_opcode_with_modifier_all ({have}/{len(ls)} launches)", "high"
            counts = mods
        else:
            st = ex.get("static_sass", {}).get(kname)
            if not st:
                rows.append({"profile_id": pid, "stage": stage_of(kname), "access_path": "", "matrix_path": "", "buffer_location": "",
                             "layout_operations": "", "atomic_path": "", "resource_summary": "", "evidence_path": "",
                             "evidence_confidence": "none", "notes": "no opcode data in report"})
                continue
            counts = collections.Counter(st["opcodes"])
            src, conf = f"static SASS ({st['method']}); counts are static instruction sites", "medium"
            static_marker = "static_sites:"
        fam = collections.Counter()
        for o, v in counts.items():
            fam[C.instruction_family(o)] += v

        def ops_in(f):
            return {o: v for o, v in counts.items() if C.instruction_family(o) == f and v}
        acc = []
        if fam["tma"]:
            acc.append("TMA[" + ",".join(f"{o}:{v}" for o, v in sorted(ops_in('tma').items())) + "]")
        if fam["async_copy_ldgsts"]:
            acc.append("cp.async[" + ",".join(f"{o}:{v}" for o, v in sorted(ops_in('async_copy_ldgsts').items())) + "]")
        for kind in ("LDG", "STG"):
            w = collections.Counter()
            for o, v in counts.items():
                if o.split(".")[0] == kind:
                    w[C.mem_width_bits(o)] += v
            if w:
                tot = sum(w.values())
                acc.append(f"{kind}[" + ",".join(f"{b}b:{100 * c / tot:.0f}%" for b, c in sorted(w.items())) + "]")
        mma = []
        for f in ("tcgen05", "wgmma", "legacy_mma"):
            ops = {o: v for o, v in ops_in(f).items() if "MMA" in o.split(".")[0]}
            if ops:
                mma.append(f"{f}[" + ",".join(f"{o}:{v}" for o, v in sorted(ops.items())) + "]")
        buf = []
        if any(o.split(".")[0] in ("LDTM", "STTM", "UTCATOMSWS") for o in counts if counts[o]):
            buf.append("TMEM")
        if fam["shared_memory"] or fam["async_copy_ldgsts"] or fam["tma"]:
            buf.append("shared")
        if fam["local_memory"]:
            buf.append("local(LDL/STL)")
        buf.append("registers")
        lay = [f"{k}:{v}" for k, v in sorted(((o, v) for o, v in counts.items() if o.split(".")[0] in ("LDSM", "STSM", "PRMT", "SHFL", "MOVM") and v))]
        atom = [f"{o}:{v}" for o, v in sorted(ops_in("atomic").items())]
        l0 = max(ls, key=lambda l: (l["metrics"].get("gpu__time_duration.sum", {}).get("value") or 0))
        m = {k: v.get("value") for k, v in l0["metrics"].items()}
        res = (f"launches={len(ls)};grid={m.get('launch__grid_size')};block={m.get('launch__block_size')};"
               f"regs={m.get('launch__registers_per_thread')};smem_per_block_B={m.get('launch__shared_mem_per_block')};"
               f"theoretical_occ_pct={m.get('sm__maximum_warps_per_active_cycle_pct')};"
               f"achieved_occ_pct={m.get('sm__warps_active.avg.pct_of_peak_sustained_active')}")
        rows.append({"profile_id": pid, "stage": stage_of(kname), "access_path": static_marker + "; ".join(acc),
                     "matrix_path": static_marker + ("; ".join(mma) or "none"), "buffer_location": "+".join(buf), "layout_operations": ";".join(lay),
                     "atomic_path": ";".join(atom), "resource_summary": res, "evidence_path": f"{hf_path}: {src}",
                     "evidence_confidence": conf, "notes": f"kernel={kname}" if stage_of(kname) != kname else ""})
    return rows


# ------------------------------------------------------------------ writers
def write_csv(path, rows, cols, gz=False):
    opener = (lambda p: gzip.open(p, "wt", newline="", compresslevel=9)) if gz else (lambda p: open(p, "w", newline=""))
    with opener(path) as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


BENCH_COLS = ["device", "architecture", "dsl", "operator", "category", "dtype", "case_id", "params_json", "mode", "torch_ms",
              "dsl_ms", "validity", "source_csv", "source_git_sha", "notes"]
PROFILE_COLS = ["profile_id", "device", "dsl", "operator", "dtype", "case_id", "params_json", "collection_level", "replay_mode",
                "report_path", "report_sha256", "report_origin", "report_revision", "benchmark_case_match_status",
                "code_match_status", "launch_count", "notes"]
PROFILE_OPT = ["extract_status", "kernel_names", "expected_launch_count", "expected_operator_launch_count", "winner_config_json", "ncu_version", "report_created",
               "profiler_cuda_version", "profiler_command_line", "benchmark_dsl_ms", "benchmark_torch_ms"]
METRIC_COLS = ["profile_id", "launch_id", "kernel_name", "stage", "raw_metric_name", "normalized_group", "value", "unit", "scope",
               "counter_kind", "extraction_method", "status", "raw_source", "notes"]
MIX_COLS = ["profile_id", "launch_id", "instruction_family", "instruction_name", "count", "count_kind", "scope", "evidence_path", "notes"]
PC_COLS = ["profile_id", "launch_id", "pc", "instruction_family", "instruction_text", "sample_count", "executed_instruction_count",
           "stall_reason", "source_line", "evidence_path", "notes"]
PATH_COLS = ["profile_id", "stage", "access_path", "matrix_path", "buffer_location", "layout_operations", "atomic_path",
             "resource_summary", "evidence_path", "evidence_confidence", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=sorted(C.DEVICES))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, a.repo)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    inv = json.load(open(a.inventory))
    bench = []
    head = build_benchmark_cases(a.device, a.repo, bench, problems)
    write_csv(out / "benchmark_cases.csv", bench, BENCH_COLS)
    profiles, extracts = build_profiles(a.device, a.repo, inv, a.extract_dir, bench, problems)
    write_csv(out / "profile_index.csv", profiles, PROFILE_COLS + PROFILE_OPT)
    mrows, xrows, prow, erows = [], [], [], []
    for p in profiles:
        ex = extracts.get(p["profile_id"])
        if not ex or ex.get("status") != "ok":
            continue
        mrows += metric_rows(p["profile_id"], ex, p["report_path"])
        xrows += mix_rows(p["profile_id"], ex, p["report_path"])
        prow += pc_rows(p["profile_id"], ex, p["report_path"])
        erows += path_rows(p["profile_id"], ex, p["report_path"])
    write_csv(out / "kernel_metrics_long.csv.gz", mrows, METRIC_COLS, gz=True)
    write_csv(out / "instruction_mix.csv", xrows, MIX_COLS)
    write_csv(out / "pc_hotspots.csv.gz", prow, PC_COLS, gz=True)
    write_csv(out / "execution_paths.csv", erows, PATH_COLS)
    json.dump({"device": a.device, "repo_head": head, "problems": problems,
               "counts": {"benchmark_cases": len(bench), "profiles": len(profiles), "kernel_metric_rows": len(mrows),
                          "instruction_mix_rows": len(xrows), "pc_hotspot_rows": len(prow), "execution_path_rows": len(erows)}},
              open(out / "_build_log.json", "w"), indent=1)
    print(a.device, {"bench": len(bench), "profiles": len(profiles), "metrics": len(mrows), "mix": len(xrows), "pc": len(prow),
                     "paths": len(erows), "problems": len(problems)})
    for p in problems[:20]:
        print("  PROBLEM", p)


if __name__ == "__main__":
    main()
