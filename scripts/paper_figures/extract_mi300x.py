"""Extract the MI300X paper-figure data package from EXISTING benchmark CSVs and profiling reports.

Reads only. It does not run rocprof-compute, rocprofv3, a benchmark, or any GPU kernel, and it does
not modify any input file. Outputs go to artifacts/paper_figures/amd/MI300X/ (see README.md there).

    python3 scripts/paper_figures/extract_mi300x.py \
        --outputs-root /root/Tilebench/outputs \
        --logs-root /root/Tilebench/results/MI300X/logs

Inputs
  results/MI300X/csv/*.csv (tracked)                       formal latency, the only latency source
  <logs-root>/time_measurement_logs, provenance, metadata  (git-ignored) verification flag + source SHAs
  <outputs-root>/rocprof_compute/MI300X/<op>/triton_<dt>/  (git-ignored) rocprof-compute reports
  <outputs-root>/profiling/MI300X/rocprof_compute_sweep/   (git-ignored) manifest, audits, provenance
  <outputs-root>/profiling/MI300X/analysis_2026-10-05/     (git-ignored) IR dumps, replay audit, PyTorch
                                                           kernel table, ATT decodes, experiments
  scripts/paper_figures/data/mi300x_diagnosis_2026-10-05.json (tracked) frozen diagnosis records
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_identity import canonical_params_json, case_id, normalize_dtype, normalize_operator, parse_params_cell  # noqa: E402
from mi300x_common import (ARCH, CATEGORY, CATEGORY_SOURCE, DEVICE, DIAGNOSIS_JSON, KNOWN_EXCLUSION, N_CU, OUT_DIR,  # noqa: E402
                           REPO, access_width_bits, isa_family, num, parse_amdgcn, read_csv, sha256_file, tree_sha256,
                           write_csv, write_json)

DSL = "triton"


# ======================================================================================= helpers
class Ctx:
    def __init__(self, outputs_root: Path, logs_root: Path, out: Path):
        self.O = outputs_root
        self.L = logs_root
        self.out = out
        self.RPC = outputs_root / "rocprof_compute" / "MI300X"
        self.SWEEP = outputs_root / "profiling" / "MI300X" / "rocprof_compute_sweep"
        self.AN = outputs_root / "profiling" / "MI300X" / "analysis_2026-10-05"
        self.csv_dir = REPO / "results" / "MI300X" / "csv"
        self.issues = collections.defaultdict(list)   # recorded into extraction_log.json for the QA summary

    def rel(self, p) -> str:
        """Logical path recorded in the tables: repo-relative for tracked files, 'outputs/...' or
        'results/MI300X/logs/...' for the git-ignored local trees."""
        p = Path(p).resolve()
        for root, prefix in ((REPO, ""), (self.O.resolve(), "outputs/"), (self.L.resolve(), "results/MI300X/logs/")):
            try:
                r = p.relative_to(root).as_posix()
                return prefix + r
            except ValueError:
                pass
        return p.as_posix()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def jdump(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pair_key(op, dt):
    return f"{op}/{dt}"


def pid(kind, op, dt, suffix=None):
    who = "torch" if kind.startswith("torch") else DSL
    base = f"{DEVICE}.{who}.{op}.{dt}.{kind.replace('torch_', '')}"
    return f"{base}.{suffix}" if suffix else base


# ============================================================================ 1. benchmark cases
def load_time_logs(ctx):
    logs, prov = {}, {}
    for f in sorted(glob.glob(str(ctx.L / "time_measurement_logs" / "*_triton.json"))):
        name = os.path.basename(f)[:-len("_triton.json")]
        op, mode = name.rsplit("_", 1)
        logs[(op, mode)] = (f, json.load(open(f)))
    for f in sorted(glob.glob(str(ctx.L / "provenance" / "*_triton.json"))):
        name = os.path.basename(f)[:-len("_triton.json")]
        op, mode = name.rsplit("_", 1)
        prov[(op, mode)] = (f, json.load(open(f)))
    return logs, prov


def benchmark_cases(ctx, base_commit):
    logs, prov = load_time_logs(ctx)
    header = ["device", "architecture", "dsl", "operator", "category", "dtype", "case_id", "params_json", "mode",
              "torch_ms", "dsl_ms", "validity", "source_csv", "source_git_sha", "notes",
              # extra provenance columns (not used for identity)
              "csv_speedup_triton", "csv_row_number", "source_csv_sha256", "measurement_source_git_sha"]
    rows, cases = [], {}
    files = sorted(ctx.csv_dir.glob("*.csv"))
    for f in files:
        stem = f.stem
        op, mode = stem.rsplit("_", 1)
        op = normalize_operator(op)
        assert mode in ("autotune", "default"), f
        rel = ctx.rel(f)
        csv_sha = sha256_file(f)
        csv_commit = git("log", "-1", "--format=%H", base_commit, "--", rel)
        lf, lrows = logs.get((op, mode), (None, None))
        pf, pj = prov.get((op, mode), (None, None))
        msrc = pj["source"]["git_sha"] if pj else ""
        with open(f, newline="", encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh), start=2):  # row 1 is the header
                params = parse_params_cell(r["params"])
                dt = normalize_dtype(r["dtype"])
                cid = case_id(op, dt, params)
                validity, note = "not_recorded", []
                if lrows is not None:
                    hits = [x for x in lrows if normalize_dtype(x["dtype"]) == dt
                            and all(x.get("params", {}).get(k) == v for k, v in params.items())]
                    if len(hits) == 1:
                        h = hits[0]
                        ok = h.get("triton_ok")
                        validity = "verified_ok" if ok is True else ("verification_failed" if ok is False else "not_recorded")
                        # cross-check: the CSV is the rounded summary of the timing log
                        for col in ("torch_ms", "triton_ms"):
                            lv = h.get(col)
                            if lv is None or f"{lv:.4f}" != r[col]:
                                ctx.issues["csv_vs_timing_log_rounding_mismatch"].append(
                                    {"csv": rel, "row": i, "column": col, "csv_value": r[col], "log_value": lv})
                        if h.get("triton_err"):
                            note.append(f"triton_err={h['triton_err']}")
                    else:
                        ctx.issues["timing_log_match_not_unique"].append({"csv": rel, "row": i, "matches": len(hits)})
                        note.append(f"timing_log_matches={len(hits)}")
                else:
                    ctx.issues["timing_log_missing"].append({"csv": rel})
                note.append("validity: triton_ok in " + (ctx.rel(lf) if lf else "n/a"))
                key = (op, dt, cid, mode)
                if key in cases:
                    ctx.issues["duplicate_case_rows"].append({"csv": rel, "row": i, "case_id": cid})
                cases[key] = r
                rows.append({"device": DEVICE, "architecture": ARCH, "dsl": DSL, "operator": op, "category": CATEGORY[op],
                             "dtype": dt, "case_id": cid, "params_json": canonical_params_json(params), "mode": mode,
                             "torch_ms": r["torch_ms"], "dsl_ms": r["triton_ms"], "validity": validity, "source_csv": rel,
                             "source_git_sha": csv_commit, "notes": "; ".join(note),
                             "csv_speedup_triton": r["speedup_triton"], "csv_row_number": i, "source_csv_sha256": csv_sha,
                             "measurement_source_git_sha": msrc})
    rows.sort(key=lambda x: (x["operator"], x["dtype"], x["mode"], x["csv_row_number"]))
    write_csv(ctx.out / "benchmark_cases.csv", header, rows)
    return rows, {"files": [ctx.rel(f) for f in files], "logs": logs, "prov": prov}


# ======================================================================= 2. inputs for profiles
def load_profile_inputs(ctx):
    sweep = json.load(open(ctx.RPC / "sweep_log.json"))
    manifest = json.load(open(ctx.SWEEP / "report_manifest.json"))
    man = {r["pair"]: r for r in manifest["reports"]}
    post = {(r["op"], r["dtype"]): r for r in json.load(open(ctx.SWEEP / "winner_replay_audit_post_fix.json"))}
    catalogue = {e["op"]: e for e in json.load(open(ctx.O / "profiling" / "MI300X" / "ncu_catalogue.json"))}
    ir = {}
    for j in glob.glob(str(ctx.AN / "triton_ir" / "*" / "*.json")):
        m = json.load(open(j))
        ir[m["hash"]] = Path(j).parent
    torch_k = json.load(open(ctx.AN / "data" / "torch_kernels.json"))
    return sweep, manifest, man, post, catalogue, ir, torch_k


def load_config_defaults(op, commit):
    import yaml  # PyYAML is a TileBench dependency
    txt = git("show", f"{commit}:tilebench/benchmarks/operators/{op}/config.yaml")
    cfg = yaml.safe_load(txt)
    return cfg.get("case_defaults") or {}, cfg


def kernel_uses_descriptor(op, kernel, commit):
    """Static source check of the profiled source commit: does @triton.jit `kernel` take TensorDescriptor
    arguments (`arg.load([..])` / `arg.store([..])`) or call tl.make_tensor_descriptor?"""
    try:
        src = git("show", f"{commit}:tilebench/benchmarks/operators/{op}/impl_triton.py")
    except subprocess.CalledProcessError:
        return None, "impl_triton.py not found at profiling commit"
    m = re.search(rf"^def {re.escape(kernel)}\s*\((.*?)\)\s*(?:->[^:]*)?:", src, re.S | re.M)
    if not m:
        return None, f"def {kernel} not found in impl_triton.py@{commit[:8]}"
    params = [p.split(":")[0].split("=")[0].strip() for p in m.group(1).split(",") if p.strip()]
    rest = src[m.end():]
    nxt = re.search(r"^(?:@|def |class )", rest, re.M)
    body = rest[: nxt.start()] if nxt else rest
    used = sorted({p for p in params if re.search(rf"\b{re.escape(p)}\.(?:load|store)\(\s*\[", body)})
    if "make_tensor_descriptor" in body:
        used.append("tl.make_tensor_descriptor")
    return bool(used), ("descriptor args: " + ",".join(used)) if used else "no descriptor load/store in kernel body"


def profiled_case_match(params_full, dt, op, bench_rows, defaults):
    """Return (status, case_id, params_json, note) for a profiled input against the autotune CSV rows."""
    cand = [r for r in bench_rows if r["operator"] == op and r["dtype"] == dt and r["mode"] == "autotune"]
    hits = [r for r in cand if all(params_full.get(k) == v for k, v in json.loads(r["params_json"]).items())]
    if len(hits) != 1:
        return ("unmatched" if not hits else "ambiguous"), "", "", f"{len(hits)} autotune CSV rows match the profiled params"
    h = hits[0]
    csv_keys = set(json.loads(h["params_json"]))
    extra = {k: v for k, v in params_full.items() if k not in csv_keys}
    diff = {k: (v, defaults.get(k, "<absent>")) for k, v in extra.items() if defaults.get(k, "<absent>") != v}
    last = max(cand, key=lambda r: r["csv_row_number"])
    pos = "largest-row (last CSV row of this dtype)" if h["csv_row_number"] == last["csv_row_number"] else f"CSV row {h['csv_row_number']}"
    if diff:
        return "matched_csv_keys_only", h["case_id"], h["params_json"], f"non-swept params differ from config case_defaults: {diff}; {pos}"
    return "matched", h["case_id"], h["params_json"], f"swept params equal the CSV row; other params equal config case_defaults; {pos}"


# ======================================================================= 3. per-pair raw readers
def kernel_table(pdir):
    return list(csv.DictReader(open(pdir / "analysis" / "workload_csv" / "kernel.csv")))


def pmc_dispatches(pdir):
    """{pass_file: [(kernel, ordinal, dispatch_fields, {counter: value_str})]} from the raw PMC CSVs."""
    out = {}
    for f in sorted((pdir / "workload").glob("results_pmc_perf_*.csv")):
        disp = collections.OrderedDict()
        for r in csv.DictReader(open(f)):
            d = disp.setdefault(int(r["Dispatch_ID"]), {"fields": r, "counters": {}})
            d["counters"][r["Counter_Name"]] = r["Counter_Value"]
        ordc = collections.Counter()
        lst = []
        for did in sorted(disp):
            k = disp[did]["fields"]["Kernel_Name"]
            lst.append((k, ordc[k], disp[did]["fields"], disp[did]["counters"]))
            ordc[k] += 1
        out[f.name] = lst
    return out


# Normalised groups: organisational labels only, NOT claims of NVIDIA/AMD equivalence.
TABLE_GROUP = {
    "System Speed-of-Light": "execution", "Roofline": "execution", "Memory Chart": "global_memory",
    "Command Processor (CPC/CPF)": "execution", "Workgroup Manager (SPI)": "occupancy", "Wavefront": "execution",
    "Compute Units - Instruction Mix": "instruction_mix", "Compute Units - Compute Pipeline": "execution",
    "Local Data Share (LDS)": "local_memory", "Instruction Cache": "cache", "Scalar L1 Data Cache": "cache",
    "Address Processing Unit and Data Return Path (TA/TD)": "global_memory", "Vector L1 Data Cache": "cache",
    "L2 Cache": "cache", "L2 Cache (per Channel)": "cache",
}
ID_GROUP = [  # metric-id prefix overrides, most specific first
    ("2.1.15", "occupancy"), ("2.1.17", "local_memory"), ("2.1.10", "matrix_pipeline"), ("2.1.1", "execution"),
    ("4.1.4", "matrix_pipeline"), ("4.1.5", "matrix_pipeline"), ("4.1.6", "matrix_pipeline"), ("4.1.7", "matrix_pipeline"), ("4.1.8", "matrix_pipeline"),
    ("6.2.", "occupancy"), ("6.1.", "occupancy"),
    ("7.1.5", "resources"), ("7.1.6", "resources"), ("7.1.7", "resources"), ("7.1.8", "resources"), ("7.1.9", "resources"),
    ("7.1.", "execution"), ("7.2.4", "stalls"), ("7.2.5", "stalls"), ("7.2.", "execution"),
    ("10.3.3", "atomics"), ("10.4.", "matrix_pipeline"), ("11.2.7", "matrix_pipeline"), ("11.2.8", "matrix_pipeline"),
    ("17.3.4", "atomics"), ("16.3.3", "atomics"),
]
SPILL_CAVEAT = ("CAVEAT: rocprof-compute 3.7.0 'Spill/Stack' metrics (10.3.4-10.3.7, 15.2.5-15.3.2) are derived from TA_BUFFER_* "
                "counters and count ordinary buffer_* instructions, which Triton emits for global memory on gfx942; they are NOT "
                "register-spill evidence (vector_add, ScratchSize 0, reports non-zero). Use AMDGCN ScratchSize / spill counts.")
CAVEAT = {
    "7.2.4": "CAVEAT: the 7.2.4 description string in this rocprof-compute version duplicates the wave-cycles description; "
             "do not treat it as definitive dependency-wait evidence.",
    "6.2.": "CAVEAT: the 6.2.x workgroup-manager 'Insufficient ...' / limit counters read zero for every MI300X profile in "
            "this suite (checked by the extraction); treat as not informative on this VF, not as proof of no stall.",
}


def metric_group(mid, table):
    for pre, g in ID_GROUP:
        if mid == pre or (pre.endswith(".") and mid.startswith(pre)):
            return g
    if table == "Vector L1 Data Cache" and mid.startswith("16.3"):
        return "cache"
    return TABLE_GROUP.get(table, "other")


def raw_counter_group(name):
    n = name
    rules = [(r"^SQ_INSTS_VALU_MFMA|^SQ_VALU_MFMA", "matrix_pipeline"), (r"^SQ_INSTS_", "instruction_mix"),
             (r"^SQ_WAIT|^SQ_INST_LEVEL|^SQ_IFETCH_LEVEL|_STALL|^SQ_VMEM_.*FULL", "stalls"),
             (r"^SQ_LDS|^SQ_ACTIVE_INST_LDS", "local_memory"), (r"ATOMIC", "atomics"),
             (r"^SQ_WAVES|^SQ_LEVEL_WAVES|^SQ_BUSY|^SQ_WAVE_CYCLES|^SQ_ACTIVE|^SQ_THREAD", "execution"),
             (r"^SPI_", "occupancy"), (r"^TA_|^TD_", "global_memory"), (r"^TCP_|^TCC_|^SQC_", "cache"),
             (r"^GRBM_|^CPC_|^CPF_", "execution")]
    for rx, g in rules:
        if re.search(rx, n):
            return g
    return "other"


def instr_family_from_counter(name):
    m = {"SQ_INSTS_VALU": "VALU (all, incl. MFMA issue)", "SQ_INSTS_MFMA": "MFMA", "SQ_INSTS_SALU": "SALU",
         "SQ_INSTS_SMEM": "SMEM", "SQ_INSTS_LDS": "LDS", "SQ_INSTS_VMEM": "VMEM", "SQ_INSTS_BRANCH": "BRANCH",
         "SQ_INSTS_GDS": "GDS", "SQ_INSTS_SENDMSG": "SENDMSG", "SQ_INSTS": "ALL", "SQ_INSTS_VSKIPPED": "VSKIPPED",
         "SQ_INSTS_VALU_CVT": "VALU_CONVERT", "SQ_INSTS_VALU_INT32": "VALU_INT32", "SQ_INSTS_VALU_INT64": "VALU_INT64",
         "SQ_WAIT_INST_ANY": "WAIT (cycles waiting for any instruction issue; quad-cycle units)",
         "SQ_INSTS_FLAT": "FLAT"}
    if name in m:
        return m[name]
    if name.startswith("SQ_INSTS_VALU_MFMA_MOPS"):
        return "MFMA_MOPS_" + name.rsplit("_", 1)[-1]
    if name.startswith("SQ_INSTS_VALU_MFMA"):
        return "MFMA_" + name.rsplit("_", 1)[-1]
    if name.startswith("SQ_INSTS_VALU_"):
        return "VALU_" + name[len("SQ_INSTS_VALU_"):]
    return None


def norm_ins(t):
    t = re.sub(r"\s+", " ", t.strip())
    if re.match(r"^s_(cbranch\w*|branch)\b", t):
        return t.split()[0]
    return t


def combined_tree_sha(dirs):
    import hashlib
    rows = sorted(f"{Path(d).name}\t{tree_sha256(d)}" for d in dirs)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def pcs_class(ins):
    op = ins.split()[0] if ins else ""
    return isa_family(op) if op else "NO_INSTRUCTION_TEXT"


# =============================================================================== 4. main build
def build(ctx: Ctx):
    base_commit = git("rev-parse", "HEAD")
    bench_rows, bench_meta = benchmark_cases(ctx, base_commit)
    sweep, manifest, man, post, catalogue, ir, torch_k = load_profile_inputs(ctx)
    replaced = {r["pair"] for r in manifest["reports"] if "supersedes" in r}

    prof_hdr = ["profile_id", "device", "dsl", "operator", "dtype", "case_id", "params_json", "collection_level",
                "replay_mode", "report_path", "report_sha256", "report_origin", "report_revision",
                "benchmark_case_match_status", "code_match_status", "launch_count", "notes",
                "profiling_source_sha", "profiled_params_full_json", "kernel_config_json"]
    km_hdr = ["profile_id", "launch_id", "kernel_name", "stage", "raw_metric_name", "normalized_group", "value", "unit",
              "scope", "counter_kind", "extraction_method", "status", "raw_source", "notes"]
    im_hdr = ["profile_id", "launch_id", "instruction_family", "instruction_name", "count", "count_kind", "scope",
              "evidence_path", "notes"]
    pc_hdr = ["profile_id", "launch_id", "pc", "instruction_family", "instruction_text", "sample_count",
              "executed_instruction_count", "stall_reason", "source_line", "evidence_path", "notes"]
    ep_hdr = ["profile_id", "stage", "access_path", "matrix_path", "buffer_location", "layout_operations", "atomic_path",
              "resource_summary", "evidence_path", "evidence_confidence", "notes"]
    P, KM, IM, PC, EP = [], [], [], [], []
    stats = collections.Counter()
    pair_summary = {}
    metric_zero_watch = collections.defaultdict(list)  # 6.2.x check across the suite
    defaults_cache = {}

    for e in sorted(sweep, key=lambda x: (x["op"], x["dtype"])):
        op, dt = normalize_operator(e["op"]), normalize_dtype(e["dtype"])
        pair = f"{e['op']}/{e['backend']}_{e['dtype']}"
        pdir = ctx.RPC / e["op"] / f"triton_{e['dtype']}"
        cap = json.load(open(pdir / "capture.json"))
        mrec = man.get(pair)
        if mrec is None:
            ctx.issues["manifest_missing_pair"].append(pair)
        src_sha = mrec["profiling_source"] if mrec else ""
        if op not in defaults_cache:
            defaults_cache[op] = load_config_defaults(op, src_sha or base_commit)[0]
        defaults = defaults_cache[op]
        params_full = cap["params"]
        if cap["params"] != e["params"]:
            ctx.issues["capture_vs_sweep_params_differ"].append(pair)
        cat_params = catalogue[e["op"]]["autotune_winner_per_dtype"][e["dtype"]]["params"]
        if cat_params != params_full:
            ctx.issues["catalogue_vs_capture_params_differ"].append(pair)
        mstatus, cid, pjson, mnote = profiled_case_match(params_full, dt, op, bench_rows, defaults)
        stats[f"case_match:{mstatus}"] += 1

        # ---- code identity evidence
        th = tree_sha256(pdir)
        tree_ok = bool(mrec) and th == mrec["tree_sha256"]
        if not tree_ok:
            ctx.issues["pair_tree_sha256_mismatch_vs_manifest"].append(pair)
        cap_ok = sha256_file(pdir / "capture.json") == (mrec or {}).get("capture_sha256")
        fo = json.load(open(ctx.AN / "replay_audit" / f"{e['op']}__{e['dtype']}__formal.json"))
        rp = json.load(open(ctx.AN / "replay_audit" / f"{e['op']}__{e['dtype']}__replay.json"))
        same_hsaco = (len(fo["launches"]) == len(rp["launches"])) and all(
            a["hsaco_sha"] == b["hsaco_sha"] and a["grid"] == b["grid"] and a["kernel"] == b["kernel"]
            and a["constexprs"] == b["constexprs"] for a, b in zip(fo["launches"], rp["launches"]))
        pf = post.get((e["op"], e["dtype"]), {})
        names_ok = cap["captured_names"] == cap["expected_names"] and cap["captured_count"] == cap["expected_count"]
        kt = kernel_table(pdir)
        n_disp = sum(int(k["dispatch_count"]) for k in kt)
        if n_disp != cap["captured_count"] or n_disp != len(fo["launches"]):
            ctx.issues["dispatch_count_mismatch"].append({"pair": pair, "kernel_csv": n_disp, "capture": cap["captured_count"],
                                                          "replay_audit": len(fo["launches"])})

        # ---- static ISA per unique code object (formal launch order)
        stages = collections.OrderedDict()
        for li, l in enumerate(fo["launches"]):
            st = stages.setdefault(l["hash"], {"kernel": l["kernel"], "launches": [], "first": l})
            st["launches"].append(li)
        name_n = collections.Counter(s["kernel"] for s in stages.values())
        name_i = collections.Counter()
        isa_text = {}
        for h, st in stages.items():
            k = st["kernel"]
            st["stage"] = k if name_n[k] == 1 else f"{k}#{name_i[k]}"
            name_i[k] += 1
            d = ir.get(h)
            st["ir_dir"] = d
            if d is None:
                ctx.issues["ir_missing_for_hash"].append({"pair": pair, "kernel": k, "hash": h})
                continue
            gcn = d / f"{st['first']['compiled_name']}.amdgcn"
            md, ins = parse_amdgcn(gcn.read_text())
            st["md"], st["ins"], st["gcn"] = md, ins, gcn
            isa_text.setdefault(k, set()).update(norm_ins(i["text"]) for i in ins)

        # ---- PC sampling rows and ISA text agreement (direct check that the sampled code is the recompiled code)
        pcs_rows = [r for r in csv.DictReader(open(pdir / "analysis" / "pc_sampling_instructions.csv")) if r["operator_kernel"] == "True"]
        n_txt = n_hit = 0
        for r in pcs_rows:
            t = norm_ins(r["instruction"].split(";")[0])
            if not t:
                continue
            n_txt += 1
            if t in isa_text.get(r["kernel_name"], set()):
                n_hit += 1
        pcs_match = (n_hit / n_txt) if n_txt else None
        code_status = ("verified" if (tree_ok and same_hsaco and names_ok and pf.get("PASS")) else "partially_verified")
        code_note = (f"{code_status}: manifest tree_sha256 reproduced={tree_ok}; capture names==expected={names_ok}; "
                     f"replay-audit hsaco formal==replay for {len(fo['launches'])}/{len(fo['launches'])} launches={same_hsaco}; "
                     f"post-fix audit PASS={pf.get('PASS')}; PC-sampled instruction text found in recompiled AMDGCN="
                     f"{n_hit}/{n_txt} (branch targets compared by opcode only: the disassembly prints offsets, the dump prints labels)")
        stats["pcs_text_checked"] += n_txt
        stats["pcs_text_found"] += n_hit

        prov_note = (f"profiled {mrec['profiled_utc'][0]}..{mrec['profiled_utc'][1]}; " if mrec else "") + \
                    ("replaced 2026-10-04 (supersedes 5b82f8af report, see PROVENANCE.md); " if pair in replaced else "")
        common = {"device": DEVICE, "dsl": DSL, "operator": op, "dtype": dt, "case_id": cid, "params_json": pjson,
                  "benchmark_case_match_status": mstatus, "profiling_source_sha": src_sha,
                  "profiled_params_full_json": jdump(params_full), "kernel_config_json": jdump(e["winner"])}
        hf_origin = f"local {ctx.rel(pdir)}; HF dataset {manifest['hf_dataset']} path {mrec['hf_path'] if mrec else ''}"
        hf_rev = ",".join(mrec["hf_uploaded_in"]) if mrec else ""

        p_c, p_p, p_s, p_t = pid("rocprof_compute", op, dt), pid("pc_sampling", op, dt), pid("static_isa", op, dt), pid("torch_kernel_trace", op, dt)
        P.append({**common, "profile_id": p_c,
                  "collection_level": f"rocprof-compute profile, full default counter set ({cap['counter_passes']} counter passes + roofline), kernel filter {cap['kernel_filter']}",
                  "replay_mode": "application replay: one process per counter pass, each does 3 unprofiled prime calls then input generation, a 512 MiB eviction and exactly one impl.run()",
                  "report_path": ctx.rel(pdir / "workload"), "report_sha256": tree_sha256(pdir / "workload"),
                  "report_origin": hf_origin, "report_revision": hf_rev, "code_match_status": code_note,
                  "launch_count": n_disp,
                  "notes": prov_note + f"report_sha256 = tree hash of workload/; whole pair dir tree hash {th} "
                           f"({'matches' if tree_ok else 'DOES NOT match'} report_manifest.json); metrics parsed from existing "
                           f"analysis/workload_csv (rocprof-compute analyze output, not re-run). {mnote}"})
        P.append({**common, "profile_id": p_p,
                  "collection_level": "rocprof-compute --experimental profile --pc-sampling (stochastic, interval 65536)",
                  "replay_mode": "single application run (same harness: 3 prime calls + 1 profiled call)",
                  "report_path": ctx.rel(pdir / "pc_sampling"), "report_sha256": tree_sha256(pdir / "pc_sampling"),
                  "report_origin": hf_origin, "report_revision": hf_rev, "code_match_status": code_note,
                  "launch_count": cap["captured_count"],
                  "notes": prov_note + f"samples on operator kernels={cap.get('pc_sampling_summary', {}).get('samples_operator')} of "
                           f"{cap.get('pc_sampling_summary', {}).get('samples_total')} total; instruction rows from "
                           f"analysis/pc_sampling_instructions.csv; launch_count is the harness launch count (PC sampling does not "
                           f"report dispatches). {mnote}"})
        P.append({**common, "profile_id": p_s,
                  "collection_level": "static compile artifacts (TTIR/TTGIR/LLIR/AMDGCN) of the formal-path winner kernels",
                  "replay_mode": "offline recompilation during the 2026-10-05 replay audit (no execution counters)",
                  "report_path": ";".join(ctx.rel(st["ir_dir"]) for st in stages.values() if st.get("ir_dir")),
                  "report_sha256": combined_tree_sha([st["ir_dir"] for st in stages.values() if st.get("ir_dir")]),
                  "report_origin": f"local {ctx.rel(ctx.AN / 'triton_ir')} (one directory per code object)",
                  "report_revision": "analysis_2026-10-05", "code_match_status": code_note, "launch_count": len(fo["launches"]),
                  "notes": "report_sha256 = sha256 over sorted '<dir>\\t<tree_sha256(dir)>' lines of the listed stage directories; "
                           "recompiled binaries are hsaco-identical between the formal autotune path and the profiling replay "
                           f"(replay_audit/{e['op']}__{e['dtype']}__formal.json). {mnote}"})
        tk = torch_k.get(pair_key(e["op"], e["dtype"]))
        P.append({**common, "dsl": "pytorch", "profile_id": p_t,
                  "collection_level": "rocprofv3 --kernel-trace --marker-trace of ONE PyTorch reference call inside a ROCTx range (after 3 prime calls)",
                  "replay_mode": "single application run",
                  "report_path": ctx.rel(ctx.AN / "data" / "torch_kernels.json") + f"#{pair_key(e['op'], e['dtype'])}",
                  "report_sha256": sha256_file(ctx.AN / "data" / "torch_kernels.json"),
                  "report_origin": "local analysis_2026-10-05/data/torch_kernels.json (derived by scripts/torch_parse.py)",
                  "report_revision": "analysis_2026-10-05",
                  "code_match_status": "not_applicable (vendor/ATen binaries; library versions in environment.json)",
                  "launch_count": tk["n_kernels"] if tk else "",
                  "notes": "RAW rocprofv3 trace CSVs were written to a session scratch directory and are NOT retained; only the "
                           "derived per-kernel table is available (kernel names truncated to 300 chars). Input = ncu_catalogue "
                           f"winner params (identical to the rocprof-compute capture params: {cat_params == params_full}). {mnote}"})
        if tk is None:
            ctx.issues["torch_trace_missing"].append(pair)
        stats["reports"] += 4
        stats["launches_rocprof_compute"] += n_disp

        # ---- kernel_metrics_long: rocprof-compute derived metrics
        kd = {k["kernel_name"]: int(k["dispatch_count"]) for k in kt}
        kmf = pdir / "analysis" / "workload_csv" / "kernel_metric.csv"
        rows = list(csv.DictReader(open(kmf)))
        avg = {(r["kernel_name"], r["metric_id"], r["metric_uuid"]): r["value"] for r in rows if r["value_name"] == "Avg"}
        src = ctx.rel(kmf)
        for r in rows:
            mid, k = r["metric_id"], r["kernel_name"]
            if r["table_name"] == "L2 Cache (per Channel)":
                stats["skipped_l2_per_channel_rows"] += 1
                continue
            vn = r["value_name"]
            if vn in ("Min", "Max", "Q1", "Median", "Q3") and kd.get(k) == 1:
                if r["value"] == avg.get((k, mid, r["metric_uuid"])):
                    stats["skipped_single_dispatch_order_stats"] += 1
                    continue
            note = [SPILL_CAVEAT] if "Spill/Stack" in r["metric_name"] else []
            for pre, c in CAVEAT.items():
                if mid == pre or (pre.endswith(".") and mid.startswith(pre)):
                    note.append(c)
            if vn.startswith("Peak"):
                note.append("peak/ceiling reference value reported by rocprof-compute, not a measurement of the kernel")
            if mid.startswith("6.2.") and vn == "Avg":
                metric_zero_watch[mid].append(r["value"])
            status = "collected" if r["value"] != "" else "unavailable_empty_in_report"
            stats[f"km_status:{status}"] += 1
            KM.append({"profile_id": p_c, "launch_id": f"agg:{k}", "kernel_name": k, "stage": k,
                       "raw_metric_name": f"{mid} | {r['metric_name']} | {vn}", "normalized_group": "atomics" if "tomic" in r["metric_name"] else metric_group(mid, r["table_name"]),
                       "value": r["value"], "unit": r["unit"],
                       "scope": f"kernel aggregate over {kd.get(k, '?')} dispatch(es); rocprof-compute normalization per_kernel; table '{r['table_name']}'",
                       "counter_kind": "rocprof_compute_derived_metric", "extraction_method": "parsed existing rocprof-compute 3.7.0 analyze CSV",
                       "status": status, "raw_source": src, "notes": " ".join(note)})
        for k in kt:
            KM.append({"profile_id": p_c, "launch_id": f"agg:{k['kernel_name']}", "kernel_name": k["kernel_name"], "stage": k["kernel_name"],
                       "raw_metric_name": "kernel.csv | duration_ns_sum", "normalized_group": "execution", "value": k["duration_ns_sum"],
                       "unit": "ns", "scope": f"sum over {k['dispatch_count']} dispatch(es) in the counter-collection run",
                       "counter_kind": "profiler_duration_diagnostic_only", "extraction_method": "parsed existing analyze CSV",
                       "status": "collected", "raw_source": ctx.rel(pdir / "analysis" / "workload_csv" / "kernel.csv"),
                       "notes": "Profiler kernel time under counter collection. Diagnostic / intra-profile fractions only; NEVER formal latency."})

        # ---- raw PMC counters per dispatch (non per-channel), dispatch fields, dynamic instruction counts
        pm = pmc_dispatches(pdir)
        disp_fields_seen = {}
        dyn = {}
        for fname, lst in pm.items():
            rsrc = ctx.rel(pdir / "workload" / fname)
            for (k, ordn, fields, counters) in lst:
                lid = f"{k}#{ordn}"
                fsig = tuple(fields[c] for c in ("Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup", "Scratch_Per_Workitem", "Arch_VGPR", "Accum_VGPR", "SGPR"))
                if lid in disp_fields_seen and disp_fields_seen[lid][0] != fsig:
                    ctx.issues["dispatch_fields_differ_across_passes"].append({"pair": pair, "launch": lid, "pass": fname})
                if lid not in disp_fields_seen:
                    disp_fields_seen[lid] = (fsig, rsrc, fields)
                for cname, cval in sorted(counters.items()):
                    if "[" in cname:
                        stats["skipped_per_instance_raw_rows"] += 1
                        continue
                    status = "collected" if cval != "" else "unavailable_empty_in_report"
                    KM.append({"profile_id": p_c, "launch_id": lid, "kernel_name": k, "stage": k, "raw_metric_name": cname,
                               "normalized_group": raw_counter_group(cname), "value": cval, "unit": "raw counter value",
                               "scope": "one dispatch; counters without [n] are device-wide totals (suffix _sum = summed over instances by rocprofiler)",
                               "counter_kind": "raw_pmc_counter", "extraction_method": "parsed existing raw rocprofiler PMC CSV",
                               "status": status, "raw_source": rsrc,
                               "notes": "TA_BUFFER_* count buffer_* instructions (Triton global memory on gfx942), not spills" if cname.startswith("TA_BUFFER") else ""})
                    fam = instr_family_from_counter(cname)
                    if fam and cval != "":
                        dyn.setdefault((lid, cname), []).append((fname, cval, fam))
        # dynamic counts summed per kernel name over its dispatches (first pass that collected the counter);
        # the per-dispatch values are in kernel_metrics_long.csv.gz (counter_kind raw_pmc_counter)
        agg = {}
        for (lid, cname), vals in dyn.items():
            fname, cval, fam = vals[0]
            k = lid.rsplit("#", 1)[0]
            a_ = agg.setdefault((k, cname), {"sum": 0.0, "n": 0, "fam": fam, "files": set(), "multi": 0})
            a_["sum"] += float(cval); a_["n"] += 1; a_["files"].add(fname); a_["multi"] += len(vals) > 1
        for (k, cname), a_ in sorted(agg.items()):
            note = "cycle count, not instructions; " if cname == "SQ_WAIT_INST_ANY" else ""
            if a_["multi"]:
                note += "counter also collected in other pass(es); value from the first pass; all values in kernel_metrics_long"
            IM.append({"profile_id": p_c, "launch_id": f"agg:{k}", "instruction_family": a_["fam"], "instruction_name": cname,
                       "count": num(a_["sum"]), "count_kind": "dynamic_hw_counter_total_summed_over_dispatches",
                       "scope": f"sum over {a_['n']} dispatch(es) of the kernel, device-wide, per-wave issue count",
                       "evidence_path": ";".join(ctx.rel(pdir / "workload" / f) for f in sorted(a_["files"])), "notes": note})
        for lid, (fsig, rsrc, fields) in sorted(disp_fields_seen.items()):
            k = fields["Kernel_Name"]
            units = {"Grid_Size": "work-items (threads), not workgroups", "Workgroup_Size": "work-items per workgroup",
                     "LDS_Per_Workgroup": "bytes", "Scratch_Per_Workitem": "bytes", "Arch_VGPR": "registers (allocation granule)",
                     "Accum_VGPR": "registers (allocation granule)", "SGPR": "registers (allocation granule)"}
            for c, u in units.items():
                KM.append({"profile_id": p_c, "launch_id": lid, "kernel_name": k, "stage": k, "raw_metric_name": f"dispatch.{c}",
                           "normalized_group": "resources" if c not in ("Grid_Size", "Workgroup_Size") else "execution",
                           "value": fields[c], "unit": u, "scope": "one dispatch (identical across counter passes; checked)",
                           "counter_kind": "dispatch_record", "extraction_method": "parsed existing raw rocprofiler PMC CSV",
                           "status": "collected", "raw_source": rsrc, "notes": ""})
            g, w = int(fields["Grid_Size"]), int(fields["Workgroup_Size"])
            KM.append({"profile_id": p_c, "launch_id": lid, "kernel_name": k, "stage": k, "raw_metric_name": "derived.workgroups",
                       "normalized_group": "execution", "value": num(g // w if w else None), "unit": "workgroups",
                       "scope": "one dispatch", "counter_kind": "derived_from_dispatch_record",
                       "extraction_method": "Grid_Size / Workgroup_Size", "status": "derived", "raw_source": rsrc,
                       "notes": f"compare with {N_CU} CUs for grid under-subscription"})
        stats["launches_raw_pmc"] += len(disp_fields_seen)

        # ---- PC sampling: per-kernel summary + hotspots
        bykern = collections.defaultdict(lambda: {"samples": 0, "issued": 0, "stalled": 0, "reasons": collections.Counter()})
        pcsrc = ctx.rel(pdir / "analysis" / "pc_sampling_instructions.csv")
        for r in pcs_rows:
            b = bykern[r["kernel_name"]]
            b["samples"] += int(r["samples"]); b["issued"] += int(r["issued"]); b["stalled"] += int(r["stalled"])
            reasons = json.loads(r["stall_reasons"] or "{}")
            for kk, vv in reasons.items():
                b["reasons"][kk] += vv
            PC.append({"profile_id": p_p, "launch_id": f"agg:{r['kernel_name']}", "pc": f"code_object {r['code_object_id']} offset {r['offset']}",
                       "instruction_family": pcs_class(r["instruction"]), "instruction_text": r["instruction"],
                       "sample_count": r["samples"], "executed_instruction_count": "",
                       "stall_reason": jdump(reasons) if reasons else "",
                       "source_line": r["source_line"], "evidence_path": pcsrc,
                       "notes": f"kernel={r['kernel_name']}; issued={r['issued']}; stalled={r['stalled']}; instruction_types={r['instruction_types']}; "
                                "stochastic samples aggregated over all launches; executed counts not available from PC sampling"})
        for k, b in sorted(bykern.items()):
            low = b["samples"] < 100
            for name, val in [("pcs.samples", b["samples"]), ("pcs.issued", b["issued"]), ("pcs.stalled", b["stalled"])] + \
                             [(f"pcs.stall_reason.{rk}", rv) for rk, rv in sorted(b["reasons"].items())]:
                KM.append({"profile_id": p_p, "launch_id": f"agg:{k}", "kernel_name": k, "stage": k, "raw_metric_name": name,
                           "normalized_group": "stalls", "value": val, "unit": "samples",
                           "scope": "all launches of the kernel in one PC-sampling run", "counter_kind": "pc_sampling_count",
                           "extraction_method": "summed from existing pc_sampling_instructions.csv", "status": "collected",
                           "raw_source": pcsrc, "notes": "LOW SAMPLE COUNT (<100): per-instruction shares unreliable" if low else ""})
        for k in sorted(set(kd) - set(bykern)):
            KM.append({"profile_id": p_p, "launch_id": f"agg:{k}", "kernel_name": k, "stage": k, "raw_metric_name": "pcs.samples",
                       "normalized_group": "stalls", "value": "", "unit": "samples", "scope": "kernel", "counter_kind": "pc_sampling_count",
                       "extraction_method": "summed from existing pc_sampling_instructions.csv", "status": "unavailable_no_samples",
                       "raw_source": pcsrc, "notes": "no PC samples attributed to this kernel (short kernel); not zero time"})
        stats["pcs_rows"] += len(pcs_rows)

        # ---- static ISA: resources, instruction mix, execution paths
        mets = {(r["kernel_name"], r["metric_id"], r["value_name"]): r["value"] for r in rows}
        for h, st in stages.items():
            k, stage = st["kernel"], st["stage"]
            l0 = st["first"]
            lid = f"static:{stage}:{h[:12]}"
            ev = ctx.rel(st["gcn"]) if st.get("gcn") else ""
            gsha = sha256_file(st["gcn"]) if st.get("gcn") else ""
            if "md" in st:
                for key, val in st["md"].items():
                    unit = {"ScratchSize": "bytes per work-item", "LDSByteSize": "bytes (static)", "codeLenInByte": "bytes",
                            "Occupancy": "waves per SIMD (compiler estimate)"}.get(key, "registers" if "gpr" in key.lower() else "count")
                    KM.append({"profile_id": p_s, "launch_id": lid, "kernel_name": k, "stage": stage, "raw_metric_name": f"amdgcn {key}",
                               "normalized_group": "occupancy" if key == "Occupancy" else "resources",
                               "value": num(val), "unit": unit, "scope": "kernel code object (static)",
                               "counter_kind": "static_compiler_metadata", "extraction_method": "regex on Triton .amdgcn dump",
                               "status": "collected" if val is not None else "unavailable_not_in_dump",
                               "raw_source": f"{ev} (sha256 {gsha})", "notes": ""})
            for key, unit in [("grid", "workgroups per dimension"), ("workgroup_size", "work-items"), ("num_warps", "wavefronts per workgroup (wave64)"),
                              ("num_stages", "software pipeline stages"), ("shared", "bytes LDS (runtime)"), ("n_regs", "registers (Triton-reported)"),
                              ("n_spills", "spilled registers (Triton-reported)")]:
                KM.append({"profile_id": p_s, "launch_id": lid, "kernel_name": k, "stage": stage, "raw_metric_name": f"launch.{key}",
                           "normalized_group": "execution" if key in ("grid", "workgroup_size", "num_warps", "num_stages") else "resources",
                           "value": jdump(l0[key]) if isinstance(l0.get(key), (list, dict)) else num(l0.get(key)), "unit": unit,
                           "scope": f"first of {len(st['launches'])} launch(es) with this code object",
                           "counter_kind": "launch_metadata", "extraction_method": "replay-audit JSON (formal path)",
                           "status": "collected" if key in l0 else "unavailable",
                           "raw_source": ctx.rel(ctx.AN / "replay_audit" / f"{e['op']}__{e['dtype']}__formal.json"), "notes": ""})
            if "ins" not in st:
                EP.append({"profile_id": p_c, "stage": stage, "access_path": "unknown", "matrix_path": "unknown", "buffer_location": "unknown",
                           "layout_operations": "unknown", "atomic_path": "unknown", "resource_summary": "unknown", "evidence_path": "",
                           "evidence_confidence": "none", "notes": "IR dump missing for this code object"})
                continue
            ins = st["ins"]
            opc = collections.Counter()
            for i in ins:
                name = i["opcode"] + (f" [{' '.join(i['mods'])}]" if i["mods"] and re.match(r"^(buffer|global|flat|scratch)_", i["opcode"]) else "")
                opc[(isa_family(i["opcode"]), name)] += 1
            for (fam, name), c in sorted(opc.items()):
                IM.append({"profile_id": p_s, "launch_id": lid, "instruction_family": fam, "instruction_name": name, "count": c,
                           "count_kind": "static_isa_opcode_count", "scope": "code object, static text (not executions)",
                           "evidence_path": ev, "notes": ""})
            ndpp = sum(1 for i in ins if i["dpp"])
            if ndpp:
                IM.append({"profile_id": p_s, "launch_id": lid, "instruction_family": "VALU_DPP", "instruction_name": "any DPP-modified VALU",
                           "count": ndpp, "count_kind": "static_isa_opcode_count", "scope": "kernel code object (static)",
                           "evidence_path": ev, "notes": "subset of VALU rows above (cross-lane data movement)"})
            # execution-path characterisation
            uses_desc, desc_note = kernel_uses_descriptor(e["op"], k, src_sha or base_commit)
            ttir = st["ir_dir"] / f"{l0['compiled_name']}.ttir"
            ttgir = st["ir_dir"] / f"{l0['compiled_name']}.ttgir"
            ttir_t = ttir.read_text() if ttir.exists() else ""
            ttg_t = ttgir.read_text() if ttgir.exists() else ""
            n_desc_ir = len(re.findall(r"tt\.descriptor_(?:load|store)|tt\.make_tensor_descriptor", ttir_t))
            loads = collections.Counter(); stores = collections.Counter(); atom = collections.Counter()
            for i in ins:
                o = i["opcode"]
                if re.match(r"^(buffer|global|flat)_load", o):
                    loads[(o, i["mods"])] += 1
                elif re.match(r"^(buffer|global|flat)_store", o):
                    stores[(o, i["mods"])] += 1
                elif re.match(r"^(buffer|global|flat)_atomic", o):
                    atom[(o, i["mods"])] += 1

            def fmt(c):
                return ", ".join(f"{o}{('[' + ' '.join(m) + ']') if m else ''} x{n}" for (o, m), n in sorted(c.items(), key=lambda x: (-x[1], x[0]))) or "none"

            def widths(c):
                w = collections.Counter()
                for (o, _), n in c.items():
                    b = access_width_bits(o)
                    w["vector(>=64b)" if b and b >= 64 else ("scalar(<=32b)" if b else "unknown")] += n
                return dict(w)
            nt_l = sum(n for (o, m), n in loads.items() if "nt" in m)
            if uses_desc:
                ap = (f"TensorDescriptor in source ({desc_note}); dumped TTIR has {n_desc_ir} descriptor ops and "
                      f"{ttir_t.count('tt.load')} tt.load -> lowered to pointer/buffer loads on gfx942 (no TMA)")
            elif uses_desc is False:
                ap = "pointer loads (tl.load) in source"
            else:
                ap = f"unknown source form ({desc_note})"
            ap += f"; static loads: {fmt(loads)} {widths(loads)}; stores: {fmt(stores)} {widths(stores)}; loads with nt modifier: {nt_l}/{sum(loads.values())}"
            mf = collections.Counter(i["opcode"] for i in ins if i["opcode"].startswith("v_mfma"))
            mp = ("MFMA: " + ", ".join(f"{o} x{n}" for o, n in sorted(mf.items()))) if mf else "not_observed (no v_mfma in static ISA)"
            mu = mets.get((k, "2.1.10", "Avg"))
            if mf and mu not in (None, ""):
                mp += f"; measured MFMA util (2.1.10 Avg, kernel-name aggregate) {mu} %"
            dsr = sum(1 for i in ins if i["opcode"].startswith("ds_read")); dsw = sum(1 for i in ins if i["opcode"].startswith("ds_write"))
            scratch = st["md"].get("ScratchSize")
            bl = (f"LDS-staged (runtime shared {l0['shared']} B; static ds_read x{dsr}, ds_write x{dsw})" if l0["shared"] and (dsr or dsw)
                  else (f"LDS allocated {l0['shared']} B (ds_read x{dsr}, ds_write x{dsw})" if l0["shared"] else "no LDS (registers + global memory)"))
            bl += f"; scratch {scratch} B/work-item" + (" (register spill to scratch)" if scratch else "")
            cl = len(re.findall(r"ttg\.convert_layout", ttg_t))
            la = len(re.findall(r"ttg\.local_alloc", ttg_t))
            perm = sum(1 for i in ins if re.match(r"^ds_(bpermute|permute|swizzle)", i["opcode"]))
            tpw = sorted(set(re.findall(r"threadsPerWarp = \[([^\]]*)\]", ttg_t)))
            lo = f"TTGIR convert_layout x{cl}, local_alloc x{la}; ds_bpermute/permute/swizzle x{perm}; DPP VALU x{ndpp}; threadsPerWarp layouts {tpw}"
            if atom:
                g_ = lambda mid, vn="Avg": mets.get((k, mid, vn), "n/a")  # noqa: E731
                apath = (f"static atomics: {fmt(atom)} (scope modifiers as emitted); kernel-name aggregate counters: "
                         f"vL1D atomic req (16.3.3) {g_('16.3.3')}, L2 atomic req (17.3.4) {g_('17.3.4')}, "
                         f"L2-Fabric atomic req (17.6.11) {g_('17.6.11')}, L2 atomic latency cycles (17.2.11) {g_('17.2.11')}, "
                         f"memory chart Fabric_L2 Atomic (3.1.46) {g_('3.1.46', 'Value')}")
            else:
                apath = "not_observed (no VMEM atomic in static ISA)"
            grid = l0["grid"]
            nwg = 1
            for gdim in grid:
                nwg *= gdim
            rs = (f"grid {grid} = {nwg} WGs ({nwg / N_CU:.3g} per CU of {N_CU}); WG {l0['workgroup_size']} threads = {l0['num_warps']} waves; "
                  f"VGPR {st['md'].get('NumVgprs')} AGPR {st['md'].get('NumAgprs')} SGPR {st['md'].get('TotalNumSgprs')}; "
                  f"LDS {l0['shared']} B; scratch {scratch} B; compiler occupancy {st['md'].get('Occupancy')} waves/SIMD; "
                  f"launches {len(st['launches'])}")
            EP.append({"profile_id": p_c, "stage": stage, "access_path": ap, "matrix_path": mp, "buffer_location": bl,
                       "layout_operations": lo, "atomic_path": apath, "resource_summary": rs,
                       "evidence_path": f"{ctx.rel(st['ir_dir'])}; {ctx.rel(ctx.AN / 'replay_audit' / (e['op'] + '__' + e['dtype'] + '__formal.json'))}",
                       "evidence_confidence": "high: static IR/ISA of the hsaco-identical recompile + launch record; source form from git show",
                       "notes": f"static_isa profile {p_s}; counters in {p_c}"})
            pair_summary.setdefault(pair_key(op, dt), {})[stage] = {"uses_descriptor": uses_desc, "n_wg": nwg,
                                                                     "narrow_loads": widths(loads).get("scalar(<=32b)", 0),
                                                                     "narrow_stores": widths(stores).get("scalar(<=32b)", 0),
                                                                     "vector_loads": widths(loads).get("vector(>=64b)", 0),
                                                                     "nt_loads": nt_l, "atomics": sum(atom.values()), "mfma": sum(mf.values()),
                                                                     "scratch": scratch}

        # ---- PyTorch kernel trace stages
        if tk and "kernels_aggregated" in tk:
            for i, kk in enumerate(tk["kernels_aggregated"]):
                lid = f"agg{i}:{kk['name'][:80]}"
                for key, unit in [("count", "dispatches"), ("dur_ns", "ns (sum over dispatches)"), ("wg", "work-items per workgroup"), ("vgpr", "registers")]:
                    KM.append({"profile_id": p_t, "launch_id": lid, "kernel_name": kk["name"], "stage": f"agg{i}", "raw_metric_name": f"trace.{key}",
                               "normalized_group": "execution" if key != "vgpr" else "resources", "value": kk.get(key, ""), "unit": unit,
                               "scope": f"aggregated over {kk.get('count')} dispatches with this kernel name inside the marked reference call",
                               "counter_kind": "profiler_duration_diagnostic_only" if key == "dur_ns" else "dispatch_record",
                               "extraction_method": "copied from analysis_2026-10-05/data/torch_kernels.json (aggregated form)",
                               "status": "collected" if key in kk else "unavailable",
                               "raw_source": ctx.rel(ctx.AN / "data" / "torch_kernels.json"),
                               "notes": "per-dispatch records were aggregated by name in the source table (15,903 eager dispatches)"})
                EP.append({"profile_id": p_t, "stage": f"agg{i}:{kk['family']}", "access_path": f"library/kernel family: {kk['family']}",
                           "matrix_path": "unknown (no ISA for this kernel)", "buffer_location": "unknown (not in aggregated record)",
                           "layout_operations": "unknown", "atomic_path": "unknown",
                           "resource_summary": f"{kk.get('count')} dispatches; WG {kk.get('wg')}; VGPR {kk.get('vgpr')}",
                           "evidence_path": ctx.rel(ctx.AN / "data" / "torch_kernels.json"), "evidence_confidence": "medium: aggregated kernel trace record",
                           "notes": f"kernel {kk['name'][:200]}"})
        elif tk:
            for i, kk in enumerate(tk["kernels"]):
                lid = f"{i}:{kk['name'][:80]}"
                for key, unit in [("dur_ns", "ns"), ("grid", "work-items"), ("wg", "work-items per workgroup"), ("vgpr", "registers"),
                                  ("agpr", "registers"), ("sgpr", "registers"), ("lds", "bytes"), ("scratch", "bytes")]:
                    KM.append({"profile_id": p_t, "launch_id": lid, "kernel_name": kk["name"], "stage": f"{i}", "raw_metric_name": f"trace.{key}",
                               "normalized_group": "execution" if key in ("dur_ns", "grid", "wg") else "resources", "value": kk[key], "unit": unit,
                               "scope": "one dispatch inside the marked reference call",
                               "counter_kind": "profiler_duration_diagnostic_only" if key == "dur_ns" else "dispatch_record",
                               "extraction_method": "copied from analysis_2026-10-05/data/torch_kernels.json", "status": "collected",
                               "raw_source": ctx.rel(ctx.AN / "data" / "torch_kernels.json"),
                               "notes": "trace duration: diagnostic/intra-call fraction only, never formal latency" if key == "dur_ns" else ""})
                EP.append({"profile_id": p_t, "stage": f"{i}:{kk['family']}", "access_path": f"library/kernel family: {kk['family']}",
                           "matrix_path": "unknown (no ISA for this kernel)", "buffer_location": f"LDS {kk['lds']} B; scratch {kk['scratch']} B",
                           "layout_operations": "unknown", "atomic_path": "unknown",
                           "resource_summary": f"grid {kk['grid']} work-items, WG {kk['wg']}; VGPR {kk['vgpr']} AGPR {kk['agpr']} SGPR {kk['sgpr']}",
                           "evidence_path": ctx.rel(ctx.AN / "data" / "torch_kernels.json"), "evidence_confidence": "medium: kernel trace record only",
                           "notes": f"kernel {kk['name'][:200]}"})
        stats["launches_torch_trace"] += tk["n_kernels"] if tk else 0

    # ---- ATT decodes of ATen kernels (dynamic hit counts of traced waves on one CU)
    for d in sorted((ctx.AN / "att_torch").glob("torch_*")):
        m = re.match(r"torch_(.+)_(fp16|fp32|bf16|int8|int32)$", d.name)
        op, dt = m.group(1), m.group(2)
        statf = sorted(d.glob("stats_*.csv"))[0]
        rows = list(csv.DictReader(open(statf)))
        kname = rows[0]["Source"] if rows else ""
        p_a = pid("torch_att", op, dt)
        P.append({"profile_id": p_a, "device": DEVICE, "dsl": "pytorch", "operator": op, "dtype": dt,
                  "case_id": next((p["case_id"] for p in P if p["profile_id"] == pid("torch_kernel_trace", op, dt)), ""),
                  "params_json": next((p["params_json"] for p in P if p["profile_id"] == pid("torch_kernel_trace", op, dt)), ""),
                  "collection_level": "rocprofv3 --att (thread trace), --att-target-cu 1, kernel include regex; decoded",
                  "replay_mode": "single application run (same PyTorch harness as the kernel trace)",
                  "report_path": ctx.rel(d), "report_sha256": tree_sha256(d), "report_origin": "local analysis_2026-10-05/att_torch",
                  "report_revision": "analysis_2026-10-05", "benchmark_case_match_status":
                      next((p["benchmark_case_match_status"] for p in P if p["profile_id"] == pid("torch_kernel_trace", op, dt)), "unmatched"),
                  "code_match_status": "not_applicable (ATen binary from torch 2.10.0+rocm7.1)", "launch_count": 1,
                  "notes": f"decoded kernel: {kname[:200]}; dispatch index from file name {statf.name} (may be a prime call; same inputs). "
                           "Only the decoded code.json + stats CSV are retained; raw .att files are not.",
                  "profiling_source_sha": "", "profiled_params_full_json":
                      next((p["profiled_params_full_json"] for p in P if p["profile_id"] == pid("torch_kernel_trace", op, dt)), ""),
                  "kernel_config_json": ""})
        stats["reports"] += 1
        opc = collections.Counter(); hits = collections.Counter()
        for r in rows:
            ins = r["Instruction"].strip()
            if not ins or ins.startswith(";"):
                continue
            o = ins.split()[0]
            toks = set(re.split(r"[\s,]+", ins)[1:])
            mods = tuple(x for x in ("sc0", "sc1", "nt", "glc", "slc") if x in toks)
            name = o + (f" [{' '.join(mods)}]" if mods and re.match(r"^(buffer|global|flat)_", o) else "")
            opc[(isa_family(o), name)] += 1
            hits[(isa_family(o), name)] += int(r["Hitcount"] or 0)
            if int(r["Hitcount"] or 0) > 0:
                PC.append({"profile_id": p_a, "launch_id": "att:traced_waves", "pc": f"code_object {r['CodeObj']} vaddr {r['Vaddr']}",
                           "instruction_family": isa_family(o), "instruction_text": ins, "sample_count": "",
                           "executed_instruction_count": r["Hitcount"], "stall_reason": "",
                           "source_line": r["Source"], "evidence_path": ctx.rel(statf),
                           "notes": f"ATT hit count of traced waves on one CU (not whole-kernel); latency={r['Latency']} stall={r['Stall']} idle={r['Idle']} cycles (summed over hits)"})
        for (fam, name), c in sorted(opc.items()):
            IM.append({"profile_id": p_a, "launch_id": "att:code_object", "instruction_family": fam, "instruction_name": name, "count": c,
                       "count_kind": "static_isa_opcode_count", "scope": "decoded code object text of the ATen kernel (static)",
                       "evidence_path": ctx.rel(statf), "notes": "[mods] = cache/scope modifiers as emitted"})
            IM.append({"profile_id": p_a, "launch_id": "att:traced_waves", "instruction_family": fam, "instruction_name": name, "count": hits[(fam, name)],
                       "count_kind": "dynamic_att_hitcount_traced_waves", "scope": "waves traced on one CU (sampled subset of the dispatch)",
                       "evidence_path": ctx.rel(statf), "notes": ""})

    # ---- 6.2.x suite-wide zero check (supports CAVEAT text)
    zero62 = {mid: all(v in ("0.0", "0", "") for v in vals) for mid, vals in sorted(metric_zero_watch.items())}
    if not all(zero62.values()):
        for r in KM:
            if r["raw_metric_name"].startswith("6.2.") and "CAVEAT" in r["notes"]:
                mid = r["raw_metric_name"].split(" | ")[0]
                if not zero62.get(mid, False):
                    r["notes"] = r["notes"].replace(CAVEAT["6.2."], "")

    P.sort(key=lambda r: r["profile_id"])
    write_csv(ctx.out / "profile_index.csv", prof_hdr, P)
    KM.sort(key=lambda r: (r["profile_id"], r["launch_id"], r["raw_source"], r["raw_metric_name"]))
    write_csv(ctx.out / "kernel_metrics_long.csv.gz", km_hdr, KM, gz=True)
    IM.sort(key=lambda r: (r["profile_id"], r["launch_id"], r["count_kind"], r["instruction_family"], r["instruction_name"]))
    write_csv(ctx.out / "instruction_mix.csv", im_hdr, IM)
    PC.sort(key=lambda r: (r["profile_id"], r["launch_id"], -int(r["sample_count"] or r["executed_instruction_count"] or 0), r["pc"]))
    write_csv(ctx.out / "pc_hotspots.csv.gz", pc_hdr, PC, gz=True)
    EP.sort(key=lambda r: (r["profile_id"], r["stage"]))
    write_csv(ctx.out / "execution_paths.csv", ep_hdr, EP)

    diag_rows = diagnosis(ctx, P, pair_summary, torch_k)
    exp_rows = experiments(ctx)
    env = environment(ctx, base_commit, manifest, bench_meta)
    write_json(ctx.out / "extraction_log.json", {"stats": dict(sorted((k, v) for k, v in stats.items() if not isinstance(v, dict))),
                                                 "metric_6_2_all_zero_across_suite": zero62,
                                                 "issues": {k: v for k, v in sorted(ctx.issues.items())},
                                                 "base_commit": base_commit})
    print(f"benchmark_cases {len(bench_rows)}  profiles {len(P)}  metrics {len(KM)}  instr {len(IM)}  pcs {len(PC)}  "
          f"paths {len(EP)}  diagnosis {len(diag_rows)}  experiments {len(exp_rows)}")
    print("issues:", {k: len(v) for k, v in ctx.issues.items()})


# ============================================================================== 5. diagnosis
MECH_METRICS = {
    "M1": "execution_paths.access_path (TensorDescriptor -> buffer/global load widths); instruction_mix static VMEM_LOAD widths; "
          "2.1.10 MFMA Util; 16.1.3/16.3.x vL1D coalescing; SQ_INSTS_VMEM; diagnostic_experiments gemm_desc_vs_ptr_*",
    "M2": "instruction_mix [nt] load modifiers (Triton static ISA vs ATen ATT); 4.1.9 HBM bandwidth; diagnostic_experiments "
          "stream_vadd_*, cache_ablation_fp32, flush_variants_fp32",
    "M3": "torch_kernel_trace kernel families (execution_paths for torch profiles); trace.dur_ns fractions",
    "M4": "10.2.0 VALU INT32 / 10.1.0 VALU; SQ_INSTS_VALU_INT32; 2.1.9 VALU Util",
    "M5": "instruction_mix static VMEM_LOAD/VMEM_STORE widths (ubyte/ushort/dword); 16.1.3 coalescing; pc_hotspots VMEM stall reasons",
    "M6": "launch.grid / derived.workgroups vs 304 CUs; 2.1.15 wave occupancy",
    "M7": "static VMEM_ATOMIC; 17.3.4 L2 Atomic Req; L2 Cache 'L2-Fabric Atomic'; pc_hotspots on atomics",
    "C": "benchmark_cases (ratio); execution_paths",
}
MECH_CONFOUNDERS = {
    "M1": "descriptor programs were authored/tuned for NVIDIA TMA; 64 KiB LDS invalidates B200-sized configs; pointer experiment also changes num_stages (fp16) and store API",
    "M2": "formal protocol flushes with a 512 MiB WRITE before each iteration; nt load policy of ATen verified by ATT for a subset of kernels only (vector_add fp32/int8, relu fp16, mean fp32, layernorm fp16, softmax fp16, interleave fp32, transpose fp32, reverse fp32)",
    "M3": "PyTorch side evidence is a single traced call per pair (no counters); library heuristics are shape-specific",
    "M4": "VALU-bound address arithmetic may be hidden by memory latency; INT32 share is a dynamic counter ratio, not cycles",
    "M5": "narrow accesses are static ISA counts; coalescing in hardware may merge them",
    "M6": "grid size is a design/config choice of the Triton program; autotune space may not contain larger grids",
    "M7": "atomic contention depends on input distribution (bin count); no LDS-privatized control",
    "C": "control/parity classification relies on the formal ratio only",
}


def diagnosis(ctx, P, pair_summary, torch_k):
    dj = json.load(open(DIAGNOSIS_JSON))
    hdr = ["device", "dsl", "operator", "dtype", "comparison_scope", "mechanism_id", "observation", "mechanism_hypothesis",
           "supporting_metric_names", "supporting_profile_ids", "alternative_explanation", "confounders", "evidence_quality",
           "review_status", "source_paths",
           "mechanism_role", "b200_table5_reference", "relationship_class", "paper_priority", "nature", "pytorch_path_mi300x",
           "trace_check_status", "trace_check_detail"]
    rows = []
    prof_by_op = collections.defaultdict(list)
    for p in P:
        prof_by_op[p["operator"]].append(p)
    for op, d in sorted(dj["operators"].items()):
        dts = sorted({p["dtype"] for p in prof_by_op[op]})
        pids = sorted(p["profile_id"] for p in prof_by_op[op])
        mechs = [("primary", d["mech"], d["primary"])]
        for m in sorted(set(re.findall(r"\((M[1-7])\)", d["secondary"]))):
            if m != d["mech"]:
                mechs.append(("secondary", m, d["secondary"]))
        for role, m, hyp in mechs:
            status, detail = trace_check(m, op, dts, pair_summary, torch_k, ctx)
            rows.append({"device": DEVICE, "dsl": DSL, "operator": op, "dtype": ";".join(dts),
                         "comparison_scope": "MI300X Triton vs MI300X PyTorch (formal autotune CSV); B200 relation from paper Table 5",
                         "mechanism_id": m, "observation": d["evidence"], "mechanism_hypothesis": hyp,
                         "supporting_metric_names": MECH_METRICS[m], "supporting_profile_ids": ";".join(pids),
                         "alternative_explanation": d["alternative_explanation"], "confounders": MECH_CONFOUNDERS[m],
                         "evidence_quality": d["conf"],
                         "review_status": "draft from 2026-10-05 analysis (not author-reviewed); text copied verbatim",
                         "source_paths": "scripts/paper_figures/data/mi300x_diagnosis_2026-10-05.json; " + "; ".join(dj["source_files"]),
                         "mechanism_role": role,
                         "b200_table5_reference": f"{d['b200_table5']['primary_diagnosis']} | PyTorch path {d['b200_table5']['pytorch_path']} | Q_o {d['b200_table5']['Q_o']}",
                         "relationship_class": f"{d['rel']}: {dj['relationship_classes'].get(d['rel'], '')}",
                         "paper_priority": d["pri"], "nature": d["nature"], "pytorch_path_mi300x": d["pt"],
                         "trace_check_status": status, "trace_check_detail": detail})
    write_csv(ctx.out / "diagnosis_evidence.csv", hdr, rows)
    return rows


def trace_check(m, op, dts, ps, torch_k, ctx):
    """Automatic check that a mechanism label is backed by extracted raw evidence for at least one dtype."""
    st = {dt: ps.get(pair_key(op, dt), {}) for dt in dts}
    fams = {}
    for dt in dts:
        t = torch_k.get(pair_key(op, dt), {})
        fams[dt] = sorted({k["family"] for k in t.get("kernels", t.get("kernels_aggregated", []))})
    if m == "M1":
        ok = [dt for dt, s in st.items() if any(v["uses_descriptor"] and v["narrow_loads"] > 0 for v in s.values())]
        return ("traced" if ok else "not_traced"), f"descriptor kernel with scalar (<=32b) static loads in dtypes {ok}"
    if m == "M2":
        tri = [dt for dt, s in st.items() if s and all(v["nt_loads"] == 0 for v in s.values())]
        aten = [dt for dt, f in fams.items() if any(x in ("ATen vectorized elementwise", "ATen reduce") for x in f)]
        att = sorted(p.name for p in (ctx.AN / "att_torch").glob(f"torch_{op}_*"))
        ok = sorted(set(tri) & set(aten))
        return ("traced" if ok else "partially_traced"), (f"Triton loads without nt in {tri}; ATen vectorized/reduce kernel in {aten}; "
                                                         f"ATT decodes for this op: {att or 'none (nt inferred from kernel template)'}")
    if m == "M3":
        vend = {dt: sorted({x for x in f if not x.startswith("ATen") and x != "other"}) for dt, f in fams.items()}
        ok = [dt for dt, v in vend.items() if v]
        return ("traced" if ok else "partially_traced"), (f"non-ATen library families per dtype: {vend}; all families: {fams}"
                                                          + ("" if ok else "; no vendor-library family in the trace: the M3 claim rests on the "
                                                             "PyTorch-side latency difference, not on a vendor-library change"))
    if m == "M4":
        return "traced_by_metric", "see kernel_metrics_long 10.2.0 (VALU INT32) and 10.1.0 (VALU) for the conv kernels"
    if m == "M5":
        ok = [dt for dt, s in st.items() if any(v["narrow_loads"] or v["narrow_stores"] for v in s.values())]
        return ("traced" if ok else "not_traced"), f"scalar (<=32b) static loads/stores in dtypes {ok}"
    if m == "M6":
        small = {dt: sorted((k, v["n_wg"]) for k, v in s.items() if v["n_wg"] < N_CU) for dt, s in st.items()}
        allwg = {dt: sorted((k, v["n_wg"]) for k, v in s.items()) for dt, s in st.items()}
        ok = [dt for dt, v in small.items() if v]
        return ("traced" if ok else "not_traced"), (f"stages with fewer workgroups than {N_CU} CUs: {small}; "
                                                    f"all stages (workgroups): {allwg}")
    if m == "M7":
        ok = [dt for dt, s in st.items() if any(v["atomics"] for v in s.values())]
        return ("traced" if ok else "not_traced"), f"static VMEM atomics in dtypes {ok}"
    return "not_applicable", "control/other class"


# ============================================================================ 6. experiments
def experiments(ctx):
    X = ctx.AN / "exp"
    hdr = ["experiment_id", "operator", "dtype", "variant", "changed_factor", "other_configuration_changes", "latency_ms",
           "reference_variant", "measurement_protocol", "correctness_status", "evidence_path", "notes"]
    proto = ("DIAGNOSTIC, not formal: tilebench.core.timer.report_benchmark(warmup=2, repeat=10, flush_l2=True -> 512 MiB write "
             "flush before each iteration, outside the timed scope), eager, mean of repeats; the formal MI300X CSVs used warmup=20, repeat=100 (see environment.json timing)")
    rows = []
    for dt in ("fp16", "fp32"):
        f = X / f"gemm_{dt}.json"
        g = json.load(open(f))
        ev = ctx.rel(f)
        eid = f"gemm_desc_vs_ptr_{dt}"
        shape = f"M,N,K={g['shape']}"
        win = jdump(g["winner"])
        rows.append({"experiment_id": eid, "operator": "matmul_fp32_fp16_fp8", "dtype": dt, "variant": "torch.matmul (hipBLASLt)",
                     "changed_factor": "reference", "other_configuration_changes": shape, "latency_ms": num(g["torch_ms"]),
                     "reference_variant": "", "measurement_protocol": proto, "correctness_status": "reference", "evidence_path": ev, "notes": ""})
        rows.append({"experiment_id": eid, "operator": "matmul_fp32_fp16_fp8", "dtype": dt, "variant": "TileBench matmul_kernel (TensorDescriptor), formal winner",
                     "changed_factor": "none (formal implementation and winner config)", "other_configuration_changes": f"{shape}; winner {win}",
                     "latency_ms": num(g["desc_winner_ms"]), "reference_variant": "torch.matmul (hipBLASLt)", "measurement_protocol": proto,
                     "correctness_status": "not_checked_against_torch_in_experiment", "evidence_path": ev, "notes": ""})
        ch = []
        if g["ptr_same_tile_cfg"]["num_stages"] != g["winner"]["num_stages"]:
            ch.append(f"num_stages {g['winner']['num_stages']} -> {g['ptr_same_tile_cfg']['num_stages']} (pointer kernel at winner stages failed: "
                      f"{'; '.join(a['error'][:90] for a in g['ptr_same_tile_attempts'])})")
        ch.append("C written with masked tl.store instead of descriptor store; explicit bounds masks on loads; B read from a [N,K] "
                  "contiguous copy made once outside timing (the descriptor path also reads a cached [N,K] copy)")
        rows.append({"experiment_id": eid, "operator": "matmul_fp32_fp16_fp8", "dtype": dt, "variant": "pointer-load kernel, same tile/grouping as winner",
                     "changed_factor": "operand load API: TensorDescriptor.load -> tl.load on pointers",
                     "other_configuration_changes": "; ".join(ch) + f"; config {jdump(g['ptr_same_tile_cfg'])}; static ISA {jdump(g['ptr_same_cfg_isa'])}",
                     "latency_ms": num(g["ptr_same_cfg_ms"]), "reference_variant": "TileBench matmul_kernel (TensorDescriptor), formal winner",
                     "measurement_protocol": proto, "correctness_status": f"max_abs_diff_vs_descriptor_output={g['ptr_same_cfg_maxabs_vs_desc']}",
                     "evidence_path": ev, "notes": "NOT a pure one-factor comparison when num_stages also changed"})
        rows.append({"experiment_id": eid, "operator": "matmul_fp32_fp16_fp8", "dtype": dt, "variant": "pointer-load kernel, best of the formal 22-config autotune space",
                     "changed_factor": "operand load API + config search", "other_configuration_changes": f"best config {jdump(g['ptr_best_cfg'])}; "
                     f"{len(g['ptr_sweep_failed'])} of the configs failed to compile/launch (listed below)",
                     "latency_ms": num(g["ptr_best_ms"]), "reference_variant": "TileBench matmul_kernel (TensorDescriptor), formal winner",
                     "measurement_protocol": proto, "correctness_status": "not_checked", "evidence_path": ev, "notes": ""})
        for fr in g["ptr_sweep_failed"]:
            rows.append({"experiment_id": eid, "operator": "matmul_fp32_fp16_fp8", "dtype": dt, "variant": "pointer-load kernel, infeasible config",
                         "changed_factor": "operand load API + config", "other_configuration_changes": jdump(fr[1]), "latency_ms": "",
                         "reference_variant": "", "measurement_protocol": proto, "correctness_status": "not_run",
                         "evidence_path": ev, "notes": f"error: {fr[2]}"})
    for dt in ("fp16", "fp32"):
        f = X / f"stream_vadd_{dt}.json"
        s = json.load(open(f)); ev = ctx.rel(f)
        eid = f"stream_vadd_config_sweep_{dt}"
        rows.append({"experiment_id": eid, "operator": "vector_add", "dtype": dt, "variant": "torch.add (ATen vectorized_elementwise_kernel)",
                     "changed_factor": "reference", "other_configuration_changes": f"n={s['n']}", "latency_ms": num(s["torch_ms"]),
                     "reference_variant": "", "measurement_protocol": proto, "correctness_status": "reference", "evidence_path": ev, "notes": ""})
        for r in s["rows"]:
            kern = r["kernel"]
            if kern == "add":
                cf, oc = "launch config only (default cache policy)", f"BLOCK={r['BLOCK']}, num_warps={r['num_warps']}"
            elif kern.startswith("add_nt"):
                cf, oc = "cache modifiers: loads .cg, store .cs", f"BLOCK={r['BLOCK']}, num_warps={r['num_warps']}"
            else:
                cf, oc = "persistent grid-stride loop (default cache policy)", f"{kern}, BLOCK={r['BLOCK']}, num_warps={r['num_warps']}"
            rows.append({"experiment_id": eid, "operator": "vector_add", "dtype": dt, "variant": f"{kern} BLOCK={r['BLOCK']} nw={r['num_warps']}",
                         "changed_factor": cf, "other_configuration_changes": oc + "; standalone kernel written for the experiment (not impl_triton.py)",
                         "latency_ms": num(r["ms"]), "reference_variant": "torch.add (ATen vectorized_elementwise_kernel)",
                         "measurement_protocol": proto, "correctness_status": "not_checked", "evidence_path": ev, "notes": ""})
        for r in s.get("errors", []):
            rows.append({"experiment_id": eid, "operator": "vector_add", "dtype": dt, "variant": f"{r.get('kernel')} BLOCK={r.get('BLOCK')} nw={r.get('num_warps')}",
                         "changed_factor": "launch config", "other_configuration_changes": "", "latency_ms": "", "reference_variant": "",
                         "measurement_protocol": proto, "correctness_status": "not_run", "evidence_path": ev, "notes": f"error: {r.get('err')}"})
    f = X / "cache_ablation_fp32.json"
    c = json.load(open(f)); ev = ctx.rel(f)
    default_isa = c["load(default)_store(default)"].get("mem_instr")
    rows.append({"experiment_id": "cache_modifier_ablation_fp32", "operator": "vector_add", "dtype": "fp32", "variant": "torch.add",
                 "changed_factor": "reference", "other_configuration_changes": "n=20971520", "latency_ms": num(c["torch_ms"]), "reference_variant": "",
                 "measurement_protocol": proto, "correctness_status": "reference", "evidence_path": ev, "notes": ""})
    for k, v in c.items():
        if k == "torch_ms":
            continue
        same = (v.get("mem_instr") == default_isa) and k != "load(default)_store(default)"
        rows.append({"experiment_id": "cache_modifier_ablation_fp32", "operator": "vector_add", "dtype": "fp32", "variant": k,
                     "changed_factor": "Triton cache_modifier on loads and/or store", "other_configuration_changes":
                     "BLOCK=2048, num_warps=4 (fixed); standalone kernel; emitted VMEM: " + " | ".join(v.get("mem_instr", [])),
                     "latency_ms": num(v.get("ms")), "reference_variant": "torch.add", "measurement_protocol": proto,
                     "correctness_status": "not_checked", "evidence_path": ev,
                     "notes": ("emitted ISA identical to the default variant: the modifier had no effect on gfx942 codegen" if same else "")
                              + (v.get("error", "") if "error" in v else "")})
    f = X / "flush_variants_fp32.json"
    fl = json.load(open(f)); ev = ctx.rel(f)
    fdesc = {"write_flush_512MiB(formal)": "512 MiB write (fill) flush - the formal protocol",
             "read_flush_512MiB": "512 MiB read (sum) flush", "write_then_read_flush": "write flush followed by read flush",
             "no_flush": "no flush (flush_l2=False)"}
    for vn, row in fl.items():
        for fn, v in row.items():
            rows.append({"experiment_id": "flush_protocol_variants_fp32", "operator": "vector_add", "dtype": "fp32", "variant": f"{fn} | {vn}",
                         "changed_factor": f"pre-iteration cache flush: {fdesc.get(vn, vn)}" + ("" if fn == "torch.add" else ""),
                         "other_configuration_changes": {"torch.add": "PyTorch reference", "triton_default": "Triton BLOCK=2048 nw=4, default cache policy",
                                                         "triton_load_cg": "Triton BLOCK=2048 nw=4, loads .cg"}[fn] + "; whole block repeated twice, second kept",
                         "latency_ms": num(v["ms"]), "reference_variant": f"torch.add | {vn}",
                         "measurement_protocol": proto.replace("flush_l2=True -> 512 MiB write flush", "flush variant as listed"),
                         "correctness_status": "not_checked", "evidence_path": ev, "notes": ""})
    write_csv(ctx.out / "diagnostic_experiments.csv", hdr, rows)
    return rows


# ============================================================================ 7. environment
def environment(ctx, base_commit, manifest, bench_meta):
    em = json.load(open(ctx.L / "metadata" / "environment.json"))
    cov = json.load(open(ctx.RPC / "coverage.json"))
    sysinfo = list(csv.DictReader(open(ctx.RPC / "vector_add" / "triton_fp32" / "workload" / "sysinfo.csv")))[0]
    import yaml
    # timing settings in force when the formal CSVs were measured: config.yaml at the measurement source commit
    # of each (operator, mode) from the provenance files, plus any CLI override recorded there
    proto = collections.Counter()
    overrides = collections.Counter()
    for (op, mode), (pf, pj) in sorted(bench_meta["prov"].items()):
        sha = pj["source"]["git_sha"]
        b = (yaml.safe_load(git("show", f"{sha}:tilebench/benchmarks/operators/{op}/config.yaml")) or {}).get("benchmark", {})
        ov = pj["run"].get("overrides") or {}
        eff = {k: ov.get(k, b.get(k)) for k in ("warmup", "repeat", "use_cuda_graph", "flush_l2")}
        proto[(mode, sha, jdump(eff))] += 1
        overrides[jdump(ov)] += 1
    proto_set = [{"mode": m, "source_git_sha": sha, "settings": json.loads(e), "operators": n} for (m, sha, e), n in sorted(proto.items())]
    inputs = {}
    for p in [ctx.RPC / "sweep_log.json", ctx.RPC / "coverage.json", ctx.SWEEP / "report_manifest.json",
              ctx.SWEEP / "winner_replay_audit_post_fix.json", ctx.SWEEP / "PROVENANCE.md", ctx.O / "profiling" / "MI300X" / "ncu_catalogue.json",
              ctx.O / "profiling" / "MI300X" / "kernel_counts.json", ctx.AN / "data" / "torch_kernels.json", ctx.L / "metadata" / "environment.json",
              DIAGNOSIS_JSON] + sorted((ctx.AN / "exp").glob("*.json")):
        inputs[ctx.rel(p)] = sha256_file(p)
    env = {
        "schema": "tilearena-paper-figures-environment/1",
        "device": DEVICE, "architecture": ARCH, "vendor": "amd", "dsls_available": ["triton"], "reference": "pytorch",
        "dsls_not_available_on_this_device": {"cutile": "NVIDIA only", "tilelang": "no MI300X formal results in this suite"},
        "gpu": em.get("gpu"), "cache": em.get("cache"),
        "rocprof_sysinfo": {k: sysinfo[k] for k in ("gpu_model", "gpu_arch", "cu_per_gpu", "simd_per_cu", "se_per_gpu", "wave_size",
                                                     "max_waves_per_cu", "max_sclk", "max_mclk", "cur_sclk", "cur_mclk", "gpu_l1", "gpu_l2",
                                                     "total_l2_chan", "num_xcd", "num_hbm_channels", "compute_partition", "memory_partition",
                                                     "rocm_version", "vbios", "amd_gpu_kernel_version")},
        "software": em.get("software"),
        "timing": {"engine": "TileBench Proton timer (roctracer backend), kernel-time sum of the operator",
                   "formal_settings_at_measurement_commits": proto_set,
                   "cli_overrides_recorded_in_provenance": dict(overrides),
                   "note": "warmup/repeat come from config.yaml at the measurement source commits (engine defaults 20/100 when "
                           "absent); the current branch's config.yaml values are NOT the ones used for the MI300X CSVs",
                   "effective": em.get("timing_mode_for_config_use_cuda_graph_true"),
                   "flush": f"{em.get('cache', {}).get('flush_buffer_mb')} MiB buffer write before each iteration, outside the timed scope"},
        "formal_benchmark": {"csv_dir": "results/MI300X/csv", "csv_files": len(bench_meta["files"]),
                             "csv_last_commit_in_this_branch": git("log", "-1", "--format=%H", base_commit, "--", "results/MI300X/csv"),
                             "formal_csv_source": manifest["formal_csv_source"],
                             "timing_logs_archive_commit": "8e991537 (archive: MI300X logs from source 4d08985a)",
                             "known_exclusion": KNOWN_EXCLUSION},
        "profiling": {"profiler": manifest["profiler"], "coverage": {k: cov[k] for k in ("expected_valid_pairs", "successful_pairs", "complete", "excluded")},
                      "profiling_source": manifest["profiling_source"], "hf_dataset": manifest["hf_dataset"], "hf_folder": manifest["hf_folder"],
                      "hf_revision": manifest["hf_revision"], "hf_commits": manifest["hf_commits"],
                      "archive_commit": "63748286 (archive: MI300X profiling from source 124fdc94)",
                      "pc_sampling": "stochastic, interval 65536", "pytorch_side": "rocprofv3 kernel trace of one call per pair + 9 ATT decodes; no PyTorch counter profiles"},
        "category_source": CATEGORY_SOURCE,
        "local_roots": {"outputs": str(ctx.O), "results_logs": str(ctx.L)},
        "input_files_sha256": inputs,
        "extraction": {"base_commit": base_commit, "scripts": {p: sha256_file(REPO / p) for p in
                       ["scripts/paper_figures/extract_mi300x.py", "scripts/paper_figures/mi300x_common.py",
                        "scripts/paper_figures/case_identity.py", "scripts/paper_figures/vendor_mi300x_diagnosis.py"]},
                       "case_id_rule": "sha256(f'{operator}|{dtype}|{params_json}'), params_json = json.dumps(params, sort_keys=True, separators=(',', ':')) "
                                       "of the CSV params cell; see scripts/paper_figures/case_identity.py"},
    }
    write_json(ctx.out / "environment.json", env)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", type=Path, default=Path("/root/Tilebench/outputs"))
    ap.add_argument("--logs-root", type=Path, default=Path("/root/Tilebench/results/MI300X/logs"))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    build(Ctx(a.outputs_root, a.logs_root, a.out))


if __name__ == "__main__":
    main()
