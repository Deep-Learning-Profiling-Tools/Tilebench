"""Offline extraction of one or more existing Nsight Compute reports (.ncu-rep) into JSON.

This script never profiles: it only imports finished reports through the NVIDIA `ncu_report`
Python module and the `ncu --import` CLI (session page, static SASS page). It must be run with an
interpreter that can import `ncu_report` (e.g. /usr/bin/python3 with --ncu-python pointing at
<nsight-compute>/extras/python).

Per report it writes <out-dir>/<operator>__<dsl>_<dtype>.json.gz containing:
  session      : parsed `ncu --import <rep> --page session` (profiler command line, versions, host)
  launches     : one record per profiled launch (range/action order) with
                   - scalar metrics (raw NVIDIA name -> value, unit, type, rollup); every launch gets
                     the CORE set, the first launch of each distinct launch shape gets the EXTENDED set
                     (all collected metrics matching GROUP_RULES), see `metric_selection` in the output
                   - dynamic per-opcode counts (instanced sass__* metrics) when collected
  pc_tables    : per-PC table (SASS text, executed warp instructions, PC-sampling counts per stall
                 reason, source file:line) for the first launch of each distinct launch shape, when
                 the report carries per-PC data
  static_sass  : per kernel, static SASS opcode histogram (from per-PC table or, when the report has
                 no per-PC metrics, from `ncu --import --page source --print-source sass --csv`)
Failures are recorded in the output (status != "ok"); a failed import never produces zeros.

Usage:
  /usr/bin/python3 scripts/paper_figures/ncu_extract.py --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu \
      --out-dir <cache>/extract/B200 <report.ncu-rep> [...]
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

SCHEMA = "nvidia_ncu_extract/1"

# Metrics recorded for EVERY launch (when collected). Also the list against which "not_collected" is
# reported for reduced/targeted profiles.
CORE_METRICS = [
    "gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__cycles_active.avg",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
    "launch__shared_mem_per_block", "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic", "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks",
    "sm__maximum_warps_per_active_cycle_pct", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum", "smsp__thread_inst_executed.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_sectors.sum",
    "sass__inst_executed_global_loads", "sass__inst_executed_global_stores",
    "sass__inst_executed_shared_loads", "sass__inst_executed_shared_stores",
    "sass__inst_executed_local_loads", "sass__inst_executed_local_stores",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
]

# normalized_group assignment: first matching rule wins. Matching is done on the metric name with a
# leading "<UNIT>.<Section>." breakdown qualifier removed; the raw name is always preserved.
GROUP_RULES = [
    ("resources", r"^launch__(registers|shared_mem|thread_count|block|grid|cluster|waves|stack|barrier_count|sm_count|func_cache|uses_)"),
    ("occupancy", r"occupancy|warps_active|maximum_warps|warps_eligible|warps_launched|ctas_active|ctas_launched"),
    ("stalls", r"issue_stalled|warp_latency|pcsamp"),
    ("atomics", r"atom|_op_red|_red_|redas|redux"),
    ("local_memory", r"local|spill"),
    ("matrix_pipeline", r"tensor|pipe_tc|_tc_|tcmma|hmma|imma|gmma|dmma|tmem|tma|utc|uma"),
    ("synchronization", r"barrier|membar|_sync|warpgroup|_bar_"),
    ("shared_memory", r"mem_shared|_shared|smem|bank_conflict|wavefronts"),
    ("global_memory", r"dram|mem_global|global_op|global_load|global_store|xbar2l1tex|l1tex2xbar|memory_throughput|bytes_per_sector|ldgsts|sectors_per_request"),
    ("cache", r"^lts__|^l1tex__|sector_hit|cache"),
    ("instruction_mix", r"inst_executed|inst_issued|thread_inst|^sass__|_pipe_|ipc|issue_active|branch"),
    ("execution", r"time_duration|cycles_elapsed|cycles_active|throughput|clock|frequency|elapsed|duration"),
]
_GROUP_RE = [(g, re.compile(p)) for g, p in GROUP_RULES]
_SKIP_RE = re.compile(r"^(device__attribute|profiler__|numa__|nvlink|nvl|c2c|pcie|PCI\.|NVL|syslts|syslrc|gcc__|lrc__|gr__|fbpa|SYSLTS|derived__pct_occupancy_per|launch__occupancy_per_|launch__(context_id|device_id|stream_id|function_pcs|kernel_name|execution_model|tpc_|sm_count|uses_)|warpsampling:)")
_ROLLUP_SKIP = re.compile(r"\.(max|min)(\.|$)")
_QUALIFIER = re.compile(r"^[A-Z][A-Z0-9_]*\.[A-Za-z]+\.")

OPCODE_METRICS = [
    "sass__inst_executed_per_opcode",
    "sass__inst_executed_per_opcode_with_modifier_all",
    "sass__inst_executed_per_opcode_with_modifier_selective",
    "sass__thread_inst_executed_true_per_opcode",
]
PC_STALLS = ["long_scoreboard", "short_scoreboard", "wait", "barrier", "mio_throttle", "lg_throttle",
             "math_pipe_throttle", "not_selected", "selected", "no_instructions", "branch_resolving",
             "dispatch_stall", "membar", "sleeping", "tex_throttle", "misc", "gmma", "drain", "imc_miss",
             "warpgroup_arrive"]


def group_of(name):
    base = _QUALIFIER.sub("", name)
    for g, rx in _GROUP_RE:
        if rx.search(base):
            return g
    return None


def keep_extended(name):
    return not _SKIP_RE.search(name) and not _ROLLUP_SKIP.search(name) and group_of(name) is not None


def metric_record(m):
    try:
        if m.num_instances() > 1 and m.has_correlation_ids():
            return None
    except Exception:
        pass
    try:
        v = m.value()
    except Exception:
        return None
    if not isinstance(v, (int, float, str)):
        return None
    rec = {"value": v}
    for key, fn in (("unit", "unit"), ("type", "metric_type"), ("subtype", "metric_subtype"), ("rollup", "rollup_operation")):
        try:
            x = getattr(m, fn)()
            rec[key] = x if isinstance(x, (str, int, float)) or x is None else str(x)
        except Exception:
            rec[key] = None
    return rec


def instances(a, name, pc_keys=False):
    try:
        m = a[name]
        n = m.num_instances()
        if not n or not m.has_correlation_ids():
            return None
    except Exception:
        return None
    cor = m.correlation_ids()
    out = {}
    for i in range(n):
        try:
            k = cor.as_uint64(i) if pc_keys else cor.as_string(i)
        except Exception:
            continue
        try:
            v = m.as_uint64(i)
        except Exception:
            try:
                v = m.as_double(i)
            except Exception:
                continue
        out[k] = out.get(k, 0) + v
    return out


def pc_table(a):
    inst = instances(a, "inst_executed", True)
    if inst is None:
        return None
    pcs = {pc: {"inst_executed": v} for pc, v in inst.items()}
    smp = instances(a, "smsp__pcsamp_sample_count", True) or {}
    for pc, v in smp.items():
        pcs.setdefault(pc, {})["samples"] = v
    for s in PC_STALLS:
        for pc, v in (instances(a, f"smsp__pcsamp_warps_issue_stalled_{s}", True) or {}).items():
            if v:
                pcs.setdefault(pc, {}).setdefault("stall", {})[s] = v
    rows = []
    for pc in sorted(pcs):
        r = {"pc": pc, **pcs[pc]}
        try:
            r["sass"] = a.sass_by_pc(pc)
        except Exception:
            r["sass"] = None
        try:
            si = a.source_info(pc)
            if si is not None:
                r["src"] = f"{si.file_name()}:{si.line()}"
        except Exception:
            pass
        rows.append(r)
    return {"pc_sampling_collected": bool(smp), "rows": rows}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_session(ncu, rep):
    rc, out, err = run([ncu, "--import", rep, "--page", "session"])
    if rc != 0:
        return {"status": "error", "error": (err or out)[-2000:]}
    info, key = {}, None
    for line in out.splitlines():
        if not line.strip() or set(line.strip()) <= set("- "):
            key = None
            continue
        if len(line) > 22 and line[:21].strip() and line[21] == " " and not line.startswith(" "):
            key, val = line[:21].strip(), line[22:].rstrip()
            info[key] = val.strip()
        elif key and line.startswith(" " * 22):
            info[key] += line[22:].rstrip()      # wrapped continuation (no separator inserted)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] in ("display_name", "compute_capability_major", "compute_capability_minor",
                                                "multiprocessor_count", "total_memory"):
                info["device__" + parts[0]] = parts[1].strip()
    cmd = info.get("Profiler Command Line", "")
    for k in ("display_name", "compute_capability_major", "compute_capability_minor", "multiprocessor_count", "total_memory"):
        if k in info and "device__" + k not in info:
            info["device__" + k] = info[k]
    return {"status": "ok", "raw": info, "command_line": cmd,
            "created": info.get("Created"), "cuda_version": info.get("CUDA Version"),
            "ncu_target_version": info.get("Nsight Compute Target"), "host": info.get("Host Name"),
            "device_name": info.get("device__display_name"),
            "compute_capability": (f"{info.get('device__compute_capability_major')}.{info.get('device__compute_capability_minor')}"
                                   if info.get("device__compute_capability_major") else None)}


def collection_from_cmd(cmd):
    """(collection_level, replay_mode, sections_or_sets) from the recorded profiler command line."""
    m = re.search(r"--replay-mode\s+(\S+)", cmd)
    replay = m.group(1) if m else "kernel(default)"
    if re.search(r"--set\s+full", cmd):
        level = "full"
    elif re.search(r"--set\s+none", cmd) and "--metrics" in cmd and replay == "kernel":
        level = "reduced"
    elif "--metrics" in cmd:
        level = "targeted"
    else:
        level = "unknown"
    m = re.search(r"--set\s+(\S+)", cmd)
    return level, replay, (m.group(1) if m else None)


def static_sass_cli(ncu, rep):
    """Static SASS opcode histogram per kernel name (first occurrence of each kernel) via the CLI."""
    rc, out, err = run([ncu, "--import", rep, "--page", "source", "--print-source", "sass", "--csv"])
    if rc != 0:
        return {"status": "error", "error": (err or out)[-2000:]}
    kernels, cur, seen = {}, None, set()
    for line in out.splitlines():
        if line.startswith('"Kernel Name"'):
            name = next(csv.reader([line]))[1]
            cur = None if name in seen else name
            if cur:
                seen.add(name)
                kernels[cur] = {}
            continue
        if cur is None or not line.startswith('"0x'):
            continue
        row = next(csv.reader([line]))
        s = re.sub(r"^@!?U?P\w+\s+", "", row[1].strip())
        if not s:
            continue
        op = s.split()[0]
        kernels[cur][op] = kernels[cur].get(op, 0) + 1
    return {"status": "ok", "method": "ncu --import --page source --print-source sass --csv", "kernels": kernels}


def static_from_pc(rows):
    hist = {}
    for r in rows:
        s = re.sub(r"^@!?U?P\w+\s+", "", (r.get("sass") or "").strip())
        if s:
            op = s.split()[0]
            hist[op] = hist.get(op, 0) + 1
    return hist


def extract(ncu_report, ncu, rep, out_path, meta):
    res = {"schema": SCHEMA, "report": rep, "meta": meta, "status": "ok", "errors": []}
    res["session"] = parse_session(ncu, rep)
    cmd = res["session"].get("command_line", "") if res["session"].get("status") == "ok" else ""
    level, replay, nset = collection_from_cmd(cmd)
    res["collection"] = {"level_from_command_line": level, "replay_mode": replay, "set": nset}
    try:
        r = ncu_report.load_report(rep)
    except Exception as e:
        res["status"] = "import_failed"
        res["errors"].append(f"load_report: {type(e).__name__}: {e}")
        json.dump(res, gzip.open(out_path, "wt"))
        return res
    launches, shapes, pc_tables, static = [], {}, {}, {}
    idx = 0
    for ri in range(r.num_ranges()):
        rng = r.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            a = rng.action_by_idx(ai)
            try:
                mangled = a.name(a.NameBase_MANGLED)
            except Exception:
                mangled = None
            rec = {"launch_id": idx, "range": ri, "action": ai, "kernel_name": a.name(), "mangled_name": mangled}
            names = list(a.metric_names())
            core = {}
            for n in CORE_METRICS:
                if n in names:
                    mr = metric_record(a[n])
                    if mr is not None:
                        core[n] = mr
            shape = (rec["kernel_name"], *(core.get(k, {}).get("value") for k in
                     ("launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__shared_mem_per_block")))
            first_of_shape = shape not in shapes
            if first_of_shape:
                shapes[shape] = idx
                ext = {}
                for n in names:
                    if n in core or not keep_extended(n):
                        continue
                    mr = metric_record(a[n])
                    if mr is not None:
                        ext[n] = mr
                rec["extended_metrics"] = ext
            rec["metrics"] = core
            rec["shape_representative"] = shapes[shape]
            rec["n_metric_names_in_report"] = len(names)
            ops = {}
            for om in OPCODE_METRICS:
                d = instances(a, om)
                if d:
                    ops[om] = d
            rec["opcode_counts"] = ops
            if first_of_shape:
                try:
                    t = pc_table(a)
                except Exception as e:
                    t = None
                    res["errors"].append(f"pc_table launch {idx}: {e}")
                if t is not None:
                    pc_tables[str(idx)] = t
                    static.setdefault(rec["kernel_name"], {"method": "ncu_report per-PC table (all PCs of the launch)",
                                                           "launch_id": idx, "opcodes": static_from_pc(t["rows"])})
            launches.append(rec)
            idx += 1
    res["launches"] = launches
    res["pc_tables"] = pc_tables
    missing_static = sorted({l["kernel_name"] for l in launches} - set(static))
    if missing_static:
        cli = static_sass_cli(ncu, rep)
        if cli.get("status") == "ok":
            for k in missing_static:
                if k in cli["kernels"]:
                    static[k] = {"method": cli["method"], "launch_id": None, "opcodes": cli["kernels"][k]}
        else:
            res["errors"].append("static_sass_cli: " + cli.get("error", "")[:500])
    res["static_sass"] = static
    res["metric_selection"] = {"core": CORE_METRICS, "group_rules": GROUP_RULES,
                               "extended": "all scalar metrics of the first launch of each launch shape whose name matches a group rule; .max/.min rollups and device/profiler/interconnect metrics skipped",
                               "opcode_metrics": OPCODE_METRICS}
    json.dump(res, gzip.open(out_path, "wt"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu", required=True, help="ncu executable used ONLY for --import")
    ap.add_argument("--ncu-python", default=None, help="directory containing ncu_report.py (default: <ncu dir>/extras/python)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-json", default=None, help="JSON {report_path: meta} merged into each output")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("reports", nargs="+")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # import-only: never touch a GPU
    sys.path.insert(0, a.ncu_python or str(Path(a.ncu).resolve().parent / "extras" / "python"))
    import ncu_report
    meta_all = json.load(open(a.meta_json)) if a.meta_json else {}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for rep in a.reports:
        p = Path(rep)
        out = Path(a.out_dir) / f"{p.parent.name}__{p.stem}.json.gz"
        if a.skip_existing and out.exists():
            continue
        try:
            res = extract(ncu_report, a.ncu, rep, str(out) + ".tmp", meta_all.get(rep, {}))
            os.replace(str(out) + ".tmp", out)
            print(f"{res['status']} {rep} launches={len(res.get('launches', []))} errors={len(res['errors'])}", flush=True)
        except Exception:
            print(f"FAILED {rep}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
