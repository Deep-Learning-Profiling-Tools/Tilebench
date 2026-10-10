"""Resolve one TileBench case into a ready-to-analyze evidence bundle.

Joins the benchmark CSV, autotune winners and saved NCU reports for one
hardware/operator/dtype, then writes per-backend extractions and brief.md.
No GPU, compiler or benchmark run is involved.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GITHUB = "Deep-Learning-Profiling-Tools/Tilebench"
HF_DATASET = "bcui2/NCU_report"
ARCHIVES = {"B200": "9455c0bd76a0b09febee3080ce95f438f600896f",
            "GH200": "4c7dc1f08b91e39dbc2b2cf4c2ec591dba0e06f9",
            "MI300X": "4c7dc1f08b91e39dbc2b2cf4c2ec591dba0e06f9"}
BACKENDS = ["tilelang", "triton", "cutile"]
NOISE_RULES = {"Low Compression Rate", "Roofline Analysis", "Tensor Core Unused", "Workload Imbalance"}
STALLS = ["long_scoreboard", "short_scoreboard", "wait", "barrier", "membar",
          "math_pipe_throttle", "mio_throttle", "lg_throttle", "tex_throttle",
          "not_selected", "branch_resolving", "dispatch_stall", "drain",
          "no_instructions", "sleeping", "misc"]
TABLE = [
    ("NCU duration (ns)", "gpu__time_duration.sum"),
    ("Grid (blocks)", "launch__grid_size"),
    ("Grid x", "launch__grid_dim_x"),
    ("Grid y", "launch__grid_dim_y"),
    ("Grid z", "launch__grid_dim_z"),
    ("Block (threads)", "launch__block_size"),
    ("Registers/thread", "launch__registers_per_thread"),
    ("Shared mem/block (B)", "launch__shared_mem_per_block"),
    ("Occupancy limit: registers (blocks/SM)", "launch__occupancy_limit_registers"),
    ("Occupancy limit: shared mem", "launch__occupancy_limit_shared_mem"),
    ("Occupancy limit: warps", "launch__occupancy_limit_warps"),
    ("Occupancy limit: blocks", "launch__occupancy_limit_blocks"),
    ("Waves/SM", "launch__waves_per_multiprocessor"),
    ("Theoretical occupancy %", "sm__maximum_warps_per_active_cycle_pct"),
    ("Achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("Eligible warps/cycle", "smsp__warps_eligible.avg.per_cycle_active"),
    ("Issue active/cycle", "smsp__issue_active.avg.per_cycle_active"),
    ("Warp instructions executed", "smsp__inst_executed.sum"),
    ("SM throughput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("SMSP active cycles (avg)", "smsp__cycles_active.avg"),
    ("Tensor pipe active %", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("tcgen05 (TMEM) tensor pipe active %", "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("FMA pipe active %", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("ALU pipe active %", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("L1/TEX throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 throughput %", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM throughput %", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM read (B)", "dram__bytes_read.sum"),
    ("DRAM write (B)", "dram__bytes_write.sum"),
    ("Global load insts", "smsp__sass_inst_executed_op_global_ld.sum"),
    ("Async copy insts (LDGSTS)", "smsp__inst_executed_op_ldgsts.sum"),
    ("Global load L1 sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
    ("Global load bytes/sector", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio"),
    ("Global load L1 hit %", "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"),
    ("Global load L1 miss sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"),
    ("L2-to-L1 read bytes", "l1tex__m_xbar2l1tex_read_bytes.sum"),
    ("TMA load insts", "smsp__sass_inst_executed_op_tma_ld.sum"),
    ("TMA load bytes (L2-to-L1)", "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"),
    ("TMA store insts", "smsp__sass_inst_executed_op_tma_st.sum"),
    ("TMA store bytes", "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum"),
    ("Global store insts", "smsp__sass_inst_executed_op_global_st.sum"),
    ("Global store L1 sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
    ("Shared load insts", "smsp__sass_inst_executed_op_shared_ld.sum"),
    ("Shared store insts", "smsp__sass_inst_executed_op_shared_st.sum"),
    ("Local load insts", "smsp__sass_inst_executed_op_local_ld.sum"),
    ("Local store insts", "smsp__sass_inst_executed_op_local_st.sum"),
    ("Shared bank conflicts (ld)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
    ("Shared bank conflicts (st)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
    ("Replay passes", "profiler__replayer_passes"),
] + [(f"Stall {s} / issue-active", f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio")
     for s in ["long_scoreboard", "short_scoreboard", "wait", "barrier", "membar",
               "math_pipe_throttle", "mio_throttle", "lg_throttle", "not_selected",
               "branch_resolving", "no_instruction", "sleeping"]]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def parse_params(text):
    out = {}
    for part in text.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def read_csv(repo, hardware, operator, mode):
    for base in (f"results/{hardware}/csv", "results/csv"):
        path = repo / base / f"{operator}_{mode}.csv"
        if path.exists():
            with path.open() as stream:
                return path, list(csv.DictReader(stream))
    return None, []


def latencies(row):
    return {k[:-3]: number(v) for k, v in row.items() if k.endswith("_ms") and number(v)}


def load_logs(repo, hardware, operator, network):
    def wanted(name):
        return name == f"{operator}_tilelang_autotune.json" or (
            name.startswith(f"{operator}_autotune") and name.endswith(".json"))

    records, sources = [], []
    for base in (f"results/{hardware}/logs/autotune_logs", "results/logs/autotune_logs", "evidence"):
        found = sorted(path for path in (repo / base).glob(f"{operator}_*autotune*.json") if wanted(path.name))
        for path in found:
            records += json.loads(path.read_text())
            sources.append({"source": str(path) + (" (unverified hardware: confirm it is a " + hardware + " log)" if base == "evidence" else ""),
                            "sha256": sha256(path)})
        if found:
            return records, sources
    commit = ARCHIVES.get(hardware)
    if not commit:
        return records, sources
    folder = f"results/{hardware}/logs/autotune_logs"
    listed = subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"{commit}:{folder}"], capture_output=True, text=True)
    if listed.returncode == 0:
        for name in sorted(n for n in listed.stdout.split() if wanted(n)):
            shown = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{folder}/{name}"], capture_output=True)
            if shown.returncode == 0:
                records += json.loads(shown.stdout)
                sources.append({"source": f"git:{commit}:{folder}/{name}", "sha256": hashlib.sha256(shown.stdout).hexdigest()})
    elif network:
        for name in (f"{operator}_autotune.json", f"{operator}_tilelang_autotune.json",
                     f"{operator}_autotune_triton-cutile-tilelang.json", f"{operator}_autotune_triton.json"):
            url = f"https://raw.githubusercontent.com/{GITHUB}/{commit}/{folder}/{name}"
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
            except OSError:
                continue
            records += json.loads(data)
            sources.append({"source": url, "sha256": hashlib.sha256(data).hexdigest()})
    return records, sources


def winners_for(records, params, dtype):
    out = {}
    for record in records:
        logged = {k: str(v) for k, v in record.get("params", {}).items()}
        if record.get("dtype") == dtype and all(logged.get(k) == v for k, v in params.items()):
            for key, value in record.items():
                if key.endswith("_autotune_cfg"):
                    out[key[:-len("_autotune_cfg")]] = value
            out.setdefault("_logged_params", record.get("params"))
            out.setdefault("_problem_size", record.get("problem_size"))
    return out


def find_report(repo, dirs, hardware, operator, backend, dtype):
    name = f"{backend}_{dtype}.ncu-rep"
    candidates = [d / name for d in dirs] + [
        d / f"NVIDIA_{hardware}" / operator / name for d in dirs] + [
        repo / "outputs/ncu" / hardware / operator / name, repo / "evidence" / name]
    return next((p for p in candidates if p.exists()), None)


def download_report(hardware, operator, backend, dtype, revision, out):
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from hf_ncu_report import fetch
        return fetch(HfApi(), hf_hub_download, hardware, operator, backend, dtype, revision, out)[0]
    except ImportError:
        pass
    import os
    token = os.environ.get("HF_TOKEN")
    token_file = Path.home() / ".cache/huggingface/token"
    if not token and token_file.exists():
        token = token_file.read_text().strip()
    remote = f"NVIDIA_{hardware}/{operator}/{backend}_{dtype}.ncu-rep"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{remote}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    out.mkdir(parents=True, exist_ok=True)
    local = out / f"{backend}_{dtype}.ncu-rep"
    with urllib.request.urlopen(request, timeout=300) as response:
        local.write_bytes(response.read())
    if local.stat().st_size < 4096:
        text = local.read_text(errors="replace")
        local.unlink()
        raise RuntimeError(f"not a report: {text[:120]}")
    (out / "download.json").write_text(json.dumps(
        {"dataset": HF_DATASET, "requested_revision": revision, "remote_path": remote,
         "local_path": str(local), "bytes": local.stat().st_size, "sha256": sha256(local)}, indent=1) + "\n")
    return local


def instances(metric):
    if metric.num_instances() == 0 or not metric.has_correlation_ids():
        return []
    ids = metric.correlation_ids()
    out = []
    for i in range(metric.num_instances()):
        try:
            key = ids.as_uint64(i)
        except Exception:
            key = None
        if key is None or metric.kind() == metric.ValueKind_STRING:
            try:
                key = ids.as_string(i)
            except Exception:
                pass
        try:
            value = metric.as_uint64(i)
        except Exception:
            value = metric.as_double(i)
        out.append((key, value))
    return out


def by_label(action, names, name):
    if name not in names:
        return {}
    metric = action[name]
    ids = metric.correlation_ids()
    return {ids.as_string(i): metric.as_uint64(i) for i in range(metric.num_instances())}


def extract_backend(report_path, out):
    import ncu_report
    from ncu_extract import extract
    out.mkdir(parents=True, exist_ok=True)
    report = ncu_report.load_report(str(report_path.resolve()))
    scalars = extract(report, report_path)
    scalars["report_sha256"] = sha256(report_path)
    (out / "ncu.json").write_text(json.dumps(scalars, indent=1) + "\n")
    summary = {"report": str(report_path), "sha256": scalars["report_sha256"], "actions": []}
    values = {}
    for ri in range(report.num_ranges()):
        current = report.range_by_idx(ri)
        for ai in range(current.num_actions()):
            action = current.action_by_idx(ai)
            names = set(action.metric_names())
            tag = f"r{ri}a{ai}"
            executed = dict(instances(action["inst_executed"])) if "inst_executed" in names else {}
            samples = dict(instances(action["smsp__pcsamp_sample_count"])) if "smsp__pcsamp_sample_count" in names else {}
            stalls = {}
            for stall in STALLS:
                name = f"smsp__pcsamp_warps_issue_stalled_{stall}"
                if name in names:
                    for pc, value in instances(action[name]):
                        if value:
                            stalls.setdefault(pc, {})[stall] = int(value)
            rows = []
            for pc in sorted(set(executed) | set(samples) | set(stalls)):
                info = None
                try:
                    info = action.source_info(pc)
                except Exception:
                    pass
                rows.append({"pc": pc, "sass": action.sass_by_pc(pc),
                             "executed": int(executed.get(pc, 0)),
                             "samples": int(samples.get(pc, 0)),
                             "stalls": stalls.get(pc, {}),
                             "file": Path(info.file_name()).name if info else None,
                             "line": info.line() if info else None})
            total_samples = sum(r["samples"] for r in rows)
            with (out / f"annotated_sass_{tag}.txt").open("w") as stream:
                stream.write(f"# {action.name()}  ({report_path.name} range {ri} action {ai})\n")
                stream.write("# offset | warp executions | PC samples (% of kernel) | source line | SASS | stalled-warp samples\n")
                base = rows[0]["pc"] if rows else 0
                for r in rows:
                    pct = 100 * r["samples"] / total_samples if total_samples else 0
                    stall_text = " ".join(f"{k}={v}" for k, v in sorted(r["stalls"].items(), key=lambda kv: -kv[1]))
                    where = f"{r['file']}:{r['line']}" if r["file"] else "-"
                    stream.write(f"{r['pc'] - base:#07x} | {r['executed']:>12} | {r['samples']:>7} ({pct:5.1f}%) | {where} | {r['sass']} | {stall_text}\n")
            hot = sorted(rows, key=lambda r: -r["samples"])[:15]
            by_opcode, by_line = {}, {}
            for r in rows:
                tokens = r["sass"].split()
                if tokens and tokens[0].startswith("@"):
                    tokens = tokens[1:]
                opcode = tokens[0].split(".")[0] if tokens else "?"
                top = max(r["stalls"], key=r["stalls"].get) if r["stalls"] else "-"
                for table, key in ((by_opcode, opcode), (by_line, f"{r['file']}:{r['line']}" if r["file"] else "-")):
                    entry = table.setdefault(key, {"samples": 0, "executed": 0, "stalls": {}})
                    entry["samples"] += r["samples"]
                    entry["executed"] += r["executed"]
                    for k, v in r["stalls"].items():
                        entry["stalls"][k] = entry["stalls"].get(k, 0) + v
            stall_totals = {}
            for r in rows:
                for k, v in r["stalls"].items():
                    stall_totals[k] = stall_totals.get(k, 0) + v
            sleeps = [r["pc"] for r in rows if "NANOSLEEP" in r["sass"]]
            signal_wait = {"sites": 0, "samples": 0, "stalls": {}}
            after_trywait = False
            for r in rows:
                tokens = [t for t in r["sass"].split() if not t.startswith("@")]
                target = int(tokens[1], 16) if len(tokens) > 1 and tokens[0] == "BRA" and tokens[1].startswith("0x") else None
                r["signal_wait"] = bool(tokens) and (tokens[0].startswith(("NANOSLEEP", "SYNCS")) or (
                    target is not None and (after_trywait or any(target <= pc <= target + 0x40 for pc in sleeps))))
                after_trywait = bool(tokens) and tokens[0].startswith("SYNCS") and "TRYWAIT" in tokens[0]
                if r["signal_wait"] and r["samples"]:
                    signal_wait["sites"] += 1
                    signal_wait["samples"] += r["samples"]
                    for k, v in r["stalls"].items():
                        signal_wait["stalls"][k] = signal_wait["stalls"].get(k, 0) + v
            opcodes = by_label(action, names, "sass__inst_executed_per_opcode")
            modifiers = by_label(action, names, "sass__inst_executed_per_opcode_with_modifier_all")
            try:
                rules = list(action.rule_results_as_dicts())
            except Exception:
                rules = []
            sources = {}
            for path, text in action.source_files().items():
                sources[path] = len(text)
                if text:
                    target = out / "embedded_source" / tag / Path(path).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
            detail = {"kernel": action.name(), "range_index": ri, "action_index": ai,
                      "static_instructions": len(rows), "pc_samples": total_samples,
                      "per_pc_executed_sum": sum(r["executed"] for r in rows),
                      "stall_sample_totals": dict(sorted(stall_totals.items(), key=lambda kv: -kv[1])),
                      "signal_wait": signal_wait,
                      "samples_by_opcode": dict(sorted(by_opcode.items(), key=lambda kv: -kv[1]["samples"])[:12]),
                      "samples_by_source_line": dict(sorted(by_line.items(), key=lambda kv: -kv[1]["samples"])[:12]),
                      "hot_pcs": [{**r, "offset": f"{r['pc'] - base:#07x}", "pc": hex(r["pc"])} for r in hot],
                      "opcodes": dict(sorted(opcodes.items(), key=lambda kv: -kv[1])),
                      "opcodes_with_modifiers": dict(sorted(modifiers.items(), key=lambda kv: -kv[1])),
                      "rules": rules, "embedded_source_bytes": sources,
                      "metric_count": len(names),
                      "missing_families": [f for f, probe in [
                          ("aggregate stall ratios", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
                          ("per-PC stall samples", "smsp__pcsamp_sample_count"),
                          ("per-PC executed counts", "inst_executed"),
                          ("eligible warps", "smsp__warps_eligible.avg.per_cycle_active"),
                          ("L1 sector counters", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
                          ("bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")]
                          if probe not in names]}
            (out / f"detail_{tag}.json").write_text(json.dumps(detail, indent=1, default=str) + "\n")
            summary["actions"].append(detail)
            for _, name in TABLE:
                if name in names:
                    try:
                        values.setdefault(tag, {})[name] = action[name].value()
                    except Exception:
                        pass
            for name in ("device__attribute_display_name", "device__attribute_multiprocessor_count"):
                if name in names:
                    values.setdefault(tag, {})[name] = action[name].value()
    summary["table"] = values
    return summary


AMD_FILES = ("capture.json", "analysis/workload_csv/kernel.csv", "analysis/workload_csv/kernel_metric.csv",
             "analysis/pc_sampling_instructions.csv")
AMD_TABLES = ("System Speed-of-Light", "Wavefront / Wavefront Launch Stats")


def find_amd(repo, dirs, operator, backend, dtype, network, revision, out):
    name = f"{backend}_{dtype}"
    for folder in [d / "AMD_MI300X" / operator / name for d in dirs] + [repo / "outputs/rocprof_compute/MI300X" / operator / name]:
        if folder.is_dir():
            return folder
    if not network:
        return None
    from huggingface_hub import hf_hub_download
    for rel in AMD_FILES:
        hf_hub_download(HF_DATASET, f"AMD_MI300X/{operator}/{name}/{rel}", repo_type="dataset", revision=revision, local_dir=out / "download")
    return out / "download" / "AMD_MI300X" / operator / name


def extract_amd(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10 ** 9)
    present = [rel for rel in AMD_FILES if (folder / rel).exists()]
    capture = json.loads((folder / AMD_FILES[0]).read_text()) if AMD_FILES[0] in present else {}
    kernels = {}
    if AMD_FILES[1] in present:
        with (folder / AMD_FILES[1]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                kernels[row["kernel_name"]] = {"kernel": row["kernel_name"], "launches": number(row["dispatch_count"]),
                                               "duration_ns_mean": number(row["duration_ns_mean"]), "metrics": {}}
    everything = []
    if AMD_FILES[2] in present:
        with (folder / AMD_FILES[2]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                value = number(row["value"])
                if value is None:
                    continue
                base = " / ".join(dict.fromkeys(part for part in (row["table_name"], row["sub_table_name"], row["metric_name"]) if part))
                key = f"{base} [{row['value_name']}]"
                everything.append({"kernel": row["kernel_name"], "metric": key, "value": value, "unit": row["unit"]})
                if base.startswith(AMD_TABLES) and row["kernel_name"] in kernels:
                    kernels[row["kernel_name"]]["metrics"][key] = (value, row["unit"])
    (out / "amd_metrics.json").write_text(json.dumps({"folder": str(folder), "metrics": everything}, indent=1) + "\n")
    if AMD_FILES[3] in present:
        sampled = {}
        with (folder / AMD_FILES[3]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["operator_kernel"] == "True":
                    sampled.setdefault(row["kernel_name"], []).append(row)
        for name, found in sampled.items():
            kernel = kernels.setdefault(name, {"kernel": name, "launches": None, "duration_ns_mean": None, "metrics": {}})
            rows, stalls, kinds, by_line = [], {}, {}, {}
            for row in sorted(found, key=lambda r: int(r["offset"], 16)):
                reasons = {k: int(v) for k, v in json.loads(row["stall_reasons"] or "{}").items()}
                where = row["source_line"].rsplit("/", 1)[-1] if row["source_line"] else "-"
                hits = int(row["samples"])
                for k, v in reasons.items():
                    stalls[k] = stalls.get(k, 0) + v
                for k, v in json.loads(row["instruction_types"] or "{}").items():
                    kinds[k] = kinds.get(k, 0) + int(v)
                by_line[where] = by_line.get(where, 0) + hits
                rows.append({"offset": row["offset"], "samples": hits, "issued": int(row["issued"]), "stalled": int(row["stalled"]),
                             "stalls": reasons, "source": where, "asm": row["instruction"]})
            total = sum(r["samples"] for r in rows)
            safe = "".join(c if c.isalnum() else "_" for c in name)[:60]
            with (out / f"sampled_asm_{safe}.txt").open("w") as stream:
                stream.write(f"# {name}  ({folder})\n# only instructions that received a PC sample; no execution counts\n"
                             "# offset | samples (% of kernel) | issued | stalled | source line | instruction | stall reasons\n")
                for r in rows:
                    stream.write(f"{r['offset']} | {r['samples']:>6} ({100 * r['samples'] / total if total else 0:5.1f}%) | {r['issued']} | {r['stalled']} | "
                                 f"{r['source']} | {r['asm']} | {' '.join(f'{k}={v}' for k, v in r['stalls'].items())}\n")
            kernel.update({"samples": total, "issued": sum(r["issued"] for r in rows), "stalled": sum(r["stalled"] for r in rows),
                           "sampled_instructions": len(rows), "listing": f"sampled_asm_{safe}.txt",
                           "stall_samples": dict(sorted(stalls.items(), key=lambda kv: -kv[1])),
                           "instruction_type_samples": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
                           "samples_by_source_line": dict(sorted(by_line.items(), key=lambda kv: -kv[1])[:12]),
                           "hot": sorted(rows, key=lambda r: -r["samples"])[:12]})
    return {"folder": str(folder), "files": present, "missing": [rel for rel in AMD_FILES if rel not in present],
            "params": capture.get("params"), "winner": capture.get("winner"), "kernels": list(kernels.values())}


def survey_reports(repo, dirs, operator):
    try:
        import ncu_report
    except ImportError:
        return []
    found = {}
    for base in list(dirs) + [repo / "evidence", repo / "outputs/ncu"]:
        for path in sorted(Path(base).glob(f"*/{operator}/*.ncu-rep")):
            found.setdefault((path.parent.parent.name.replace("NVIDIA_", ""), path.stem), path)
    rows = []
    for (hardware, stem), path in sorted(found.items()):
        backend, _, dtype = stem.partition("_")
        try:
            report = ncu_report.load_report(str(path.resolve()))
        except Exception:
            continue
        actions = [report.range_by_idx(ri).action_by_idx(ai) for ri in range(report.num_ranges())
                   for ai in range(report.range_by_idx(ri).num_actions())]

        def value(action, name):
            return action[name].value() if name in set(action.metric_names()) else None

        def joined(name):
            return "/".join(str(value(a, name)) for a in actions)

        def total(name):
            values = [value(a, name) for a in actions]
            return sum(v for v in values if v is not None) if any(v is not None for v in values) else None

        duration = total("gpu__time_duration.sum")
        rows.append({"hardware": hardware, "backend": backend, "dtype": dtype, "kernels": len(actions),
                     "threads": joined("launch__block_size"), "regs": joined("launch__registers_per_thread"),
                     "shared": joined("launch__shared_mem_per_block"), "insts": total("smsp__inst_executed.sum"),
                     "shared_st": total("smsp__sass_inst_executed_op_shared_st.sum"),
                     "shared_conflicts": (total("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum") or 0)
                     + (total("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum") or 0),
                     "us": duration / 1000 if duration else None})
    return rows


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1e6 else f"{value:.6g}"
    return str(value)


def write_brief(out, case, extracted):
    lines = [f"# Case bundle: {case['hardware']} {case['operator']} {case['dtype']} {case['params_text']}", "",
             "Generated by tb_case.py from saved evidence. Benchmark latency comes from the CSV;",
             "NCU durations are profiler measurements and are not interchangeable with it.", ""]
    lat = case["latency_ms"]
    best = min(lat.values()) if lat else None
    lines += ["## Benchmark row", "", f"Source: `{case['csv']}` (row params `{case['params_text']}`, dtype `{case['dtype']}`)", "",
              "| Backend | Latency (ms) | vs fastest | Winner config |", "|---|---:|---:|---|"]
    for name, value in sorted(lat.items(), key=lambda kv: kv[1]):
        cfg = case["winners"].get(name)
        lines.append(f"| {name} | {value:g} | {value / best:.3f}x | {json.dumps(cfg) if cfg else '-'} |")
    if case.get("logged_params"):
        lines += ["", f"Full logged shape parameters: `{json.dumps(case['logged_params'])}`; problem_size {case['winners'].get('_problem_size')}"]
    lines += ["", f"Winner-log sources: {', '.join(s['source'] for s in case['winner_sources']) or 'NOT FOUND'}", ""]
    for title, key in (("Scale trend (same dtype, all shapes)", "scale_trend"),
                       ("Dtype trend (same shape)", "dtype_trend"),
                       ("Other hardware (same shape/dtype)", "hardware_trend")):
        rows = case[key]
        if not rows:
            continue
        cols = sorted({b for r in rows for b in r["latency_ms"]})
        lines += [f"## {title}", "", "| Case | " + " | ".join(f"{c} ms" for c in cols) + " | tilelang/triton | tilelang/cutile | winners |",
                  "|---|" + "---:|" * (len(cols) + 2) + "---|"]
        previous = None
        for r in rows:
            l = r["latency_ms"]
            ratio = lambda a, b: f"{l[a] / l[b]:.2f}" if a in l and b in l else "-"
            w = "; ".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in r.get("winners", {}).items() if not k.startswith("_"))
            w, previous = ("same as above" if w and w == previous else w), w
            lines.append(f"| {r['label']} | " + " | ".join(f"{l[c]:g}" if c in l else "-" for c in cols)
                         + f" | {ratio('tilelang', 'triton')} | {ratio('tilelang', 'cutile')} | {w} |")
        lines.append("")
    if extracted:
        columns = [(b, tag) for b, s in extracted.items() for tag in s["table"]]
        lines += ["## NCU side-by-side (exact metric names)", "",
                  "| Metric | " + " | ".join(f"{b} {t}" for b, t in columns) + " |", "|---|" + "---:|" * len(columns)]
        for label, name in TABLE:
            cells = [extracted[b]["table"][t].get(name) for b, t in columns]
            if any(c is not None for c in cells):
                lines.append(f"| {label} `{name}` | " + " | ".join(fmt(c) for c in cells) + " |")
        lines.append("")
        lines += ["## Per-kernel roll-up", "",
                  "| Backend | Kernel | NCU us | % of backend | Grid | Threads | Regs | Warp insts | Insts/warp | Insts/CTA | Global ld/warp | Global st/warp |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b, s in extracted.items():
            total = sum(t.get("gpu__time_duration.sum") or 0 for t in s["table"].values())
            for a in s["actions"]:
                tag = f"r{a['range_index']}a{a['action_index']}"
                t = s["table"].get(tag, {})
                grid, block = t.get("launch__grid_size"), t.get("launch__block_size")
                warps = grid * -(-block // 32) if grid and block else None
                per = lambda name: fmt(t[name] / warps) if warps and t.get(name) is not None else "n/a"
                d = t.get("gpu__time_duration.sum")
                lines.append(f"| {b} | `{a['kernel'][:36]}` {tag} | {fmt(d / 1000) if d else 'n/a'} | {fmt(100 * d / total) if d and total else 'n/a'} | "
                             f"{fmt(grid)} | {fmt(block)} | {fmt(t.get('launch__registers_per_thread'))} | {fmt(t.get('smsp__inst_executed.sum'))} | "
                             f"{per('smsp__inst_executed.sum')} | {fmt(t['smsp__inst_executed.sum'] / grid) if grid and t.get('smsp__inst_executed.sum') else 'n/a'} | {per('smsp__sass_inst_executed_op_global_ld.sum')} | {per('smsp__sass_inst_executed_op_global_st.sum')} |")
            csv_ms = case["latency_ms"].get(b)
            lines.append(f"| {b} | **sum of captured kernels** |  {fmt(total / 1000)} | | | | | | | | | CSV {fmt(csv_ms * 1000) if csv_ms else 'n/a'} us |")
        lines += ["", "Kernel durations are profiler times; use them to apportion a gap between kernels, and use the CSV for the gap itself. "
                  "A captured sum far below the CSV means untimed work (host, torch preprocessing, uncaptured kernels).", ""]
        size = case["winners"].get("_problem_size")
        firsts = {}
        for b, s in extracted.items():
            merged = {}
            for t in s["table"].values():
                for name, value in t.items():
                    if name.endswith(".sum") and isinstance(value, (int, float)):
                        merged[name] = merged.get(name, 0) + value
                    else:
                        merged.setdefault(name, value)
            firsts[b] = merged
        lines += ["## Normalized work and first-order time model", ""]
        if size:
            lines += [f"Per 1000 units of the logged problem_size ({size:,}). For some operators this is one dimension, not the element count: "
                      "compare the columns with each other and do not quote the absolute values unless problem_size matches the elements processed. "
                      "Counts are warp-level instructions.", "",
                      "| Per 1000 elements | " + " | ".join(firsts) + " |", "|---|" + "---:|" * len(firsts)]
            for label, name in (("Warp instructions", "smsp__inst_executed.sum"),
                                ("Global load insts", "smsp__sass_inst_executed_op_global_ld.sum"),
                                ("Global load L1 sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
                                ("Global load L1 miss sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"),
                                ("L2-to-L1 read bytes", "l1tex__m_xbar2l1tex_read_bytes.sum"),
                                ("Global store insts", "smsp__sass_inst_executed_op_global_st.sum"),
                                ("Shared load+store insts", "smsp__sass_inst_executed_op_shared.sum")):
                cells = [t.get(name) for t in firsts.values()]
                if any(c is not None for c in cells):
                    lines.append(f"| {label} | " + " | ".join(fmt(1000 * c / size) if c is not None else "n/a" for c in cells) + " |")
            opcodes = {}
            for b, s in extracted.items():
                merged = {}
                for a in s["actions"]:
                    for k, v in a["opcodes"].items():
                        merged[k] = merged.get(k, 0) + v
                opcodes[b] = merged
            ranked = sorted({k for o in opcodes.values() for k in o}, key=lambda k: -max(o.get(k, 0) for o in opcodes.values()))[:12]
            for k in ranked:
                lines.append(f"| `{k}` | " + " | ".join(fmt(1000 * o.get(k, 0) / size) if o else "not collected" for o in opcodes.values()) + " |")
            if any(len(s["actions"]) > 1 for s in extracted.values()):
                lines.append("")
                lines.append("Multi-kernel operator: the rows above sum every captured kernel per backend.")
            lines.append("")
        model = {b: (t.get("smsp__inst_executed.sum"), t.get("smsp__issue_active.avg.per_cycle_active"),
                     t.get("gpu__time_duration.sum")) for b, t in firsts.items()}
        usable = {b: v for b, v in model.items() if all(v)}
        if len(usable) > 1 and all(len(s["actions"]) == 1 for s in extracted.values()):
            ref = min(usable, key=lambda b: case["latency_ms"].get(b, float("inf")))
            ri, rr, rd = usable[ref]
            lines += [f"First-order check only: time is roughly instructions executed / issue rate. It reproduces the gap almost by construction and does not say why the issue rate differs; the sample table below does. Relative to {ref}:", "",
                      "| Backend | Instruction ratio | Issue-rate ratio | Predicted time ratio | NCU duration ratio | CSV ratio |", "|---|---:|---:|---:|---:|---:|"]
            for b, (i, r, d) in usable.items():
                csv_ratio = case["latency_ms"][b] / case["latency_ms"][ref] if b in case["latency_ms"] and ref in case["latency_ms"] else None
                lines.append(f"| {b} | {i / ri:.3f} | {r / rr:.3f} | {(i / ri) / (r / rr):.3f} | {d / rd:.3f} | {fmt(csv_ratio)} |")
            lines += ["", "A backend whose gap is mostly instruction ratio does extra work; one whose gap is mostly issue-rate ratio is waiting. "
                      "A large residual between predicted and measured means the model is missing something (multi-kernel, tails, launch).", ""]
            sampled = {b: s["actions"][0] for b, s in extracted.items() if s["actions"] and s["actions"][0].get("pc_samples")}
            if ref in sampled and len(sampled) > 1:
                reasons = sorted({k for a in sampled.values() for k in a["stall_sample_totals"]},
                                 key=lambda k: -max(a["stall_sample_totals"].get(k, 0) for a in sampled.values()))[:7]
                lines += ["### Where the extra time sits (PC samples)", "",
                          "Samples are taken at a fixed rate, so a sample count is time. The second row of each backend is its excess over "
                          f"{ref}: the stall reasons that hold the excess are where the gap is spent, whatever the instruction counts say.", "",
                          "| Backend | Samples | Samples/us | " + " | ".join(reasons) + " | no stall recorded | at signal waits |", "|---|---:|---:|" + "---:|" * (len(reasons) + 2)]
                base_a = sampled[ref]
                for b, a in sampled.items():
                    us = (firsts[b].get("gpu__time_duration.sum") or 0) / 1000
                    row = [a["stall_sample_totals"].get(k, 0) for k in reasons]
                    rest = a["pc_samples"] - sum(a["stall_sample_totals"].values())
                    lines.append(f"| {b} | {a['pc_samples']:,} | {fmt(a['pc_samples'] / us) if us else 'n/a'} | " + " | ".join(f"{v:,}" for v in row) + f" | {rest:,} | {a.get('signal_wait', {}).get('samples', 0):,} |")
                    if b != ref:
                        base_rest = base_a["pc_samples"] - sum(base_a["stall_sample_totals"].values())
                        lines.append(f"| {b} minus {ref} | {a['pc_samples'] - base_a['pc_samples']:+,} | | "
                                     + " | ".join(f"{v - base_a['stall_sample_totals'].get(k, 0):+,}" for k, v in zip(reasons, row)) + f" | {rest - base_rest:+,} | "
                                     f"{a.get('signal_wait', {}).get('samples', 0) - base_a.get('signal_wait', {}).get('samples', 0):+,} |")
                lines += ["", "The last column is not an extra reason: it counts the samples, already included under the reasons to its left, that sit on "
                          "`NANOSLEEP`/`SYNCS` sites or on the branch that follows a `TRYWAIT` or enters a sleep loop. Those are a warp waiting for a signal from another warp or from "
                          "a bulk (TMA) copy, usually recorded as long_scoreboard. Subtract them before reading long_scoreboard as waiting on a load; "
                          "each backend's section gives the split by reason.", ""]
        for backend, summary in extracted.items():
            backend_total = sum(t.get("gpu__time_duration.sum") or 0 for t in summary["table"].values())
            for a in summary["actions"]:
                tag = f"r{a['range_index']}a{a['action_index']}"
                share = (summary["table"].get(tag, {}).get("gpu__time_duration.sum") or 0) / backend_total if backend_total else 1
                if share < 0.05 and len(summary["actions"]) > 1:
                    lines += [f"## {backend}: `{a['kernel'][:60]}` ({tag})", "",
                              f"Minor kernel ({100 * share:.1f}% of this backend's captured time); details in `{backend}/detail_{tag}.json` and `{backend}/annotated_sass_{tag}.txt`.", ""]
                    continue
                lines += [f"## {backend}: `{a['kernel']}` ({tag})", "",
                          f"Report `{summary['report']}` sha256 `{summary['sha256'][:16]}`; {a['static_instructions']} static instructions, {a['pc_samples']} PC samples."]
                if a["missing_families"] and not a["static_instructions"]:
                    lines.append("**No SASS or PC samples in this capture.** If another hardware's report for the same backend exists, "
                                 "bundle it as a proxy for the generated code and label conclusions drawn from it as inferred.")
                if a["missing_families"]:
                    lines.append(f"**Reduced capture, not collected:** {', '.join(a['missing_families'])}.")
                empty = [p for p, n in a["embedded_source_bytes"].items() if not n]
                if empty:
                    lines.append(f"**Embedded source empty for:** {', '.join(empty)} (kernel body must be inferred from SASS/current source).")
                top = list(a["opcodes"].items())[:14]
                lines += ["", "Dynamic warp-level opcode executions: " + (", ".join(f"{k} {v:,}" for k, v in top) or "not collected in this capture"), ""]
                if a["stall_sample_totals"]:
                    lines += ["Stalled-warp samples by reason: " + ", ".join(f"{k} {v:,}" for k, v in a["stall_sample_totals"].items()), ""]
                waiting = a.get("signal_wait") or {}
                if waiting.get("samples"):
                    lines += [f"Of these, {waiting['samples']:,} samples sit on {waiting['sites']} signal-wait sites (`NANOSLEEP`/`SYNCS`, or the branch after a `TRYWAIT` or into a sleep loop): "
                              + ", ".join(f"{k} {v:,}" for k, v in sorted(waiting["stalls"].items(), key=lambda kv: -kv[1]))
                              + ". They are a warp waiting for a signal, not for a load.", ""]
                if a["hot_pcs"] and a["pc_samples"]:
                    lines += ["| Offset | % samples | Executions | Source | SASS | Top stalls |", "|---|---:|---:|---|---|---|"]
                    for r in a["hot_pcs"][:10]:
                        stall_text = ", ".join(f"{k} {v}" for k, v in sorted(r["stalls"].items(), key=lambda kv: -kv[1])[:3])
                        where = f"{r['file']}:{r['line']}" if r["file"] else "-"
                        lines.append(f"| {r['offset']} | {100 * r['samples'] / a['pc_samples']:.1f} | {r['executed']:,} | {where} | `{r['sass']}` | {stall_text} |")
                    lines.append("")
                for title, key in (("opcode", "samples_by_opcode"), ("source line", "samples_by_source_line")):
                    if a["pc_samples"] and a[key]:
                        cells = []
                        for name, entry in list(a[key].items())[:8]:
                            top = max(entry["stalls"], key=entry["stalls"].get) if entry["stalls"] else "-"
                            cells.append(f"{name} {100 * entry['samples'] / a['pc_samples']:.1f}% ({top})")
                        lines += [f"PC samples by {title} (dominant stall): " + ", ".join(cells), ""]
                by_exec = sorted(a["samples_by_source_line"].items(), key=lambda kv: -kv[1].get("executed", 0))[:8]
                if a.get("per_pc_executed_sum"):
                    lines += ["Warp-instruction executions by source line: " + ", ".join(
                        f"{name} {entry.get('executed', 0):,}" for name, entry in by_exec), ""]
                    counter = summary["table"].get(tag, {}).get("smsp__inst_executed.sum")
                    if counter and abs(a["per_pc_executed_sum"] - counter) / counter > 0.1:
                        lines += [f"**Counter mismatch:** per-PC executions sum to {a['per_pc_executed_sum']:,} but `smsp__inst_executed.sum` is {int(counter):,}. "
                                  "Spin-wait loops iterate differently across replay passes; use the counter for totals and treat per-line splits as approximate.", ""]
                load_counter = summary["table"].get(tag, {}).get("smsp__sass_inst_executed_op_global_ld.sum")
                if load_counter == 0 and a["opcodes"].get("LDG"):
                    lines += [f"**Load counter reads zero:** `smsp__sass_inst_executed_op_global_ld.sum` is 0 although `LDG` executed {a['opcodes']['LDG']:,} times "
                              "(seen with 256-bit loads). Use the opcode count or L1 sector counters for loads.", ""]
                for rule in a["rules"]:
                    message = rule.get("rule_message", {})
                    if message.get("title") and message["title"] not in NOISE_RULES:
                        lines.append(f"- NCU rule (hypothesis, verify against SASS): **{message['title']}**: {message.get('message', '')[:170]}")
                lines += ["", f"Full detail: `{backend}/detail_{tag}.json`, `{backend}/annotated_sass_{tag}.txt`, `{backend}/ncu.json`, `{backend}/embedded_source/`.", ""]
    for backend, amd in (case.get("amd") or {}).items():
        lines += [f"## {backend} on MI300X: rocprof-compute profile", "",
                  f"Folder `{amd['folder']}`. " + (f"Capture shape `{json.dumps(amd['params'])}`, winner `{json.dumps(amd['winner'])}` "
                  "(from the profile's own `capture.json`; use this config for the IR)." if amd["winner"] else
                  "No `capture.json` here, so take the shape and winner from the benchmark row above."),
                  "AMD terms: a wavefront is 64 work-items (the warp), a workgroup is the block, LDS is shared memory, "
                  "VALU/SALU are the vector and scalar ALUs, VMEM is global memory access, MFMA is the matrix unit.", ""]
        if amd["missing"]:
            lines += [f"**Not available:** {', '.join(amd['missing'])}. What they would give is absent below, not zero.", ""]
        for k in amd["kernels"]:
            lines += [f"### `{k['kernel']}`", "",
                      f"Launches {fmt(k.get('launches'))}, mean duration {fmt(k.get('duration_ns_mean'))} ns (profiler time, not benchmark latency).", ""]
            shown = {n: v for n, v in k["metrics"].items() if n.endswith("[Avg]")}
            if shown:
                lines += ["| Metric (exact name) | Value | Unit |", "|---|---:|---|"]
                lines += [f"| `{n}` | {fmt(v[0])} | {v[1]} |" for n, v in shown.items()] + [""]
            if k.get("samples"):
                lines += [f"PC samples: {k['samples']:,} over {k['sampled_instructions']} sampled instructions "
                          f"({k['issued']:,} issuing, {k['stalled']:,} stalled). Only instructions that received a sample are listed, "
                          "and there are no execution counts, so instruction totals cannot be derived here.", "",
                          "Stalled samples by reason: " + ", ".join(f"{a} {b:,}" for a, b in k["stall_samples"].items()), "",
                          "Samples by instruction type: " + ", ".join(f"{a} {b:,}" for a, b in k["instruction_type_samples"].items()), "",
                          "Samples by source line: " + ", ".join(f"{a} {b:,}" for a, b in k["samples_by_source_line"].items()), "",
                          "| Offset | % samples | Issued | Stalled | Source | Instruction | Stall reasons |", "|---|---:|---:|---:|---|---|---|"]
                for r in k["hot"]:
                    lines.append(f"| {r['offset']} | {100 * r['samples'] / k['samples']:.1f} | {r['issued']} | {r['stalled']} | {r['source']} | "
                                 f"`{r['asm'][:70]}` | {', '.join(f'{a} {b}' for a, b in r['stalls'].items())} |")
                lines += ["", f"Full listing: `{backend}/{k['listing']}`; every metric: `{backend}/amd_metrics.json`. "
                          "The complete assembly comes from `triton_ir.py --hardware MI300X` (`.amdgcn`); match by offset and source line.", ""]
    if case.get("all_reports"):
        lines += ["## Every saved report for this operator", "",
                  "Launch facts for each backend, dtype and hardware. A row whose block size, registers or shared-memory traffic "
                  "differs from its siblings is a natural experiment: the same backend compiled or tuned differently.", "",
                  "| Hardware | Backend | Dtype | Kernels | Threads | Regs | Shared B | Warp insts | Shared stores | Shared bank conflicts (ld+st) | NCU us |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in case["all_reports"]:
            lines.append(f"| {r['hardware']} | {r['backend']} | {r['dtype']} | {r['kernels']} | {r['threads']} | {r['regs']} | {r['shared']} | "
                         f"{fmt(r['insts'])} | {fmt(r['shared_st'])} | {fmt(r.get('shared_conflicts'))} | {fmt(r['us'])} |")
        lines.append("")
    if case.get("contrasts"):
        lines += ["## Suggested contrast cases", "",
                  "The gap changes materially in these neighbouring cases. Bundling the same backend there and diffing it "
                  "against this case is usually the fastest way to isolate what the gap depends on.", ""]
        lines += [f"- {c}" for c in case["contrasts"]] + [""]
    if case["notes"]:
        lines += ["## Bundle notes", ""] + [f"- {n}" for n in case["notes"]] + [""]
    (out / "brief.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware")
    parser.add_argument("operator")
    parser.add_argument("dtype")
    parser.add_argument("--params", help="CSV params selector, e.g. 'M=640'; default is the largest shape")
    parser.add_argument("--mode", default="autotune", choices=["autotune", "default"])
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="TileBench checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, action="append", default=[])
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()
    repo, out, notes = args.repo.resolve(), args.out, []
    out.mkdir(parents=True, exist_ok=True)
    csv_path, rows = read_csv(repo, args.hardware, args.operator, args.mode)
    if not rows:
        parser.error(f"No {args.mode} CSV for {args.operator} on {args.hardware} under {repo}/results")
    same_dtype = [r for r in rows if r.get("dtype") == args.dtype]
    if csv_path.parent == repo / "results" / "csv":
        notes.append(f"The benchmark CSV is `results/csv`, which does not name its hardware. Confirm it is a {args.hardware} run; "
                     "no other hardware's results are in this checkout, so there is no hardware trend table.")
    if not same_dtype:
        parser.error(f"dtype {args.dtype} not in {csv_path}; available: {sorted({r.get('dtype') for r in rows})}")
    records, sources = load_logs(repo, args.hardware, args.operator, not args.no_network)
    if not records:
        notes.append("No autotune winner log found locally, in git, or in the archive; configurations unknown.")
    if args.params:
        wanted = parse_params(args.params)
        chosen = [r for r in same_dtype if all(parse_params(r["params"]).get(k) == v for k, v in wanted.items())]
        if len(chosen) != 1:
            parser.error(f"--params matched {len(chosen)} rows; available: {[r['params'] for r in same_dtype]}")
        row = chosen[0]
    else:
        sized = [(winners_for(records, parse_params(r["params"]), args.dtype).get("_problem_size") or 0, i, r)
                 for i, r in enumerate(same_dtype)]
        row = max(sized, key=lambda t: (t[0], t[1]))[2]
        notes.append("Shape defaulted to the largest case, which is what the released NCU profiles capture. "
                     "Confirm against the captured grid dimensions before trusting the join.")

    def trend(hardware_rows, selector):
        out_rows = []
        for r in hardware_rows:
            if selector(r):
                out_rows.append({"label": f"{r['params']} {r['dtype']}", "latency_ms": latencies(r),
                                 "winners": winners_for(records, parse_params(r["params"]), r["dtype"])})
        return out_rows

    case = {"hardware": args.hardware, "operator": args.operator, "dtype": args.dtype, "mode": args.mode,
            "params_text": row["params"], "csv": str(csv_path.relative_to(repo)), "csv_sha256": sha256(csv_path),
            "latency_ms": latencies(row), "csv_row": row,
            "winners": winners_for(records, parse_params(row["params"]), args.dtype), "winner_sources": sources,
            "scale_trend": trend(rows, lambda r: r["dtype"] == args.dtype),
            "dtype_trend": trend(rows, lambda r: r["params"] == row["params"]),
            "hardware_trend": [], "notes": notes}
    for other in ("B200", "GH200", "MI300X"):
        if other != args.hardware:
            other_path, other_rows = read_csv(repo, other, args.operator, args.mode)
            if other_path == csv_path:
                continue
            for r in other_rows:
                if r["params"] == row["params"] and r.get("dtype") == args.dtype:
                    case["hardware_trend"].append({"label": other, "latency_ms": latencies(r)})
    here = case["latency_ms"]
    contrasts = []
    for a, b in (("tilelang", "triton"), ("tilelang", "cutile"), ("cutile", "triton")):
        if a not in here or b not in here:
            continue
        ratio = here[a] / here[b]
        for kind, rows_ in (("dtype", case["dtype_trend"]), ("hardware", case["hardware_trend"])):
            for r in rows_:
                l = r["latency_ms"]
                if a in l and b in l and r["label"] != f"{row['params']} {args.dtype}":
                    other = l[a] / l[b]
                    if max(other / ratio, ratio / other) > 1.25:
                        if kind == "dtype":
                            command = f"tb_case.py {args.hardware} {args.operator} {r['label'].split()[-1]} --backends {a},{b}"
                        else:
                            command = f"tb_case.py {r['label']} {args.operator} {args.dtype} --backends {a},{b}" if r["label"] in ("B200", "GH200") else "no NCU profiles on this hardware"
                        contrasts.append(f"{a}/{b} is {ratio:.2f} here but {other:.2f} for {kind} `{r['label']}` ({command})")
        series = sorted(r["latency_ms"][a] / r["latency_ms"][b] for r in case["scale_trend"]
                        if a in r["latency_ms"] and b in r["latency_ms"])
        if len(series) >= 5:
            median = series[len(series) // 2]
            if max(ratio / median, median / ratio) > 1.25:
                contrasts.append(f"{a}/{b} is {ratio:.2f} at this shape but the median over all shapes is {median:.2f}: "
                                 "the profiled shape is an outlier, so separate the steady gap from whatever is specific to this shape")
    case["contrasts"] = contrasts
    case["logged_params"] = case["winners"].get("_logged_params")
    extracted = {}
    if args.hardware in ("B200", "GH200"):
        for backend in [b for b in args.backends.split(",") if b]:
            path = find_report(repo, args.reports_dir + [repo / "evidence"], args.hardware, args.operator, backend, args.dtype)
            if path is None and not args.no_network:
                try:
                    path = download_report(args.hardware, args.operator, backend, args.dtype,
                                           args.dataset_revision, out / backend)
                except Exception as error:
                    notes.append(f"{backend}: no local report and download failed ({type(error).__name__}: {error}).")
            if path is None:
                notes.append(f"{backend}: no saved NCU report available for this dtype.")
                continue
            try:
                summary = extract_backend(Path(path), out / backend)
            except ImportError:
                notes.append("NCU Python API (ncu_report) not importable; reports located but not extracted.")
                break
            if summary["actions"]:
                extracted[backend] = summary
            else:
                notes.append(f"{backend}: {path} contains no profiled kernels (not a valid report?).")
    else:
        case["amd"] = {}
        for backend in [b for b in args.backends.split(",") if b in case["latency_ms"]]:
            try:
                folder = find_amd(repo, args.reports_dir, args.operator, backend, args.dtype, not args.no_network,
                                  args.dataset_revision, out / backend)
            except Exception as error:
                notes.append(f"{backend}: no local MI300X profile and download failed ({type(error).__name__}: {error}).")
                continue
            if folder is None:
                notes.append(f"{backend}: no saved MI300X profile available for this dtype.")
                continue
            case["amd"][backend] = extract_amd(folder, out / backend)
        notes.append(f"{args.hardware} profiles come from rocprof-compute, not Nsight Compute: sampled instructions carry no execution "
                     "counts, and the NCU counter names in ncu-analysis do not apply. Use the AMD section of that reference.")
    case["reports"] = {b: {"report": s["report"], "sha256": s["sha256"]} for b, s in extracted.items()}
    case["all_reports"] = survey_reports(repo, args.reports_dir, args.operator)
    (out / "case.json").write_text(json.dumps(case, indent=1) + "\n")
    write_brief(out, case, extracted)
    print(json.dumps({"brief": str(out / "brief.md"), "backends": list(extracted), "notes": notes}, indent=1))


if __name__ == "__main__":
    main()
