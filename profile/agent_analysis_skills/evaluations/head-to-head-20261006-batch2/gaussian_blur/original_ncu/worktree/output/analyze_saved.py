"""Bounded offline extraction for the supplied gaussian-blur case."""
import csv
import hashlib
import json
from pathlib import Path

import ncu_report

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
TAGS = ("tilelang", "triton", "cutile")
KEYS = (
    "gpu__time_duration.sum", "launch__grid_dim_x", "launch__grid_dim_y",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
    "launch__shared_mem_per_block_static", "launch__occupancy_limit_registers",
    "launch__waves_per_multiprocessor", "launch__sm_count",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum", "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__icc_request_hit_rate.pct", "sm__cycles_active.avg",
    "sm__cycles_active.min", "sm__cycles_active.max",
    "derived__memory_l2_theoretical_sectors_global_excessive",
) + tuple("smsp__average_warps_issue_stalled_" + s + "_per_issue_active.ratio"
          for s in ("long_scoreboard", "lg_throttle", "wait", "math_pipe_throttle",
                    "not_selected", "barrier", "short_scoreboard", "dispatch_stall"))
records = []
inventory = []
summary = {}
for tag in TAGS:
    path = "evidence/" + tag + "_fp32.ncu-rep"
    rep = ncu_report.load_report(str(ROOT / path))
    for ri in range(rep.num_ranges()):
        rng = rep.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            a = rng.action_by_idx(ai)
            ident = dict(backend=tag, report=path, range_index=ri,
                         action_index=ai, kernel=a.name())
            names = list(a.metric_names())
            inventory.append({**ident, "metric_count": len(names),
                              "pmsampling_metrics": [n for n in names if n.startswith("pmsampling:")]})
            summary[tag] = {}
            for name in KEYS:
                if name in names:
                    m = a[name]
                    records.append({**ident, "metric": name, "value": m.value(), "unit": m.unit()})
                    summary[tag][name] = m.value()
            rules = a.rule_results_as_dicts()
            (OUT / ("rules_" + tag + ".json")).write_text(json.dumps(rules, indent=2))
            sources = dict(a.source_files())
            (OUT / ("source_inventory_" + tag + ".json")).write_text(json.dumps(sources, indent=2))
            for idx, (name, content) in enumerate(sources.items()):
                if content:
                    (OUT / ("embedded_" + tag + "_" + str(idx) + Path(name).suffix)).write_text(content)
            pcs = set()
            hotspots = []
            for name in names:
                if not name.startswith("smsp__pcsamp_"):
                    continue
                m = a[name]
                records.append({**ident, "metric": name, "value": m.value(), "unit": m.unit()})
                if not m.has_correlation_ids():
                    continue
                ids = m.correlation_ids()
                instances = []
                for i in range(m.num_instances()):
                    pc = ids.as_uint64(i)
                    value = m.as_uint64(i)
                    pcs.add(pc)
                    if value:
                        rec = {**ident, "metric": name, "value": value, "unit": m.unit(),
                               "instance_index": i, "correlation_id": pc}
                        records.append(rec)
                        si = a.source_info(pc)
                        instances.append({**rec, "sass": a.sass_by_pc(pc),
                                          "source_file": si.file_name() if si else None,
                                          "source_line": si.line() if si else None})
                instances.sort(key=lambda r: r["value"], reverse=True)
                hotspots.extend(instances[:5])
            (OUT / ("hotspots_" + tag + ".json")).write_text(json.dumps(hotspots, indent=2))
            sass = ["Kernel Name: " + a.name()]
            for pc in sorted(pcs):
                instruction = a.sass_by_pc(pc)
                si = a.source_info(pc)
                source = f"  {si.file_name()}:{si.line()}" if si else ""
                if instruction:
                    sass.append(f"0x{pc:x} {instruction.strip()}{source}")
            (OUT / ("sass_" + tag + ".txt")).write_text("\n".join(sass) + "\n")

(OUT / "selected_records.json").write_text(json.dumps(records, indent=2) + "\n")
(OUT / "action_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
(OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
manifest = json.loads((ROOT / "evidence/reports.json").read_text())
hashes = [{**row, "actual_sha256": hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()}
          for row in manifest]
(OUT / "hash_verification.json").write_text(json.dumps(hashes, indent=2) + "\n")
with (ROOT / "results/B200/csv/gaussian_blur_autotune.csv").open() as f:
    case = next(r for r in csv.DictReader(f) if r["params"] == "input_rows=10240" and r["dtype"] == "fp32")
winners = []
for name in ("gaussian_blur_autotune.json", "gaussian_blur_tilelang_autotune.json"):
    winners.extend(r for r in json.loads((ROOT / "evidence" / name).read_text())
                   if r["params"]["input_rows"] == 10240 and r["dtype"] == "fp32")
(OUT / "benchmark_case.json").write_text(json.dumps({"csv": case, "winners": winners}, indent=2) + "\n")
for tag in TAGS:
    print(tag, {k: v for k, v in summary[tag].items() if k in KEYS[:12]})
print("Records:", len(records))
