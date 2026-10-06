"""Saved-input extraction only; this script does not execute kernels."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

import ncu_report

from tools.csv_ratios import compare_references
from tools.sass_listing import parse_listing

OUT = Path("output")
OP = Path("tilebench/benchmarks/operators/2d_max_pooling")
BACKENDS = ("tilelang", "triton", "cutile")

def kernel_ast(text):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "max_pool2d_kernel")
    return ast.dump(node, include_attributes=False)

provenance = {"reader_version": None, "reports": []}
records = []
opcode_records = []
selected_names = {
    "device__attribute_display_name", "device__attribute_compute_capability_major",
    "device__attribute_compute_capability_minor", "device__attribute_multiprocessor_count",
    "gpu__time_duration.sum", "smsp__inst_executed.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__bytes.sum.per_second",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed", "l1tex__t_sector_hit_rate.pct",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "lts__t_sector_hit_rate.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "launch__block_size",
    "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z", "launch__grid_size",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks", "launch__occupancy_limit_warps", "launch__waves_per_multiprocessor",
    "sm__warps_active.sum.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct",
    "smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.sum.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_alu.sum.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.sum.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum", "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_shared_stsm.sum", "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum", "profiler__replayer_passes",
}
for backend in BACKENDS:
    path = Path(f"evidence/{backend}_fp32.ncu-rep")
    report = ncu_report.load_report(str(path))
    provenance["reader_version"] = report.get_version()
    sources = []
    for ri in range(report.num_ranges()):
        for ai in range(report.range_by_idx(ri).num_actions()):
            action = report.range_by_idx(ri).action_by_idx(ai)
            metric = action["sass__inst_executed_per_opcode"]
            for index in range(metric.num_instances()):
                opcode_records.append({"backend": backend, "report": str(path), "range_index": ri,
                    "action_index": ai, "kernel": action.name(), "metric": metric.name(),
                    "instance_index": index, "correlation_id": metric.correlation_ids().value(index),
                    "value": metric.value(index), "unit": metric.unit()})
            for index, (original, source) in enumerate(dict(action.source_files()).items()):
                entry = {"original_path": original, "bytes": len(source),
                         "range_index": ri, "action_index": ai}
                if source and original.endswith(f"impl_{backend}.py"):
                    target = OUT / f"{backend}_embedded.py"
                    target.write_text(source)
                    current = (OP / f"impl_{backend}.py").read_text()
                    entry["saved_as"] = str(target)
                    entry["operator_kernel_ast_matches_current"] = kernel_ast(source) == kernel_ast(current)
                    entry["file_matches_current"] = source == current
                    diff = OUT / f"{backend}_source.diff"
                    diff.write_text("".join(difflib.unified_diff(source.splitlines(True), current.splitlines(True),
                        fromfile="embedded", tofile="current")))
                sources.append(entry)
    data = json.loads((OUT / f"{backend}_metrics.json").read_text())
    for metric in data["metrics"]:
        if metric["metric"] in selected_names:
            records.append({"backend": backend, **metric})
    listing = OUT / f"{backend}_sass.txt"
    if listing.exists():
        parsed = parse_listing(listing.read_text())
        (OUT / f"{backend}_sass.json").write_text(json.dumps(parsed, indent=2) + "\n")
    expected = next(x for x in json.loads(Path("evidence/reports.json").read_text()) if x["backend"] == backend)
    provenance["reports"].append({**expected, "hash_matches": expected["sha256"] == data["report_sha256"],
                                  "actions": data["actions"], "sources": sources})

(OUT / "selected_records.json").write_text(json.dumps(records + opcode_records, indent=2) + "\n")
(OUT / "opcode_records.json").write_text(json.dumps(opcode_records, indent=2) + "\n")
(OUT / "capture_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
csv = compare_references("results/B200/csv/2d_max_pooling_autotune.csv",
                        {"params": "H=640", "dtype": "fp32"}, "tilelang_ms", ["triton_ms", "cutile_ms"])
(OUT / "csv_comparison.json").write_text(json.dumps(csv, indent=2) + "\n")
selected = {}
for name in ("2d_max_pooling_autotune", "2d_max_pooling_tilelang_autotune"):
    rows = json.loads(Path(f"evidence/{name}.json").read_text())
    matches = [r for r in rows if r["params"]["H"] == 640 and r["dtype"] == "fp32"]
    assert len(matches) == 1
    selected[name] = matches[0]
(OUT / "winner_records.json").write_text(json.dumps(selected, indent=2) + "\n")
print(json.dumps({"reader_version": provenance["reader_version"], "records": len(records),
                  "source_checks": [{"backend": p["backend"], "hash_matches": p["hash_matches"],
                                     "sources": p["sources"]} for p in provenance["reports"]]}, indent=2))
