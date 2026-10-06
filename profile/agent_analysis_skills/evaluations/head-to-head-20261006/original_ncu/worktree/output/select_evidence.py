import ast
import csv
import json
from pathlib import Path

names = {
    "gpu__time_duration.sum", "device__attribute_multiprocessor_count",
    "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z", "launch__grid_size",
    "launch__block_size", "launch__registers_per_thread", "launch__registers_per_thread_allocated",
    "launch__occupancy_limit_registers", "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic", "launch__waves_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__pcsamp_sample_count", "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_selected", "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "sm__cycles_active.avg", "sm__cycles_active.max", "sm__cycles_active.min",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "memory_l2_theoretical_sectors_global", "memory_l2_theoretical_sectors_global_ideal",
}
records = []
inventory = []
for backend in ("tilelang", "triton", "cutile"):
    data = json.loads(Path(f"output/metrics_all_{backend}.json").read_text())
    expected = next(r for r in json.loads(Path("evidence/reports.json").read_text()) if r["backend"] == backend)
    assert data["report_sha256"] == expected["sha256"]
    inventory.append({"backend": backend, "report_sha256": data["report_sha256"], "actions": data["actions"]})
    for record in data["metrics"]:
        if record["metric"] in names:
            records.append({"backend": backend, **record})
    assert names <= {r["metric"] for r in records if r["backend"] == backend}
Path("output/selected_records.json").write_text(json.dumps(records, indent=2) + "\n")
Path("output/action_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

rows = [r for r in csv.DictReader(open("results/B200/csv/2d_max_pooling_autotune.csv"))
        if r["params"] == "H=640" and r["dtype"] == "fp32"]
assert len(rows) == 1
row = rows[0]
latencies = {b: float(row[f"{b}_ms"]) for b in ("tilelang", "triton", "cutile")}
csv_result = {"path": "results/B200/csv/2d_max_pooling_autotune.csv", "row": row,
              "latency_ms": latencies,
              "ordered_latency_ratios": {f"{a}/{b}": latencies[a]/latencies[b]
                                         for a in latencies for b in latencies if a != b},
              "stored_triton_vs_cutile": {"value": row["triton_vs_cutile"],
                  "actual_definition": "cutile_ms / triton_ms",
                  "calculated": latencies["cutile"]/latencies["triton"]}}
Path("output/csv_case.json").write_text(json.dumps(csv_result, indent=2) + "\n")

def kernel_body(text):
    tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "max_pool2d_kernel")
    return ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False)

correspondence = []
for backend in ("triton", "cutile"):
    embedded = json.loads(Path(f"output/source_files_{backend}.json").read_text())
    path, text = next((p, t) for p, t in embedded.items() if p.endswith(f"impl_{backend}.py"))
    current_path = f"tilebench/benchmarks/operators/2d_max_pooling/impl_{backend}.py"
    current = Path(current_path).read_text()
    correspondence.append({"backend": backend, "embedded_path": path,
                           "current_path": current_path, "whole_file_equal": current == text,
                           "kernel_body_ast_equal": kernel_body(current) == kernel_body(text)})
correspondence.append({"backend": "tilelang", "kernel_body_ast_equal": None,
                       "reason": "Generated CUDA is named in line information but has no embedded content; geometry and SASS constants support correspondence only."})
Path("output/source_correspondence.json").write_text(json.dumps(correspondence, indent=2) + "\n")
print(json.dumps({"records":len(records), "csv":csv_result, "correspondence":correspondence}, indent=2))
