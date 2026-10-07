"""Saved-report extraction only; never launches GPU work."""

import ast
import csv
import hashlib
import json
from pathlib import Path

import ncu_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
METRICS = """
gpu__time_duration.sum launch__grid_dim_x launch__grid_dim_y launch__grid_dim_z
launch__grid_size launch__block_size launch__waves_per_multiprocessor
launch__registers_per_thread launch__shared_mem_per_block
launch__shared_mem_per_block_static launch__shared_mem_per_block_dynamic
launch__occupancy_limit_blocks launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem launch__occupancy_limit_warps
sm__maximum_warps_per_active_cycle_pct
sm__warps_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum dram__bytes_write.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct lts__throughput.avg.pct_of_peak_sustained_elapsed
smsp__inst_executed.sum smsp__issue_active.avg.pct_of_peak_sustained_active
smsp__warps_eligible.avg.per_cycle_active
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
sm__cycles_active.avg sm__cycles_active.max sm__cycles_active.min
device__attribute_multiprocessor_count device__attribute_display_name
""".split()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


records = []
inventory = []
for entry in json.loads((ROOT / "evidence/reports.json").read_text()):
    backend = entry["backend"]
    path = ROOT / entry["path"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == entry["sha256"], (backend, "hash mismatch")
    report = ncu_report.load_report(str(path))
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            identity = dict(backend=backend, report=entry["path"], range_index=ri,
                            action_index=ai, kernel=action.name())
            names = set(action.metric_names())
            inventory.append({**identity, "sha256": sha,
                              "dataset_revision": entry["dataset_revision"],
                              "missing_metrics": sorted(set(METRICS) - names),
                              "pm_metrics": sorted(n for n in names if n.startswith("pmsampling:")),
                              "pcsamp_metrics": sorted(n for n in names if "pcsamp" in n)})
            for name in METRICS:
                if name in names:
                    metric = action[name]
                    records.append({**identity, "metric": name, "value": metric.value(),
                                    "unit": metric.unit()})
            hotspots = []
            for name in sorted(names):
                if not name.startswith("smsp__pcsamp_"):
                    continue
                metric = action[name]
                records.append({**identity, "metric": name, "value": metric.value(),
                                "unit": metric.unit()})
                if "warps_issue_stalled" not in name or not metric.has_correlation_ids():
                    continue
                ids = metric.correlation_ids()
                for i in range(metric.num_instances()):
                    value = metric.as_uint64(i)
                    if not value:
                        continue
                    pc = ids.as_uint64(i)
                    records.append({**identity, "metric": name, "value": value,
                                    "unit": metric.unit(), "instance_index": i,
                                    "correlation_id": pc})
                    si = action.source_info(pc)
                    hotspots.append({"metric": name, "value": value, "pc": hex(pc),
                                     "sass": action.sass_by_pc(pc),
                                     "file": si.file_name() if si else None,
                                     "line": si.line() if si else None})
            dump(OUT / f"stall_hotspots_{backend}.json",
                 sorted(hotspots, key=lambda r: r["value"], reverse=True))
            dump(OUT / f"rules_{backend}.json", list(action.rule_results_as_dicts()))
            source_files = dict(action.source_files())
            dump(OUT / f"source_inventory_{backend}.json",
                 [{"file": k, "characters": len(v),
                   "sha256": hashlib.sha256(v.encode()).hexdigest() if v else None}
                  for k, v in source_files.items()])
            for name, content in source_files.items():
                basename = Path(name).name
                if basename in {"impl_triton.py", "impl_cutile.py", "tvm_kernels.cu",
                                "tcgen05mma.h", "tcgen_05.h", "barrier.h"} and content:
                    (OUT / f"embedded_{backend}_{basename}.txt").write_text(content)
                if basename in {"impl_triton.py", "impl_cutile.py"} and content:
                    current = ROOT / "tilebench/benchmarks/operators/batched_matmul" / basename
                    def functions(text):
                        return {n.name: ast.dump(n, include_attributes=False)
                                for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
                    old, new = functions(content), functions(current.read_text())
                    dump(OUT / f"source_correspondence_{backend}.json",
                         {"embedded_file": name, "current_file": str(current.relative_to(ROOT)),
                          "whole_file_equal": content == current.read_text(),
                          "function_matches": {name: old.get(name) == new.get(name)
                                               for name in sorted(set(old) | set(new))}})
            bases = action["launch__function_pcs"]
            listing = []
            ptx = []
            mappings = []
            for i in range(bases.num_instances()):
                base = bases.as_uint64(i)
                cor = bases.correlation_ids().as_uint64(i) if bases.has_correlation_ids() else None
                records.append({**identity, "metric": "launch__function_pcs", "value": base,
                                "unit": bases.unit(), "instance_index": i, "correlation_id": cor})
                # Instructions are 16 bytes; stop at the end of this saved function.
                for offset in range(0, 1024 * 1024, 16):
                    pc = base + offset
                    sass = action.sass_by_pc(pc)
                    if not sass:
                        break
                    listing.append(f"0x{pc:x} {sass.strip()}")
                    pt = action.ptx_by_pc(pc)
                    if pt:
                        ptx.append(f"0x{pc:x} {pt}")
                    si = action.source_info(pc)
                    if si:
                        mappings.append({"pc": hex(pc), "sass": sass.strip(),
                                         "file": si.file_name(), "line": si.line()})
                else:
                    raise RuntimeError("SASS extraction exceeded bound")
            (OUT / f"sass_{backend}.txt").write_text("\n".join(listing) + "\n")
            (OUT / f"ptx_{backend}.txt").write_text("\n".join(ptx) + "\n")
            dump(OUT / f"source_mapping_{backend}.json", mappings)

dump(OUT / "selected_records.json", records)
dump(OUT / "capture_inventory.json", inventory)
with (ROOT / "results/B200/csv/batched_matmul_autotune.csv").open() as stream:
    rows = [r for r in csv.DictReader(stream) if r["params"] == "M=640" and r["dtype"] == "fp16"]
assert len(rows) == 1
winners = {}
for path in ["evidence/batched_matmul_autotune.json", "evidence/batched_matmul_tilelang_autotune.json"]:
    winners[path] = [r for r in json.loads((ROOT / path).read_text())
                     if r["params"]["M"] == 640 and r["dtype"] == "fp16"]
dump(OUT / "selected_case.json", {"csv": rows[0], "winner_logs": winners,
                                 "tilelang_over_triton": float(rows[0]["tilelang_ms"]) / float(rows[0]["triton_ms"]),
                                 "tilelang_over_cutile": float(rows[0]["tilelang_ms"]) / float(rows[0]["cutile_ms"]),
                                 "tilelang_over_torch": float(rows[0]["tilelang_ms"]) / float(rows[0]["torch_ms"])})
print(f"Extracted {len(records)} selected records; all report hashes match.")
