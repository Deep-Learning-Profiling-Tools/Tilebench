"""Assemble comparison records from the exact saved-input exports."""
import json
from pathlib import Path

OUT = Path("output")
records = json.loads((OUT / "selected_records.json").read_text())
csv = json.loads((OUT / "csv_comparison.json").read_text())
provenance = json.loads((OUT / "capture_provenance.json").read_text())
winners = json.loads((OUT / "winner_records.json").read_text())
names = ("tilelang", "triton", "cutile")

def observed(backends, metrics, opcodes=()):
    return [r for r in records if r["backend"] in backends and
            (r["metric"] in metrics or (r["metric"] == "sass__inst_executed_per_opcode"
                                        and r.get("correlation_id") in opcodes))]

def value(backend, metric):
    return next(r["value"] for r in records if r["backend"] == backend and r["metric"] == metric)

configs = {
    "tilelang": winners["2d_max_pooling_tilelang_autotune"]["tilelang_autotune_cfg"],
    "triton": winners["2d_max_pooling_autotune"]["triton_autotune_cfg"],
    "cutile": winners["2d_max_pooling_autotune"]["cutile_autotune_cfg"],
}
implementations = []
for backend in names:
    p = next(r for r in provenance["reports"] if r["backend"] == backend)
    implementations.append({
        "backend": backend,
        "benchmark": {"path": csv["csv"], "csv_line": csv["selected_row"]["csv_line"],
                      "column": backend + "_ms", "value": csv["selected_row"]["case"][backend + "_ms"], "unit": "ms"},
        "selected_configuration": configs[backend],
        "source_path": f"tilebench/benchmarks/operators/2d_max_pooling/impl_{backend}.py",
        "profile": {k: p[k] for k in ("path", "sha256", "dataset_revision", "hash_matches")},
        "action_identity": {k: p["actions"][0][k] for k in ("report", "range_index", "action_index", "kernel")},
        "source_correspondence": "inferred" if backend == "tilelang" else "observed kernel AST match",
        "selected_configuration_correspondence": (
            "1x512 inferred from grid and shape strides/output ownership; full source missing" if backend == "tilelang" else
            "1x512 supported by grid and emitted CTA-column shift 0x9; 128 threads observed" if backend == "triton" else
            "3/2/1/4/128 constant suffix and 512x80x3 grid match; occupancy hint 8 is recorded winner, not proven capture metadata"),
        "provenance_limits": "No complete capture-time revision/compiler/library/cache/replay manifest; reader version is not capture version.",
    })

mechanisms = [
    {
        "claim": "Triton performs more address/bounds and output-layout work than TileLang; net performance difference remains parity-scale.",
        "status": "inferred",
        "code_difference": ["impl_triton.py:9 runtime shape arguments and :39 masked loads", "impl_tilelang.py:45 specialized unrolled fragment",
                            "Saved Triton STSM/BAR/LDS.128/STG.E.128 versus TileLang scalar STG.E"],
        "expected_consequences": "More integer/predicate instructions and shared conversion, but fewer global-store instructions; similar input traffic.",
        "observations": observed(["tilelang", "triton"], ["smsp__inst_executed.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
            "smsp__sass_inst_executed_op_global_st.sum", "smsp__inst_executed_op_shared_stsm.sum", "smsp__sass_inst_executed_op_shared_ld.sum",
            "smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.sum.pct_of_peak_sustained_active"], ["ISETP", "IMAD", "BAR"]),
        "alternative": "Memory delivery/dependencies limit TileLang; vector-store benefit and overlap can hide Triton's extra instructions.",
        "limits": "No critical-path latency share or proven compiler defect; CSV TileLang/Triton=0.9140 is within +/-10% parity band.",
    },
    {
        "claim": "cuTile's selected generic gather/output layout entails extra indexing work and a lower register residency ceiling.",
        "status": "inferred",
        "code_difference": ["impl_cutile.py:43 generic 3D ct.gather and :46 ct.store", "4x128 ownership", "SASS IMAD.X/LEA bounds path and STS/BAR/LDS.128 output conversion"],
        "expected_consequences": "Higher instruction and ALU work, greater register allocation, fewer resident CTAs; tile still reduces CTA count and padded columns.",
        "observations": observed(list(names), ["smsp__inst_executed.sum", "launch__registers_per_thread", "launch__occupancy_limit_registers",
            "sm__warps_active.sum.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct",
            "sm__inst_executed_pipe_alu.sum.pct_of_peak_sustained_active", "launch__grid_size",
            "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum"], ["IMAD", "LEA", "STS", "BAR"]),
        "alternative": "Transaction/reuse inefficiency also contributes; eligible warp count is higher than TileLang despite lower residency, so occupancy alone cannot explain latency.",
        "limits": "Separate indexing, output conversion and residency time shares are unresolved; no spill observed and no config sweep performed.",
    },
    {
        "claim": "cuTile's row-interleaved ownership increases input sectors and HBM reads despite better L1 hit rate.",
        "status": "inferred",
        "code_difference": "Saved cuTile lane-index SASS splits tid into tid/4 columns and tid mod 4 rows; TileLang/Triton use adjacent output columns with stride-2 input access.",
        "expected_consequences": "More sector demand per global-load request; cache hit percentage need not imply less traffic.",
        "observations": observed(list(names), ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sector_hit_rate.pct", "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__cycles_active.avg.pct_of_peak_sustained_elapsed"]),
        "alternative": "Instruction scheduling and inter-CTA cache reuse can affect observed sectors/HBM and explain part of the difference; HBM traffic alone does not explain latency ratio.",
        "limits": "Request width/participation differs and replay/cache equivalence is unproven; no universal sectors/request ideal or temporal bandwidth share asserted.",
    },
]
result = {
    "case": {"hardware": "NVIDIA B200", "operator": "2d_max_pooling", "dtype": "fp32", "tuning_mode": "autotune",
             "shape": {"N": 4, "C": 128, "H": 640, "W": 640, "H_out": 320, "W_out": 320},
             "semantics": {"kernel_size": 3, "stride": 2, "padding": 1, "out_of_bounds_value": "-inf", "useful_outputs": 52428800}},
    "implementations": implementations,
    "timing": {"benchmark_ratios": csv["comparisons"], "profiler_durations": observed(list(names), ["gpu__time_duration.sum"]),
               "profiler_ratios": {"tilelang_over_triton": value("tilelang", "gpu__time_duration.sum") / value("triton", "gpu__time_duration.sum"),
                                   "tilelang_over_cutile": value("tilelang", "gpu__time_duration.sum") / value("cutile", "gpu__time_duration.sum")}},
    "mechanisms": mechanisms,
    "smallest_additional_evidence": "Matched capture manifest and per-PC dependency timeline; no collection authorized or performed.",
}
(OUT / "comparison_evidence.json").write_text(json.dumps(result, indent=2) + "\n")

coverage = []
for lens, metrics, conclusion, missing in [
    ("Useful work and instruction path", ["smsp__inst_executed.sum", "sass__inst_executed_per_opcode"],
     "Matched useful outputs, current/embedded code and SASS identify extra address/predicate/layout work; dynamic counts remain warp executions.", "Full TileLang capture source and compiler manifest unavailable."),
    ("Launch and residency", ["launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__occupancy_limit_registers", "sm__warps_active.sum.pct_of_peak_sustained_active"],
     "cuTile register allocation lowers residency ceiling; all grids have many waves.", "Residency's latency share unmeasured."),
    ("Memory movement and reuse", ["dram__bytes_read.sum", "dram__bytes_write.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"],
     "Equal TileLang/Triton input transactions; cuTile expands sectors/HBM traffic; shared conflicts do not rank the backends.", "Capture cache-control equivalence unproven."),
    ("Issue dependencies and synchronization", ["smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.sum.pct_of_peak_sustained_active", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
     "Output barriers observed; TileLang remains more exposed to load dependencies per issue. Ratios are not elapsed-time shares.", "Critical-path attribution unavailable."),
    ("Compute pipeline", ["sm__inst_executed_pipe_alu.sum.pct_of_peak_sustained_active"],
     "Higher integer/address work and ALU activity support cuTile issue-pressure contribution; max pooling requires no tensor cores.", "No isolated indexing latency measurement."),
    ("Balance and temporal behavior", ["launch__grid_size", "launch__waves_per_multiprocessor", "gpu__time_duration.sum"],
     "One pooling kernel per saved report matches source stages; large grids exclude a simple undersized-launch story, but cannot quantify tails.", "No temporal tail/critical-path analysis or host launch measurement performed."),
]:
    coverage.append({"lens": lens, "status": "insufficient" if lens == "Balance and temporal behavior" else "supported",
                     "metrics": metrics, "missing_metrics": [], "evidence": observed(list(names), metrics),
                     "conclusion": conclusion, "limits": missing})
(OUT / "diagnostic_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

required = {"metric", "value", "unit", "backend", "report", "range_index", "action_index", "kernel"}
assert all(required <= r.keys() for r in records)
assert all(r["hash_matches"] for r in provenance["reports"])
assert all(len(r["actions"]) == 1 for r in provenance["reports"])
assert csv["status"] == "ok"
assert all(not json.loads((OUT / f"{b}_metrics.json").read_text())["errors"] for b in names)
assert all(not json.loads((OUT / f"{b}_sass.json").read_text())["warnings"] for b in names)
print(json.dumps({"exact_records": len(records), "mechanisms": len(mechanisms), "coverage_lenses": len(coverage), "validation": "passed"}))
