"""Extract metrics, NCU rules, and dynamic opcodes from transpose reports."""

from __future__ import annotations

import json
from pathlib import Path

import ncu_report


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = RUN_DIR / "reports"
ANALYSIS_DIR = RUN_DIR / "analysis"

METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "grid_blocks": "launch__grid_size",
    "block_threads": "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_mem_per_block": "launch__shared_mem_per_block",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "alu_active_pct": "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "executed_warp_instructions": "inst_executed",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    "global_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "global_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "shared_instructions": "smsp__sass_inst_executed_op_shared.sum",
    "shared_load_instructions": "smsp__sass_inst_executed_op_shared_ld.sum",
    "shared_store_instructions": "smsp__sass_inst_executed_op_shared_st.sum",
    "shared_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "shared_load_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "shared_store_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "shared_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "shared_load_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "shared_store_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "long_scoreboard": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "short_scoreboard": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "mio_throttle": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
}


def safe(action, metric_name: str):
    try:
        return action[metric_name].value()
    except Exception:
        return None


def opcode_counts(action) -> dict[str, float]:
    metric = action["sass__inst_executed_per_opcode"]
    labels = metric.correlation_ids()
    counts = {}
    for index in range(metric.num_instances()):
        counts[labels.as_string(index)] = metric.as_double(index)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def rules(action) -> list[dict]:
    output = []
    for rule in action.rule_results_as_dicts():
        message = rule.get("rule_message", {})
        speedup = rule.get("speedup_estimation", {})
        output.append(
            {
                "identifier": rule.get("rule_identifier"),
                "title": message.get("title"),
                "message": message.get("message"),
                "estimated_speedup_pct": speedup.get("speedup"),
            }
        )
    return output


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype in ("fp16", "int8"):
        for backend in ("tilelang", "triton", "cutile"):
            report = REPORT_DIR / f"full_{backend}_{dtype}.ncu-rep"
            action = ncu_report.load_report(str(report)).range_by_idx(0).action_by_idx(0)
            rows.append(
                {
                    "backend": backend,
                    "dtype": dtype,
                    "kernel": action.name(),
                    **{key: safe(action, metric) for key, metric in METRICS.items()},
                    "opcodes": opcode_counts(action),
                    "rules": rules(action),
                }
            )

    (ANALYSIS_DIR / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    columns = ["backend", "dtype", *METRICS]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column)) for column in columns))
    (ANALYSIS_DIR / "summary.tsv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
