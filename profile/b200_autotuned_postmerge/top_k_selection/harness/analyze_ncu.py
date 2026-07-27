"""Extract key metrics, dynamic opcode mixes and stall reasons from the NCU reports."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
import ncu_report  # noqa: E402

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "analysis"

BACKENDS = ("tilelang", "triton", "cutile")

KEY = {
    "duration_us": "gpu__time_duration.sum",
    "cycles": "gpc__cycles_elapsed.max",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "regs_per_thread": "launch__registers_per_thread",
    "inst_executed_warp": "smsp__inst_executed.sum",
    "inst_thread": "smsp__thread_inst_executed.sum",
    "ipc": "sm__inst_executed.avg.per_cycle_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "alu_active_pct": "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "xu_active_pct": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "fma_active_pct": "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "lsu_active_pct": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "branch_inst": "sm__sass_branch_targets.sum",
    "branch_div": "sm__sass_branch_targets_threads_divergent.sum",
    "l1_sectors_global_ld": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1_sectors_global_st": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1_requests_global_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1_requests_global_st": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
}

STALLS = [
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
]


def val(action, name):
    m = action.metric_by_name(name)
    if m is None:
        return None
    try:
        return m.value()
    except Exception:
        return None


def opcode_counts(action, metric="sass__inst_executed_per_opcode") -> Counter:
    m = action.metric_by_name(metric)
    if m is None:
        return Counter()
    out: Counter = Counter()
    try:
        for i in range(m.num_instances()):
            name = m.correlation_ids().as_string(i)
            out[name] += m.as_double(i)
    except Exception:
        pass
    return out


def load(path: Path):
    r = ncu_report.load_report(str(path))
    rng = r.range_by_idx(0)
    # pick the bitonic step kernel (the only profiled launch)
    return rng.action_by_idx(rng.num_actions() - 1)


def base(op: str) -> str:
    return op.split(".")[0]


def table(title: str, rows: dict[str, dict[str, object]], backends) -> str:
    lines = [f"### {title}", ""]
    width = max(len(k) for k in rows) + 2
    lines.append("metric".ljust(width) + "".join(b.rjust(16) for b in backends))
    for k, per in rows.items():
        cells = []
        for b in backends:
            v = per.get(b)
            if isinstance(v, float):
                cells.append(f"{v:,.3f}".rjust(16))
            elif v is None:
                cells.append("-".rjust(16))
            else:
                cells.append(f"{v:,}".rjust(16) if isinstance(v, int) else str(v).rjust(16))
        lines.append(k.ljust(width) + "".join(cells))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = []
    dump = {}

    for tag in ("highN", "lowN"):
        actions = {}
        for b in BACKENDS:
            p = RUN / "reports" / f"full_{b}_{tag}.ncu-rep"
            if p.exists():
                actions[b] = load(p)
        present = list(actions)

        rows = {}
        for label, metric in KEY.items():
            rows[label] = {b: val(a, metric) for b, a in actions.items()}
        report.append(table(f"Key metrics ({tag})", rows, present))

        srows = {}
        for metric in STALLS:
            label = metric.split("issue_stalled_")[1].split("_per_issue")[0]
            vals = {b: val(a, metric) for b, a in actions.items()}
            if any(v for v in vals.values()):
                srows[label] = vals
        report.append(table(f"Warp stall reasons, warps per issue-active cycle ({tag})", srows, present))

        # dynamic opcode counts
        hists = {b: opcode_counts(a) for b, a in actions.items()}
        keys = Counter()
        for h in hists.values():
            keys.update(h)
        lines = [f"### Dynamic executed warp instructions per opcode ({tag})", ""]
        width = max((len(k) for k in keys), default=10) + 2
        lines.append("opcode".ljust(width) + "".join(b.rjust(16) for b in present))
        for op, _ in keys.most_common(40):
            lines.append(
                op.ljust(width)
                + "".join(f"{int(hists[b].get(op, 0)):,}".rjust(16) for b in present)
            )
        lines.append(
            "TOTAL".ljust(width)
            + "".join(f"{int(sum(hists[b].values())):,}".rjust(16) for b in present)
        )
        lines.append("")
        report.append("\n".join(lines))

        # grouped by base opcode
        g = {}
        for b, h in hists.items():
            gg: Counter = Counter()
            for op, c in h.items():
                gg[base(op)] += c
            g[b] = gg
        gkeys = Counter()
        for gg in g.values():
            gkeys.update(gg)
        lines = [f"### Dynamic warp instructions, grouped by base opcode ({tag})", ""]
        width = max((len(k) for k in gkeys), default=10) + 2
        lines.append("opcode".ljust(width) + "".join(b.rjust(16) for b in present))
        for op, _ in gkeys.most_common(30):
            lines.append(
                op.ljust(width)
                + "".join(f"{int(g[b].get(op, 0)):,}".rjust(16) for b in present)
            )
        lines.append("")
        report.append("\n".join(lines))

        dump[tag] = {
            "key": {k: {b: rows[k][b] for b in present} for k in rows},
            "stalls": {k: {b: srows[k][b] for b in present} for k in srows},
            "opcodes": {b: {k: int(v) for k, v in hists[b].items()} for b in present},
        }

    text = "\n".join(report)
    (OUT / "ncu_metrics.txt").write_text(text)
    (OUT / "ncu_metrics.json").write_text(json.dumps(dump, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
