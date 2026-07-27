"""Compare the four matmul kernels: SOL, tensor-pipe, stalls, occupancy."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
import ncu_report  # noqa: E402

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "analysis"
BACKENDS = ("tilelang", "tilelang_ws", "triton", "cutile")
import os
PREFIX = os.environ.get("PREFIX", "")

KEY = {
    "duration_us": "gpu__time_duration.sum",
    "cycles": "gpc__cycles_elapsed.max",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "waves": "launch__waves_per_multiprocessor",
    "regs_per_thread": "launch__registers_per_thread",
    "smem_per_block": "launch__shared_mem_per_block_static",
    "smem_dynamic": "launch__shared_mem_per_block_dynamic",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_hmma_pct": "sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_active",
    "tensor_pipe_imma_pct": "sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_active",
    "mem_tensor_active_pct": "sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "shared_pipe_pct": "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active",
    "alu_pct": "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "lsu_pct": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "adu_pct": "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    "inst_executed_warp": "smsp__inst_executed.sum",
    "ipc_active": "sm__inst_executed.avg.per_cycle_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "active_warps_per_sched": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "dram_bytes_read": "dram__bytes_read.sum",
}

STALLS = [
    "barrier", "long_scoreboard", "short_scoreboard", "wait", "no_instruction",
    "not_selected", "math_pipe_throttle", "mio_throttle", "lg_throttle",
    "dispatch_stall", "drain", "branch_resolving", "membar", "imc_miss",
    "tex_throttle", "misc", "sleeping",
]


def val(action, name):
    m = action.metric_by_name(name)
    if m is None:
        return None
    try:
        return m.value()
    except Exception:
        return None


def opcodes(action):
    m = action.metric_by_name("sass__inst_executed_per_opcode")
    out = Counter()
    if m is None:
        return out
    try:
        for i in range(m.num_instances()):
            out[m.correlation_ids().as_string(i)] += m.as_double(i)
    except Exception:
        pass
    return out


def load(p):
    r = ncu_report.load_report(str(p))
    rng = r.range_by_idx(0)
    return rng.action_by_idx(rng.num_actions() - 1)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def main():
    OUT.mkdir(exist_ok=True)
    lines = []
    for dt in ("fp16", "fp8"):
        acts = {}
        for b in BACKENDS:
            p = RUN / "reports" / f"{PREFIX}{b}_{dt}.ncu-rep"
            if p.exists():
                acts[b] = load(p)
        if not acts:
            continue
        names = list(acts)
        w = 26
        lines.append(f"### {dt}  (M=N={os.environ.get('SHAPE_M','4096')}, K={os.environ.get('SHAPE_K','2048')}, matched tuned tile)\n")
        lines.append("metric".ljust(w) + "".join(n.rjust(15) for n in names))
        for label, metric in KEY.items():
            lines.append(label.ljust(w) + "".join(fmt(val(acts[n], metric)).rjust(15) for n in names))
        lines.append("")
        lines.append("stall (warps/issue-active)".ljust(w) + "".join(n.rjust(15) for n in names))
        for s in STALLS:
            mn = f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"
            vals = {n: val(acts[n], mn) for n in names}
            if any(v for v in vals.values()):
                lines.append(s.ljust(w) + "".join(fmt(vals[n]).rjust(15) for n in names))
        lines.append("")
        hists = {n: opcodes(acts[n]) for n in names}
        keys = Counter()
        for h in hists.values():
            keys.update(h)
        if keys:
            lines.append("dynamic warp inst by opcode".ljust(w) + "".join(n.rjust(15) for n in names))
            for op, _ in keys.most_common(22):
                lines.append(op.ljust(w) + "".join(f"{int(hists[n].get(op, 0)):,}".rjust(15) for n in names))
            lines.append("TOTAL".ljust(w) + "".join(f"{int(sum(hists[n].values())):,}".rjust(15) for n in names))
        lines.append("")
    text = "\n".join(lines)
    (OUT / f"ncu_metrics{PREFIX and '_' + PREFIX.strip('_')}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
