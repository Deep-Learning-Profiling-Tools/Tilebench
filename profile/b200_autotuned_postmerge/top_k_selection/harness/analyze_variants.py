"""Side-by-side table for the four TileLang variants vs Triton / cuTile."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ir import base, ptx_hist, sass_hist  # noqa: E402
from analyze_ncu import KEY, load, opcode_counts, val  # noqa: E402

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "analysis"

COLS = [
    ("v0_baseline", RUN / "reports/variant_v0_baseline.ncu-rep", "ir/variants/tilelang_v0_baseline_N1048576"),
    ("v1_shift", RUN / "reports/variant_v1_shift.ncu-rep", "ir/variants/tilelang_v1_shift_N1048576"),
    ("v3_novalid_only", RUN / "reports/variant_v3_novalid_only.ncu-rep", "ir/variants/tilelang_v3_novalid_only_N1048576"),
    ("v2_shift_novalid", RUN / "reports/variant_v2_shift_novalid.ncu-rep", "ir/variants/tilelang_v2_shift_novalid_N1048576"),
    ("triton", RUN / "reports/full_triton_highN.ncu-rep", "ir/triton_N1048576"),
    ("cutile", RUN / "reports/full_cutile_highN.ncu-rep", None),
]

METRICS = [
    "duration_us", "cycles", "inst_executed_warp", "ipc", "sm_throughput_pct",
    "mem_throughput_pct", "alu_active_pct", "xu_active_pct", "occupancy_pct",
    "l1_requests_global_ld", "l1_sectors_global_ld",
]
OPCODES = ["IMAD", "ISETP", "MUFU", "I2F", "F2I", "BRA", "BSSY", "BSYNC",
           "IADD3", "SEL", "LOP3", "VIADD", "LEA", "IABS", "LDG", "STG"]


def main() -> None:
    acts = {n: load(p) for n, p, _ in COLS}
    names = [n for n, _, _ in COLS]
    w = 24
    lines = ["### NCU metrics: TileLang variants vs Triton / cuTile (N = 1,048,576, stride 16)", ""]
    lines.append("metric".ljust(w) + "".join(n.rjust(18) for n in names))
    for m in METRICS:
        lines.append(
            m.ljust(w)
            + "".join(f"{val(acts[n], KEY[m]):,.1f}".rjust(18) for n in names)
        )

    hists = {n: opcode_counts(acts[n]) for n in names}
    grouped = {}
    for n, h in hists.items():
        g: Counter = Counter()
        for op, c in h.items():
            g[base(op)] += c
        grouped[n] = g
    lines += ["", "### Dynamic warp instructions by base opcode", ""]
    lines.append("opcode".ljust(w) + "".join(n.rjust(18) for n in names))
    for op in OPCODES:
        lines.append(op.ljust(w) + "".join(f"{int(grouped[n].get(op, 0)):,}".rjust(18) for n in names))
    lines.append("TOTAL".ljust(w) + "".join(f"{int(sum(hists[n].values())):,}".rjust(18) for n in names))

    lines += ["", "### Static code size", ""]
    lines.append(
        "kind".ljust(w) + "".join(n.rjust(18) for n in names)
    )
    ptx_tot, ptx_div, sass_tot = {}, {}, {}
    for n, _, stem in COLS:
        if stem and (RUN / (stem + ".ptx")).exists():
            p = ptx_hist(RUN / (stem + ".ptx"))
            ptx_tot[n] = sum(p.values())
            ptx_div[n] = sum(c for o, c in p.items() if o.startswith(("div.", "rem.")))
        else:
            ptx_tot[n] = ptx_div[n] = None
        sass_path = RUN / ((stem + ".sass") if stem else "ir/cutile_N1048576.sass")
        sass_tot[n] = sum(sass_hist(sass_path).values())
    for label, d in (("ptx_instructions", ptx_tot), ("ptx_div_rem", ptx_div), ("sass_instructions", sass_tot)):
        lines.append(
            label.ljust(w)
            + "".join(("-" if d[n] is None else f"{d[n]:,}").rjust(18) for n in names)
        )
    lines.append("")

    text = "\n".join(lines)
    (OUT / "variant_comparison.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
