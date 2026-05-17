"""Extract metrics from .ncu-rep files and generate per-op comparison.md
plus a global tilebench_run/ncu/SUMMARY.md.

For each report we pull:
  - Duration (us)
  - Memory Throughput %, DRAM %, L1/TEX %, L2 %, Compute (SM) %
  - Achieved memory bandwidth (Tbyte/s or Gbyte/s)
  - Block Size, Registers/Thread, Static/Dynamic Shared Mem per Block,
    Block Limit (registers, shmem)
  - SOLBottleneck rule text (NCU's own verdict)
"""
import csv
import io
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
NCU = "/usr/local/cuda/bin/ncu"
NCU_DIR = ROOT / "tilebench_run" / "ncu"

WANTED = {
    "Duration", "Memory Throughput", "DRAM Throughput",
    "L1/TEX Cache Throughput", "L2 Cache Throughput",
    "Compute (SM) Throughput",
    "Block Size", "Registers Per Thread",
    "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block",
    "Block Limit Registers", "Block Limit Shared Mem",
}


def parse_report(rep: Path) -> dict:
    proc = subprocess.run(
        [NCU, "--import", str(rep), "--csv", "--page", "details"],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[-400:]}

    metrics: dict[str, tuple[str, str]] = {}
    bottleneck = None
    bw_value = None
    bw_unit = None
    rdr = csv.DictReader(io.StringIO(proc.stdout))
    for row in rdr:
        m = (row.get("Metric Name") or "").strip()
        u = (row.get("Metric Unit") or "").strip()
        v = (row.get("Metric Value") or "").strip()
        rule = (row.get("Rule Name") or "").strip()
        desc = (row.get("Rule Description") or "").strip()
        if m == "Memory Throughput" and u in {"Tbyte/s", "Gbyte/s"}:
            bw_value, bw_unit = v, u
            continue
        if rule == "SOLBottleneck" and desc and bottleneck is None:
            bottleneck = desc[:300]
            continue
        if m in WANTED and m not in metrics:
            metrics[m] = (v, u)
    metrics["__bottleneck__"] = (bottleneck or "", "")
    if bw_value is not None:
        metrics["__memory_bandwidth__"] = (bw_value, bw_unit)
    return metrics


def duration_us(metrics: dict) -> float | None:
    """Return Duration in microseconds regardless of NCU's reported unit."""
    if not metrics or metrics.get("error") or "Duration" not in metrics:
        return None
    v_str, unit = metrics["Duration"]
    try:
        v = float(v_str.replace(",", ""))
    except Exception:
        return None
    scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}.get(unit, 1.0)
    return v * scale


def fmt(metrics: dict, key: str) -> str:
    if key not in metrics:
        return "—"
    v, u = metrics[key]
    if not v:
        return "—"
    v_str = v.replace(",", "")
    if u in {"%", "us", "byte", "byte/block", "Kbyte/block",
             "register/thread", "block", "Kbyte", "Gbyte/s", "Tbyte/s"}:
        return f"{v_str} {u}".strip() if u else v_str
    return v_str


def collect_ops() -> dict:
    ops = {}
    for sub in sorted(NCU_DIR.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        reports = sorted(sub.glob("*.ncu-rep"))
        if not reports:
            continue
        per_pair = {}
        for rep in reports:
            stem = rep.stem
            m = re.match(r"^(triton|cutile)(?:_(.+))?$", stem)
            if not m:
                continue
            backend = m.group(1)
            tag = m.group(2) or "default"
            per_pair[(tag, backend)] = parse_report(rep)
        ops[sub.name] = per_pair
    return ops


def write_op_doc(op: str, per_pair: dict, catalogue_entry: dict) -> None:
    out_dir = NCU_DIR / op
    headline = []
    headline.append(f"# NCU Comparison: {op}")
    headline.append("")
    headline.append(
        "**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  "
    )
    headline.append(
        "**Profile method:** `--set full --import-source on`, "
        "`--launch-skip 3 --launch-count 1`, autotune-winner cfg at "
        "sweep-max input.  "
    )
    headline.append("")

    headline.append("## Test cases (sweep-max per dtype)")
    headline.append("")
    headline.append("| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |")
    headline.append("|---|---|---|---|")
    dtypes = catalogue_entry.get("dtypes", [])
    for dt in dtypes:
        params = catalogue_entry["default_params_per_dtype"].get(dt, {})
        winner = catalogue_entry["autotune_winner_per_dtype"].get(dt, {})
        t = winner.get("triton") if winner else None
        c = winner.get("cutile") if winner else None
        headline.append(
            f"| {dt} | `{params}` | `{t or '(default)'}` | `{c or '(default)'}` |"
        )
    headline.append("")

    headline.append("## Headline (per dtype, both backends)")
    headline.append("")
    cols = [
        ("Backend",                 "__backend__"),
        ("Duration",                "Duration"),
        ("Mem Tput %",              "Memory Throughput"),
        ("DRAM %",                  "DRAM Throughput"),
        ("L1 %",                    "L1/TEX Cache Throughput"),
        ("L2 %",                    "L2 Cache Throughput"),
        ("Compute %",               "Compute (SM) Throughput"),
        ("Mem BW",                  "__memory_bandwidth__"),
        ("Block Sz",                "Block Size"),
        ("Regs",                    "Registers Per Thread"),
        ("Static Shm",              "Static Shared Memory Per Block"),
        ("Dyn Shm",                 "Dynamic Shared Memory Per Block"),
        ("Blk Lim (R/S)",           "__block_lim__"),
    ]
    headline.append("| dtype | " + " | ".join(c[0] for c in cols) + " |")
    headline.append("|" + "---|" * (len(cols) + 1))

    for dt in dtypes:
        for backend in ("triton", "cutile"):
            metrics = per_pair.get((dt, backend))
            if not metrics:
                continue
            if metrics.get("error"):
                row_cells = [f"_(parse error)_"] + [""] * (len(cols) - 1)
            else:
                row_cells = []
                for header, key in cols:
                    if key == "__backend__":
                        row_cells.append(backend)
                    elif key == "__block_lim__":
                        r = fmt(metrics, "Block Limit Registers")
                        s = fmt(metrics, "Block Limit Shared Mem")
                        row_cells.append(f"{r} / {s}")
                    elif key == "Duration":
                        d = duration_us(metrics)
                        row_cells.append(f"{d:.2f} us" if d is not None else "—")
                    else:
                        row_cells.append(fmt(metrics, key))
            headline.append("| " + dt + " | " + " | ".join(row_cells) + " |")
    headline.append("")

    headline.append("## Key findings (auto-derived)")
    headline.append("")
    derived = []
    for dt in dtypes:
        t = per_pair.get((dt, "triton"))
        c = per_pair.get((dt, "cutile"))
        if not t or not c or t.get("error") or c.get("error"):
            continue
        t_dur = duration_us(t)
        c_dur = duration_us(c)
        if t_dur is None or c_dur is None or t_dur <= 0 or c_dur <= 0:
            continue
        if t_dur < c_dur:
            ratio = c_dur / t_dur
            derived.append(
                f"- **{dt}**: Triton is **{ratio:.2f}× faster** "
                f"({t_dur:.1f} µs vs {c_dur:.1f} µs)."
            )
        else:
            ratio = t_dur / c_dur
            derived.append(
                f"- **{dt}**: cuTile is **{ratio:.2f}× faster** "
                f"({c_dur:.1f} µs vs {t_dur:.1f} µs)."
            )
    if derived:
        headline.extend(derived)
    else:
        headline.append("- _(no successfully-parsed pair to compare)_")
    headline.append("")

    headline.append("## NCU's own bottleneck verdict")
    headline.append("")
    for (dt, backend), metrics in sorted(per_pair.items()):
        if metrics.get("error"):
            continue
        bn = metrics.get("__bottleneck__", ("", ""))[0]
        if bn:
            short = bn.split(":")[0]
            headline.append(f"- **{dt} / {backend}** — {short}")
    headline.append("")

    headline.append("## Reports")
    headline.append("")
    for rep in sorted(out_dir.glob("*.ncu-rep")):
        headline.append(f"- `{rep.name}`")
    headline.append("")

    headline.append(
        "## Notes\n\n"
        "Bottleneck verdicts above come from NCU's own SOLBottleneck rule "
        "(headline `OPT` recommendation). For per-section detail, open the "
        ".ncu-rep in `ncu-ui` or run "
        "`ncu --import <file> --page details | less`.\n"
    )

    (out_dir / "comparison.md").write_text("\n".join(headline))


def write_summary(ops: dict, catalogue: list) -> None:
    cat_by_op = {c["op"]: c for c in catalogue}
    lines = [
        "# NCU Sweep Summary — all operators",
        "",
        "**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  ",
        "**Profile method:** autotune-winner cfg at sweep-max input case, "
        "`--set full --import-source on`, `--launch-skip 3 --launch-count 1`",
        "",
        "Per-operator detail: `tilebench_run/ncu/<op>/comparison.md` and the "
        "`<backend>_<dtype>.ncu-rep` files in that directory.",
        "",
        "## Headline duration table (µs)",
        "",
        "| op | dtype | Triton (µs) | cuTile (µs) | Triton:cuTile |",
        "|---|---|---:|---:|---:|",
    ]
    for op in sorted(ops):
        per_pair = ops[op]
        dtypes = cat_by_op.get(op, {}).get("dtypes", [])
        for dt in dtypes:
            t = per_pair.get((dt, "triton"))
            c = per_pair.get((dt, "cutile"))
            td = duration_us(t); cd = duration_us(c)
            if td is None and cd is None:
                continue
            if td and cd and cd > 0:
                ratio = cd / td
                ratio_str = f"{ratio:.2f}×"
            else:
                ratio_str = "—"
            td_s = f"{td:.1f}" if td else "—"
            cd_s = f"{cd:.1f}" if cd else "—"
            lines.append(f"| {op} | {dt} | {td_s} | {cd_s} | {ratio_str} |")
    lines.append("")

    failures_path = NCU_DIR / "sweep_failures.md"
    if failures_path.exists():
        lines.append("## Failed pairs")
        lines.append("")
        lines.append(failures_path.read_text())
    (NCU_DIR / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    catalogue = json.loads((ROOT / "tilebench_run" / "ncu_catalogue.json").read_text())
    cat_by_op = {c["op"]: c for c in catalogue}
    ops = collect_ops()
    print(f"parsed {sum(len(v) for v in ops.values())} reports across {len(ops)} ops")
    for op, per_pair in ops.items():
        write_op_doc(op, per_pair, cat_by_op.get(op, {"dtypes": [], "default_params_per_dtype": {}, "autotune_winner_per_dtype": {}}))
    write_summary(ops, catalogue)
    print(f"wrote {len(ops)} comparison.md files + SUMMARY.md")


if __name__ == "__main__":
    main()
