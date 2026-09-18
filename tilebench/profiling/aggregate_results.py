"""Aggregate per-op sweep CSVs into per-op dtype-grouped CSVs.

Input:  results/csv/<op>_default.csv  + results/csv/<op>_autotune.csv
Output: results/aggregate/<op>.csv

For each operator, the output CSV has one row per (dtype, mode) combination
with the per-case geometric mean of each timing column. Rows where the backend
wrote nan (skipped / verification-failed cases) are excluded from the mean.

Columns:
  dtype, mode, n_cases, torch_ms, triton_ms, cutile_ms,
  speedup_triton, speedup_cutile, triton_vs_cutile

Speedup / ratio columns are computed from the aggregated geometric means
(ratio-of-geomeans, which equals the geometric mean of the per-case ratios)
— i.e. they're exactly torch_ms / triton_ms etc. computed
on the values that appear in the same row. This makes the table
self-readable: every ratio matches the obvious division of the columns
to its left. Per-case ratios from the source CSV are intentionally NOT
re-averaged here.

  speedup_triton   = torch_ms  / triton_ms     (>1 => Triton faster than Torch)
  speedup_cutile   = torch_ms  / cutile_ms
  triton_vs_cutile = cutile_ms / triton_ms     (>1 => Triton faster than cuTile)
"""
import csv
import math
from collections import defaultdict
from pathlib import Path
from tilebench.paths import REPO_ROOT

ROOT = REPO_ROOT
CSV_DIR = ROOT / "results" / "csv"
OUT_DIR = ROOT / "results" / "aggregate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEAN_COLS = ("torch_ms", "triton_ms", "cutile_ms")


def _parse_float(s: str):
    if not s or s.lower() in {"nan", "inf", "-inf"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def aggregate_one_op(op: str) -> bool:
    """Build the aggregate CSV for `op`. Returns True if any data was written."""
    rows_out = []
    for mode in ("default", "autotune"):
        path = CSV_DIR / f"{op}_{mode}.csv"
        if not path.exists():
            continue
        try:
            rows = list(csv.DictReader(open(path)))
        except Exception:
            continue
        if not rows:
            continue
        # group by dtype
        by_dt: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_dt[r.get("dtype", "?")].append(r)
        for dt, group in by_dt.items():
            agg = {"dtype": dt, "mode": mode, "n_cases": len(group)}
            means = {}
            for col in MEAN_COLS:
                vals = [_parse_float(r.get(col, "")) for r in group]
                vals = [v for v in vals if v is not None and v > 0]
                # Geometric mean (better for averaging latencies across different
                # problem sizes; ratio-of-geomeans == geomean of the per-case ratios).
                m = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None
                means[col] = m
                agg[col] = f"{m:.6g}" if m is not None else ""
            # Ratio columns are computed from the means above (ratio-of-means),
            # so the table is self-readable: speedup_triton matches the obvious
            # division of torch_ms by triton_ms shown in the same row.
            def _ratio(num, den):
                if num is None or den is None or den == 0: return ""
                return f"{num / den:.6g}"
            agg["speedup_triton"]   = _ratio(means["torch_ms"],  means["triton_ms"])
            agg["speedup_cutile"]   = _ratio(means["torch_ms"],  means["cutile_ms"])
            agg["triton_vs_cutile"] = _ratio(means["cutile_ms"], means["triton_ms"])
            rows_out.append(agg)

    if not rows_out:
        return False
    out_path = OUT_DIR / f"{op}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dtype", "mode", "n_cases", *MEAN_COLS,
                           "speedup_triton", "speedup_cutile",
                           "triton_vs_cutile"]
        )
        writer.writeheader()
        # Sort rows: all default rows first, then all autotune rows; within
        # each block, dtype ascending.
        rows_out.sort(key=lambda r: (0 if r["mode"] == "default" else 1, r["dtype"]))
        writer.writerows(rows_out)
    return True


def main() -> None:
    ops = set()
    for p in CSV_DIR.glob("*_default.csv"):
        ops.add(p.stem.removesuffix("_default"))
    for p in CSV_DIR.glob("*_autotune.csv"):
        ops.add(p.stem.removesuffix("_autotune"))
    for p in CSV_DIR.glob("*_summary.csv"):
        ops.add(p.stem.removesuffix("_summary"))
    ops.discard("")

    wrote = 0
    skipped = []
    for op in sorted(ops):
        if aggregate_one_op(op):
            wrote += 1
        else:
            skipped.append(op)
    print(f"wrote {wrote} aggregate CSV files to {OUT_DIR}")
    if skipped:
        print(f"skipped (no per-op csv data found): {skipped}")


if __name__ == "__main__":
    main()
