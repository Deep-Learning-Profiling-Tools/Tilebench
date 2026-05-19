"""Aggregate per-op sweep CSVs into per-op dtype-grouped CSVs.

Input:  results/csv/<op>_default.csv  + results/csv/<op>_autotune.csv
Output: results/aggregate/<op>.csv

For each operator, the output CSV has one row per (dtype, mode) combination
with the per-case mean of each timing column. Rows where the backend wrote
nan (skipped / verification-failed cases) are excluded from the mean.

Columns:
  dtype, mode, n_cases, torch_ms, triton_ms, cutile_ms,
  speedup_triton, speedup_cutile
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
CSV_DIR = ROOT / "results" / "csv"
OUT_DIR = ROOT / "results" / "aggregate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ("torch_ms", "triton_ms", "cutile_ms",
                "speedup_triton", "speedup_cutile")


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
            for col in NUMERIC_COLS:
                vals = [_parse_float(r.get(col, "")) for r in group]
                vals = [v for v in vals if v is not None]
                agg[col] = f"{sum(vals) / len(vals):.6g}" if vals else ""
            rows_out.append(agg)

    if not rows_out:
        return False
    out_path = OUT_DIR / f"{op}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dtype", "mode", "n_cases", *NUMERIC_COLS]
        )
        writer.writeheader()
        # Sort rows: dtype ascending, default before autotune
        rows_out.sort(key=lambda r: (r["dtype"], 0 if r["mode"] == "default" else 1))
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
