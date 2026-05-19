"""Aggregate per-op sweep CSVs into per-op dtype-grouped CSVs.

Input:  results/csv/<op>_default.csv  + results/csv/<op>_autotune.csv
Output: results/aggregate/<op>.csv

For each operator, the output CSV has one row per (dtype, mode) combination
with the per-case mean of each timing column. Rows where the backend wrote
nan (skipped / verification-failed cases) are excluded from the mean.

Columns:
  dtype, mode, n_cases, torch_ms, triton_ms, cutile_ms,
  speedup_triton, speedup_cutile, triton_vs_cutile

`triton_vs_cutile = triton_ms / cutile_ms`. >1 means Triton is slower than
cuTile (takes more time); <1 means Triton is faster.
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
            means = {}
            for col in NUMERIC_COLS:
                vals = [_parse_float(r.get(col, "")) for r in group]
                vals = [v for v in vals if v is not None]
                m = sum(vals) / len(vals) if vals else None
                means[col] = m
                agg[col] = f"{m:.6g}" if m is not None else ""
            # Mean-of-per-case Triton/cuTile ratio (computed per case, then
            # averaged — more meaningful than mean(triton_ms) / mean(cutile_ms)
            # when per-case latencies span orders of magnitude).
            ratios = []
            for r in group:
                t = _parse_float(r.get("triton_ms", ""))
                c = _parse_float(r.get("cutile_ms", ""))
                if t is not None and c not in (None, 0):
                    ratios.append(t / c)
            agg["triton_vs_cutile"] = (
                f"{sum(ratios) / len(ratios):.6g}" if ratios else ""
            )
            rows_out.append(agg)

    if not rows_out:
        return False
    out_path = OUT_DIR / f"{op}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dtype", "mode", "n_cases", *NUMERIC_COLS,
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
