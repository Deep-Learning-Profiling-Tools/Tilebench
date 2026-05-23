"""Cross-validate the TMA-descriptor Triton refactor against the in-repo
baseline.

Inputs:
  results/csv/<op>_default.csv      <- baseline (pre-TMA exp/timing snapshot)
  results/csv/<op>_autotune.csv     <- baseline (autotune)
  results/descriptor_results/csv/<op>_default.csv   <- descriptor (post-TMA, this run)
  results/descriptor_results/csv/<op>_autotune.csv  <- descriptor (autotune)

Outputs:
  results/descriptor_results/compare/case_comparison.csv
      One row per (op, mode, dtype, params). Shows baseline vs descriptor
      torch_ms / triton_ms / cutile_ms / speedup_triton / speedup_cutile,
      plus per-backend delta-percent (descriptor - baseline) / baseline.

  results/descriptor_results/compare/aggregate_comparison.csv
      One row per (op, mode, dtype). Means across cases for each side,
      plus delta-percent of the mean.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
BASELINE_CSV = ROOT / "results" / "csv"
DESCRIPTOR_CSV = ROOT / "results" / "descriptor_results" / "csv"
OUT_DIR = ROOT / "results" / "descriptor_results" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = ("torch_ms", "triton_ms", "cutile_ms", "speedup_triton", "speedup_cutile")


def _f(s):
    if not s or s.lower() in {"nan", "inf", "-inf"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pct_delta(d, b):
    """Return (descriptor - baseline) / baseline * 100; '' if unmeaningful."""
    if d is None or b is None or b == 0:
        return ""
    return f"{(d - b) / b * 100:.2f}"


def _load_csv(p):
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def main():
    # Discover ops by looking at descriptor side (the source of truth for
    # what was rerun).
    ops = sorted({
        p.stem.removesuffix("_default").removesuffix("_autotune")
        for p in DESCRIPTOR_CSV.glob("*.csv")
        if p.name.endswith("_default.csv") or p.name.endswith("_autotune.csv")
    })

    case_rows = []
    agg_rows = []

    for op in ops:
        for mode in ("default", "autotune"):
            base = _load_csv(BASELINE_CSV / f"{op}_{mode}.csv")
            desc = _load_csv(DESCRIPTOR_CSV / f"{op}_{mode}.csv")
            if not base and not desc:
                continue

            # Index by (dtype, params)
            base_idx = {(r.get("dtype"), r.get("params")): r for r in base}
            desc_idx = {(r.get("dtype"), r.get("params")): r for r in desc}
            all_keys = sorted(set(base_idx) | set(desc_idx))

            # Aggregate accumulators: per dtype
            agg_acc = defaultdict(lambda: {c: ([], []) for c in NUM_COLS})

            for (dt, pa) in all_keys:
                b = base_idx.get((dt, pa))
                d = desc_idx.get((dt, pa))
                bp = {c: _f((b or {}).get(c, "")) for c in NUM_COLS}
                dp = {c: _f((d or {}).get(c, "")) for c in NUM_COLS}

                row = {
                    "operator": op,
                    "mode": mode,
                    "dtype": dt,
                    "params": pa,
                    "baseline_present": "1" if b else "0",
                    "descriptor_present": "1" if d else "0",
                }
                for c in NUM_COLS:
                    row[f"baseline_{c}"] = f"{bp[c]:.6g}" if bp[c] is not None else ""
                    row[f"descriptor_{c}"] = f"{dp[c]:.6g}" if dp[c] is not None else ""
                # Per-backend delta-pct (descriptor vs baseline)
                for c in ("torch_ms", "triton_ms", "cutile_ms"):
                    row[f"{c}_delta_pct"] = _pct_delta(dp[c], bp[c])
                # Triton-only descriptor-vs-baseline speedup: > 1 if descriptor
                # is faster, < 1 if descriptor is slower. Only Triton is
                # affected by the TMA refactor, so we report just this ratio.
                if bp["triton_ms"] is not None and dp["triton_ms"] is not None and dp["triton_ms"] > 0:
                    row["descriptor_vs_baseline_speedup"] = f"{bp['triton_ms'] / dp['triton_ms']:.6g}"
                else:
                    row["descriptor_vs_baseline_speedup"] = ""
                case_rows.append(row)

                # accumulate aggregate
                for c in NUM_COLS:
                    if bp[c] is not None:
                        agg_acc[dt][c][0].append(bp[c])
                    if dp[c] is not None:
                        agg_acc[dt][c][1].append(dp[c])

            # Emit aggregate rows
            for dt in sorted(agg_acc):
                acc = agg_acc[dt]
                arow = {"operator": op, "mode": mode, "dtype": dt,
                        "n_cases": len(acc["torch_ms"][0])}
                for c in NUM_COLS:
                    bvals, dvals = acc[c]
                    bm = sum(bvals) / len(bvals) if bvals else None
                    dm = sum(dvals) / len(dvals) if dvals else None
                    arow[f"baseline_{c}"] = f"{bm:.6g}" if bm is not None else ""
                    arow[f"descriptor_{c}"] = f"{dm:.6g}" if dm is not None else ""
                for c in ("torch_ms", "triton_ms", "cutile_ms"):
                    bvals, dvals = acc[c]
                    bm = sum(bvals) / len(bvals) if bvals else None
                    dm = sum(dvals) / len(dvals) if dvals else None
                    arow[f"{c}_delta_pct"] = _pct_delta(dm, bm)
                # Triton-only descriptor-vs-baseline speedup (ratio of
                # mean baseline_triton_ms / mean descriptor_triton_ms).
                bvals, dvals = acc["triton_ms"]
                bm = sum(bvals) / len(bvals) if bvals else None
                dm = sum(dvals) / len(dvals) if dvals else None
                if bm is not None and dm is not None and dm > 0:
                    arow["descriptor_vs_baseline_speedup"] = f"{bm / dm:.6g}"
                else:
                    arow["descriptor_vs_baseline_speedup"] = ""
                agg_rows.append(arow)

    # Write outputs
    if case_rows:
        case_header = list(case_rows[0].keys())
        with open(OUT_DIR / "case_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=case_header)
            w.writeheader()
            w.writerows(case_rows)
        print(f"wrote {len(case_rows)} case rows -> compare/case_comparison.csv")

    if agg_rows:
        agg_header = list(agg_rows[0].keys())
        with open(OUT_DIR / "aggregate_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=agg_header)
            w.writeheader()
            w.writerows(agg_rows)
        print(f"wrote {len(agg_rows)} aggregate rows -> compare/aggregate_comparison.csv")


if __name__ == "__main__":
    main()
