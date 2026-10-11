"""Extract the bank-conflict analysis table from the NCU reps.

Per (op, backend, dtype) — primary kernel = heaviest-duration action in the
rep (same convention as tilebench_run/ncu_writeup.py headline metrics):

  conflict_score C_LSU = 100 * l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
                             / l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
  branch_eff  = valid branch efficiency (%): uniform branch targets share
  ipc_gap     = sm__inst_issued.avg.per_cycle_active
              - sm__inst_executed.avg.per_cycle_active

Diagnosis (mutually exclusive, evaluated in order; thresholds documented here
because the appendix-D figures are qualitative severity bands):
  Confounded          branch_eff < 98.0  (divergence confounds attribution)
  Severe/Moderate     C_LSU >= 10
  Mild                1 <= C_LSU < 10
  Likely              0.1 <= C_LSU < 1
  No direct conflict  C_LSU < 0.1

Writes Figures/_data_ncu_conflict.csv.
Usage:  PYTHONPATH=. python scripts/analysis/ncu_conflict_data.py
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

for _c in glob.glob("/opt/nvidia/nsight-compute/*/extras/python"):
    sys.path.insert(0, _c)
import ncu_report  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NCU_DIR = REPO / "tilebench_run" / "ncu"
OUT = REPO / "Figures" / "_data_ncu_conflict.csv"

M_CONF = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
M_WAVE = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
M_ISSUED = "sm__inst_issued.avg.per_cycle_active"
M_EXEC = "sm__inst_executed.avg.per_cycle_active"
M_DUR = "gpu__time_duration.sum"
BRANCH_CANDIDATES = [
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__average_branch_targets_threads_uniform.pct",
]


def _val(action, name):
    try:
        m = action[name]
    except Exception:
        return None
    try:
        return m.value()
    except Exception:
        return None


def _branch_eff(action) -> float | None:
    for name in BRANCH_CANDIDATES:
        v = _val(action, name)
        if v is not None and v > 0.0:
            return float(v)
    return None


def diagnose(c: float | None, branch_eff: float | None) -> str:
    if c is None:
        return "unavailable"
    if branch_eff is not None and branch_eff < 98.0:
        return "Confounded"
    if c >= 10:
        return "Severe/Moderate"
    if c >= 1:
        return "Mild"
    if c >= 0.1:
        return "Likely"
    return "No direct conflict"


def main() -> None:
    rows = []
    for rep in sorted(NCU_DIR.glob("*/*.ncu-rep")):
        op = rep.parent.name
        backend, dtype = rep.stem.split("_", 1)
        try:
            ctx = ncu_report.load_report(str(rep))
            rng = ctx.range_by_idx(0)
        except Exception as e:
            print(f"  skip {rep}: {e}")
            continue
        best, best_dur = None, -1.0
        for i in range(rng.num_actions()):
            a = rng.action_by_idx(i)
            dur = _val(a, M_DUR) or 0.0
            if dur > best_dur:
                best, best_dur = a, dur
        if best is None:
            continue
        conf = _val(best, M_CONF)
        wave = _val(best, M_WAVE)
        c = (100.0 * conf / wave) if (conf is not None and wave and wave > 0) else None
        be = _branch_eff(best)
        issued, executed = _val(best, M_ISSUED), _val(best, M_EXEC)
        ipc_gap = (issued - executed) if (issued is not None and executed is not None) else None
        rows.append({
            "op": op, "backend": backend, "dtype": dtype,
            "kernel": best.name(), "duration_ns": best_dur,
            "conflicts": conf, "shared_wavefronts": wave,
            "conflict_score": c, "branch_eff": be, "ipc_gap": ipc_gap,
            "diag": diagnose(c, be),
        })
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
