"""Paper Fig 5: LLM TokenCost@10 and TokenEfficiency@10 by (model, backend),
with best-speedup values re-based against the CURRENT torch baselines.

The archived LLM aggregates (results/B200/figures/evaluation/v7/_data_llm.csv)
store best_speedup measured against torch at LLM-run time (mid-June). The
generated kernels' own latencies are fixed, so re-basing against today's
torch is exact:  speedup_new = speedup_old * (torch_new / torch_old).

The stored per-iteration speedup is the sweep-max case's measurement
(tools/llm_codegen/generate.py picks the largest verify-clean combo), so the
re-basing factor is computed at the sweep-max case:

  factor(op) = geomean over dtypes of
               torch_ms_now(sweep-max) / torch_ms_jun15(sweep-max)

torch_ms_jun15 comes from results/csv/<op>_default.csv at commit
OLD_COMMIT (last main commit before 2026-06-16, the LLM-run era).
Factors within +-5% of 1.0 are clamped to 1.0 (re-measurement noise).

Aggregation follows the paper's all-equal-weight-geomean convention:
per (model, backend), TokenCost@10 and TokenEfficiency@10 are geometric
means across operators (efficiency over ops with a verify-clean kernel).

Writes Figures/Figure5.pdf and Figures/_data_llm_rebased.csv.
Usage:  PYTHONPATH=. python scripts/analysis/fig_llm_token.py
"""
from __future__ import annotations

import csv
import io
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.tensors import infer_problem_size                    # noqa: E402
from scripts.analysis.paper_style import BACKEND_COLOR         # noqa: E402
from scripts.analysis.bench_data import _parse_params, geomean  # noqa: E402

OLD_COMMIT = "35226a956751f46892559ff0ab01aa13d08982fe"
LLM_CSV = REPO / "results" / "B200" / "figures" / "evaluation" / "v7" / "_data_llm.csv"
FIG_DIR = REPO / "Figures"
NOISE_BAND = 0.05


def _sweepmax_torch(csv_text: str, op: str) -> dict[str, float]:
    """{dtype: torch_ms at the max-problem_size case} from a summary CSV."""
    cfg = yaml.safe_load(
        (REPO / "benchmarks" / "operators" / op / "config.yaml").read_text())
    case_defaults = cfg.get("case_defaults") or {}
    best: dict[str, tuple[int, float]] = {}
    for rec in csv.DictReader(io.StringIO(csv_text)):
        try:
            torch_ms = float(rec["torch_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        params = dict(case_defaults)
        params.update(_parse_params(rec.get("params", "")))
        psize = infer_problem_size(op, params)
        dt = rec.get("dtype", "?")
        if dt not in best or psize > best[dt][0]:
            best[dt] = (psize, torch_ms)
    return {dt: ms for dt, (ps, ms) in best.items()}


def torch_factor(op: str) -> tuple[float, str]:
    """(torch_now / torch_jun15) at sweep-max, geomean over dtypes."""
    new_path = REPO / "results" / "csv" / f"{op}_default.csv"
    if not new_path.exists():
        return 1.0, "no current csv"
    try:
        old_text = subprocess.run(
            ["git", "show", f"{OLD_COMMIT}:results/csv/{op}_default.csv"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return 1.0, "no jun-15 csv (renamed op?)"
    new_t = _sweepmax_torch(new_path.read_text(), op)
    old_t = _sweepmax_torch(old_text, op)
    ratios = [new_t[dt] / old_t[dt] for dt in new_t if dt in old_t and old_t[dt] > 0]
    if not ratios:
        return 1.0, "no matching dtypes"
    f = geomean(ratios)
    if abs(f - 1.0) <= NOISE_BAND:
        return 1.0, f"clamped ({f:.3f})"
    return f, f"applied ({f:.3f})"


def main() -> None:
    rows = list(csv.DictReader(LLM_CSV.open()))
    ops = sorted({r["op"] for r in rows})
    factors = {}
    for op in ops:
        f, why = torch_factor(op)
        factors[op] = f
        if f != 1.0:
            print(f"  rebase {op}: torch factor {why}")

    rebased = []
    for r in rows:
        best_old = float(r["best_speedup"])
        cost = float(r["token_cost"])
        best_new = best_old * factors[r["op"]]
        eff = (best_new / (cost / 1e6)) if cost > 0 else 0.0
        rebased.append({
            "op": r["op"], "model": r["model"], "backend": r["backend"],
            "token_cost": cost, "best_speedup_old": best_old,
            "torch_factor": factors[r["op"]],
            "best_speedup": best_new, "token_efficiency": eff,
        })
    with (FIG_DIR / "_data_llm_rebased.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rebased[0].keys()))
        w.writeheader(); w.writerows(rebased)

    MODELS = ["gpt-5.5", "claude-opus-4-7"]
    MODEL_LABEL = {"gpt-5.5": "GPT-5.5", "claude-opus-4-7": "Claude Opus 4.7"}
    cost_agg, eff_agg = {}, {}
    for m in MODELS:
        for b in ("triton", "cutile"):
            sub = [r for r in rebased if r["model"] == m and r["backend"] == b]
            cost_agg[(m, b)] = geomean([r["token_cost"] / 1e6
                                        for r in sub if r["token_cost"] > 0])
            eff_agg[(m, b)] = geomean([r["token_efficiency"]
                                       for r in sub if r["token_efficiency"] > 0])
            n_eff = sum(1 for r in sub if r["token_efficiency"] > 0)
            print(f"{MODEL_LABEL[m]:16s} {b:6s} cost {cost_agg[(m,b)]:.3f}M "
                  f"eff {eff_agg[(m,b)]:.1f} (n_eff={n_eff})")

    fig, (axc, axe) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    width = 0.38
    for ax, agg, hatched in ((axc, cost_agg, False), (axe, eff_agg, True)):
        for mi, m in enumerate(MODELS):
            for bi, b in enumerate(("triton", "cutile")):
                x = mi + (bi - 0.5) * (width + 0.03)
                ax.bar(x, agg[(m, b)], width=width,
                       color=BACKEND_COLOR[b], edgecolor="black",
                       linewidth=0.6, hatch="//" if hatched else None,
                       label=("Triton" if b == "triton" else "cuTile")
                             if (mi == 0 and ax is axc) else None)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)
    axc.set_title("TokenCost@10", fontsize=10)
    axc.set_ylabel("Geomean TokenCost@10 per operator (M tokens)", fontsize=8.5)
    axe.set_title("TokenEfficiency@10", fontsize=10)
    axe.set_ylabel(r"Geomean TokenEfficiency@10 ($\times$ / 1M tokens)",
                   fontsize=8.5)
    axc.legend(loc="upper left", fontsize=8.5, frameon=True,
               edgecolor="lightgray")

    fig.tight_layout()
    out = FIG_DIR / "Figure5.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
