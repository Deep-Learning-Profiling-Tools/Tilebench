"""Paper Fig 17: iterative refinement trajectory for one operator (softmax).

Data: the archived LLM-track trajectory snapshot
results/B200/figures/evaluation/v7/_data_llm_traj.csv (the raw
run_summary.json artifacts no longer exist on this machine). softmax's
torch baseline is unchanged, so the stored speedup_vs_torch values remain
valid for this figure.

Writes Figures/Figure17.pdf.
Usage:  PYTHONPATH=. python scripts/analysis/fig_llm.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.paper_style import BACKEND_COLOR  # noqa: E402

TRAJ_CSV = REPO / "results" / "B200" / "figures" / "evaluation" / "v7" / "_data_llm_traj.csv"
FIG_DIR = REPO / "Figures"


def load_traj() -> list[dict]:
    rows = []
    for r in csv.DictReader(TRAJ_CSV.open()):
        rows.append({
            "op": r["op"], "model": r["model"], "backend": r["backend"],
            "iter": int(r["iter"]),
            "skipped": r["skipped"] == "True",
            "verify_clean": r["verify_clean"] == "True",
            "frozen_at_this_iter": r["frozen_at_this_iter"] == "True",
            "speedup_vs_torch": (float(r["speedup_vs_torch"])
                                 if r["speedup_vs_torch"] else None),
        })
    return rows


def fig_rq4_c(traj_rows: list[dict], op: str = "softmax") -> None:
    by_mb: dict[tuple, list[dict]] = defaultdict(list)
    for r in traj_rows:
        if r["op"] != op or r["skipped"]:
            continue
        by_mb[(r["model"], r["backend"])].append(r)
    if not by_mb:
        print(f"no trajectory data for op={op}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    MODEL_STYLE = {"gpt-5.5": "-", "claude-opus-4-7": "--"}
    MODEL_MARK = {"gpt-5.5": "o", "claude-opus-4-7": "^"}
    MODEL_LABEL = {"gpt-5.5": "GPT-5.5", "claude-opus-4-7": "Claude Opus 4.7"}
    order = [("gpt-5.5", "triton"), ("gpt-5.5", "cutile"),
             ("claude-opus-4-7", "triton"), ("claude-opus-4-7", "cutile")]

    all_speedups, plot_data, fail_data = [], [], []
    for (m, b) in order:
        items = sorted(by_mb.get((m, b), []), key=lambda r: r["iter"])
        if not items:
            continue
        xs, ys = [], []
        for it in items:
            sp = it["speedup_vs_torch"]
            if it["verify_clean"] and sp and sp > 0:
                xs.append(it["iter"]); ys.append(sp); all_speedups.append(sp)
            else:
                fail_data.append((it["iter"], m, b))
        plot_data.append((m, b, xs, ys, items))

    y_floor = max(0.05, min(all_speedups) / 2.5)
    for (m, b, xs, ys, items) in plot_data:
        if not xs:
            continue
        ax.plot(xs, ys, marker=MODEL_MARK[m], color=BACKEND_COLOR[b],
                linestyle=MODEL_STYLE[m], lw=1.6, markersize=6, alpha=0.9,
                label=f"{MODEL_LABEL[m]} + {b.capitalize()}")
        frozen = [it for it in items if it["frozen_at_this_iter"]]
        if frozen:
            ax.axvline(frozen[0]["iter"], color=BACKEND_COLOR[b],
                       ls=MODEL_STYLE[m], lw=0.6, alpha=0.35)

    y_jitter = {("gpt-5.5", "triton"): 1.00, ("gpt-5.5", "cutile"): 0.82,
                ("claude-opus-4-7", "triton"): 0.67,
                ("claude-opus-4-7", "cutile"): 0.55}
    for (it_idx, m, b) in fail_data:
        ax.scatter(it_idx, y_floor * y_jitter[(m, b)], marker="x", s=55,
                   color=BACKEND_COLOR[b],
                   linewidth=1.4, alpha=0.85, zorder=5)
    if fail_data:
        ax.text(0.02, 0.03, "$\\times$ = verify failed (no speedup recorded)",
                transform=ax.transAxes, fontsize=8, color="gray",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          linewidth=0.4, boxstyle="round,pad=0.25", alpha=0.85))

    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(9.6, 1.02, "PyTorch parity (1$\\times$)", color="gray",
            fontsize=8, ha="right", va="bottom")
    ax.set_xlim(-0.3, 10)
    ax.set_yscale("log")
    ax.set_ylim(y_floor * 0.4, max(all_speedups) * 1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Speedup vs PyTorch  ($T_{\mathrm{torch}}/T_{b}$)")
    ax.set_title(f"Iterative convergence: {op}", fontsize=11)
    ax.legend(loc="lower right", frameon=True, edgecolor="lightgray",
              fontsize=8.5)
    ax.grid(True, alpha=0.18, which="major")

    fig.tight_layout()
    out = FIG_DIR / "Figure17.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_rq4_c(load_traj())
