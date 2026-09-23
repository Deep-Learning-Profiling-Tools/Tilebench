"""Regenerate the NCU bank-conflict appendix figures (paper Figs 10–15)
from Figures/_data_ncu_conflict.csv (built by ncu_conflict_data.py).

  Figures/Figure10.pdf       (Fig 10)
  Figures/Figure11.pdf      (Fig 11)
  Figures/Figure12.pdf   (Fig 12)
  Figures/Figure13.pdf     (Fig 13)
  Figures/Figure14.pdf      (Fig 14)
  Figures/Figure15.pdf                (Fig 15)

Usage:  PYTHONPATH=. python scripts/analysis/fig_ncu_suite.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.paper_style import BACKEND_COLOR  # noqa: E402

FIG_DIR = REPO / "Figures"
DATA = FIG_DIR / "_data_ncu_conflict.csv"

DIAG_ORDER = ["Severe/Moderate", "Mild", "Likely", "Confounded",
              "No direct conflict"]
DIAG_COLOR = {
    "Severe/Moderate": "#7b0d7b",
    "Mild": "#c2185b",
    "Likely": "#f06292",
    "Confounded": "#f8bbd0",
    "No direct conflict": "#6fa8d6",
}
DIAG_MARKER = {
    "No direct conflict": "o",
    "Likely": "x",
    "Mild": "s",
    "Severe/Moderate": "+",
    "Confounded": "D",
}
DTYPE_ORDER = ["fp16", "fp32", "bf16", "int8", "int32",
               "fp8_e4m3fn", "fp8_e5m2"]
DTYPE_COLOR = {d: c for d, c in zip(DTYPE_ORDER, plt.cm.tab10.colors)}

C_FLOOR = 1e-3  # display floor for log axes


def load() -> list[dict]:
    rows = []
    for r in csv.DictReader(DATA.open()):
        r["conflict_score"] = float(r["conflict_score"]) if r["conflict_score"] else None
        r["branch_eff"] = float(r["branch_eff"]) if r["branch_eff"] else None
        r["ipc_gap"] = float(r["ipc_gap"]) if r["ipc_gap"] else None
        rows.append(r)
    return rows


def fig_top_conflict_bar(rows: list[dict]) -> None:
    scored = [r for r in rows if r["conflict_score"]]
    top = sorted(scored, key=lambda r: -r["conflict_score"])[:20]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ys = np.arange(len(top))[::-1]
    for y, r in zip(ys, top):
        ax.barh(y, r["conflict_score"], height=0.72,
                color=BACKEND_COLOR[r["backend"]], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['op']}/{r['backend']}/{r['dtype']}" for r in top],
                       fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Bank conflicts / shared wavefront (%)", fontsize=10)
    ax.set_title("Top 20 bank-conflict cases", fontsize=11)
    ax.grid(True, axis="x", alpha=0.25, which="both", zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR["triton"], label="triton"),
               plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR["cutile"], label="cutile")]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, edgecolor="lightgray")
    fig.tight_layout()
    out = FIG_DIR / "Figure10.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_diagnosis_stacked(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for yi, backend in enumerate(("triton", "cutile")):
        left = 0.0
        for diag in DIAG_ORDER:
            n = sum(1 for r in rows
                    if r["backend"] == backend and r["diag"] == diag)
            ax.barh(1 - yi, n, left=left, height=0.62,
                    color=DIAG_COLOR[diag],
                    label=diag if yi == 0 else None)
            left += n
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["triton", "cutile"], fontsize=10)
    ax.set_xlabel("# reports", fontsize=10)
    ax.set_title("Automated NCU bank-conflict diagnosis", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "Figure11.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_conflict_heatmap(rows: list[dict]) -> None:
    by_rb: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["conflict_score"] is not None:
            by_rb[(r["op"], r["backend"])][r["dtype"]] = r["conflict_score"]
    ranked = sorted(by_rb.items(),
                    key=lambda kv: -max(kv[1].values()))[:25]
    dtypes = [d for d in DTYPE_ORDER
              if any(d in v for _, v in ranked)]
    mat = np.full((len(ranked), len(dtypes)), np.nan)
    for i, (_, v) in enumerate(ranked):
        for j, d in enumerate(dtypes):
            if d in v:
                mat[i, j] = v[d]

    try:
        import seaborn as sns
        cmap = sns.color_palette("rocket_r", as_cmap=True)
    except Exception:
        cmap = plt.get_cmap("magma_r")
    cmap = cmap.copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("conflict score (%)", fontsize=10)
    ax.set_xticks(range(len(dtypes)))
    ax.set_xticklabels(dtypes, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([f"{op} / {b}" for (op, b), _ in ranked], fontsize=9)
    ax.set_xlabel("dtype", fontsize=11)
    ax.set_title("Top 25 bank-conflict operator/backend rows", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "Figure12.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_backend_comparison(rows: list[dict]) -> None:
    by_od: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["conflict_score"] is not None:
            by_od[(r["op"], r["dtype"])][r["backend"]] = r["conflict_score"]

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    seen: set[str] = set()
    for (op, dtype), v in by_od.items():
        if "triton" not in v or "cutile" not in v:
            continue
        x = max(v["triton"], C_FLOOR)
        y = max(v["cutile"], C_FLOOR)
        ax.scatter(x, y, s=42, alpha=0.8, color=DTYPE_COLOR.get(dtype, "gray"),
                   edgecolors="black", linewidth=0.3,
                   label=dtype if dtype not in seen else None, zorder=4)
        seen.add(dtype)
    lims = (C_FLOOR * 0.7, 300)
    xx = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)
    ax.plot(xx, xx, ls="--", color="black", lw=0.8, alpha=0.7, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Triton conflict score (%)", fontsize=10)
    ax.set_ylabel("cuTile conflict score (%)", fontsize=10)
    ax.set_title("Backend comparison", fontsize=11)
    ax.grid(True, alpha=0.2, which="major")
    ax.legend(title="dtype", fontsize=9, title_fontsize=10, loc="upper left",
              frameon=True, edgecolor="lightgray")
    fig.tight_layout()
    out = FIG_DIR / "Figure13.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _scatter_diag(ax, rows, xkey):
    seen_b, seen_d = set(), set()
    for r in rows:
        x, c = r[xkey], r["conflict_score"]
        if x is None or c is None:
            continue
        ax.scatter(x, max(c, C_FLOOR),
                   marker=DIAG_MARKER.get(r["diag"], "o"), s=34,
                   color=BACKEND_COLOR[r["backend"]], alpha=0.8,
                   linewidths=1.2, zorder=4)
        seen_b.add(r["backend"]); seen_d.add(r["diag"])
    handles = [plt.Line2D([0], [0], ls="", marker="o", color=BACKEND_COLOR[b],
                          label=b) for b in ("cutile", "triton") if b in seen_b]
    handles += [plt.Line2D([0], [0], ls="", marker=DIAG_MARKER[d],
                           color="gray", label=d)
                for d in DIAG_MARKER if d in seen_d]
    ax.legend(handles=handles, fontsize=7, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)


def fig_branch_confounder(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    rs = [r for r in rows if r["branch_eff"] is not None]
    _scatter_diag(ax, rs, "branch_eff")
    ax.axvline(98.0, color="gray", ls=":", lw=1.0, alpha=0.8)
    ax.axvline(100.0, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Branch efficiency (%)", fontsize=10)
    ax.set_ylabel("Conflict score (%)", fontsize=10)
    ax.set_title("Branch-divergence confounder", fontsize=11)
    ax.grid(True, alpha=0.2, which="major")
    fig.tight_layout()
    out = FIG_DIR / "Figure14.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_ipc_gap(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    rs = [r for r in rows if r["ipc_gap"] is not None]
    _scatter_diag(ax, rs, "ipc_gap")
    ax.set_yscale("log")
    ax.set_xlabel("Issued/executed IPC gap", fontsize=10)
    ax.set_ylabel("Conflict score (%)", fontsize=10)
    ax.set_title("IPC gap is not a conflict proxy", fontsize=11)
    ax.grid(True, alpha=0.2, which="major")
    fig.tight_layout()
    out = FIG_DIR / "Figure15.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = load()
    print(f"loaded {len(rows)} rows")
    fig_top_conflict_bar(rows)
    fig_diagnosis_stacked(rows)
    fig_conflict_heatmap(rows)
    fig_backend_comparison(rows)
    fig_branch_confounder(rows)
    fig_ipc_gap(rows)


if __name__ == "__main__":
    main()
