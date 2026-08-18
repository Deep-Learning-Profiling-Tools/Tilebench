"""Regenerate the main evaluation figures (paper Figs 2, 3, 4, 8a, 8b, 9)
from the 8-column summary CSVs on main.

All figures use autotuned-mode data except Fig 3 (which compares default
vs autotuned by construction). Output filenames match the paper's
\\includegraphics targets:
  Figures/Figure2.pdf       (Fig 2, autotuned mode)
  Figures/Figure3.pdf   (Fig 3)
  Figures/Figure4.pdf(Fig 4, autotuned sweep-max)
  Figures/Figure8a.pdf          (Fig 8a, autotuned mode)
  Figures/Figure8b.pdf               (Fig 8b, per-category R;
                                               filename is historical)
  Figures/Figure9.pdf         (Fig 9)

Usage:  PYTHONPATH=. python scripts/analysis/fig_evaluation.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.paper_style import (      # noqa: E402
    CAT_ORDER, CAT_COLOR, CAT_MARKER, BACKEND_COLOR, CATEGORY,
    legend_categories,
)
from scripts.analysis.bench_data import (       # noqa: E402
    load_main_table, per_op_mean_speedup, per_op_mean_R, geomean,
)

FIG_DIR = REPO / "Figures"
FIG_DIR.mkdir(exist_ok=True)


def fig_rq1_a(per_op: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    xy_max = 1.0
    for op, d in per_op.items():
        tx, ty = d["triton"], d["cutile"]
        if tx <= 0 or ty <= 0:
            continue
        cat = d["category"]
        ax.scatter(tx, ty, marker=CAT_MARKER[cat], color=CAT_COLOR[cat],
                   s=80, alpha=0.85, edgecolors="black", linewidth=0.5, zorder=4)
        xy_max = max(xy_max, tx, ty)

    lim = (0.05, max(50.0, xy_max * 1.2))
    xx = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 100)
    ax.plot(xx, xx, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
    ax.text(lim[1] * 0.8, lim[1] * 0.95, "y = x", color="gray",
            fontsize=9, ha="right", va="top", style="italic")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":", alpha=0.6)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":", alpha=0.6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Triton speedup vs PyTorch")
    ax.set_ylabel("cuTile speedup vs PyTorch")
    ax.grid(True, alpha=0.18, which="major")
    legend_categories(ax, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "Figure2.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_rq1_b(per_op: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    triton_vals = [d["triton"] for d in per_op.values() if d["triton"] > 0]
    cutile_vals = [d["cutile"] for d in per_op.values() if d["cutile"] > 0]

    bp = ax.boxplot([triton_vals, cutile_vals], positions=[1, 2],
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="black", lw=0.8),
                    capprops=dict(color="black", lw=0.8))
    for patch, b in zip(bp["boxes"], ("triton", "cutile")):
        patch.set_facecolor(BACKEND_COLOR[b])
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(0)
    for x_center, vals, b in [(1, triton_vals, "triton"),
                              (2, cutile_vals, "cutile")]:
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x_center + jitter, vals, s=18, alpha=0.75,
                   color=BACKEND_COLOR[b], edgecolors="black",
                   linewidth=0.25, zorder=4)

    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", ls=":", lw=0.7, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Triton", "cuTile"], fontsize=11)
    ax.set_ylabel(r"Speedup vs PyTorch ($\times$)")
    ax.set_title("Autotuned mode (vs PyTorch)", fontsize=11)
    ax.grid(True, alpha=0.18, axis="y", which="major")
    for label, b in zip(ax.get_xticklabels(), ("triton", "cutile")):
        label.set_color(BACKEND_COLOR[b])

    fig.tight_layout()
    out = FIG_DIR / "Figure8a.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


CAT_SHORT = {
    "Stencil/Conv": "Stencil/Conv",
    "Matrix Mult./Attn": "MatMul/Attn",
    "Reduction/Norm": "Reduce/Norm",
    "Point-wise": "Point-wise",
    "Data Layout": "Data Layout",
}


def fig_rq1_c(main_rows: list[dict]) -> None:
    """Per-category default-mode roofline utilization: mean bars + per-op dots.
    (Kept under the historical Figure8b.pdf name used by the paper.)"""
    by_cat: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: {"triton": defaultdict(list), "cutile": defaultdict(list)})
    for r in main_rows:
        if r["mode"] != "autotune" or r["R"] is None or r["R"] <= 0:
            continue
        by_cat[r["category"]][r["backend"]][r["op"]].append(r["R"])

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.36
    rng = np.random.default_rng(2)
    for ci, cat in enumerate(CAT_ORDER):
        for bi, b in enumerate(("triton", "cutile")):
            per_op_R = {op: geomean(v)
                        for op, v in by_cat[cat][b].items()}
            vals = list(per_op_R.values())
            if not vals:
                continue
            x = ci + (bi - 0.5) * (width + 0.04)
            ax.bar(x, geomean(vals), width=width,
                   color=BACKEND_COLOR[b], edgecolor="black", linewidth=0.8,
                   alpha=0.9, zorder=2,
                   label=("Triton" if b == "triton" else "Cutile") if ci == 0 else None)
            jitter = rng.uniform(-width * 0.35, width * 0.35, size=len(vals))
            ax.scatter(x + jitter, vals, s=26, color=BACKEND_COLOR[b],
                       edgecolors="black", linewidth=0.6, zorder=4)

    n_per_cat = {cat: len({op for b in ("triton", "cutile")
                           for op in by_cat[cat][b]}) for cat in CAT_ORDER}
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels([f"{CAT_SHORT[c]}\n(n={n_per_cat[c]})" for c in CAT_ORDER],
                       fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"Roofline utilization $R$ (autotuned)", fontsize=13)
    ax.grid(True, alpha=0.18, axis="y")
    ax.legend(loc="upper right", frameon=True, edgecolor="lightgray", fontsize=12)

    fig.tight_layout()
    out = FIG_DIR / "Figure8b.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_rq2_a(per_op_R: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for (op, b), d in per_op_R.items():
        x0, y0 = d["default_R"], d["autotune_R"]
        ax.annotate(
            "", xy=(x0, y0), xytext=(x0, x0),
            arrowprops=dict(arrowstyle="->", color=BACKEND_COLOR[b],
                            lw=1.0, alpha=0.55),
        )
        ax.scatter(x0, y0, marker=("o" if b == "triton" else "^"),
                   color=BACKEND_COLOR[b], s=42, alpha=0.85,
                   edgecolors="black", linewidth=0.4, zorder=4)

    xx = np.linspace(0, 1, 50)
    ax.plot(xx, xx, ls="--", color="gray", lw=0.8, alpha=0.7)
    ax.text(0.98, 0.96, "y = x", color="gray", fontsize=9,
            ha="right", va="top", style="italic")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Default roofline utilization $R$")
    ax.set_ylabel(r"Autotuned roofline utilization $R$")
    ax.grid(True, alpha=0.18)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=BACKEND_COLOR["triton"],
                   markeredgecolor="black", markersize=8, label="Triton"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=BACKEND_COLOR["cutile"],
                   markeredgecolor="black", markersize=8, label="cuTile"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              edgecolor="lightgray", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "Figure3.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_rq2_b(main_rows: list[dict]) -> None:
    by_ob: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"default": [], "autotune": []})
    for r in main_rows:
        if r["R"] is None or r["R"] <= 0:
            continue
        by_ob[(r["op"], r["backend"])][r["mode"]].append(r["R"])

    triton_gains, cutile_gains = [], []
    for (op, b), d in by_ob.items():
        if not d["default"] or not d["autotune"]:
            continue
        n = min(len(d["default"]), len(d["autotune"]))
        gains = [d["autotune"][i] / d["default"][i]
                 for i in range(n) if d["default"][i] > 0]
        if not gains:
            continue
        (triton_gains if b == "triton" else cutile_gains).append(geomean(gains))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    bp = ax.boxplot([triton_gains, cutile_gains], positions=[1, 2],
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="black", lw=0.8),
                    capprops=dict(color="black", lw=0.8))
    for patch, b in zip(bp["boxes"], ("triton", "cutile")):
        patch.set_facecolor(BACKEND_COLOR[b])
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(1)
    for x_center, vals, b in [(1, triton_gains, "triton"),
                              (2, cutile_gains, "cutile")]:
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x_center + jitter, vals, s=18, alpha=0.75,
                   color=BACKEND_COLOR[b], edgecolors="black",
                   linewidth=0.25, zorder=4)

    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", ls=":", lw=0.7, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Triton", "cuTile"], fontsize=11)
    for label, b in zip(ax.get_xticklabels(), ("triton", "cutile")):
        label.set_color(BACKEND_COLOR[b])
    ax.set_ylabel(r"Autotune gain  $R_{\mathrm{autotune}} / R_{\mathrm{default}}$  ($\times$)")
    ax.set_title("autotuned / default", fontsize=11)
    ax.grid(True, alpha=0.18, axis="y", which="major")

    fig.tight_layout()
    out = FIG_DIR / "Figure9.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _fmt_param_val(v) -> str:
    if isinstance(v, (int, float)) and v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return str(v)


def _compact_params(params_str: str) -> str:
    parts = []
    for part in params_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            parts.append(f"{k}={_fmt_param_val(int(v))}")
        except ValueError:
            parts.append(part)
    return ", ".join(parts)


def fig_rq3_top20(main_rows: list[dict]) -> None:
    """Paper Fig 4: top-20 autotuned-mode latency gaps at the NCU-profiled
    sweep-max input.

    The (op, dtype) case is the one recorded in tilebench_run/ncu_catalogue.json
    (`default_params_per_dtype`) — i.e. exactly the case that has an NCU report
    — so the figure and the RQ3 profiling analysis refer to the same input.
    Falls back to the max-`infer_problem_size` case only when the catalogue has
    no entry (never the case for the 110 (op, dtype) pairs on main).
    """
    import json
    from data.tensors import infer_problem_size
    from scripts.analysis.bench_data import _parse_params
    import yaml

    catalogue = json.loads(
        (REPO / "tilebench_run" / "ncu_catalogue.json").read_text())
    ncu_case = {(c["op"], dt): c["default_params_per_dtype"][dt]
                for c in catalogue for dt in c["dtypes"]}

    by_od: dict[tuple, list[dict]] = defaultdict(list)
    for r in main_rows:
        if r["mode"] != "autotune":
            continue
        by_od[(r["op"], r["dtype"], r["params"])].append(r)

    case_defaults_cache: dict[str, dict] = {}

    def full_params(op: str, params_str: str) -> dict:
        if op not in case_defaults_cache:
            cfg = yaml.safe_load(
                (REPO / "benchmarks" / "operators" / op / "config.yaml").read_text())
            case_defaults_cache[op] = cfg.get("case_defaults") or {}
        p = dict(case_defaults_cache[op])
        p.update(_parse_params(params_str))
        return p

    def _is_ncu_case(op: str, dtype: str, fp: dict) -> bool:
        cat = ncu_case.get((op, dtype))
        if cat is None:
            return False
        return all(str(fp.get(k)) == str(v) for k, v in cat.items())

    best_case: dict[tuple, dict] = {}
    for (op, dtype, params_str), rs in by_od.items():
        t = next((r for r in rs if r["backend"] == "triton"), None)
        c = next((r for r in rs if r["backend"] == "cutile"), None)
        if t is None or c is None:
            continue
        fp = full_params(op, params_str)
        key = (op, dtype)
        gap = c["kernel_ms"] / t["kernel_ms"]
        entry = {
            "psize": infer_problem_size(op, fp), "params": params_str,
            "gap": max(gap, 1.0 / gap),
            "winner": "triton" if gap >= 1.0 else "cutile",
            "ncu": _is_ncu_case(op, dtype, fp),
        }
        prev = best_case.get(key)
        if prev is None:
            best_case[key] = entry
        elif entry["ncu"] and not prev["ncu"]:
            best_case[key] = entry              # NCU-profiled case wins outright
        elif entry["ncu"] == prev["ncu"] and entry["psize"] > prev["psize"]:
            best_case[key] = entry              # fallback: largest problem size
    missing = [k for k, v in best_case.items() if not v["ncu"]]
    if missing:
        print(f"  WARNING fig_rq3_top20: no NCU-catalogue case for {missing}; "
              f"used max problem_size instead")

    top = sorted(best_case.items(), key=lambda kv: -kv[1]["gap"])[:20]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ys = np.arange(len(top))[::-1]
    max_gap = max(d["gap"] for _, d in top)
    for y, ((op, dtype), d) in zip(ys, top):
        color = BACKEND_COLOR[d["winner"]]
        ax.barh(y, d["gap"] - 1.0, left=1.0, height=0.72, color=color,
                edgecolor="none", zorder=3)
        label = "Triton" if d["winner"] == "triton" else "cuTile"
        ax.text(d["gap"] * 1.05, y, f"{label} {d['gap']:.1f}×",
                va="center", ha="left", fontsize=11, fontweight="bold",
                color=color, clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{op} / {dtype} / {_compact_params(d['params'])}"
         for (op, dtype), d in top], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.95, max_gap * 1.45)
    ticks = [t for t in (1, 2, 3, 4, 6, 8) if t <= max_gap * 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}×" for t in ticks])
    ax.set_xlabel(r"Autotuned-mode latency gap (slower / faster) [$\times$]")
    ax.set_title("Top 20 Triton vs cuTile latency gaps (autotuned mode, NCU-profiled sweep-max input)",
                 fontsize=12)
    ax.grid(True, axis="x", alpha=0.25, which="both", zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR["triton"],
                             label="Triton faster"),
               plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR["cutile"],
                             label="cuTile faster")]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              edgecolor="lightgray", fontsize=11)

    fig.tight_layout()
    out = FIG_DIR / "Figure4.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = load_main_table()
    print(f"loaded {len(rows)} rows")
    per_op_autotune = per_op_mean_speedup(rows, mode="autotune")
    fig_rq1_a(per_op_autotune)
    fig_rq1_b(per_op_autotune)
    fig_rq1_c(rows)
    fig_rq2_a(per_op_mean_R(rows))
    fig_rq2_b(rows)
    fig_rq3_top20(rows)


if __name__ == "__main__":
    main()
