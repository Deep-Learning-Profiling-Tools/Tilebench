"""Figure 2 (RQ1): category x (device, DSL) geometric-mean speedup over the local PyTorch baseline."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAME = "fig_rq1_cross_accelerator"
COLUMNS = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"),
           ("GH200", "tilelang"), ("MI300X", "triton"), ("MI300X", "cutile"), ("MI300X", "tilelang")]


def compute(D):
    rows = list(D.cat_order) + ["Overall"]
    vals, counts = {}, {}
    for r in rows:
        ops = D.operators() if r == "Overall" else [o for o in D.operators() if D.categories[o] == r]
        for dev, dsl in COLUMNS:
            if not D.supported(dev, dsl):
                vals[(r, dev, dsl)] = None
                continue
            s, n = D.speedup_group(dev, dsl, ops)
            vals[(r, dev, dsl)] = s
            counts[(r, dev, dsl)] = n
    return rows, vals, counts


def luminance(rgba):
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot(D, out_root):
    PS.apply()
    rows, vals, counts = compute(D)
    vmax = 2.0  # log2(4x)
    norm = PS.log2_norm(vmax)
    gap = 0.35
    xpos, x = [], 0.0
    for i, (dev, dsl) in enumerate(COLUMNS):
        if i and COLUMNS[i - 1][0] != dev:
            x += gap
        xpos.append(x)
        x += 1.0
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 2.3))
    ax = fig.add_axes([0.165, 0.04, 0.70, 0.70])
    cax = fig.add_axes([0.895, 0.10, 0.012, 0.58])
    nrows = len(rows)
    for ri, r in enumerate(rows):
        y = nrows - 1 - ri + (0 if r == "Overall" else 0.25)
        for ci, (dev, dsl) in enumerate(COLUMNS):
            v = vals[(r, dev, dsl)]
            x0 = xpos[ci]
            if v is None:
                ax.add_patch(Rectangle((x0 + 0.03, y + 0.04), 0.94, 0.92, facecolor=PS.NA_FILL, edgecolor="white", lw=0.8,
                                       hatch="////", zorder=1))
                ax.text(x0 + 0.5, y + 0.5, "N/A", ha="center", va="center", fontsize=6.0, color=PS.MUTED, zorder=2)
                continue
            c = PS.DIVERGING(norm(math.log2(v)))
            ax.add_patch(Rectangle((x0 + 0.03, y + 0.04), 0.94, 0.92, facecolor=c, edgecolor="white", lw=0.8, zorder=1))
            ax.text(x0 + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if luminance(c) < 0.45 else PS.INK, fontweight="bold" if r == "Overall" else "normal", zorder=2)
    ax.set_xlim(-0.05, xpos[-1] + 1.05)
    ax.set_ylim(-0.05, nrows + 0.3)
    ax.set_yticks([nrows - 1 - ri + (0 if r == "Overall" else 0.25) + 0.5 for ri, r in enumerate(rows)])
    ax.set_yticklabels([PS.CATEGORY_SHORT.get(r, r) for r in rows])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "Overall":
            lab.set_fontweight("bold")
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0, pad=2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.axhline(1.12, color=PS.GRID, lw=0.6, xmin=0.0, xmax=1.0)
    ytop = nrows + 0.25
    for i, (dev, dsl) in enumerate(COLUMNS):
        ax.text(xpos[i] + 0.5, ytop + 0.12, PS.DSL_LABEL[dsl], ha="center", va="bottom", fontsize=6.6, color=PS.INK, clip_on=False)
    for dev in PS.DEVICES:
        cs = [xpos[i] for i, (d, _) in enumerate(COLUMNS) if d == dev]
        ax.plot([cs[0] + 0.08, cs[-1] + 0.92], [ytop + 0.72, ytop + 0.72], color="#9A9FA4", lw=0.6, clip_on=False)
        ax.text((cs[0] + cs[-1] + 1) / 2, ytop + 0.80, dev, ha="center", va="bottom", fontsize=7.5, fontweight="bold", clip_on=False)
    ticks = [0.25, 0.5, 1, 2, 4]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=PS.DIVERGING)
    cb = fig.colorbar(sm, cax=cax, ticks=[math.log2(t) for t in ticks])
    cb.ax.set_yticklabels([f"{t:g}×" for t in ticks])
    cb.ax.tick_params(labelsize=6.2, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("speedup vs. PyTorch (GM)", fontsize=6.4, labelpad=3)
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    table = [{"row": r, "device": d, "dsl": s, "speedup": vals[(r, d, s)], "n_operators": counts.get((r, d, s))}
             for r in rows for d, s in COLUMNS]
    return paths, layout, table


def main(out_root=FD.PLOTS):
    D = FD.Data("autotune")
    paths, layout, table = plot(D, out_root)
    m = D.manifest
    manifest = {
        "figure": NAME, "rq": "RQ1", "script": "scripts/paper_figures/plot_rq1.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"]),
        "metric_formula": "S[o,b,d] = GM_cases(torch_ms/dsl_ms) over valid autotune cases; cell = GM over the row's operators of S[o,b,d]",
        "aggregation_order": ["geometric mean over input cases within an operator", "geometric mean over operators (category or all 45)"],
        "colour_scale": "diverging, log2(speedup) in [-2, 2] centred at 0 (1x)",
        "case_coverage": m["intersections_case_id_v2"]["per_device_valid_autotune"],
        "excluded_cases": {"invalid_or_missing_rows": len(D.excluded), "known_unsupported": m["known_unsupported"],
                           "not_available": "cuTile and TileLang on MI300X (N/A cells, not zero)", "nki": "no finalized NKI results; not shown"},
        "selection_criteria": "all 45 operators; all supported device/DSL combinations; autotuned results",
        "profiling_evidence_ids": [], "known_limitations": m["device_limitations"],
        "plotted_values": table, "na_cells": [r for r in table if r["speedup"] is None],
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
