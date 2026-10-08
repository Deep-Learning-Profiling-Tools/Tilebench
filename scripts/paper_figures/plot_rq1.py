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
           ("GH200", "tilelang"), ("MI300X", "triton")]          # the seven supported device/DSL combinations
ROW_LABEL = {"Point-wise": "Point-wise", "Reduction/Normalization": "Reduction/\nNormalization",
             "Matrix Multiplication/Attention": "Matrix Multiplication/\nAttention", "Stencil/Convolution": "Stencil/\nConvolution",
             "Data Layout": "Data Layout", "Overall": "Overall (45 operators)"}
GAP = 0.3          # horizontal gap between device groups, in cell widths


def compute(D):
    rows = list(D.cat_order) + ["Overall"]
    vals, counts = {}, {}
    for r in rows:
        ops = D.operators() if r == "Overall" else [o for o in D.operators() if D.categories[o] == r]
        for dev, dsl in COLUMNS:
            vals[(r, dev, dsl)], counts[(r, dev, dsl)] = D.speedup_group(dev, dsl, ops)
    return rows, vals, counts


def luminance(rgba):
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def xpositions():
    xs, x = [], 0.0
    for i, (dev, _) in enumerate(COLUMNS):
        if i and COLUMNS[i - 1][0] != dev:
            x += GAP
        xs.append(x)
        x += 1.0
    return xs, x


def plot(D, out_root):
    PS.apply()
    rows, vals, counts = compute(D)
    norm = PS.log2_norm(2.0)                       # colour: log2 speedup, +-2 (1/4x .. 4x)
    xpos, xmax = xpositions()
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 2.2))
    ax = fig.add_axes([0.205, 0.02, 0.665, 0.74])
    cax = fig.add_axes([0.905, 0.08, 0.014, 0.62])
    n = len(rows)
    ys = {r: n - 1 - i + (0 if r == "Overall" else 0.2) for i, r in enumerate(rows)}
    for r in rows:
        for ci, (dev, dsl) in enumerate(COLUMNS):
            v = vals[(r, dev, dsl)]
            c = PS.DIVERGING(norm(max(-2.0, min(2.0, math.log2(v)))))
            ax.add_patch(Rectangle((xpos[ci] + 0.03, ys[r] + 0.05), 0.94, 0.9, facecolor=c, edgecolor="white", lw=0.8))
            ax.text(xpos[ci] + 0.5, ys[r] + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if luminance(c) < 0.45 else PS.INK, fontweight="bold" if r == "Overall" else "normal")
    ax.set_xlim(-0.02, xmax + 0.02)
    ax.set_ylim(0, n + 0.2)
    ax.axis("off")
    for r in rows:
        ax.text(-0.12, ys[r] + 0.5, ROW_LABEL[r], ha="right", va="center", fontsize=7.5, linespacing=1.05,
                fontweight="bold" if r == "Overall" else "normal")
    top = n + 0.2
    for ci, (dev, dsl) in enumerate(COLUMNS):
        ax.text(xpos[ci] + 0.5, top + 0.08, PS.DSL_LABEL[dsl], ha="center", va="bottom", fontsize=7.5, clip_on=False)
    for dev in PS.DEVICES:
        cs = [xpos[i] for i, (d, _) in enumerate(COLUMNS) if d == dev]
        ax.plot([cs[0] + 0.06, cs[-1] + 0.94], [top + 0.62] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
        ax.text((cs[0] + cs[-1] + 1) / 2, top + 0.7, dev, ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
    ticks = [0.25, 0.5, 1, 2, 4]
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.DIVERGING), cax=cax, ticks=[math.log2(t) for t in ticks])
    cb.ax.set_yticklabels(["1/4×", "1/2×", "1×", "2×", "4×"])
    cb.ax.tick_params(labelsize=7, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("speedup vs. local\nPyTorch (GM)", fontsize=7, labelpad=3, linespacing=1.05)
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    table = [{"row": r, "device": d, "dsl": s, "speedup": vals[(r, d, s)], "n_operators": counts[(r, d, s)]} for r in rows for d, s in COLUMNS]
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
        "colour_scale": "diverging, log2(speedup) in [-2, 2] centred at 0 (1x); printed values are not clipped",
        "columns": [f"{d}:{s}" for d, s in COLUMNS], "case_coverage": m["intersections_case_id_v2"]["per_device_valid_autotune"],
        "excluded_cases": {"invalid_or_missing_rows": len(D.excluded), "known_unsupported": m["known_unsupported"],
                           "not_shown": "cuTile and TileLang do not run on MI300X (no column)", "nki": "no finalized NKI results; not shown"},
        "selection_criteria": "all 45 operators; the seven supported device/DSL combinations; autotuned results",
        "profiling_evidence_ids": [], "known_limitations": m["device_limitations"],
        "plotted_values": table,
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
