"""Figure 2 (RQ1): category x (device, DSL) proximity to the modeled, device-specific hybrid SOL reference (T_SOL / T_k).

T_SOL comes from the algorithm-aware compute mode of each operator and dtype (sol_modes.py) and the hybrid peaks
(sol_data.py: published dense rates for matmul_fp32_fp16_fp8, PR #323 empirical peaks otherwise); T_k is the formal autotuned latency. Cell = GM over the row's operators of each operator's GM over
its valid cases. The two rows under Overall split the 45 operators into memory-only targets (approved decision M2) and
targets with a compute term, as M2 requires."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import sol_data as SD  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAME = "fig_rq1_cross_accelerator"
COLUMNS = SD.COLUMNS          # the seven supported device/DSL combinations
MEM, CMP = "Memory-only targets", "Compute+memory targets"
ROW_LABEL = {"Point-wise": "Point-wise", "Reduction/Normalization": "Reduction/\nNormalization",
             "Matrix Multiplication/Attention": "Matrix Multiplication/\nAttention", "Stencil/Convolution": "Stencil/\nConvolution",
             "Data Layout": "Data Layout", "Overall": "Overall (45 operators)"}
GAP = 0.3          # horizontal gap between device groups, in cell widths
VMIN = 0.01        # colour floor of the log scale (printed values are not clipped)


def compute(S):
    rows = list(S.cat_order) + ["Overall", MEM, CMP]
    groups = {r: [o for o in S.operators() if S.categories[o] == r] for r in S.cat_order}
    groups.update({"Overall": S.operators(), MEM: S.memory_only_ops, CMP: [o for o in S.operators() if o not in S.memory_only_ops]})
    vals, counts = {}, {}
    for r in rows:
        for dev, dsl in COLUMNS:
            vals[(r, dev, dsl)], counts[(r, dev, dsl)] = S.r_group(dev, dsl, groups[r])
    return rows, groups, vals, counts


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


def plot(S, out_root):
    PS.apply()
    rows, groups, vals, counts = compute(S)
    norm = PS.proximity_norm(VMIN)
    xpos, xmax = xpositions()
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 2.78))
    ax = fig.add_axes([0.215, 0.02, 0.645, 0.79])
    cax = fig.add_axes([0.885, 0.08, 0.014, 0.66])
    ys, y = {}, 0.0
    for r in reversed(rows):                       # bottom-up: the two subgroup rows, a gap, Overall, a gap, the categories
        ys[r] = y
        y += {CMP: 0.8, MEM: 1.05, "Overall": 1.2}.get(r, 1.0)
    ytop = y
    for r in rows:
        sub = r in (MEM, CMP)
        h = 0.72 if sub else 0.9
        for ci, (dev, dsl) in enumerate(COLUMNS):
            v = vals[(r, dev, dsl)]
            c = PS.proximity_color(v, norm)
            ax.add_patch(Rectangle((xpos[ci] + 0.03, ys[r] + 0.04), 0.94, h, facecolor=c, edgecolor="white", lw=0.8))
            ax.text(xpos[ci] + 0.5, ys[r] + 0.04 + h / 2, PS.proximity_label(v), ha="center", va="center", fontsize=7 if sub else 7.5,
                    color="white" if luminance(to_rgba(c)) < 0.45 else PS.INK, fontweight="bold" if r == "Overall" else "normal")
    ax.set_xlim(-0.02, xmax + 0.02)
    ax.set_ylim(0, ytop)
    ax.axis("off")
    for r in rows:
        sub = r in (MEM, CMP)
        lab = (f"memory-only targets ({len(groups[r])})" if r == MEM else f"with compute term ({len(groups[r])})") if sub \
            else ROW_LABEL[r]
        ax.text(-0.12, ys[r] + (0.4 if sub else 0.49), lab, ha="right", va="center", fontsize=7 if sub else 7.5, linespacing=1.05,
                fontweight="bold" if r == "Overall" else "normal", color=PS.MUTED if sub else PS.INK)
    top = ytop
    for ci, (dev, dsl) in enumerate(COLUMNS):
        ax.text(xpos[ci] + 0.5, top + 0.08, PS.DSL_LABEL[dsl], ha="center", va="bottom", fontsize=7.5, clip_on=False)
    for dev in PS.DEVICES:
        cs = [xpos[i] for i, (d, _) in enumerate(COLUMNS) if d == dev]
        ax.plot([cs[0] + 0.06, cs[-1] + 0.94], [top + 0.62] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
        ax.text((cs[0] + cs[-1] + 1) / 2, top + 0.7, dev, ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
    ticks = [0.01, 0.03, 0.1, 0.3, 1]
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.PROXIMITY), cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(["≤0.01", "0.03", "0.1", "0.3", "1"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=7, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("proximity to modeled SOL\nT_SOL / T_k (GM)", fontsize=7, labelpad=3, linespacing=1.1)
    assert not any(v > 1 for v in vals.values()), "an aggregate exceeds 1: add the >1 key before publishing"
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    table = [{"row": r, "device": d, "dsl": s, "sol_proximity": vals[(r, d, s)], "n_operators": counts[(r, d, s)],
              "operators": len(groups[r])} for r in rows for d, s in COLUMNS]
    return paths, layout, table, groups


def main(out_root=FD.PLOTS):
    S = SD.Sol()
    D = FD.Data("autotune")
    paths, layout, table, groups = plot(S, out_root)
    m = D.manifest
    prov = json.load(open(SD.SOL / "sol_provenance.json"))
    manifest = {
        "figure": NAME, "rq": "RQ1", "script": "scripts/paper_figures/plot_rq1.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"]),
        "sol_inputs": SD.sol_input_hashes(),
        "sol_code_sha256": SD.sol_code_hashes(),
        "metric": "Proximity to modeled SOL (T_SOL / T_k)",
        "metric_formula": "R[o,b,d,c] = T_SOL[o,d,c] / T_k[o,b,d,c]; T_SOL = max(F / P_peak[mode(o,dtype)], Q / BW_peak) (memory-only "
                          "targets: Q / BW_peak); cell = GM over the row's operators of GM over each operator's valid cases",
        "aggregation_order": ["geometric mean of case-level R within an operator", "geometric mean over operators (category, all 45, or "
                              "an M2 subgroup)"],
        "colour_scale": f"sequential log scale from {VMIN} (floor) to 1 = modeled SOL; values > 1 drawn in amber; printed values not clipped",
        "peaks": prov["peaks"], "mode_decisions": SD.SM.DECISIONS,
        "columns": [f"{d}:{s}" for d, s in COLUMNS],
        "case_coverage": prov["coverage_cases"], "operator_coverage": prov["coverage_operators"],
        "row_operators": {r: len(v) for r, v in groups.items()},
        "memory_only_subgroup": {"operators": groups[MEM], "rule": "approved decision M2: memory-only target Q / BW_peak; reported "
                                 "beside the overall aggregate; not a compute+memory roofline"},
        "conditional_memory_dominance": prov["conditional_memory_dominance"],
        "sensitivity": "sol/sol_sensitivity_mi300x_bf16.csv (every row with and without the 100 MI300X bf16 conditional cases)",
        "above_one_cases": prov["above_one"]["n_cases"],
        "excluded_cases": {"invalid_or_missing_rows": len(D.excluded), "known_unsupported": m["known_unsupported"],
                           "missing_peak_mode_mappings": prov["missing_peak_mode_mappings"],
                           "not_shown": "cuTile and TileLang do not run on MI300X (no column)", "nki": "no finalized NKI results; not shown"},
        "selection_criteria": "all 45 operators; the seven supported device/DSL combinations; autotuned results",
        "interpretation": "each column is compared with its own device's hybrid SOL reference (published dense compute rates for the direct GEMM, "
                          "empirically calibrated rates otherwise); differences between columns are "
                          "differences in proximity, not absolute latency or hardware capability",
        "profiling_evidence_ids": [], "known_limitations": m["device_limitations"],
        "plotted_values": table,
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
