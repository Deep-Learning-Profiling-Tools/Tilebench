"""Appendix figures A1-A5 (Evaluation Appendix). Reads only artifacts/paper_figures/combined/.

All figures are drawn at the ACL text width (plot_style.DOUBLE_COL_IN = 6.30 in) with text of at least 6 pt."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

W = PS.DOUBLE_COL_IN
S_COLS = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"),
          ("GH200", "tilelang"), ("MI300X", "triton")]
D_COLS = [("triton", "B200", "GH200"), ("triton", "B200", "MI300X"), ("cutile", "B200", "GH200"), ("tilelang", "B200", "GH200")]
ALIAS = {"matmul_fp32_fp16_fp8": "matmul", "block_sparse_attention": "block_sparse_attn", "linear_self_attention": "linear_self_attn"}


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def base_manifest(name, script, inputs, D):
    return {"figure": name, "rq": "Evaluation Appendix", "script": script, "source_git_commit": FD.git_head(),
            "source_data_files": FD.input_hashes(inputs), "known_limitations": D.manifest["device_limitations"]}


def finish(manifest, fig, out_root, name):
    paths, layout = PS.save(fig, out_root, "appendix", name)
    plt.close(fig)
    manifest["outputs"] = {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}
    manifest["layout"] = layout
    PS.write_manifest(out_root, name, manifest)
    return manifest


def row_layout(D, gap=0.6):
    ops = D.operators()
    ys, y, prev = {}, 0.0, None
    for o in ops:
        if prev is not None and D.categories[o] != D.categories[prev]:
            y += gap
        ys[o] = y
        y += 1.0
        prev = o
    return ops, ys, y


def axes_in(fig, H, x0, y0, w, h):
    """Axes placed in inches from the lower-left corner of a W x H figure."""
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def category_labels(fig, ax, D, ops, ys, ymax, x_in, H, fontsize=6.5):
    cats = []
    for o in ops:
        if not cats or cats[-1][0] != D.categories[o]:
            cats.append([D.categories[o], ys[o], ys[o]])
        cats[-1][2] = ys[o]
    pos = ax.get_position()
    for c, a, b in cats:
        yc = ymax - (a + b + 1) / 2
        fig.text(x_in / W, pos.y0 + pos.height * yc / ymax, PS.CATEGORY_SHORT[c], rotation=90, ha="left", va="center",
                 fontsize=fontsize, fontweight="bold", color=PS.CATEGORY_COLORS[c])


# ---------------------------------------------------------------------------------------------- A1
def fig_a1(D, out_root):
    name = "fig_a1_performance_atlas"
    ops, ys, ymax = row_layout(D)
    S = {(o, d, s): D.speedup_op(d, s, o) for o in ops for d, s in S_COLS}
    DL = {(o,) + c: D.delta_op(c[0], c[1], c[2], o) for o in ops for c in D_COLS}
    H, bot, top = 8.7, 0.62, 0.95
    fig = plt.figure(figsize=(W, H))
    axL = axes_in(fig, H, 1.42, bot, 2.98, H - bot - top)
    axR = axes_in(fig, H, 4.62, bot, 1.64, H - bot - top)
    nS, nD = PS.log2_norm(4.0), PS.log2_norm(2.0)
    xl = [j + 0.25 * (j >= 3) + 0.25 * (j >= 6) for j in range(len(S_COLS))]
    xr = [j + 0.25 * (j >= 2) for j in range(len(D_COLS))]
    for o in ops:
        y = ymax - ys[o] - 1
        for j, (d, s) in enumerate(S_COLS):
            v, n = S[(o, d, s)]
            c = PS.DIVERGING(nS(max(-4, min(4, math.log2(v)))))
            axL.add_patch(Rectangle((xl[j] + 0.03, y + 0.05), 0.94, 0.9, facecolor=c, edgecolor="white", lw=0.5))
            axL.text(xl[j] + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if lum(c) < 0.45 else PS.INK)
        for j, c_ in enumerate(D_COLS):
            v, n = DL[(o,) + c_]
            c = PS.DIVERGING(nD(max(-2, min(2, v))))
            axR.add_patch(Rectangle((xr[j] + 0.03, y + 0.05), 0.94, 0.9, facecolor=c, edgecolor="white", lw=0.5))
            axR.text(xr[j] + 0.5, y + 0.5, f"{2 ** v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if lum(c) < 0.45 else PS.INK)
    for ax, xm in ((axL, xl[-1] + 1), (axR, xr[-1] + 1)):
        ax.set_xlim(0, xm)
        ax.set_ylim(0, ymax)
        ax.axis("off")
    for o in ops:
        axL.text(-0.12, ymax - ys[o] - 0.5, o, ha="right", va="center", fontsize=6.5)
    category_labels(fig, axL, D, ops, ys, ymax, 0.04, H)
    for j, (d, s) in enumerate(S_COLS):
        axL.text(xl[j] + 0.5, ymax + 0.25, PS.DSL_LABEL[s], ha="center", va="bottom", fontsize=6.5, clip_on=False)
    for dev in PS.DEVICES:
        js = [j for j, (d, _) in enumerate(S_COLS) if d == dev]
        a, b = xl[js[0]], xl[js[-1]] + 1
        axL.plot([a + 0.06, b - 0.06], [ymax + 1.55] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
        axL.text((a + b) / 2, ymax + 1.65, dev, ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
    for j, (s, a, b) in enumerate(D_COLS):
        axR.text(xr[j] + 0.5, ymax + 0.25, f"{PS.DSL_LABEL[s]}\n{b}\nvs. {a}", ha="center", va="bottom", fontsize=6.5, clip_on=False,
                 linespacing=1.05)
    fig.text(1.42 / W, (H - 0.08) / H, "A  Speedup vs. local PyTorch (GM over cases)", ha="left", va="top", fontsize=7.5, fontweight="bold")
    fig.text(4.62 / W, (H - 0.08) / H, "B  Change in relative speedup", ha="left", va="top", fontsize=7.5, fontweight="bold")
    for cax_x, cax_w, norm, ticks, labels, lab in (
            (1.6, 2.6, nS, [-4, -2, 0, 2, 4], ["1/16×", "1/4×", "1×", "4×", "16×"], "speedup vs. local PyTorch (colour clipped at 1/16× and 16×)"),
            (4.7, 1.48, nD, [-2, -1, 0, 1, 2], ["1/4×", "1/2×", "1×", "2×", "4×"], "S(dev 2) / S(dev 1) over matched cases")):
        cax = axes_in(fig, H, cax_x, 0.36, cax_w, 0.07)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.DIVERGING), cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels(labels)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        cax.set_title(lab, fontsize=6.5, pad=2)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"], D)
    m.update({"metric_formula": {"left": "S[o,b,d] = GM_cases(torch_ms/dsl_ms) (valid autotune cases)",
                                 "right": "Δ = log2(S_dev2/S_dev1), both S over the case_id_v2 valid on both devices; cell shows 2^Δ; not an absolute hardware speedup"},
              "aggregation_order": ["GM over input cases within operator"], "colour_scale": {"left": "log2 S clipped to ±4", "right": "Δ clipped to ±2"},
              "case_coverage": {"left": {f"{o}|{d}|{s}": S[(o, d, s)][1] for o in ops for d, s in S_COLS},
                                "right_matched_cases": {f"{o}|{s}|{b}/{a}": DL[(o, s, a, b)][1] for o in ops for s, a, b in D_COLS}},
              "excluded_cases": {"invalid_or_missing_rows": len(D.excluded), "known_unsupported": D.manifest["known_unsupported"],
                                 "not_shown": "cuTile/TileLang on MI300X do not exist; Triton MI300X/B200 for matmul fp8 excluded by matching"},
              "selection_criteria": "all 45 operators", "profiling_evidence_ids": [],
              "plotted_values": {"left": [{"operator": o, "device": d, "dsl": s, "speedup": S[(o, d, s)][0]} for o in ops for d, s in S_COLS],
                                 "right": [{"operator": o, "dsl": s, "from": a, "to": b, "delta_log2": DL[(o, s, a, b)][0]} for o in ops for s, a, b in D_COLS]}})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A2
A2_CANDIDATES = ["matmul_fp32_fp16_fp8", "flash_decode", "linear_self_attention", "top_k_selection", "vector_add", "streamk_matmul"]
A2_GRID = {"flash_decode": ("seq_len", None), "linear_self_attention": ("M", "D"), "top_k_selection": ("N", "k"),
           "streamk_matmul": ("n", "m"), "matmul_fp32_fp16_fp8": ("K", None), "vector_add": ("n", None)}
A2_MAX_TICKS = 12          # panels with more ungrouped cases label every second case (all cases are still drawn)


def a2_variation(D, op):
    rng = []
    for (dev, dsl, o), c in D.cases.items():
        if o != op:
            continue
        by = {}
        for cid, (t, d, dt, p) in c.items():
            by.setdefault(dt, []).append(math.log2(t / d))
        rng += [max(v) - min(v) for v in by.values()]
    rng.sort()
    return rng[len(rng) // 2]


def fmt_num(v):
    if v >= 1 << 20 and v % (1 << 20) == 0:
        return f"{v >> 20}M"
    if v >= 1024 and v % 1024 == 0:
        return f"{v >> 10}k"
    return str(v)


def fig_a2(D, out_root):
    name = "fig_a2_shape_dtype"
    var = {o: a2_variation(D, o) for o in A2_CANDIDATES}
    shape_sel = [o for o in A2_CANDIDATES if var[o] >= 0.5 and o != "matmul_fp32_fp16_fp8"]
    selected = shape_sel + ["matmul_fp32_fp16_fp8"]
    panels = []
    for op in selected:
        for dt in sorted({D.cases[k][c][2] for k in D.cases if k[2] == op for c in D.cases[k]}):
            panels.append((op, dt))
    ncol, nrow = 2, math.ceil(len(panels) / 2)
    H = 1.36 * nrow + 0.15
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, H))
    fig.subplots_adjust(left=0.105, right=0.995, top=1 - 0.3 / H, bottom=0.42 / H, hspace=1.05, wspace=0.04)
    norm = PS.log2_norm(4.0)
    values, ticks_shown = [], {}
    for idx, (ax, (op, dt)) in enumerate(zip(axes.flat, panels)):
        outer_k, inner_k = A2_GRID[op][1], A2_GRID[op][0]
        allp = {}
        for (dev, dsl, o), c in D.cases.items():
            if o == op:
                for cid, (t, d, dtt, p) in c.items():
                    if dtt == dt:
                        allp[cid] = json.loads(p)
        order = sorted(allp, key=lambda cid: ((allp[cid][outer_k] if outer_k else 0), allp[cid][inner_k]))
        mat = np.full((len(S_COLS), len(order)), np.nan)
        for i, (dev, dsl) in enumerate(S_COLS):
            for j, cid in enumerate(order):
                c = D.case(dev, dsl, op, cid)
                if c is not None:
                    mat[i, j] = math.log2(c[0] / c[1])
                    values.append({"operator": op, "dtype": dt, "device": dev, "dsl": dsl, "case_id_v2": cid, "speedup": c[0] / c[1]})
        cm = PS.DIVERGING.copy()
        cm.set_bad(PS.NA_FILL)
        ax.imshow(np.clip(mat, -4, 4), aspect="auto", cmap=cm, norm=norm, interpolation="nearest")
        for i in range(len(S_COLS)):
            if np.all(np.isnan(mat[i])):
                ax.text(len(order) / 2 - 0.5, i, "N/A (unsupported dtype)" if S_COLS[i][0] == "MI300X" else "N/A", ha="center",
                        va="center", fontsize=6, color=PS.MUTED)
        ax.set_yticks(range(len(S_COLS)))
        ax.set_yticklabels([f"{d} {PS.DSL_SHORT[s]}" for d, s in S_COLS] if idx % ncol == 0 else [], fontsize=6.5)
        step = 2 if (outer_k is None and len(order) > A2_MAX_TICKS) else 1
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([fmt_num(allp[c][inner_k]) if j % step == 0 else "" for j, c in enumerate(order)], fontsize=6, rotation=90)
        ticks_shown[f"{op}|{dt}"] = {"cases": len(order), "label_every": step}
        if outer_k:
            groups = []
            for j, cid in enumerate(order):
                v = allp[cid][outer_k]
                if not groups or groups[-1][0] != v:
                    groups.append([v, j, j])
                groups[-1][2] = j
            for v, a, b in groups:
                if a:
                    ax.axvline(a - 0.5, color="white", lw=1.4)
                ax.text((a + b) / 2, -0.7, f"{outer_k}={fmt_num(v)}", ha="center", va="bottom", fontsize=6, color=PS.INK)
        ax.tick_params(length=0, pad=1.5)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.axhline(2.5, color="white", lw=1.4)
        ax.axhline(5.5, color="white", lw=1.4)
        ax.set_title(f"{ALIAS.get(op, op)} · {dt}", loc="left", fontsize=7, fontweight="bold", pad=12)
        ax.set_xlabel(inner_k, fontsize=6.5, labelpad=1)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    assert len(panels) < nrow * ncol, "A2 places its colour bar in the empty last panel slot"
    pos = list(axes.flat)[-1].get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 + 0.55 * pos.height, 0.8 * pos.width, 0.07 * pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.DIVERGING), cax=cax, orientation="horizontal", ticks=[-4, -2, 0, 2, 4])
    cb.ax.set_xticklabels(["1/16×", "1/4×", "1×", "4×", "16×"])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("speedup vs. local PyTorch (per input case)", fontsize=6.5, labelpad=2)
    cb.outline.set_linewidth(0.4)
    fig.text(pos.x0 + 0.1 * pos.width, pos.y0 + 0.02 * pos.height, "T = Triton, C = cuTile, TL = TileLang", fontsize=6.5, color=PS.MUTED)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "comparison_manifest.json"], D)
    m.update({"metric_formula": "per input case: torch_ms/dsl_ms (formal autotuned CSV), log2 colour scale clipped to ±4",
              "aggregation_order": ["none (each cell is one input case)"],
              "selection_criteria": {"rule": "candidates with median within-(device,DSL,dtype) log2-speedup range >= 0.5 are shown for shape sensitivity; matmul_fp32_fp16_fp8 is kept for dtype sensitivity with ALL its dtypes",
                                     "variation_median_log2_range": var, "selected": selected,
                                     "not_selected": {o: f"median log2 range {var[o]:.2f} < 0.5" for o in A2_CANDIDATES if o not in selected}},
              "tick_labels": ticks_shown,
              "case_coverage": {f"{op}|{dt}": sum(1 for v in values if v["operator"] == op and v["dtype"] == dt) for op, dt in panels},
              "excluded_cases": {"MI300X matmul fp8_e4m3fn": "unsupported dtype (row shown as N/A)", "invalid_or_missing_rows": len(D.excluded)},
              "x_axis": {op: {"inner": A2_GRID[op][0], "outer_group": A2_GRID[op][1]} for op in selected},
              "profiling_evidence_ids": [], "plotted_values": values})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A3
PATH_COLORS = {"tcgen05": "#397E8A", "wgmma": "#8DB3BC", "legacy_mma": "#C58458", "mfma": "#9276A5", "none": "#FFFFFF",
               "unknown": "#EDEDEA", "n/a": "#EDEDEA"}
PATH_LABEL = {"tcgen05": "tcgen05", "wgmma": "WGMMA", "legacy_mma": "legacy MMA (HMMA/IMMA)", "mfma": "MFMA", "none": "no matrix instruction",
              "unknown": "unknown / no profile", "n/a": "unsupported"}
SECONDARY = ("spill", "atom.global", "atom.shared", "atomic", "ldsm/stsm", "convert_layout")


def staging(flags):
    f = set(flags)
    if "TMEM" in f and "smem" in f:
        return "TMEM+SMEM"
    if "TMEM" in f:
        return "TMEM"
    if "smem" in f:
        return "SMEM"
    if "LDS" in f:
        return "LDS"
    return "–"


def fig_a3(D, out_root):
    name = "fig_a3_execution_paths"
    rows = FD.read_csv(FD.COMBINED / "execution_path_matrix.csv")
    keys = []
    for r in rows:
        if (r["operator"], r["dtype"]) not in keys:
            keys.append((r["operator"], r["dtype"]))
    cell = {(r["operator"], r["dtype"], r["device"], r["dsl"]): r for r in rows}
    n = len(keys)
    H = 0.29 * n + 1.25
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, H, 1.42, 0.62, W - 1.47, 0.29 * n)
    xs = [j + 0.25 * (j >= 3) + 0.25 * (j >= 6) for j in range(len(S_COLS))]
    secondary = []
    for i, (op, dt) in enumerate(keys):
        y = n - 1 - i
        for j, (dev, dsl) in enumerate(S_COLS):
            r = cell[(op, dt, dev, dsl)]
            mp = r["matrix_path"]
            static = r["evidence_kind"] in ("static_sass", "static_isa")
            ax.add_patch(Rectangle((xs[j] + 0.04, y + 0.07), 0.92, 0.86, facecolor=PATH_COLORS.get(mp, "#EDEDEA"),
                                   edgecolor="#6E747A" if mp not in ("n/a", "unknown") else "white", lw=0.6,
                                   ls=(0, (2, 1.5)) if static else "-", hatch="////" if mp == "n/a" else None))
            if mp in ("n/a", "unknown"):
                ax.text(xs[j] + 0.5, y + 0.5, "unsupported" if mp == "n/a" else "?", ha="center", va="center", fontsize=6.5, color=PS.MUTED)
                continue
            fl = [f for f in r["flags"].split(";") if f]
            dark = mp in ("tcgen05", "mfma")
            ax.text(xs[j] + 0.5, y + 0.66, r["operand_path"], ha="center", va="center", fontsize=6, color="white" if dark else PS.INK)
            ax.text(xs[j] + 0.5, y + 0.31, staging(fl), ha="center", va="center", fontsize=6, color="white" if dark else "#4A4F54")
            secondary.append({"operator": op, "dtype": dt, "device": dev, "dsl": dsl, "evidence_kind": r["evidence_kind"],
                              **{f: int(f in fl) for f in SECONDARY}})
    ax.set_xlim(0, xs[-1] + 1)
    ax.set_ylim(0, n)
    ax.axis("off")
    for i, (op, dt) in enumerate(keys):
        ax.text(-0.1, n - 1 - i + 0.5, f"{ALIAS.get(op, op)} · {dt}", ha="right", va="center", fontsize=6.5)
    for j, (dev, dsl) in enumerate(S_COLS):
        ax.text(xs[j] + 0.5, n + 0.1, PS.DSL_LABEL[dsl], ha="center", va="bottom", fontsize=6.5, clip_on=False)
    for dev in PS.DEVICES:
        js = [j for j, (d, _) in enumerate(S_COLS) if d == dev]
        a, b = xs[js[0]], xs[js[-1]] + 1
        ax.plot([a + 0.06, b - 0.06], [n + 0.75] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
        ax.text((a + b) / 2, n + 0.82, dev, ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
    h = [Patch(facecolor=PATH_COLORS[k], edgecolor="#6E747A", lw=0.5, label=PATH_LABEL[k]) for k in ("tcgen05", "wgmma", "legacy_mma", "mfma", "none")]
    h += [Patch(facecolor="white", edgecolor="#6E747A", lw=0.6, ls=(0, (2, 1.5)), label="static SASS/ISA only"),
          Patch(facecolor=PATH_COLORS["n/a"], hatch="////", edgecolor="white", label="unsupported")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.0)
    fig.text(1.42 / W, 0.56 / H, "Top line: operand load path (TMA, cp.async, LDG, desc→ptr = TensorDescriptor lowered to pointer loads, pointer).\n"
             "Bottom line: on-chip staging (SMEM, TMEM, LDS; – = none observed).", fontsize=6, color=PS.MUTED, ha="left", va="top",
             linespacing=1.15)
    tab = Path(out_root) / "tables" / "fig_a3_secondary_flags.csv"
    tab.parent.mkdir(parents=True, exist_ok=True)
    with open(tab, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(secondary[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(secondary)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["execution_path_matrix.csv"], D)
    m.update({"metric_formula": "categorical: matrix-instruction family (colour), operand load path and on-chip staging (labels), derived from execution_paths.csv (NVIDIA dynamic opcode-with-modifier counts or static SASS; MI300X static ISA + source form)",
              "aggregation_order": ["union over the kernels (stages) of the profiled run"],
              "selection_criteria": "15 representative operator/dtype pairs covering GEMM, attention, convolution, scatter, histogram, pooling and streaming",
              "evidence_semantics": "absence of a label or flag means it was not observed in the available evidence; static-only cells (dashed) show code present in the binary, not executed counts; TMA = UTMALDG operand loads only; atom.shared counts only > 64 shared atomics per CTA (tcgen05 TMEM-allocator bookkeeping excluded)",
              "secondary_flags_table": {"path": str(tab.relative_to(out_root)), "sha256": PS.sha256(tab), "columns": list(SECONDARY)},
              "case_coverage": len(rows), "excluded_cases": {"MI300X matmul fp8": "unsupported"},
              "profiling_evidence_ids": [r["profile_id"] for r in rows if r["profile_id"]], "plotted_values": rows})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A4
def fig_a4(D, out_root):
    name = "fig_a4_within_device_matrix"
    ops, ys, ymax = row_layout(D)
    cols = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"), ("GH200", "tilelang")]
    SL = {}
    for o in ops:
        for dev in ("B200", "GH200"):
            ids = D.matched_ids(o, [(dev, s) for s in ("triton", "cutile", "tilelang")])
            g = {s: D.latency_gm(dev, s, o, ids) for s in ("triton", "cutile", "tilelang")}
            mn = min(g.values())
            for s in g:
                SL[(o, dev, s)] = (g[s] / mn, len(ids))
    norm = LogNorm(vmin=1.0, vmax=8.0)
    H, bot, top = 7.9, 0.15, 0.55
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, H, 1.62, bot, 3.62, H - bot - top)
    xs = [j * 0.95 + 0.3 * (j >= 3) for j in range(6)]
    for o in ops:
        y = ymax - ys[o] - 1
        for j, (dev, s) in enumerate(cols):
            v, n = SL[(o, dev, s)]
            near = v <= 1.05
            c = (0.97, 0.96, 0.94, 1.0) if near else PS.SEQUENTIAL(norm(min(v, 8.0)))
            ax.add_patch(Rectangle((xs[j] + 0.04, y + 0.06), 0.87, 0.88, facecolor=c, edgecolor="white", lw=0.5))
            if v == 1.0:
                ax.add_patch(Rectangle((xs[j] + 0.06, y + 0.09), 0.83, 0.82, facecolor="none", edgecolor="#4A4F54", lw=0.8))
            ax.text(xs[j] + 0.475, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (not near and lum(c) < 0.5) else (PS.MUTED if near and v > 1.0 else PS.INK))
    ax.set_xlim(0, xs[-1] + 0.95)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    for o in ops:
        ax.text(-0.12, ymax - ys[o] - 0.5, o, ha="right", va="center", fontsize=7)
    category_labels(fig, ax, D, ops, ys, ymax, 0.06, H, fontsize=7)
    for j, (dev, s) in enumerate(cols):
        ax.text(xs[j] + 0.475, ymax + 0.25, PS.DSL_LABEL[s], ha="center", va="bottom", fontsize=7, clip_on=False)
    for a, b, t in ((xs[0], xs[2] + 0.95, "B200"), (xs[3], xs[5] + 0.95, "GH200")):
        ax.text((a + b) / 2, ymax + 1.55, t, ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
        ax.plot([a + 0.06, b - 0.06], [ymax + 1.45] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
    cax = axes_in(fig, H, 5.55, H - top - 2.4, 0.09, 2.2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.SEQUENTIAL), cax=cax, ticks=[1, 2, 4, 8])
    cb.ax.set_yticklabels(["1×", "2×", "4×", "≥8×"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("slowdown vs. fastest DSL\non the same device", fontsize=7, labelpad=3, linespacing=1.1)
    ax2 = axes_in(fig, H, 5.42, H - top - 3.15, 0.85, 0.5)
    ax2.axis("off")
    ax2.add_patch(Rectangle((0.0, 0.55), 0.22, 0.35, facecolor=(0.97, 0.96, 0.94), edgecolor="#4A4F54", lw=0.8))
    ax2.text(0.3, 0.72, "fastest", va="center", fontsize=7)
    ax2.add_patch(Rectangle((0.0, 0.05), 0.22, 0.35, facecolor=(0.97, 0.96, 0.94), edgecolor="white", lw=0.5))
    ax2.text(0.3, 0.22, "≤ 1.05×", va="center", fontsize=7)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"], D)
    m.update({"metric_formula": "slowdown = GM_cases(latency_DSL) / min_DSL GM_cases(latency_DSL), cases valid for all three NVIDIA DSLs on the device",
              "aggregation_order": ["GM over matched cases", "ratio to the per-operator minimum"], "near_parity_threshold": 1.05,
              "case_coverage": {f"{o}|{d}": SL[(o, d, 'triton')][1] for o in ops for d in ("B200", "GH200")},
              "excluded_cases": {"invalid_or_missing_rows": len(D.excluded)}, "selection_criteria": "all 45 operators",
              "profiling_evidence_ids": [],
              "plotted_values": [{"operator": o, "device": d, "dsl": s, "slowdown": SL[(o, d, s)][0]} for o in ops for d, s in cols]})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A5
A5_INSTR_CASES = [("weight_dequant/bf16", "weight_dequant\nBF16"), ("destindex/int8", "destindex\nINT8"),
                  ("cross_entropy/fp16", "cross_entropy\nFP16"), ("moe_topk_gating/fp16", "moe_topk_gating\nFP16"),
                  ("flash_decode/fp32", "flash_decode\nFP32")]
A5_CONFOUNDED = {("weight_dequant/bf16", "GH200"): "GH200 Triton uses a different autotuned configuration (confounder)"}
FLUSH = [("write_flush_512MiB(formal)", "write flush\n(formal)"), ("read_flush_512MiB", "read flush"), ("no_flush", "no flush")]


class A5Ev:
    def __init__(self):
        self.rows = FD.read_csv(FD.COMBINED / "figure_evidence.csv")
        self.axes = {}
        self.axis = None

    def find(self, case, dev, dsl, metric, exact=True):
        hits = [r for r in self.rows if r["figure"] == "a5" and r["case"] == case and r["device"] == dev and r["dsl"] == dsl
                and (r["metric_name"] == metric if exact else metric in r["metric_name"])]
        assert len(hits) == 1, (case, dev, dsl, metric, len(hits))
        self.axes.setdefault(self.axis, []).append(hits[0]["evidence_id"])
        return hits[0]

    def num(self, *a, **k):
        v = self.find(*a, **k)["value"]
        return None if v == "" else float(v)


def fig_a5(D, out_root):
    name = "fig_a5_profiling_evidence"
    E = A5Ev()
    H = 6.1
    fig = plt.figure(figsize=(W, H))
    col = PS.DSL_COLORS
    FS = 6.5

    def title(ax, t):
        ax.set_title(t, loc="left", fontsize=7, pad=3, fontweight="bold" if t[:3] in ("A  ", "B  ", "C  ") else "normal")

    # ---- A: dynamic instruction count relative to Triton on the same device
    E.axis = "A_nvidia_dynamic_instruction_ratio"
    ax = axes_in(fig, H, 0.5, 4.62, 5.75, 1.2)
    xt, xl = [], []
    for ci, (case, lab) in enumerate(A5_INSTR_CASES):
        for di, dev in enumerate(("B200", "GH200")):
            x = ci * 2.6 + di * 1.05
            t = E.num(case, dev, "triton", "smsp__inst_executed.sum")
            for si, s in enumerate(("cutile", "tilelang")):
                v = E.num(case, dev, s, "smsp__inst_executed.sum")
                r = v / t
                ax.bar(x + (si - 0.5) * 0.42, r, width=0.4, color=col[s], edgecolor="white", lw=0.3)
                ax.text(x + (si - 0.5) * 0.42, r * 1.07 if r >= 1 else r / 1.07, f"{r:.1f}", ha="center",
                        va="bottom" if r >= 1 else "top", fontsize=6)
            xt.append(x)
            xl.append(dev + ("†" if (case, dev) in A5_CONFOUNDED else ""))
        ax.text(ci * 2.6 + 0.525, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS, linespacing=1.1)
    ax.set_yscale("log", base=2)
    ax.axhline(1, color="#9A9FA4", lw=0.6)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=FS)
    ax.set_ylim(1 / 8, 64)
    ax.set_yticks([1 / 8, 1 / 2, 1, 2, 8, 32])
    ax.set_yticklabels(["1/8×", "1/2×", "1×", "2×", "8×", "32×"])
    ax.minorticks_off()
    ax.tick_params(labelsize=FS)
    ax.set_ylabel("instructions\nrelative to Triton", fontsize=FS, linespacing=1.0)
    title(ax, "A  Indexing and reduction overhead: executed instructions (same device, profiled input)")
    ax.legend(handles=[Patch(color=col["cutile"], label="cuTile / Triton"), Patch(color=col["tilelang"], label="TileLang / Triton")],
              loc="upper right", fontsize=FS, ncol=2, borderaxespad=0.2)
    ax.text(0.0, -0.5, "† " + A5_CONFOUNDED[("weight_dequant/bf16", "GH200")], transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color=PS.MUTED)

    # ---- B: matrix operand delivery
    dts = [("fp32", "FP32"), ("fp16", "FP16"), ("fp8_e4m3fn", "FP8")]
    E.axis = "B_gh200_sts_per_wgmma"
    b1 = axes_in(fig, H, 0.5, 2.38, 1.55, 1.05)
    for i, (dt, _) in enumerate(dts):
        for si, s in enumerate(("triton", "cutile")):
            c = f"matmul_fp32_fp16_fp8/{dt}"
            st = E.num(c, "GH200", s, "sass__inst_executed_per_opcode_with_modifier_all[STS*]")
            w = E.num(c, "GH200", s, "sass__inst_executed_per_opcode[family=wgmma]")
            x = i + (si - 0.5) * 0.38
            r = st / w
            b1.bar(x, r, width=0.36, color=col[s], edgecolor="white", lw=0.3)
            b1.text(x, r + 0.35, f"{r:.2f}", ha="center", va="bottom", fontsize=6)
    b1.set_xticks(range(3))
    b1.set_xticklabels([d for _, d in dts], fontsize=FS)
    b1.set_ylim(0, 19)
    b1.tick_params(labelsize=FS)
    b1.set_ylabel("STS per WGMMA", fontsize=FS)
    title(b1, "B  GH200 matmul: shared-memory\nstores per WGMMA")
    E.axis = "B_b200_tma_load_bytes"
    b2 = axes_in(fig, H, 2.62, 2.38, 1.55, 1.05)
    for i, (dt, _) in enumerate(dts):
        for si, s in enumerate(("triton", "cutile", "tilelang")):
            v = E.num(f"matmul_fp32_fp16_fp8/{dt}", "B200", s, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum")
            x = i + (si - 1) * 0.28
            if v is None:
                b2.text(x, 0.4, "n/c", ha="center", va="bottom", fontsize=6, color=PS.MUTED, rotation=90)
                continue
            b2.bar(x, v / 1e9, width=0.26, color=col[s], edgecolor="white", lw=0.3)
            if v == 0:
                b2.text(x, 0.4, "0", ha="center", va="bottom", fontsize=6)
    b2.set_xticks(range(3))
    b2.set_xticklabels([d for _, d in dts], fontsize=FS)
    b2.set_ylim(0, 24)
    b2.tick_params(labelsize=FS)
    b2.set_ylabel("TMA load bytes (GB)", fontsize=FS)
    title(b2, "B200 matmul: TMA operand\nload bytes")
    b2.legend(handles=[Patch(color=col[s], label=PS.DSL_LABEL[s]) for s in ("triton", "cutile", "tilelang")], fontsize=6, loc="upper right",
              handlelength=1.0, borderaxespad=0.1)
    E.axis = "B_mi300x_diagnostic_latency"
    b3 = axes_in(fig, H, 4.72, 2.38, 1.5, 1.05)
    variants = [("torch.matmul (hipBLASLt)", "PyTorch", PS.DSL_COLORS["pytorch"], None),
                ("TileBench matmul_kernel (TensorDescriptor), formal winner", "descriptor (winner)", "#2F4E63", None),
                ("pointer-load kernel, same tile/grouping as winner", "pointer variant", col["triton"], None)]
    for i, dt in enumerate(("fp32", "fp16")):
        for j, (var, _, cc, _) in enumerate(variants):
            v = E.num(f"matmul_fp32_fp16_fp8/{dt}", "MI300X", "pytorch" if "torch" in var else "triton",
                      f"diagnostic:gemm_desc_vs_ptr_{dt}:{var}")
            x = i * 1.15 + (j - 1) * 0.33
            hatch = "////" if (dt == "fp16" and j == 2) else None
            b3.bar(x, v, width=0.31, color=cc, edgecolor="white", lw=0.3, hatch=hatch)
            b3.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    b3.set_xticks([0, 1.15])
    b3.set_xticklabels(["FP32", "FP16"], fontsize=FS)
    b3.set_ylim(0, 46)
    b3.tick_params(labelsize=FS)
    b3.set_ylabel("latency (ms)", fontsize=FS)
    title(b3, "MI300X matmul (diagnostic):\ndescriptor vs. pointer loads")
    b3.legend(handles=[Patch(color=c_, label=l_) for _, l_, c_, _ in variants] +
              [Patch(facecolor=col["triton"], hatch="////", edgecolor="white", label="also num_stages 3→2")],
              fontsize=6, loc="upper right", handlelength=1.0, borderaxespad=0.1)

    # ---- C: memory access and latency hiding
    E.axis = "C_nvidia_1d_conv_load_traffic_ratio"
    c1 = axes_in(fig, H, 0.95, 0.45, 1.25, 1.1)
    metrics = [("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "load requests", "o", "none"),
               ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "L1 load sectors", "o", "fill"),
               ("dram__bytes_read.sum", "DRAM read bytes", "D", "none")]
    rowsC = [("B200", "cutile"), ("B200", "tilelang"), ("GH200", "cutile"), ("GH200", "tilelang")]
    for yi, (dev, s) in enumerate(rowsC):
        y = 3 - yi
        for k, (mname, _, mk, fill) in enumerate(metrics):
            r = E.num("1d_conv/fp16", dev, s, mname) / E.num("1d_conv/fp16", dev, "triton", mname)
            c1.scatter(r, y + 0.2 - 0.2 * k, marker=mk, s=16, facecolor=col[s] if fill == "fill" else "white", edgecolor=col[s], linewidth=0.9, zorder=3)
    c1.set_xscale("log", base=2)
    c1.set_xlim(0.7, 24)
    c1.axvline(1, color="#9A9FA4", lw=0.6)
    c1.set_xticks([1, 2, 4, 8, 16])
    c1.set_xticklabels(["1×", "2×", "4×", "8×", "16×"])
    c1.minorticks_off()
    c1.set_yticks(range(4))
    c1.set_yticklabels([f"{d} {PS.DSL_LABEL[s]}" for d, s in reversed(rowsC)], fontsize=FS)
    c1.set_ylim(-0.5, 4.6)
    c1.tick_params(labelsize=FS)
    c1.grid(axis="x", color=PS.GRID, lw=0.35)
    c1.set_xlabel("ratio to Triton (same device)", fontsize=FS)
    title(c1, "C  1d_conv FP16:\nload traffic")
    c1.legend(handles=[Line2D([], [], marker=mk, ls="", markerfacecolor="#6E747A" if fill == "fill" else "white", markeredgecolor="#6E747A",
                              markersize=4, label=lab) for _, lab, mk, fill in metrics],
              fontsize=6, loc="upper right", handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    E.axis = "C_nvidia_occupancy_pct"
    c2 = axes_in(fig, H, 2.62, 0.45, 1.55, 1.1)
    issue = []
    for di, dev in enumerate(("B200", "GH200")):
        for si, s in enumerate(("triton", "cutile", "tilelang")):
            a = E.num("1d_conv/fp16", dev, s, "sm__warps_active.avg.pct_of_peak_sustained_active")
            th = E.num("1d_conv/fp16", dev, s, "sm__maximum_warps_per_active_cycle_pct")
            x = di * 1.15 + (si - 1) * 0.3
            c2.bar(x, a, width=0.28, color=col[s], edgecolor="white", lw=0.3)
            c2.plot([x - 0.14, x + 0.14], [th, th], color=PS.INK, lw=0.9)
            issue.append((x, E.num("1d_conv/fp16", dev, s, "smsp__issue_active.avg.pct_of_peak_sustained_active")))
    E.axis = "C_nvidia_occupancy_pct"
    c2.set_xticks([0, 1.15])
    c2.set_xticklabels(["B200", "GH200"], fontsize=FS)
    c2.set_ylim(0, 60)
    c2.tick_params(labelsize=FS)
    c2.set_ylabel("achieved occupancy (%)", fontsize=FS)
    title(c2, "1d_conv FP16: achieved (bar) and\ntheoretical (line) occupancy")
    for x, iv in issue:
        c2.text(x, -0.27, f"{iv:.0f}", transform=c2.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=PS.MUTED)
    c2.text(-0.02, -0.27, "issue %", transform=c2.transAxes, ha="right", va="top", fontsize=6, color=PS.MUTED)
    E.axis = "C_mi300x_diagnostic_latency"
    c3 = axes_in(fig, H, 4.72, 0.45, 1.5, 1.1)
    vs = [("torch.add", "PyTorch (ATen)", PS.DSL_COLORS["pytorch"], "o", True), ("triton_default", "Triton, default loads", col["triton"], "o", False),
          ("triton_load_cg", "Triton, .cg loads", col["triton"], "o", True)]
    for yi, (fk, flab) in enumerate(FLUSH):
        y = 2 - yi
        for vk, _, cc, mk, filled in vs:
            v = E.num("vector_add/fp32", "MI300X", "pytorch" if vk.startswith("torch") else "triton",
                      f"diagnostic:flush_protocol_variants_fp32:{vk} | {fk}")
            c3.scatter(v * 1000, y, marker=mk, s=20, facecolor=cc if filled else "white", edgecolor=cc, linewidth=0.9, zorder=3)
    c3.set_yticks(range(3))
    c3.set_yticklabels([lab for _, lab in reversed(FLUSH)], fontsize=FS, linespacing=1.0)
    c3.set_ylim(-0.6, 2.6)
    c3.set_xlim(40, 110)
    c3.tick_params(labelsize=FS)
    c3.grid(axis="x", color=PS.GRID, lw=0.35)
    c3.set_xlabel("latency (µs)", fontsize=FS)
    title(c3, "MI300X vector_add (diagnostic):\nload modifier × cache flush")
    c3.legend(handles=[Line2D([], [], marker=mk, ls="", markerfacecolor=cc if filled else "white", markeredgecolor=cc, markersize=4, label=lab)
                       for _, lab, cc, mk, filled in vs], fontsize=6, loc="lower right", handletextpad=0.2, borderaxespad=0.1)

    used = sorted({i for v in E.axes.values() for i in v})
    abl = [r for r in E.rows if r["figure"] == "a5" and "cache_modifier_ablation_fp32" in r["metric_name"]]
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["figure_evidence.csv"], D)
    m.update({"metric_formula": {"A": "smsp__inst_executed.sum(X) / smsp__inst_executed.sum(Triton), same device, profiled input, summed over the run() launches",
                                 "B GH200": "dynamic warp-level STS (sass__inst_executed_per_opcode_with_modifier_all, STS.*) / dynamic warp-level WGMMA family count, one matmul_kernel launch",
                                 "B B200": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum (sum over launches)",
                                 "B MI300X": "diagnostic latency (warmup 2 / repeat 10, 512 MiB write flush), not formal",
                                 "C NVIDIA traffic": "ratio to Triton on the same device of global-load requests, L1 global-load sectors and DRAM read bytes",
                                 "C occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active (bar), sm__maximum_warps_per_active_cycle_pct (line), smsp__issue_active.avg.pct_of_peak_sustained_active (text)",
                                 "C MI300X": "diagnostic latency of torch.add and a standalone Triton add kernel (BLOCK 2048, 4 warps) with default or .cg loads under three flush protocols; mean of 10 repeats, block repeated twice, second kept"},
              "aggregation_order": ["single profiled input per operator/dtype"],
              "measurement_kinds_per_axis": {"A": ["dynamic counter (NVIDIA)"], "B GH200": ["dynamic SASS counts"], "B B200": ["dynamic counter"],
                                             "B MI300X": ["diagnostic_experiment latency"], "C traffic": ["dynamic counters"],
                                             "C occupancy": ["dynamic counter", "launch_config"], "C MI300X": ["diagnostic_experiment latency"]},
              "axes_evidence": {k: sorted(set(v)) for k, v in E.axes.items()},
              "missing_shown_as": "n/c (not collected): B200 TileLang FP16/FP8 matmul reports are reduced collections",
              "confounders": {f"{c}|{d}": t for (c, d), t in A5_CONFOUNDED.items()} | {
                  "GH200 STS per WGMMA": "Triton reads a B operand transposed outside the timed region; cuTile loads B as [K, N] and re-lays it out for TF32 and FP8, not FP16",
                  "MI300X GEMM": "FP32 pointer variant keeps the winner tile and stages but also writes C with a masked tl.store; FP16 also lowers num_stages 3 -> 2 (pointer kernel exceeds LDS at 3 stages)",
                  "MI300X vector_add": "the cache_modifier_ablation_fp32 experiment is not plotted: its default-store run (131.8 us) is not reproduced by the ISA-identical .cg-store run (98.9 us) or the launch-config sweep (97-105 us); store modifiers (.cs -> sc0 nt, .wt -> sc0 sc1, .cg -> no ISA change) stay at 97.6-98.9 us"},
              "not_plotted_evidence_ids": [r["evidence_id"] for r in abl],
              "selection_criteria": "evidence for the three mechanisms of Figure 3 plus the cases moved out of it (flash_decode in A; vector_add in C)",
              "profiling_evidence_ids": used, "case_coverage": "profiled maximum-input case of each operator/dtype", "excluded_cases": {},
              "plotted_values": [r for r in E.rows if r["evidence_id"] in set(used)]})
    return finish(m, fig, out_root, name)


def main(out_root=FD.PLOTS, which=("a1", "a2", "a3", "a4", "a5")):
    PS.apply()
    D = FD.Data("autotune")
    out = {}
    for w in which:
        out[w] = {"a1": fig_a1, "a2": fig_a2, "a3": fig_a3, "a4": fig_a4, "a5": fig_a5}[w](D, out_root)
    return out


if __name__ == "__main__":
    main(which=tuple(sys.argv[1:]) or ("a1", "a2", "a3", "a4", "a5"))
