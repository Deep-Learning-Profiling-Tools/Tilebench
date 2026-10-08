"""Appendix figures A1-A5 (Evaluation Appendix). Reads only artifacts/paper_figures/combined/."""
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

S_COLS = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"),
          ("GH200", "tilelang"), ("MI300X", "triton")]
D_COLS = [("triton", "B200", "GH200"), ("triton", "B200", "MI300X"), ("cutile", "B200", "GH200"), ("tilelang", "B200", "GH200")]


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


def row_layout(D, gap=0.55):
    ops = D.operators()
    ys, y, prev = {}, 0.0, None
    for o in ops:
        if prev is not None and D.categories[o] != D.categories[prev]:
            y += gap
        ys[o] = y
        y += 1.0
        prev = o
    return ops, ys, y


# ---------------------------------------------------------------------------------------------- A1
def fig_a1(D, out_root):
    name = "fig_a1_performance_atlas"
    ops, ys, ymax = row_layout(D)
    S = {(o, d, s): D.speedup_op(d, s, o) for o in ops for d, s in S_COLS}
    DL = {(o,) + c: D.delta_op(c[0], c[1], c[2], o) for o in ops for c in D_COLS}
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 8.6))
    axL = fig.add_axes([0.185, 0.075, 0.46, 0.845])
    axR = fig.add_axes([0.675, 0.075, 0.27, 0.845])
    nS = PS.log2_norm(4.0)
    nD = PS.log2_norm(2.0)
    for o in ops:
        y = ymax - ys[o] - 1
        for j, (d, s) in enumerate(S_COLS):
            v, n = S[(o, d, s)]
            x0 = j + (0.3 if j >= 3 else 0) + (0.3 if j >= 6 else 0)
            if v is None:
                axL.add_patch(Rectangle((x0 + 0.03, y + 0.05), 0.94, 0.9, facecolor=PS.NA_FILL, hatch="////", edgecolor="white", lw=0.5))
                axL.text(x0 + 0.5, y + 0.5, "N/A", ha="center", va="center", fontsize=5.5, color=PS.MUTED)
                continue
            c = PS.DIVERGING(nS(max(-4, min(4, math.log2(v)))))
            axL.add_patch(Rectangle((x0 + 0.03, y + 0.05), 0.94, 0.9, facecolor=c, edgecolor="white", lw=0.5))
            axL.text(x0 + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if lum(c) < 0.45 else PS.INK)
        for j, c_ in enumerate(D_COLS):
            v, n = DL[(o,) + c_]
            x0 = j + (0.3 if j >= 2 else 0)
            if v is None:
                axR.add_patch(Rectangle((x0 + 0.03, y + 0.05), 0.94, 0.9, facecolor=PS.NA_FILL, hatch="////", edgecolor="white", lw=0.5))
                axR.text(x0 + 0.5, y + 0.5, "N/A", ha="center", va="center", fontsize=5.5, color=PS.MUTED)
                continue
            c = PS.DIVERGING(nD(max(-2, min(2, v))))
            axR.add_patch(Rectangle((x0 + 0.03, y + 0.05), 0.94, 0.9, facecolor=c, edgecolor="white", lw=0.5))
            axR.text(x0 + 0.5, y + 0.5, f"{2 ** v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if lum(c) < 0.45 else PS.INK)
    for ax, ncol in ((axL, 7 + 0.6), (axR, 4 + 0.3)):
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, ymax)
        ax.axis("off")
    for o in ops:
        axL.text(-0.12, ymax - ys[o] - 0.5, o, ha="right", va="center", fontsize=5.8)
    cats = []
    for o in ops:
        if not cats or cats[-1][0] != D.categories[o]:
            cats.append([D.categories[o], ys[o], ys[o]])
        cats[-1][2] = ys[o]
    for c, a, b in cats:
        yc = ymax - (a + b + 1) / 2
        fig.text(0.012, axL.get_position().y0 + axL.get_position().height * yc / ymax, PS.CATEGORY_SHORT[c], rotation=90,
                 ha="left", va="center", fontsize=6.5, fontweight="bold", color=PS.CATEGORY_COLORS[c])
    hdrL = [(0, 3, "B200"), (3.3, 6.3, "GH200"), (6.6, 7.6, "MI300X")]
    for a, b, t in hdrL:
        axL.text((a + b) / 2, ymax + 1.35, t, ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
        axL.plot([a + 0.08, b - 0.08], [ymax + 1.25] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
    for j, (d, s) in enumerate(S_COLS):
        x0 = j + (0.3 if j >= 3 else 0) + (0.3 if j >= 6 else 0)
        axL.text(x0 + 0.5, ymax + 0.2, PS.DSL_LABEL[s], ha="center", va="bottom", fontsize=6.0, clip_on=False)
    axL.text(7.6 / 2, ymax + 2.9, "Speedup vs. local PyTorch (GM over cases)", ha="center", va="bottom", fontsize=7, clip_on=False)
    for j, (s, a, b) in enumerate(D_COLS):
        x0 = j + (0.3 if j >= 2 else 0)
        axR.text(x0 + 0.5, ymax + 0.2, f"{PS.DSL_LABEL[s]}\n{b}\nvs. {a}", ha="center", va="bottom", fontsize=5.7, clip_on=False, linespacing=1.05)
    axR.text(4.3 / 2, ymax + 2.9, "Change in relative speedup\n2^Δ, Δ = log2(S_dev2/S_dev1), matched cases", ha="center",
             va="bottom", fontsize=6.4, clip_on=False, linespacing=1.15)
    cax1 = fig.add_axes([0.235, 0.035, 0.36, 0.009])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nS, cmap=PS.DIVERGING), cax=cax1, orientation="horizontal",
                      ticks=[-4, -2, 0, 2, 4])
    cb.ax.set_xticklabels(["1/16×", "1/4×", "1×", "4×", "16×"])
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)
    cax2 = fig.add_axes([0.70, 0.035, 0.22, 0.009])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=nD, cmap=PS.DIVERGING), cax=cax2, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
    cb2.ax.set_xticklabels(["1/4×", "1/2×", "1×", "2×", "4×"])
    cb2.ax.tick_params(labelsize=5.8, length=2)
    cb2.outline.set_linewidth(0.4)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"], D)
    m.update({"metric_formula": {"left": "S[o,b,d] = GM_cases(torch_ms/dsl_ms) (valid autotune cases)",
                                 "right": "Δ = log2(S_dev2/S_dev1), both S over the case_id_v2 valid on both devices; cell shows 2^Δ; not an absolute hardware speedup"},
              "aggregation_order": ["GM over input cases within operator"], "colour_scale": {"left": "log2 S clipped to ±4", "right": "Δ clipped to ±2"},
              "case_coverage": {"left": {f"{o}|{d}|{s}": S[(o, d, s)][1] for o in ops for d, s in S_COLS},
                                "right_matched_cases": {f"{o}|{s}|{b}/{a}": DL[(o, s, a, b)][1] for o in ops for s, a, b in D_COLS}},
              "excluded_cases": {"invalid_or_missing_rows": len(D.excluded), "known_unsupported": D.manifest["known_unsupported"],
                                 "na_cells": "cuTile/TileLang on MI300X do not exist; Triton MI300X/B200 for matmul fp8 excluded by matching"},
              "selection_criteria": "all 45 operators", "profiling_evidence_ids": [],
              "plotted_values": {"left": [{"operator": o, "device": d, "dsl": s, "speedup": S[(o, d, s)][0]} for o in ops for d, s in S_COLS],
                                 "right": [{"operator": o, "dsl": s, "from": a, "to": b, "delta_log2": DL[(o, s, a, b)][0]} for o in ops for s, a, b in D_COLS]}})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A2
A2_CANDIDATES = ["matmul_fp32_fp16_fp8", "flash_decode", "linear_self_attention", "top_k_selection", "vector_add", "streamk_matmul"]
A2_GRID = {"flash_decode": ("seq_len", None), "linear_self_attention": ("M", "D"), "top_k_selection": ("N", "k"),
           "streamk_matmul": ("n", "m"), "matmul_fp32_fp16_fp8": ("K", None), "vector_add": ("n", None)}


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
        dts = sorted({D.cases[k][c][2] for k in D.cases if k[2] == op for c in D.cases[k]})
        for dt in dts:
            panels.append((op, dt))
    ncol = 2
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(PS.DOUBLE_COL_IN, 1.12 * nrow + 0.7))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.085, hspace=0.95, wspace=0.17)
    norm = PS.log2_norm(4.0)
    values = []
    for ax, (op, dt) in zip(axes.flat, panels):
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
                        va="center", fontsize=5.5, color=PS.MUTED)
        ax.set_yticks(range(len(S_COLS)))
        ax.set_yticklabels([f"{d} {PS.DSL_SHORT[s]}" for d, s in S_COLS], fontsize=5.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([fmt_num(allp[c][inner_k]) for c in order], fontsize=5.5, rotation=90)
        if outer_k:
            groups = []
            for j, cid in enumerate(order):
                v = allp[cid][outer_k]
                if not groups or groups[-1][0] != v:
                    groups.append([v, j, j])
                groups[-1][2] = j
            for v, a, b in groups:
                if a:
                    ax.axvline(a - 0.5, color="white", lw=1.2)
                ax.text((a + b) / 2, -0.85, f"{outer_k}={fmt_num(v)}", ha="center", va="bottom", fontsize=5.5, color=PS.INK)
        ax.tick_params(length=0, pad=1.5)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.axhline(2.5, color="white", lw=1.2)
        ax.axhline(5.5, color="white", lw=1.2)
        ax.set_title(f"{op} · {dt}", loc="left", fontsize=6.6, fontweight="bold", pad=10 if outer_k else 3)
        ax.set_xlabel(inner_k, fontsize=6.0, labelpad=1)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    assert len(panels) < nrow * ncol, "A2 places its colour bar in the empty last panel slot"
    pos = list(axes.flat)[-1].get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 + 0.55 * pos.height, 0.8 * pos.width, 0.06 * pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.DIVERGING), cax=cax, orientation="horizontal", ticks=[-4, -2, 0, 2, 4])
    cb.ax.set_xticklabels(["1/16×", "1/4×", "1×", "4×", "16×"])
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.set_label("speedup vs. local PyTorch (per input case)", fontsize=6.2, labelpad=1)
    cb.outline.set_linewidth(0.4)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "comparison_manifest.json"], D)
    m.update({"metric_formula": "per input case: torch_ms/dsl_ms (formal autotuned CSV), log2 colour scale clipped to ±4",
              "aggregation_order": ["none (each cell is one input case)"],
              "selection_criteria": {"rule": "candidates with median within-(device,DSL,dtype) log2-speedup range >= 0.5 are shown for shape sensitivity; matmul_fp32_fp16_fp8 is kept for dtype sensitivity with ALL its dtypes",
                                     "variation_median_log2_range": var, "selected": selected,
                                     "not_selected": {o: f"median log2 range {var[o]:.2f} < 0.5" for o in A2_CANDIDATES if o not in selected}},
              "case_coverage": {f"{op}|{dt}": sum(1 for v in values if v["operator"] == op and v["dtype"] == dt) for op, dt in panels},
              "excluded_cases": {"MI300X matmul fp8_e4m3fn": "unsupported dtype (row shown as N/A)", "invalid_or_missing_rows": len(D.excluded)},
              "x_axis": {op: {"inner": A2_GRID[op][0], "outer_group": A2_GRID[op][1]} for op in selected},
              "profiling_evidence_ids": [], "plotted_values": values})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A3
PATH_COLORS = {"tcgen05": "#397E8A", "wgmma": "#7FA9B3", "legacy_mma": "#C58458", "mfma": "#9276A5", "none": "#FFFFFF",
               "unknown": "#EDEDEA", "n/a": "#EDEDEA"}
PATH_LABEL = {"tcgen05": "tcgen05", "wgmma": "WGMMA", "legacy_mma": "legacy MMA (HMMA/IMMA)", "mfma": "MFMA", "none": "no matrix instr.",
              "unknown": "unknown / no profile", "n/a": "unsupported"}
FLAG_ABBR = {"TMEM": "TMEM", "smem": "SMEM", "LDS": "LDS", "spill": "spill", "atom.global": "atom.g", "atom.shared": "atom.s",
             "atomic": "atomic", "ldsm/stsm": "ldsm", "convert_layout": "cvt"}


def fig_a3(D, out_root):
    name = "fig_a3_execution_paths"
    rows = FD.read_csv(FD.COMBINED / "execution_path_matrix.csv")
    keys = []
    for r in rows:
        k = (r["operator"], r["dtype"])
        if k not in keys:
            keys.append(k)
    cols = S_COLS
    cell = {(r["operator"], r["dtype"], r["device"], r["dsl"]): r for r in rows}
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 0.42 * len(keys) + 1.3))
    ax = fig.add_axes([0.21, 0.17, 0.78, 0.74])
    n = len(keys)
    for i, (op, dt) in enumerate(keys):
        y = n - 1 - i
        for j, (dev, dsl) in enumerate(cols):
            x0 = j + (0.25 if j >= 3 else 0) + (0.25 if j >= 6 else 0)
            r = cell[(op, dt, dev, dsl)]
            mp = r["matrix_path"]
            static = r["evidence_kind"] in ("static_sass", "static_isa")
            ax.add_patch(Rectangle((x0 + 0.04, y + 0.06), 0.92, 0.88, facecolor=PATH_COLORS.get(mp, "#EDEDEA"),
                                   edgecolor="#6E747A" if mp not in ("n/a", "unknown") else "white",
                                   lw=0.6, ls=(0, (2, 1.5)) if static else "-", hatch="////" if mp in ("n/a",) else None, alpha=0.9))
            if mp == "n/a":
                ax.text(x0 + 0.5, y + 0.5, "unsupported", ha="center", va="center", fontsize=5.5, color=PS.MUTED)
                continue
            if mp == "unknown":
                ax.text(x0 + 0.5, y + 0.5, "?", ha="center", va="center", fontsize=6, color=PS.MUTED)
                continue
            dark = mp in ("tcgen05", "mfma")
            fl = [FLAG_ABBR.get(f, f) for f in r["flags"].split(";") if f]
            line1 = r["operand_path"]
            lines, cur = [], ""
            for f in fl:                                  # wrap the flags at ~12 characters per line
                if cur and len(cur) + 1 + len(f) > 12:
                    lines.append(cur)
                    cur = f
                else:
                    cur = f"{cur} {f}".strip()
            lines.append(cur or "–")
            ax.text(x0 + 0.5, y + (0.70 if len(lines) == 1 else 0.76), line1, ha="center", va="center", fontsize=5.5,
                    color="white" if dark else PS.INK)
            ax.text(x0 + 0.5, y + 0.33 if len(lines) == 1 else y + 0.36, "\n".join(lines), ha="center", va="center", fontsize=5.5,
                    color="white" if dark else "#4A4F54", linespacing=0.95)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, n)
    ax.axis("off")
    for i, (op, dt) in enumerate(keys):
        ax.text(-0.08, n - 1 - i + 0.5, f"{op} · {dt}", ha="right", va="center", fontsize=6.0)
    for j, (dev, dsl) in enumerate(cols):
        x0 = j + (0.25 if j >= 3 else 0) + (0.25 if j >= 6 else 0)
        ax.text(x0 + 0.5, n + 0.12, PS.DSL_LABEL[dsl], ha="center", va="bottom", fontsize=6.0)
    for a, b, t in ((0, 3, "B200"), (3.25, 6.25, "GH200"), (6.5, 7.5, "MI300X")):
        ax.text((a + b) / 2, n + 0.8, t, ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.plot([a + 0.08, b - 0.08], [n + 0.72] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
    h = [Patch(facecolor=PATH_COLORS[k], edgecolor="#6E747A", lw=0.5, label=PATH_LABEL[k]) for k in ("tcgen05", "wgmma", "legacy_mma", "mfma", "none")]
    h += [Patch(facecolor="white", edgecolor="#6E747A", lw=0.6, ls=(0, (2, 1.5)), label="static SASS/ISA only"),
          Patch(facecolor=PATH_COLORS["n/a"], hatch="////", edgecolor="white", label="unsupported")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.55, 0.0), handlelength=1.4, columnspacing=1.0)
    fig.text(0.20, 0.135, "Line 1: operand load path, union over the kernels of the run (TMA = TMA loads; cp.async; LDG = ordinary global loads;\n"
             "desc→ptr = TensorDescriptor lowered to pointer loads; pointer = pointer loads). Line 2: SMEM / LDS / TMEM use, spill (local or scratch),\n"
             "atom.g / atom.s (global / shared atomics at scale), ldsm (LDSM/STSM layout operations), cvt (convert_layout).",
             fontsize=5.6, color=PS.MUTED, ha="left", va="top", linespacing=1.25)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["execution_path_matrix.csv"], D)
    m.update({"metric_formula": "categorical: matrix-instruction class, operand path, staging and flags derived from execution_paths.csv (NVIDIA dynamic opcode-with-modifier counts or static SASS; MI300X static ISA + source form)",
              "aggregation_order": ["union over the kernels (stages) of the profiled run"],
              "selection_criteria": "15 representative operator/dtype pairs covering GEMM, attention, convolution, scatter, histogram, pooling and streaming",
              "evidence_semantics": "absence of a flag means it was not observed in the available evidence; static-only cells (dashed) show code paths present in the binary, not executed counts",
              "case_coverage": len(rows), "excluded_cases": {"MI300X matmul fp8": "unsupported"}, "profiling_evidence_ids": [r["profile_id"] for r in rows if r["profile_id"]],
              "plotted_values": rows})
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
    fig = plt.figure(figsize=(PS.SINGLE_COL_IN, 8.4))
    ax = fig.add_axes([0.40, 0.07, 0.56, 0.86])
    for o in ops:
        y = ymax - ys[o] - 1
        for j, (dev, s) in enumerate(cols):
            v, n = SL[(o, dev, s)]
            x0 = j + (0.3 if j >= 3 else 0)
            near = v <= 1.05
            c = (0.97, 0.96, 0.94, 1.0) if near else PS.SEQUENTIAL(norm(min(v, 8.0)))
            ax.add_patch(Rectangle((x0 + 0.04, y + 0.05), 0.92, 0.9, facecolor=c, edgecolor="white", lw=0.5))
            if v == 1.0:
                ax.add_patch(Rectangle((x0 + 0.06, y + 0.08), 0.88, 0.84, facecolor="none", edgecolor="#4A4F54", lw=0.7))
            ax.text(x0 + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if (not near and lum(c) < 0.5) else (PS.MUTED if near and v > 1.0 else PS.INK))
    ax.set_xlim(0, 6.3)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    for o in ops:
        ax.text(-0.1, ymax - ys[o] - 0.5, o, ha="right", va="center", fontsize=5.6)
    for j, (dev, s) in enumerate(cols):
        x0 = j + (0.3 if j >= 3 else 0)
        ax.text(x0 + 0.5, ymax + 0.25, PS.DSL_SHORT[s], ha="center", va="bottom", fontsize=6.0)
    for a, b, t in ((0, 3, "B200"), (3.3, 6.3, "GH200")):
        ax.text((a + b) / 2, ymax + 1.35, t, ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.plot([a + 0.08, b - 0.08], [ymax + 1.25] * 2, color="#9A9FA4", lw=0.6, clip_on=False)
    cats = []
    for o in ops:
        if not cats or cats[-1][0] != D.categories[o]:
            cats.append([D.categories[o], ys[o], ys[o]])
        cats[-1][2] = ys[o]
    for c, a, b in cats:
        yc = ymax - (a + b + 1) / 2
        fig.text(0.015, ax.get_position().y0 + ax.get_position().height * yc / ymax, PS.CATEGORY_SHORT[c], rotation=90,
                 ha="left", va="center", fontsize=6.2, fontweight="bold", color=PS.CATEGORY_COLORS[c])
    cax = fig.add_axes([0.42, 0.045, 0.50, 0.009])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PS.SEQUENTIAL), cax=cax, orientation="horizontal", ticks=[1, 2, 4, 8])
    cb.ax.set_xticklabels(["1×", "2×", "4×", "≥8×"])
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("slowdown vs. fastest DSL on the device\n(outlined = fastest; ≤1.05× drawn neutral)", fontsize=5.8, labelpad=1,
                 linespacing=1.1)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"], D)
    m.update({"metric_formula": "slowdown = GM_cases(latency_DSL) / min_DSL GM_cases(latency_DSL), cases valid for all three NVIDIA DSLs on the device",
              "aggregation_order": ["GM over matched cases", "ratio to the per-operator minimum"], "near_parity_threshold": 1.05,
              "case_coverage": {f"{o}|{d}": SL[(o, d, 'triton')][1] for o in ops for d in ("B200", "GH200")},
              "excluded_cases": {"invalid_or_missing_rows": len(D.excluded)}, "selection_criteria": "all 45 operators",
              "profiling_evidence_ids": [],
              "plotted_values": [{"operator": o, "device": d, "dsl": s, "slowdown": SL[(o, d, s)][0]} for o in ops for d, s in cols]})
    return finish(m, fig, out_root, name)


# ---------------------------------------------------------------------------------------------- A5
def fig_a5(D, out_root):
    name = "fig_a5_profiling_evidence"
    ev = FD.read_csv(FD.COMBINED / "figure_evidence.csv")
    used = []
    axes_ev, cur = {}, {"axis": None}   # evidence ids per numeric axis (validate_plots checks vendor/semantics homogeneity)

    def get(fig_, case, dev, dsl, sub):
        hits = [r for r in ev if r["figure"] == fig_ and r["case"] == case and r["device"] == dev and r["dsl"] == dsl and sub in r["metric_name"]]
        if not hits:
            return None
        r = hits[0]
        used.append(r["evidence_id"])
        axes_ev.setdefault(cur["axis"], []).append(r["evidence_id"])
        return float(r["value"]) if r["value"] not in ("", None) else None

    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 6.9))
    gs = fig.add_gridspec(3, 3, left=0.085, right=0.985, top=0.95, bottom=0.12, hspace=0.95, wspace=0.42,
                          width_ratios=[1.25, 1.0, 1.0])
    colors = {"cutile": PS.DSL_COLORS["cutile"], "tilelang": PS.DSL_COLORS["tilelang"], "triton": PS.DSL_COLORS["triton"]}

    # ---- A: indexing / reduction (NVIDIA within-device instruction ratios)
    ax = fig.add_subplot(gs[0, 0:2])
    cur["axis"] = "A_nvidia_dynamic_instruction_ratio"
    cases = [("weight_dequant/bf16", "weight_dequant\nbf16"), ("destindex/int8", "destindex\nint8"),
             ("cross_entropy/fp16", "cross_entropy\nfp16"), ("moe_topk_gating/fp16", "moe_topk_gating\nfp16")]
    xpos, labels, k = [], [], 0
    for ci, (case, lab) in enumerate(cases):
        for di, dev in enumerate(("B200", "GH200")):
            x = ci * 3.0 + di * 1.1
            t = get("a5", case, dev, "triton", "smsp__inst_executed.sum")
            for si, s in enumerate(("cutile", "tilelang")):
                v = get("a5", case, dev, s, "smsp__inst_executed.sum")
                r = v / t if (v is not None and t) else None
                if r is None:
                    ax.text(x + (si - 0.5) * 0.42, 1.05, "n/c", ha="center", fontsize=5.5, color=PS.MUTED)
                    continue
                ax.bar(x + (si - 0.5) * 0.42, r, width=0.40, color=colors[s], edgecolor="white", lw=0.3)
                ax.text(x + (si - 0.5) * 0.42, r * 1.06, f"{r:.1f}", ha="center", va="bottom", fontsize=5.5)
            xpos.append(x)
            labels.append(dev + ("†" if (case, dev) == ("weight_dequant/bf16", "GH200") else ""))
        ax.text(ci * 3.0 + 0.55, -0.20, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.8, linespacing=1.45)
    ax.set_yscale("log", base=2)
    ax.axhline(1, color="#9A9FA4", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylim(0.25, 64)
    ax.set_yticks([0.25, 1, 4, 16, 64])
    ax.set_yticklabels(["1/4×", "1×", "4×", "16×", "64×"])
    ax.set_ylabel("dynamic warp instr.\nratio to Triton", fontsize=6.2)
    ax.text(0.0, -0.50, "† GH200 Triton uses a different autotuned config, which inflates its instruction count (confounder, "
            "not an architecture effect)", transform=ax.transAxes, fontsize=5.5, color=PS.MUTED, va="top")
    ax.set_title("A  Indexing and reduction overhead (NVIDIA, smsp__inst_executed.sum, same device)", loc="left", fontsize=7, fontweight="bold")
    ax.legend(handles=[Patch(color=colors["cutile"], label="cuTile / Triton"), Patch(color=colors["tilelang"], label="TileLang / Triton")],
              loc="upper right", fontsize=5.8, ncol=2)
    axm = fig.add_subplot(gs[0, 2])
    axm.axis("off")
    axm.set_title("MI300X (Triton, static ISA)", loc="left", fontsize=6.6, fontweight="bold")
    da = [r for r in ev if r["case"] == "destindex/int8" and r["device"] == "MI300X" and "access_path" in r["metric_name"]]
    used += [r["evidence_id"] for r in da]
    nb = da[0]["value"].split("buffer_store_byte x")[1].split(" ")[0] if da else "?"
    axm.text(0.0, 0.97, f"destindex int8:\n{nb} buffer_store_byte per lane,\nno vector store (traced)\n\n"
             "weight_dequant (control):\nconstexpr N → multiply-high\nindex math\n\nmoe_topk_gating (control):\nsingle-wave rows\n(DPP + ds_bpermute)",
             transform=axm.transAxes, ha="left", va="top", fontsize=5.8, linespacing=1.2)
    used += [r["evidence_id"] for r in ev if r["device"] == "MI300X" and r["case"].startswith("destindex") and "diagnosis" in r["metric_name"]]

    # ---- B: matrix execution and operand delivery
    axb1 = fig.add_subplot(gs[1, 0])
    cur["axis"] = "B_b200_tma_load_bytes"
    dts = ["fp32", "fp16", "fp8_e4m3fn"]
    for i, dt in enumerate(dts):
        for si, s in enumerate(("triton", "cutile", "tilelang")):
            v = get("a5", f"matmul_fp32_fp16_fp8/{dt}", "B200", s, "tma_ld")
            x = i * 1.2 + (si - 1) * 0.3
            if v is None:
                axb1.text(x, 0.6, "n/c", ha="center", fontsize=5.5, color=PS.MUTED, rotation=90)
                continue
            axb1.bar(x, v / 1e9, width=0.28, color=colors[s], edgecolor="white", lw=0.3)
            if v == 0:
                axb1.text(x, 0.6, "0", ha="center", fontsize=5.5, color=PS.INK)
    axb1.set_xticks([i * 1.2 for i in range(3)])
    axb1.set_xticklabels(["FP32", "FP16", "FP8"], fontsize=6)
    axb1.set_ylabel("TMA load bytes (GB)", fontsize=6.2)
    axb1.set_title("B  Matrix operand delivery\nB200 matmul: TMA bytes", loc="left", fontsize=6.8, fontweight="bold")
    axb1.legend(handles=[Patch(color=colors[s], label=PS.DSL_LABEL[s]) for s in ("triton", "cutile", "tilelang")], fontsize=5.6,
                loc="upper right")
    axb1.text(0.0, -0.24, "TileLang FP32: legacy HMMA via cp.async\n(0 B TMA); FP16/FP8 TileLang reports\nare reduced collections (n/c)", transform=axb1.transAxes,
              fontsize=5.5, color=PS.MUTED, va="top")
    axb2 = fig.add_subplot(gs[1, 1])
    cur["axis"] = "B_gh200_sts_per_wgmma"
    for i, dt in enumerate(dts):
        for si, s in enumerate(("triton", "cutile")):
            st = get("a5", f"matmul_fp32_fp16_fp8/{dt}", "GH200", s, "[STS*]")
            w = get("a5", f"matmul_fp32_fp16_fp8/{dt}", "GH200", s, "family=wgmma")
            x = i * 1.0 + (si - 0.5) * 0.36
            if not w:
                axb2.text(x, 0.05, "no WGMMA", ha="center", fontsize=5.5, color=PS.MUTED, rotation=90)
                continue
            r = st / w
            axb2.bar(x, r, width=0.34, color=colors[s], edgecolor="white", lw=0.3)
            axb2.text(x, r + 0.2, f"{r:.2f}", ha="center", va="bottom", fontsize=5.5)
    axb2.set_xticks(range(3))
    axb2.set_xticklabels(["FP32", "FP16", "FP8"], fontsize=6)
    axb2.set_ylabel("STS per WGMMA (dynamic)", fontsize=6.2)
    axb2.set_ylim(0, 18.5)
    axb2.set_title("GH200 matmul: operand re-layout", loc="left", fontsize=6.8, fontweight="bold")
    axb3 = fig.add_subplot(gs[1, 2])
    exps = [r for r in ev if r["measurement_kind"] == "diagnostic_experiment" and r["case"].startswith("matmul")]
    used += [r["evidence_id"] for r in exps]
    axes_ev["B_mi300x_diagnostic_latency"] = [r["evidence_id"] for r in exps]
    order = [("torch.matmul", "PyTorch"), ("formal winner", "desc. (formal winner)"), ("same tile", "pointer, same tile")]
    for i, dt in enumerate(("fp32", "fp16")):
        for j, (key, lab) in enumerate(order):
            hit = [r for r in exps if r["case"] == f"matmul_fp32_fp16_fp8/{dt}" and key in r["metric_name"]]
            if not hit:
                continue
            v = float(hit[0]["value"])
            x = i * 1.15 + (j - 1) * 0.33
            col = PS.DSL_COLORS["pytorch"] if j == 0 else ("#2F4E63" if j == 1 else PS.DSL_COLORS["triton"])
            axb3.bar(x, v, width=0.31, color=col, edgecolor="white", lw=0.3)
            axb3.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    axb3.set_ylim(0, 31)
    axb3.set_xticks([0, 1.15])
    axb3.set_xticklabels(["FP32", "FP16"], fontsize=6)
    axb3.set_ylabel("latency (ms), diagnostic run", fontsize=6.2)
    axb3.set_title("MI300X: descriptor vs pointer", loc="left", fontsize=6.8, fontweight="bold")
    axb3.legend(handles=[Patch(color=PS.DSL_COLORS["pytorch"], label="PyTorch"), Patch(color="#2F4E63", label="TensorDesc (winner)"),
                         Patch(color=PS.DSL_COLORS["triton"], label="pointer, same tile")], fontsize=5.5, loc="upper right")
    axb3.text(0.0, -0.24, "diagnostic run, warmup 2/repeat 10\n(not formal). FP32 variant: same\ntile and stages, masked tl.store.\nFP16 variant also sets num_stages\n3→2 (not a one-factor change).",
              transform=axb3.transAxes, fontsize=5.5, color=PS.MUTED, va="top")

    # ---- C: convolution / memory access / latency hiding
    axc1 = fig.add_subplot(gs[2, 0])
    cur["axis"] = "C_nvidia_load_sectors_per_request"
    for di, dev in enumerate(("B200", "GH200")):
        for si, s in enumerate(("triton", "cutile", "tilelang")):
            v = get("a5", "1d_conv/fp16", dev, s, "sectors_pipe_lsu_mem_global_op_ld")
            x = di * 1.2 + (si - 1) * 0.3
            axc1.bar(x, v, width=0.28, color=colors[s], edgecolor="white", lw=0.3)
            axc1.text(x, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    axc1.set_xticks([0, 1.2])
    axc1.set_xticklabels(["B200", "GH200"], fontsize=6)
    axc1.set_ylabel("global load sectors / request", fontsize=6.2)
    axc1.set_ylim(0, 20)
    axc1.set_title("C  Gathered staging, latency hiding\n1d_conv fp16: load coalescing", loc="left", fontsize=6.8, fontweight="bold")
    axc2 = fig.add_subplot(gs[2, 1])
    cur["axis"] = "C_nvidia_occupancy_pct"
    for di, dev in enumerate(("B200", "GH200")):
        for si, s in enumerate(("triton", "cutile", "tilelang")):
            a = get("a5", "1d_conv/fp16", dev, s, "warps_active")
            th = get("a5", "1d_conv/fp16", dev, s, "maximum_warps")
            iss = get("a5", "1d_conv/fp16", dev, s, "issue_active")
            x = di * 1.2 + (si - 1) * 0.3
            axc2.bar(x, a, width=0.28, color=colors[s], edgecolor="white", lw=0.3)
            axc2.plot([x - 0.14, x + 0.14], [th, th], color=PS.INK, lw=0.8)
            axc2.text(x, max(a, th) + 1.2, f"{iss:.0f}", ha="center", va="bottom", fontsize=5.5, color=PS.MUTED)
    axc2.set_xticks([0, 1.2])
    axc2.set_xticklabels(["B200", "GH200"], fontsize=6)
    axc2.set_ylabel("achieved occupancy (%)", fontsize=6.2)
    axc2.set_ylim(0, 62)
    axc2.set_title("bar: achieved, line: theoretical;\nnumber: issue-active %", loc="left", fontsize=6.0)
    axc3 = fig.add_subplot(gs[2, 2])
    ca = [r for r in ev if r["measurement_kind"] == "diagnostic_experiment" and r["case"] == "vector_add/fp32"]
    used += [r["evidence_id"] for r in ca]
    axes_ev["C_mi300x_diagnostic_latency"] = [r["evidence_id"] for r in ca]
    lab = {"torch.add": "PyTorch", "load(default)_store(default)": "default", "load.cg_store(default)": "ld.cg",
           "load(default)_store.cs": "st.cs", "load.cg_store.cs": "ld.cg+st.cs"}
    vals = [(lab[r["metric_name"].split(":")[-1]], float(r["value"])) for r in ca if r["metric_name"].split(":")[-1] in lab]
    for i, (l_, v) in enumerate(vals):
        axc3.barh(i, v * 1000, color=PS.DSL_COLORS["pytorch"] if l_ == "PyTorch" else PS.DSL_COLORS["triton"], height=0.6)
        axc3.text(v * 1000 + 3, i, f"{v * 1000:.0f}", va="center", fontsize=5.5)
    axc3.set_yticks(range(len(vals)))
    axc3.set_yticklabels([v[0] for v in vals], fontsize=5.8)
    axc3.invert_yaxis()
    axc3.set_xlabel("latency (µs), diagnostic run", fontsize=6.2)
    axc3.set_xlim(0, 160)
    axc3.set_title("MI300X vector_add fp32:\nload cache policy", loc="left", fontsize=6.8, fontweight="bold")
    axc3.text(0.0, -0.34, "n = 20,971,520; BLOCK 2048, 4 warps;\nwarmup 2/repeat 10 (diagnostic);\nld.cg emits nt loads (as ATen);\nst.cg has no ISA effect",
              transform=axc3.transAxes, fontsize=5.5, color=PS.MUTED, va="top")
    for a_ in (ax, axb1, axb2, axb3, axc1, axc2, axc3):
        a_.tick_params(labelsize=5.8)
    m = base_manifest(name, "scripts/paper_figures/plot_appendix.py", ["figure_evidence.csv"], D)
    m.update({"metric_formula": {"A": "smsp__inst_executed.sum(X) / smsp__inst_executed.sum(Triton), same device, profiled case",
                                 "B200 TMA": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum (sum over launches)",
                                 "GH200 relayout": "dynamic STS count / dynamic WGMMA count (sass__inst_executed_per_opcode*)",
                                 "MI300X": "diagnostic latencies (report_benchmark warmup 2 / repeat 10), not formal",
                                 "C": "l1tex sectors/requests (global loads); sm__warps_active (achieved) vs sm__maximum_warps_per_active_cycle_pct (theoretical); smsp__issue_active"},
              "aggregation_order": ["single profiled case per operator/dtype"],
              "measurement_kinds_per_axis": {"A": ["dynamic counter (NVIDIA)"], "B200 TMA": ["dynamic counter"], "GH200": ["dynamic SASS counts"],
                                             "MI300X B/C": ["diagnostic_experiment latency"], "C1/C2": ["dynamic counter", "launch_config"]},
              "missing_shown_as": "n/c (not collected): B200 TileLang FP16/FP8 matmul reports are reduced collections",
              "selection_criteria": "high-quality evidence from the diagnosis tables for the three mechanisms", "profiling_evidence_ids": sorted(set(used)),
              "axes_evidence": {k: sorted(set(v)) for k, v in axes_ev.items()},
              "text_only_evidence": {"A_mi300x_static_isa_text": sorted(r["evidence_id"] for r in ev if r["device"] == "MI300X" and r["case"].startswith("destindex")
                                                                    and ("access_path" in r["metric_name"] or "diagnosis" in r["metric_name"]))},
              "case_coverage": "profiled maximum-input case of each operator/dtype", "excluded_cases": {},
              "plotted_values": [r for r in ev if r["evidence_id"] in set(used)]})
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
