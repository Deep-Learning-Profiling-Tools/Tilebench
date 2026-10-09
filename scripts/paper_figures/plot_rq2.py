"""Figure 3 (RQ2): three cross-device case studies (matrix operand delivery, indexing overhead, memory access and latency
hiding). Left of each row: formal speedup over the local PyTorch baseline at the profiled input on every device. Right:
two NVIDIA-only dynamic counters (one axis per counter, B200 and GH200 only) and one line of MI300X static-ISA or
kernel-trace evidence. Every number comes from benchmark_cases_normalized.csv.gz or figure_evidence.csv."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "fig_rq2_cross_device_diagnosis"
W, H = PS.DOUBLE_COL_IN, 4.95
ROW_H = 1.62                           # inches per case row
NV = ("B200", "GH200")


class Ev:
    def __init__(self):
        self.rows = FD.read_csv(FD.COMBINED / "figure_evidence.csv")
        self.used = []

    def get(self, case, dev, dsl, metric, fig="rq2"):
        """Exact metric name, or a prefix when it ends with '*'."""
        pre = metric.endswith("*")
        hits = [r for r in self.rows if r["figure"] == fig and r["case"] == case and r["device"] == dev and r["dsl"] == dsl
                and (r["metric_name"].startswith(metric[:-1]) if pre else r["metric_name"] == metric)]
        assert len(hits) == 1, (case, dev, dsl, metric, len(hits))
        self.used.append(hits[0]["evidence_id"])
        return hits[0]["value"]

    def num(self, *a, **k):
        v = self.get(*a, **k)
        assert v != "", a
        return float(v)


def case_rows(E):
    """Per case: the two NVIDIA counters (label, unit scale, per (dev, dsl) value, optional per-bar note) and the text lines."""
    g = E.num
    out = {}
    c = "matmul_fp32_fp16_fp8/fp32"
    sts = "sass__inst_executed_per_opcode_with_modifier_all[STS*]"
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path*")
    nld = acc.split("global_load_dword x")[1].split(" ")[0]
    out[c] = {
        "title": "A  Matrix operand delivery (matmul, FP32)", "dsls": ("triton", "cutile"),
        "metrics": [("TMA load bytes (GB)", {(d, s): g(c, d, s, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum") / 1e9
                                             for d in NV for s in ("triton", "cutile")}, "{:.1f}", None),
                    ("Shared-memory stores (M)", {(d, s): g(c, d, s, sts) / 1e6 for d in NV for s in ("triton", "cutile")},
                     "{:.2f}", None)],
        "mi300x": f"MI300X (ROCm / ISA evidence): TensorDescriptor loads lowered to {nld} scalar 32-bit global_load_dword; "
                  f"MFMA util. {g(c, 'MI300X', 'triton', '2.1.10 | MFMA Utilization | Avg*'):.1f}%",
        "caveat": "Confounders: Triton reads a B operand transposed before timing; cuTile tiles differ (256×256 on B200, 128×128 on GH200)",
    }
    c = "destindex/int8"
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path[copy_by_dest_kernel#0]")
    nb = acc.split("buffer_store_byte x")[1].split(" ")[0]
    width = {}
    for d in NV:
        for s in ("triton", "cutile"):
            mix = E.get(c, d, s, "opcode_mix[STG]")
            ops = [x.split(":")[0] for x in mix.split(";")]
            assert len(ops) == 1, mix
            width[(d, s)] = {"STG.E.128": "128-bit", "STG.E.U8": "8-bit", "STG.E.U16": "16-bit", "STG.E": "32-bit"}[ops[0]]
    sec = {(d, s): g(c, d, s, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") for d in NV for s in ("triton", "cutile")}
    assert max(sec.values()) / min(sec.values()) < 1.01, sec          # same store traffic: the request count is the difference
    out[c] = {
        "title": "B  Indexing overhead (destindex, INT8)", "dsls": ("triton", "cutile"),
        "metrics": [("Executed instructions (M)", {(d, s): g(c, d, s, "smsp__inst_executed.sum") / 1e6 for d in NV for s in ("triton", "cutile")},
                     "{:.1f}", None),
                    ("Global store instructions (M)", {(d, s): g(c, d, s, "sass__inst_executed_per_opcode_with_modifier_all[STG*]") / 1e6
                                                       for d in NV for s in ("triton", "cutile")}, "{:.2f}", width)],
        "mi300x": f"MI300X (ISA evidence): Triton stores each row with {nb} per-lane buffer_store_byte, no vector store",
        "caveat": f"Triton and cuTile write the same {min(sec.values()) / 1e6:.2f} M store sectors on both devices; "
                  "the PyTorch baseline (index_copy) differs by vendor",
    }
    c = "1d_conv/fp16"
    kt = E.get(c, "MI300X", "pytorch", "kernel_trace.kernels")
    n_tr = kt.count("transpose")
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path*")
    ld = acc.split("static loads: ")[1].split(" x")[0]
    for d in NV:
        for s in ("triton", "tilelang"):
            assert "U16" in E.get(c, d, s, "opcode_mix[LDG]"), (d, s)       # same 16-bit load width for both DSLs
    dram = [g(c, d, s, "dram__bytes_read.sum") for d in NV for s in ("triton", "tilelang")]
    theo = {(d, s): g(c, d, s, "sm__maximum_warps_per_active_cycle_pct") for d in NV for s in ("triton", "tilelang")}
    out[c] = {
        "title": "C  Memory access and latency hiding (1d_conv, FP16)", "dsls": ("triton", "tilelang"),
        "metrics": [("L1 global-load sectors (M)", {(d, s): g(c, d, s, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / 1e6
                                                    for d in NV for s in ("triton", "tilelang")}, "{:.0f}", None),
                    ("Achieved vs. Theoretical\nOccupancy (%)", {(d, s): g(c, d, s, "sm__warps_active.avg.pct_of_peak_sustained_active")
                                                for d in NV for s in ("triton", "tilelang")}, "{:.1f}", None)],
        "theoretical": theo,
        "mi300x": f"MI300X (ISA / PyTorch trace): PyTorch runs MIOpen implicit GEMM plus {n_tr} transpose kernels; Triton loads with {ld}",
        "caveat": f"NVIDIA Triton and TileLang (B200, GH200): 16-bit loads and equal DRAM reads ({min(dram) / 1e9:.2f} GB); "
                  "TileLang uses a different kernel body on Hopper",
    }
    return out


def speedups(D):
    sel = json.load(open(FD.COMBINED / "rq2_case_selection.json"))["selected"]
    prof = FD.read_csv(FD.COMBINED / "profile_index_normalized.csv")
    res, values = {}, []
    for s in sel:
        op, dt = s["operator"], s["dtype"]
        cids = set()
        for dsl in s["dsls"]:
            for dev in PS.DEVICES:
                if D.supported(dev, dsl):
                    kind = "rocprof_compute" if dev == "MI300X" else "ncu_"
                    hits = {p["case_id_v2"] for p in prof if p["device"] == dev and p["dsl"] == dsl and p["operator"] == op
                            and p["dtype"] == dt and p["report_kind"].startswith(kind)}
                    assert len(hits) == 1, (dev, dsl, op, dt, hits)
                    cids |= hits
        assert len(cids) == 1, f"profiled cases differ across devices for {op}/{dt}"
        cid = cids.pop()
        for dsl in s["dsls"]:
            for dev in PS.DEVICES:
                if not D.supported(dev, dsl):
                    continue
                c = D.case(dev, dsl, op, cid)
                assert c is not None, (dev, dsl, op, cid)
                res[(s["case"], dsl, dev)] = c[0] / c[1]
                values.append({"case": s["case"], "dsl": dsl, "device": dev, "case_id_v2": cid, "torch_ms": c[0], "dsl_ms": c[1],
                               "speedup": c[0] / c[1]})
    return sel, res, values


def fig_axes(fig, x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def slope_panel(ax, case, dsls, S):
    ax.set_yscale("log", base=2)
    ax.set_ylim(1 / 48, 11)
    ax.set_xlim(-0.35, 2.45)
    ax.axhline(1.0, color="#9A9FA4", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks([1 / 16, 1 / 4, 1, 4])
    ax.set_yticklabels(["1/16×", "1/4×", "1×", "4×"])
    ax.minorticks_off()
    ax.grid(axis="y", color=PS.GRID, lw=0.35)
    ax.set_xticks(range(3))
    ax.set_xticklabels(PS.DEVICES)
    ax.tick_params(axis="x", length=0)
    pts = {dsl: [(i, S[(case, dsl, d)]) for i, d in enumerate(PS.DEVICES) if (case, dsl, d) in S] for dsl in dsls}
    for dsl, p in pts.items():
        ax.plot([x for x, _ in p], [y for _, y in p], color=PS.DSL_COLORS[dsl], lw=1.2, marker="o", ms=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)
    for i in range(3):                                   # value labels: higher point above, lower point below
        here = sorted(((p[k][1], dsl) for dsl, p in pts.items() for k in range(len(p)) if p[k][0] == i))
        for rank, (v, dsl) in enumerate(here):
            above = rank == len(here) - 1
            ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 4.5 if above else -4.5), textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=7, color=PS.DSL_COLORS[dsl])
    ax.legend(handles=[Line2D([], [], color=PS.DSL_COLORS[s], lw=1.2, marker="o", ms=3.5, label=PS.DSL_LABEL[s]) for s in dsls],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=7, handlelength=1.4, handletextpad=0.4, borderaxespad=0.1,
              ncol=2, columnspacing=0.8)
    ax.set_ylabel("speedup vs.\nlocal PyTorch", fontsize=7, labelpad=1, linespacing=1.0)
    return pts


def bar_panel(ax, label, vals, fmt, notes, dsls, show_y, theoretical=None):
    keys = [(d, s) for d in NV for s in dsls]
    ys = [3.15, 2.15, 1.0, 0.0]                         # B200 pair above the GH200 pair, with a gap between devices
    vmax = max(vals.values())
    for y, k in zip(ys, keys):
        v = vals[k]
        lab = fmt.format(v) + (f" ({notes[k]})" if notes else "")
        if theoretical:                # occupancy (as Figure A5): grey theoretical limit behind a narrower achieved bar
            t = theoretical[k]
            ax.barh(y, t, height=0.78, color=PS.OCC_THEORETICAL_COLOR, edgecolor="none", zorder=1)
            ax.barh(y, v, height=0.78 * PS.OCC_ACHIEVED_RATIO, color=PS.DSL_COLORS[k[1]], edgecolor="none", zorder=2)
            ax.text(max(v, t) + vmax * 0.04, y, f"{v:.1f} / {t:.1f}", va="center", ha="left", fontsize=7)
            continue
        ax.barh(y, v, height=0.78, color=PS.DSL_COLORS[k[1]], edgecolor="white", lw=0.3)
        ax.text(v + vmax * 0.04, y, lab, va="center", ha="left", fontsize=7)
    ax.set_xlim(0, vmax * (1.75 if notes else 1.45))
    ax.set_ylim(-0.6, 3.75)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d} {PS.DSL_LABEL[s]}" for d, s in keys] if show_y else [])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#BFC3C7")
    ax.set_title(label, fontsize=7.5, loc="left", pad=2)
    if theoretical:
        PS.occupancy_legend(ax, dsls, fontsize=7, loc="upper right", bbox_to_anchor=(1.02, -0.01), ncol=2, handlelength=1.4,
                            handletextpad=0.35, columnspacing=0.8, borderaxespad=0.0)


def plot(out_root):
    PS.apply()
    D = FD.Data("autotune")
    sel, S, values = speedups(D)
    E = Ev()
    C = case_rows(E)
    fig = plt.figure(figsize=(W, H))
    # one shared heading over the counter columns: the bars are NVIDIA NCU data (B200/GH200 only); MI300X is text below
    hx0, hx1 = 2.45, 6.2
    fig.text((hx0 + hx1) / 2 / W, (H - 0.07) / H, "NVIDIA Profiling (NCU)", ha="center", va="top", fontsize=7.5, fontweight="bold")
    fig.add_artist(Line2D([hx0 / W, hx1 / W], [(H - 0.215) / H] * 2, color="#9A9FA4", lw=0.6))
    plotted = {}
    for i, s in enumerate(sel):
        case = s["case"]
        R = C[case]
        top = H - 0.05 - i * ROW_H
        fig.text(0.1 / W, (top - 0.02) / H, R["title"], ha="left", va="top", fontsize=8, fontweight="bold")
        ay, ah = top - 1.13, 0.80
        ax = fig_axes(fig, 0.66, ay, 1.6, ah)
        plotted[case] = {"speedup": slope_panel(ax, case, R["dsls"], S)}
        for j, (lab, vals, fmt, notes) in enumerate(R["metrics"]):
            bx = fig_axes(fig, 3.18 + j * 1.6, ay, 1.18, ah)
            bar_panel(bx, lab, vals, fmt, notes, R["dsls"], show_y=(j == 0), theoretical=R.get("theoretical") if j == 1 else None)
            plotted[case][lab] = {f"{d}:{s_}": v for (d, s_), v in vals.items()}
        if R.get("theoretical"):
            plotted[case]["Theoretical occupancy (%)"] = {f"{d}:{s_}": v for (d, s_), v in R["theoretical"].items()}
        fig.text(0.1 / W, (ay - 0.2) / H, R["mi300x"], ha="left", va="top", fontsize=7, color=PS.INK)
        fig.text(0.1 / W, (ay - 0.34) / H, R["caveat"], ha="left", va="top", fontsize=7, color=PS.MUTED)
        if i:
            fig.add_artist(Line2D([0.1 / W, (W - 0.1) / W], [(top + 0.06) / H] * 2, color=PS.GRID, lw=0.6))
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    return sel, paths, layout, values, sorted(set(E.used)), C, plotted


def main(out_root=FD.PLOTS):
    sel, paths, layout, values, used, C, plotted = plot(out_root)
    D = FD.Data("autotune")
    selection = json.load(open(FD.COMBINED / "rq2_case_selection.json"))
    manifest = {
        "figure": NAME, "rq": "RQ2", "script": "scripts/paper_figures/plot_rq2.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "profile_index_normalized.csv", "figure_evidence.csv",
                                              "rq2_case_selection.json"]),
        "metric_formula": {"left": "torch_ms / dsl_ms from the formal autotuned CSV at the case_id_v2 captured by every profile of the case "
                                   "(one input case, identical on all devices; not an operator aggregate)",
                           "right": "NVIDIA dynamic counters summed over the profiled run() launches (B200 and GH200 only); "
                                    "MI300X evidence is static ISA or kernel trace, shown as text"},
        "aggregation_order": ["none: one input case per row"],
        "case_selection": {"rule": selection["selection_rules"], "selected": [{k: x[k] for k in ("case", "mechanism", "why", "caveats")}
                                                                             for x in selection["selected"]],
                           "not_selected": selection["not_selected"]},
        "case_coverage": {x["case"]: {"case_id_v2": next(v["case_id_v2"] for v in values if v["case"] == x["case"])} for x in sel},
        "excluded_cases": {"cuTile/TileLang on MI300X": "not available"},
        "measurement_kinds_per_axis": {"left": ["formal benchmark latency"],
                                       "right": ["dynamic NCU counters (one counter per axis, NVIDIA only)",
                                                 "occupancy (as Figure A5): grey background bar = theoretical limit "
                                                 "(sm__maximum_warps_per_active_cycle_pct, from the resource and launch configuration); narrower "
                                                 "DSL-coloured foreground bar = achieved (sm__warps_active.avg.pct_of_peak_sustained_active); label = "
                                                 "achieved / theoretical; no reference line"],
                                       "text": ["MI300X static ISA (execution_paths.csv)", "MI300X PyTorch kernel trace", "rocprof-compute MFMA utilization"]},
        "evidence_text": {c: {"mi300x": C[c]["mi300x"], "caveat": C[c]["caveat"]} for c in C},
        "occupancy_style": PS.occupancy_style(),
        "profiling_evidence_ids": used,
        "known_limitations": D.manifest["device_limitations"] | {
            "figure": ["the right-hand counters describe the profiled launches; they support, but do not prove, the attribution of the latency change",
                       "B200 and GH200 use different benchmark protocols and cuda-tile versions",
                       "MI300X cannot be compared numerically with the NVIDIA counters (different counters and static vs dynamic counts)"]},
        "plotted_values": {"speedup": values, "counters": plotted},
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
