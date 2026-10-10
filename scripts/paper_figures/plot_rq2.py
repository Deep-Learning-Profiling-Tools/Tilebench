"""Figure 3 (RQ2): explaining performance differences across accelerators, for three cases with distinct mechanisms.

Left of each row: proximity to the modeled, device-specific hybrid SOL reference (T_SOL / T_k, sol_data.py) at the input case
captured by every profile, on every device. Right: the hardware/compiler evidence for that case. NVIDIA dynamic NCU
counters of the profiled run() are shown as a small table (B200 and GH200 only); MI300X static ISA, rocprof counters and
kernel traces are separate text lines and never share a numeric column with NVIDIA counters. Observations,
interpretation and confounders are on separate lines. Every number comes from sol_cases.csv.gz or figure_evidence.csv."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import sol_data as SD  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "fig_rq2_cross_device_diagnosis"
W = PS.DOUBLE_COL_IN
NV = ("B200", "GH200")
TX0, LABEL_W, COL_W = 2.55, 1.3, 0.58   # evidence table: left edge, label column, value columns (inches)
LINE = 0.125                           # text line pitch at 7 pt (inches)
TITLE_H, GAP_H = 0.24, 0.12            # case title band and the gap before the next case (inches)


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
    """Per case: the NVIDIA evidence table (label -> {(dev, dsl): (value, text)}) and the text lines."""
    g = E.num
    out = {}
    c = "matmul_fp32_fp16_fp8/fp32"
    D2 = ("triton", "cutile")
    sts = "sass__inst_executed_per_opcode_with_modifier_all[STS*]"
    ldsm = "sass__inst_executed_per_opcode_with_modifier_all[LDSM*]"
    fam = {}
    for d in NV:
        for s in D2:
            t5, wg = g(c, d, s, "sass__inst_executed_per_opcode[family=tcgen05]"), g(c, d, s, "sass__inst_executed_per_opcode[family=wgmma]")
            assert (t5 > 0) != (wg > 0), (d, s, t5, wg)
            fam[(d, s)] = ("tcgen05" if t5 > 0 else "WGMMA", "tcgen05" if t5 > 0 else "WGMMA")
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path*")
    nld = acc.split("global_load_dword x")[1].split(" ")[0]
    out[c] = {
        "title": "A  Matrix operand delivery (matmul, FP32)", "dsls": D2,
        "table": [
            ("MMA instruction family", fam),
            ("TMA load bytes (GB)", {(d, s): (v, f"{v:.1f}") for d in NV for s in D2
                                     for v in [g(c, d, s, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum") / 1e9]}),
            ("STS / LDSM instr. (M)", {(d, s): ((a, b), f"{a:.2f} / {b:.2f}") for d in NV for s in D2
                                       for a, b in [(g(c, d, s, sts) / 1e6, g(c, d, s, ldsm) / 1e6)]}),
        ],
        "mi300x": f"MI300X Triton (static ISA): TensorDescriptor loads lowered to {nld} scalar\n32-bit global_load_dword; MFMA "
                  f"utilization {g(c, 'MI300X', 'triton', '2.1.10 | MFMA Utilization | Avg*'):.1f}% (rocprof-compute)",
        "interpretation": "Interpretation: on GH200, cuTile adds one STS and one LDSM per WGMMA,\nconsistent with an on-chip "
                          "operand re-layout; B200 feeds tcgen05 via TMA.",
        "caveat": "Confounders: Triton reads B pre-transposed (outside timing); cuTile tiles\n256×256 (B200) vs. 128×128 (GH200); "
                  "cuda-tile 1.3.0 vs. 1.5.0.",
        "basis": "T_SOL: TF32-class MMA term\n(MI300X: XF32); compute-bound",
    }
    c = "destindex/int8"
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path[copy_by_dest_kernel#0]")
    nb = acc.split("buffer_store_byte x")[1].split(" ")[0]
    width = {}
    for d in NV:
        for s in D2:
            mix = E.get(c, d, s, "opcode_mix[STG]")
            ops = [x.split(":")[0] for x in mix.split(";")]
            assert len(ops) == 1, mix
            width[(d, s)] = {"STG.E.128": "128-bit", "STG.E.U8": "8-bit", "STG.E.U16": "16-bit", "STG.E": "32-bit"}[ops[0]]
    sec = {(d, s): g(c, d, s, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") for d in NV for s in D2}
    assert max(sec.values()) / min(sec.values()) < 1.01, sec          # same store traffic: the request count is the difference
    out[c] = {
        "title": "B  Indexing overhead (destindex, INT8)", "dsls": D2,
        "table": [
            ("Executed instructions (M)", {(d, s): (v, f"{v:.1f}") for d in NV for s in D2 for v in [g(c, d, s, "smsp__inst_executed.sum") / 1e6]}),
            ("Global store instr. (M)", {(d, s): (v, f"{v:.2f}") for d in NV for s in D2
                                         for v in [g(c, d, s, "sass__inst_executed_per_opcode_with_modifier_all[STG*]") / 1e6]}),
            ("Store width", {k: (v, v) for k, v in width.items()}),
        ],
        "store_sectors_M": {f"{d}:{s}": v / 1e6 for (d, s), v in sec.items()},
        "mi300x": f"MI300X Triton (static ISA): each row is stored with {nb} per-lane buffer_store_byte,\nno vector store",
        "interpretation": f"Interpretation: for the same {min(sec.values()) / 1e6:.2f} M store sectors, cuTile issues 16× more "
                          "(byte-wide)\nstores and about 17× more instructions than Triton on both NVIDIA GPUs.",
        "caveat": "Confounders: NVIDIA counts are dynamic, MI300X counts static; the memory-only\ntarget does not model index "
                  "arithmetic.",
        "basis": "T_SOL: memory-only (Q / BW)",
    }
    c = "1d_conv/fp16"
    D2 = ("triton", "tilelang")
    kt = E.get(c, "MI300X", "pytorch", "kernel_trace.kernels")
    n_tr = kt.count("transpose")
    acc = E.get(c, "MI300X", "triton", "execution_paths.access_path*")
    ld = acc.split("static loads: ")[1].split(" x")[0]
    vi = g(c, "MI300X", "triton", "SQ_INSTS_VALU_INT32") / g(c, "MI300X", "triton", "SQ_INSTS_VALU")
    for d in NV:
        for s in D2:
            assert "U16" in E.get(c, d, s, "opcode_mix[LDG]"), (d, s)       # same 16-bit load width for both DSLs
    occ = {(d, s): (g(c, d, s, "sm__warps_active.avg.pct_of_peak_sustained_active"), g(c, d, s, "sm__maximum_warps_per_active_cycle_pct"))
           for d in NV for s in D2}
    out[c] = {
        "title": "C  Memory access and latency hiding (1d_conv, FP16)", "dsls": D2,
        "table": [
            ("L1 global-load sectors (M)", {(d, s): (v, f"{v:.0f}") for d in NV for s in D2
                                            for v in [g(c, d, s, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / 1e6]}),
            ("Occupancy ach. / limit (%)", {k: (v, f"{v[0]:.1f} / {v[1]:.1f}") for k, v in occ.items()}),
            ("Issue active (%)", {(d, s): (v, f"{v:.1f}") for d in NV for s in D2
                                  for v in [g(c, d, s, "smsp__issue_active.avg.pct_of_peak_sustained_active")]}),
        ],
        "dram_read_GB": {f"{d}:{s}": g(c, d, s, "dram__bytes_read.sum") / 1e9 for d in NV for s in D2},
        "mi300x": f"MI300X Triton (static ISA, rocprof): {ld} loads; {100 * vi:.0f}% of VALU instr. are\nINT32. PyTorch "
                  f"(MIOpen implicit GEMM + {n_tr} transposes) is not part of T_SOL / T_k.",
        "interpretation": f"Interpretation: with the same 16-bit loads and {min(g(c, d, s, 'dram__bytes_read.sum') for d in NV for s in D2) / 1e9:.2f} GB "
                          "of DRAM reads, TileLang\ntouches 7–10× more L1 sectors; on B200 it reaches 6.2% of an 18.8% "
                          "occupancy limit.",
        "caveat": "Confounders: TileLang uses a different kernel body on Hopper; no controlled\nexperiment isolates the MI300X "
                  "difference.",
        "basis": "T_SOL: max(FP16-MMA, HBM) terms;\ncompute-bound on B200 and MI300X,\nmemory-bound on GH200",
    }
    return out


def proximities(S):
    sel = json.load(open(FD.COMBINED / "rq2_case_selection.json"))["selected"]
    prof = FD.read_csv(FD.COMBINED / "profile_index_normalized.csv")
    res, values = {}, []
    for s in sel:
        op, dt = s["operator"], s["dtype"]
        cids = set()
        for dsl in s["dsls"]:
            for dev in PS.DEVICES:
                if dsl in FD.DSL_SUPPORT[dev]:
                    kind = "rocprof_compute" if dev == "MI300X" else "ncu_"
                    hits = {p["case_id_v2"] for p in prof if p["device"] == dev and p["dsl"] == dsl and p["operator"] == op
                            and p["dtype"] == dt and p["report_kind"].startswith(kind)}
                    assert len(hits) == 1, (dev, dsl, op, dt, hits)
                    cids |= hits
        assert len(cids) == 1, f"profiled cases differ across devices for {op}/{dt}"
        cid = cids.pop()
        for dsl in s["dsls"]:
            for dev in PS.DEVICES:
                if dsl not in FD.DSL_SUPPORT[dev]:
                    continue
                r = S.case(dev, dsl, op, cid)
                assert r is not None, (dev, dsl, op, cid)
                res[(s["case"], dsl, dev)] = r["R_SOL"]
                values.append({"case": s["case"], "dsl": dsl, "device": dev, "case_id_v2": cid, "T_SOL_ms": r["T_SOL_ms"],
                               "dsl_ms": r["dsl_ms"], "sol_proximity": r["R_SOL"], "bound": r["bound"],
                               "compute_mode": r["compute_mode"], "peak_key": r["peak_key"]})
    return sel, res, values


AX_H, LEGEND_H, XTICK_H = 0.84, 0.2, 0.24    # left proximity panel (inches)
TEXT_KEYS = (("mi300x", PS.INK), ("interpretation", PS.INK), ("caveat", PS.MUTED))


def nlines(t):
    return t.count("\n") + 1


def left_height(Rc):
    return LEGEND_H + AX_H + XTICK_H + LINE * nlines(Rc["basis"])


def right_height(Rc):
    return 0.18 + LINE + 0.01 + LINE * len(Rc["table"]) + sum(0.035 + LINE * nlines(Rc[k]) for k, _ in TEXT_KEYS)


def proximity_panel(ax, case, dsls, R):
    ax.set_yscale("log")
    ax.set_ylim(0.004, 3.2)
    ax.set_xlim(-0.35, 2.45)
    ax.axhline(1.0, color=PS.INK, lw=0.7, zorder=1)
    ax.text(2.45, 1.08, "modeled SOL", ha="right", va="bottom", fontsize=7, color=PS.INK)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.grid(axis="y", color=PS.GRID, lw=0.35)
    ax.set_xticks(range(3))
    ax.set_xticklabels(PS.DEVICES)
    ax.tick_params(axis="x", length=0)
    pts = {dsl: [(i, R[(case, dsl, d)]) for i, d in enumerate(PS.DEVICES) if (case, dsl, d) in R] for dsl in dsls}
    for dsl, p in pts.items():
        ax.plot([x for x, _ in p], [y for _, y in p], color=PS.DSL_COLORS[dsl], lw=1.2, marker="o", ms=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)
    above = []
    for i in range(3):        # value labels: the higher of two points above, the lower below; a single point near the
        here = sorted(((p[k][1], dsl) for dsl, p in pts.items() for k in range(len(p)) if p[k][0] == i))   # SOL line below
        for rank, (v, dsl) in enumerate(here):
            up = rank == len(here) - 1 and not (len(here) == 1 and 0.35 < v < 1.0)
            lab = PS.proximity_label(v) + ("†" if v > 1 else "")
            above += [v] if v > 1 else []
            ax.annotate(lab, (i, v), xytext=(0, 4.5 if up else -4.5), textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=7, color=PS.DSL_COLORS[dsl])
    ax.legend(handles=[Line2D([], [], color=PS.DSL_COLORS[s], lw=1.2, marker="o", ms=3.5, label=PS.DSL_LABEL[s]) for s in dsls],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=7, handlelength=1.4, handletextpad=0.4, borderaxespad=0.1,
              ncol=2, columnspacing=0.8)
    ax.set_ylabel("T_SOL / T_k", fontsize=7, labelpad=1)
    return pts, above


def evidence_table(fig, H, Rc, dsls, y_top):
    """NVIDIA-only table with its top at y_top (inches from the bottom); returns the y below it."""
    cols = [(d, s) for d in NV for s in dsls]
    x = [TX0 + LABEL_W + (j + 0.5) * COL_W for j in range(len(cols))]
    fig.text(TX0 / W, y_top / H, "NVIDIA NCU, profiled run()", ha="left", va="top", fontsize=7, fontweight="bold")
    for k, d in enumerate(NV):
        xa, xb = x[2 * k] - 0.5 * COL_W + 0.04, x[2 * k + 1] + 0.5 * COL_W - 0.04
        fig.text((xa + xb) / 2 / W, y_top / H, d, ha="center", va="top", fontsize=7, fontweight="bold")
        fig.add_artist(Line2D([xa / W, xb / W], [(y_top - 0.135) / H] * 2, color="#9A9FA4", lw=0.5))
    y = y_top - 0.16
    for j, (d, s) in enumerate(cols):
        fig.text(x[j] / W, y / H, PS.DSL_LABEL[s], ha="center", va="top", fontsize=7, color=PS.DSL_COLORS[s])
    y -= LINE + 0.01
    for lab, vals in Rc["table"]:
        fig.text(TX0 / W, y / H, lab, ha="left", va="top", fontsize=7, color=PS.INK)
        for j, k in enumerate(cols):
            fig.text(x[j] / W, y / H, vals[k][1], ha="center", va="top", fontsize=7, color=PS.INK)
        y -= LINE
    return y


def plot(out_root):
    PS.apply()
    S = SD.Sol()
    sel, R, values = proximities(S)
    E = Ev()
    C = case_rows(E)
    blocks = [TITLE_H + max(left_height(C[s["case"]]), right_height(C[s["case"]])) for s in sel]
    above_any = any(v > 1 for v in R.values())
    H = round(0.06 + sum(blocks) + GAP_H * (len(sel) - 1) + (0.2 if above_any else 0.0), 2)
    fig = plt.figure(figsize=(W, H))
    plotted, above_one = {}, []
    top = H - 0.06
    for i, s in enumerate(sel):
        case = s["case"]
        Rc = C[case]
        if i:
            fig.add_artist(Line2D([0.1 / W, (W - 0.1) / W], [(top + GAP_H / 2) / H] * 2, color=PS.GRID, lw=0.6))
        fig.text(0.1 / W, top / H, Rc["title"], ha="left", va="top", fontsize=8, fontweight="bold")
        y0 = top - TITLE_H - LEGEND_H - AX_H
        ax = fig.add_axes([0.66 / W, y0 / H, 1.62 / W, AX_H / H])
        pts, ab = proximity_panel(ax, case, Rc["dsls"], R)
        above_one += [{"case": case, "value": v} for v in ab]
        fig.text(0.1 / W, (y0 - XTICK_H) / H, Rc["basis"], ha="left", va="top", fontsize=7, color=PS.MUTED, linespacing=1.1)
        y = evidence_table(fig, H, Rc, Rc["dsls"], top - TITLE_H)
        for key, color in TEXT_KEYS:
            fig.text(TX0 / W, (y - 0.035) / H, Rc[key], ha="left", va="top", fontsize=7, color=color, linespacing=1.1)
            y -= 0.035 + LINE * nlines(Rc[key])
        plotted[case] = {"table": {lab: {f"{d}:{s_}": v[0] for (d, s_), v in vals.items()} for lab, vals in Rc["table"]},
                         "table_text": {lab: {f"{d}:{s_}": v[1] for (d, s_), v in vals.items()} for lab, vals in Rc["table"]}}
        top -= blocks[i] + GAP_H
    if above_one:
        fig.text(0.1 / W, 0.04 / H, "† Above the modeled SOL (kept, not clipped): the B200 TF32 ceiling is a sustained "
                 "library-GEMM rate, not a hardware bound (SOL audit).", ha="left", va="bottom", fontsize=7, color=PS.MUTED)
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    return sel, paths, layout, values, sorted(set(E.used)), C, plotted, above_one


def main(out_root=FD.PLOTS):
    sel, paths, layout, values, used, C, plotted, above_one = plot(out_root)
    D = FD.Data("autotune")
    selection = json.load(open(FD.COMBINED / "rq2_case_selection.json"))
    prov = json.load(open(SD.SOL / "sol_provenance.json"))
    manifest = {
        "figure": NAME, "rq": "RQ2", "script": "scripts/paper_figures/plot_rq2.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "profile_index_normalized.csv", "figure_evidence.csv",
                                              "rq2_case_selection.json"]),
        "sol_inputs": SD.sol_input_hashes(), "sol_code_sha256": SD.sol_code_hashes(),
        "metric": "Proximity to modeled SOL (T_SOL / T_k)",
        "metric_formula": {"left": "T_SOL / T_k at the case_id_v2 captured by every profile of the case (one input case, identical on all "
                                   "devices; not an operator aggregate); T_SOL from the frozen algorithm-level compute mode and the "
                                   "device's hybrid peaks (published dense rates for matmul_fp32_fp16_fp8, PR #323 "
                                   "empirical peaks otherwise); T_k = formal autotuned latency",
                           "right": "NVIDIA dynamic counters summed over the profiled run() launches (B200 and GH200 only), as a table; "
                                    "MI300X static ISA, rocprof counters and kernel traces as separate text"},
        "aggregation_order": ["none: one input case per row"],
        "case_selection": {"rule": selection["selection_rules"], "selected": [{k: x[k] for k in ("case", "mechanism", "why", "caveats")}
                                                                             for x in selection["selected"]],
                           "not_selected": selection["not_selected"],
                           "sol_note": "the three cases are retained from the speedup version; each has a defensible SOL target "
                                       "(sol_mode_manifest.json): matmul FP32 = TF32-class MMA (MI300X XF32, decision D1), "
                                       "destindex INT8 = memory-only (M2, no arithmetic on the data), 1d_conv FP16 = FP16 MMA"},
        "case_coverage": {x["case"]: {"case_id_v2": next(v["case_id_v2"] for v in values if v["case"] == x["case"])} for x in sel},
        "excluded_cases": {"cuTile/TileLang on MI300X": "not available"},
        "measurement_kinds_per_axis": {"left": ["formal benchmark latency", "modeled SOL target (no measurement)"],
                                       "right": ["dynamic NCU counters and dynamic SASS counts (NVIDIA only, one table per case)",
                                                 "occupancy as 'achieved / theoretical limit' (sm__warps_active.avg.pct_of_peak_sustained_active / "
                                                 "sm__maximum_warps_per_active_cycle_pct)"],
                                       "text": ["MI300X static ISA (execution_paths.csv)", "MI300X rocprof counters", "MI300X PyTorch kernel trace"]},
        "evidence_text": {c: {k: C[c][k] for k in ("mi300x", "interpretation", "caveat", "basis")} for c in C},
        "above_one": {"values": above_one, "audit": "sol/sol_above_one_cases.csv; the calibrated MMA ceiling is a sustained library-GEMM "
                      "rate (power-capped telemetry), not a hardware bound"},
        "peaks": prov["peaks"],
        "profiling_evidence_ids": used,
        "known_limitations": D.manifest["device_limitations"] | {
            "figure": ["the counters describe the profiled launches; they support, but do not prove, the attribution of the change in proximity",
                       "B200 and GH200 use different benchmark protocols and cuda-tile versions",
                       "MI300X cannot be compared numerically with the NVIDIA counters (different counters and static vs dynamic counts)",
                       "T_SOL is a modeled hybrid reference (published dense GEMM rates, empirical sustained rates otherwise), "
                       "not a proven bound"]},
        "plotted_values": {"sol_proximity": values, "evidence": plotted},
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
