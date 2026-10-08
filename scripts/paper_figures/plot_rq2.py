"""Figure 3 (RQ2): cross-device performance changes at the profiled case + device-native evidence."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

NAME = "fig_rq2_cross_device_diagnosis"


class Ev:
    def __init__(self):
        self.rows = FD.read_csv(FD.COMBINED / "figure_evidence.csv")
        self.used = []

    def get(self, case, dev, dsl, metric_sub, fig="rq2"):
        """metric_sub: substring of the metric name, or '=name' for an exact match."""
        exact = metric_sub.startswith("=")
        hits = [r for r in self.rows if r["figure"] == fig and r["case"] == case and r["device"] == dev and r["dsl"] == dsl
                and ((r["metric_name"] == metric_sub[1:]) if exact else (metric_sub in r["metric_name"]))]
        assert len(hits) >= 1, (case, dev, dsl, metric_sub)
        r = hits[0]
        self.used.append(r["evidence_id"])
        return r["value"]

    def num(self, *a, **k):
        v = self.get(*a, **k)
        return float(v) if v != "" else None


def evidence_text(E):
    """Device-native evidence per case (<= 28 characters per line); every number comes from figure_evidence.csv."""
    g = E.num
    T = {}
    c = "matmul_fp32_fp16_fp8/fp32"
    T[(c, "B200")] = ("T, C: TMA → tcgen05\n"
                      f"TMA: C {g(c,'B200','cutile','tma_ld')/1e9:.1f} GB, T {g(c,'B200','triton','tma_ld')/1e9:.1f} GB\n"
                      "(C: larger tile)")
    st, w = g(c, "GH200", "cutile", "[STS*]"), g(c, "GH200", "cutile", "family=wgmma")
    T[(c, "GH200")] = (f"T, C: TMA → WGMMA ({w/1e6:.1f}M)\n"
                       f"C per WGMMA: {st/w:.2f} STS +\n"
                       f"{g(c,'GH200','cutile','[LDSM*]')/w:.2f} LDSM (operand re-layout)")
    T[(c, "MI300X")] = ("T: TensorDescriptor lowered\n"
                        "to pointer loads (no TMA)\n"
                        f"MFMA util. {g(c,'MI300X','triton','MFMA Utilization'):.1f}%")
    c = "block_sparse_attention/fp16"
    T[(c, "B200")] = (f"C: tcgen05, {g(c,'B200','cutile','shared_mem_per_block')/1024:.0f} KiB smem/CTA\n"
                      f"T: legacy HMMA, {g(c,'B200','triton','shared_mem_per_block')/1024:.0f} KiB")
    T[(c, "GH200")] = (f"C: WGMMA, {g(c,'GH200','cutile','shared_mem_per_block')/1024:.0f} KiB smem/CTA\n"
                       f"T: legacy HMMA, {g(c,'GH200','triton','shared_mem_per_block')/1024:.0f} KiB")
    rs = E.get(c, "MI300X", "triton", "resource_summary")
    vg = rs.split("VGPR ")[1].split(" ")[0]
    ag = rs.split("AGPR ")[1].split(" ")[0]
    T[(c, "MI300X")] = (f"T: {vg} VGPR + {ag} AGPR,\n"
                        f"{g(c,'MI300X','triton','Scratch Allocation'):.0f} B scratch/work-item\n"
                        f"MFMA util. {g(c,'MI300X','triton','MFMA Utilization'):.1f}%")
    c = "destindex/int8"
    for dev in ("B200", "GH200"):
        T[(c, dev)] = (f"store sectors/req: C {g(c,dev,'cutile','sectors_pipe_lsu_mem_global_op_st'):.1f}, "
                       f"T {g(c,dev,'triton','sectors_pipe_lsu_mem_global_op_st'):.0f}\n"
                       f"C: {g(c,dev,'cutile','inst_executed.sum')/g(c,dev,'triton','inst_executed.sum'):.1f}× T instructions")
    acc = E.get(c, "MI300X", "triton", "access_path")
    nbyte = acc.split("buffer_store_byte x")[1].split(" ")[0] if "buffer_store_byte x" in acc else "?"
    T[(c, "MI300X")] = f"T: per-element byte stores\n(buffer_store_byte ×{nbyte},\nstatic ISA)"
    c = "flash_decode/fp32"
    for dev in ("B200", "GH200"):
        T[(c, dev)] = (f"{g(c,dev,'cutile','grid_size'):.0f} CTAs; C: {g(c,dev,'cutile','inst_executed.sum')/g(c,dev,'triton','inst_executed.sum'):.1f}× T instr.\n"
                       f"long-scoreboard stall/issue:\nC {g(c,dev,'cutile','long_scoreboard'):.1f}, T {g(c,dev,'triton','long_scoreboard'):.1f}")
    rs = E.get(c, "MI300X", "triton", "resource_summary")
    T[(c, "MI300X")] = (f"T: {rs.split('= ')[1].split(' WGs')[0]} WGs "
                        f"({float(rs.split('(')[1].split(' per')[0]):.2f} per CU)\nserial, latency-bound loop")
    c = "1d_conv/fp16"
    T[(c, "B200")] = (f"load sectors/req:\nTL {g(c,'B200','tilelang','sectors_pipe_lsu_mem_global_op_ld'):.1f}, T {g(c,'B200','triton','sectors_pipe_lsu_mem_global_op_ld'):.1f}\n"
                      f"TL occ. {g(c,'B200','tilelang','warps_active'):.1f}% (theor. {g(c,'B200','tilelang','maximum_warps'):.1f}%)")
    T[(c, "GH200")] = (f"load sectors/req:\nTL {g(c,'GH200','tilelang','sectors_pipe_lsu_mem_global_op_ld'):.1f}, T {g(c,'GH200','triton','sectors_pipe_lsu_mem_global_op_ld'):.1f}\n"
                       f"TL occ. {g(c,'GH200','tilelang','warps_active'):.1f}% (Hopper body)")
    T[(c, "MI300X")] = (f"T: INT32 = {100*g(c,'MI300X','triton','=SQ_INSTS_VALU_INT32')/g(c,'MI300X','triton','=SQ_INSTS_VALU'):.0f}% of VALU instr.\n"
                        "PyTorch: MIOpen igemm\n+ layout transposes")
    E.get(c, "MI300X", "triton", "diagnosis:M3")
    c = "vector_add/fp32"
    for dev in ("B200", "GH200"):
        T[(c, dev)] = (f"T: {g(c,dev,'triton','sectors_pipe_lsu_mem_global_op_ld'):.0f} sectors/req loads,\n"
                       f"DRAM {g(c,dev,'triton','dram_throughput'):.0f}% of peak")
    E.get(c, "MI300X", "triton", "diagnosis:M2")
    E.get(c, "MI300X", "triton", "access_path")
    T[(c, "MI300X")] = "T: no nt loads (ATen: nt);\n.cg loads recover (Fig. A5)"
    return T


def plot(out_root):
    PS.apply()
    D = FD.Data("autotune")
    sel = json.load(open(FD.COMBINED / "rq2_case_selection.json"))["selected"]
    prof = FD.read_csv(FD.COMBINED / "profile_index_normalized.csv")
    E = Ev()
    T = evidence_text(E)

    def profiled_case(dev, dsl, op, dt):
        kind = "rocprof_compute" if dev == "MI300X" else "ncu_"
        hits = [p for p in prof if p["device"] == dev and p["dsl"] == dsl and p["operator"] == op and p["dtype"] == dt
                and p["report_kind"].startswith(kind)]
        return hits[0]["case_id_v2"] if hits else None

    rows, values = [], []
    for s in sel:
        op, dt = s["operator"], s["dtype"]
        cids = {}
        for dsl in s["dsls"]:
            for dev in PS.DEVICES:
                if not D.supported(dev, dsl):
                    continue
                cid = profiled_case(dev, dsl, op, dt)
                cids[(dev, dsl)] = cid
        assert len({v for v in cids.values() if v}) == 1, f"profiled cases differ across devices for {op}/{dt}"
        cid = next(v for v in cids.values() if v)
        params = json.loads(next(r["params_full_json"] for r in D.rows if r["case_id_v2"] == cid))
        for dsl in s["dsls"]:
            pts = {}
            for dev in PS.DEVICES:
                if not D.supported(dev, dsl):
                    continue
                c = D.case(dev, dsl, op, cid)
                pts[dev] = None if c is None else c[0] / c[1]
                values.append({"case": s["case"], "dsl": dsl, "device": dev, "case_id_v2": cid,
                               "torch_ms": None if c is None else c[0], "dsl_ms": None if c is None else c[1],
                               "speedup": pts[dev]})
            rows.append((s, dsl, pts, params))

    n = len(rows)
    fig = plt.figure(figsize=(PS.DOUBLE_COL_IN, 4.1))
    axA = fig.add_axes([0.19, 0.16, 0.215, 0.73])
    axB = fig.add_axes([0.425, 0.16, 0.575, 0.73])
    ys = []
    y, prev = 0.0, None
    for s, dsl, pts, _ in rows:
        if prev is not None and s["case"] != prev:
            y += 0.55
        ys.append(y)
        prev = s["case"]
        y += 1.0
    ymax = y
    for (s, dsl, pts, _), yy in zip(rows, ys):
        yy = ymax - yy - 0.5
        xs = [v for v in pts.values() if v is not None]
        order = [(pts[d], {"B200": 0.17, "GH200": 0.0, "MI300X": -0.17}[d]) for d in PS.DEVICES if pts.get(d) is not None]
        axA.plot([p[0] for p in order], [yy + p[1] for p in order], color="#C9CCCF", lw=0.9, zorder=1)
        off = {"B200": 0.17, "GH200": 0.0, "MI300X": -0.17}
        for dev, v in pts.items():
            if v is None:
                continue
            axA.scatter([v], [yy + off[dev]], marker=PS.DEVICE_MARKERS[dev], s=22 if dev != "MI300X" else 26, color=PS.DSL_COLORS[dsl],
                        edgecolor="white", linewidth=0.5, zorder=3)
        axA.text(-0.03, yy, PS.DSL_LABEL[dsl], transform=axA.get_yaxis_transform(), ha="right", va="center", fontsize=6.4,
                 color=PS.DSL_COLORS[dsl], fontweight="bold", clip_on=False)
    axA.set_xscale("log", base=2)
    axA.axvline(1.0, color="#9A9FA4", lw=0.6, ls=(0, (2, 2)), zorder=0)
    lo = min(v["speedup"] for v in values if v["speedup"])
    hi = max(v["speedup"] for v in values if v["speedup"])
    axA.set_xlim(2 ** math.floor(math.log2(lo) - 0.2), 2 ** math.ceil(math.log2(hi) + 0.2))
    ticks = [t for t in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32) if axA.get_xlim()[0] <= t <= axA.get_xlim()[1]]
    axA.set_xticks(ticks)
    # grid line at every power of two, label every second one (1/16, 1/4, 1, 4, ...) so labels do not collide
    axA.set_xticklabels([("" if round(math.log2(t)) % 2 else (f"{t:g}×" if t >= 1 else f"1/{int(1/t)}×")) for t in ticks])
    axA.minorticks_off()
    axA.set_ylim(0, ymax)
    axA.set_yticks([])
    axA.spines["left"].set_visible(False)
    axA.set_xlabel("speedup vs. local PyTorch at the profiled case", fontsize=6.6)
    axA.grid(axis="x", color=PS.GRID, lw=0.4)
    # case labels
    groups = {}
    for (s, dsl, pts, params), yy in zip(rows, ys):
        groups.setdefault(s["case"], []).append((ymax - yy - 0.5, params, s))
    for case, items in groups.items():
        yc = sum(i[0] for i in items) / len(items)
        op, dt = case.split("/")
        axA.text(0.008, yc, f"{op.replace('matmul_fp32_fp16_fp8', 'matmul').replace('block_sparse_attention', 'block_sparse_attn')}\n{dt}",
                 transform=blended_transform_factory(fig.transFigure, axA.transData), ha="left", va="center", fontsize=6.4,
                 color=PS.INK, clip_on=False, linespacing=1.15)
    axA.set_title("A  Performance at the profiled case", loc="left", fontsize=7.5, fontweight="bold", x=(0.008 - 0.19) / 0.215, pad=12)
    # panel B: evidence matrix
    axB.set_xlim(0, 3)
    axB.set_ylim(0, ymax + 0.35)
    axB.axis("off")
    for j, dev in enumerate(PS.DEVICES):
        axB.text(j + 0.04, ymax - 0.05, dev, ha="left", va="bottom", fontsize=7.0, fontweight="bold")
    for case, items in groups.items():
        top = max(i[0] for i in items) + 0.5
        bot = min(i[0] for i in items) - 0.5
        axB.plot([0.02, 2.98], [top + 0.27, top + 0.27], color=PS.GRID, lw=0.5) if case != list(groups)[0] else None
        for j, dev in enumerate(PS.DEVICES):
            txt = T[(case, dev)]
            axB.text(j + 0.04, (top + bot) / 2, txt, ha="left", va="center", fontsize=6.0, color=PS.INK, linespacing=1.25)
    axB.set_title("B  Device-native evidence (counters are not comparable across vendors)", loc="left", fontsize=7.5,
                  fontweight="bold", x=0.0, pad=12)
    handles = [Line2D([], [], marker=PS.DEVICE_MARKERS[d], ls="", color="#6E747A", markersize=4.2, label=d) for d in PS.DEVICES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, handletextpad=0.2, columnspacing=0.7,
               bbox_to_anchor=(0.19 + 0.215 / 2, 0.0), frameon=False)
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    return paths, layout, values, sorted(set(E.used)), T


def main(out_root=FD.PLOTS):
    paths, layout, values, used, T = plot(out_root)
    D = FD.Data("autotune")
    sel = json.load(open(FD.COMBINED / "rq2_case_selection.json"))
    manifest = {
        "figure": NAME, "rq": "RQ2", "script": "scripts/paper_figures/plot_rq2.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "profile_index_normalized.csv", "figure_evidence.csv",
                                              "rq2_case_selection.json"]),
        "metric_formula": "panel A: torch_ms/dsl_ms of the formal CSV at the case_id_v2 captured by the profiles (single case, not an operator aggregate); panel B: device-native profiler/compiler evidence as text",
        "aggregation_order": ["none (single profiled case per operator/dtype)"],
        "case_coverage": {"cases": [s["case"] for s in sel["selected"]], "profiled_case_identical_across_devices": True},
        "excluded_cases": {"not_selected": sel["not_selected"]}, "selection_criteria": sel["selection_rules"],
        "selection_manifest": "artifacts/paper_figures/combined/rq2_case_selection.json",
        "profiling_evidence_ids": used, "evidence_text": {f"{k[0]}|{k[1]}": v for k, v in T.items()},
        "measurement_kinds_per_axis": {"A.x": ["formal_csv_latency_ratio"], "B": ["text only; no numeric axis"]},
        "known_limitations": D.manifest["device_limitations"] | {"figure": [
            "B200/GH200 cuTile versions differ; autotuned winners may differ between devices",
            "TileLang uses another kernel body on Hopper for 1d_conv",
            "Triton matmul caches the transposed B outside the timed region",
            "some B200 NCU reports omit auxiliary launches and have no recorded source SHA",
            "GH200 flash_decode: cuTile long-scoreboard ratio is NOT higher than Triton's; only instruction expansion is shown",
            "MI300X vector_add cache-policy recovery comes from a diagnostic (non-formal) experiment"]},
        "plotted_values": values, "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
