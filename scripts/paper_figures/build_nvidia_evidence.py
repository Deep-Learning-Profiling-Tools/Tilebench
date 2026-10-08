"""Build environment.json (per device), cross_device_pairs.csv and diagnosis_evidence.csv (per device)
from the normalized tables written by build_nvidia_tables.py. Offline; reads only artifacts and git refs.

Usage:
  python scripts/paper_figures/build_nvidia_evidence.py --repo . --root artifacts/paper_figures/nvidia \
      --inventory-dir <cache>
"""
import argparse
import collections
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvidia_common as C  # noqa: E402

csv.field_size_limit(1 << 30)
DEVS = ("B200", "GH200")

# ----------------------------------------------------------------------------------------------- environment
SOFTWARE = {
    "B200": {
        "benchmark_campaign": {
            "csv_dir": "results/B200/csv",
            "timing": "Proton, CUDA graph",
            "final_measurement_warmup_repeat": "not recorded; config.yaml at every result commit has 20/100 (developer_guide.md, Multi-Architecture Status)",
            "l2_eviction": "PyTorch/Triton/cuTile columns: fixed 64 MB buffer for 41 operators; 2x L2 (~253 MB) for cross_entropy, flash_decode, moe_topk_gating, linear_self_attention; TileLang columns: depends on run date (developer_guide.md)",
            "benchmark_source": "not recorded (CSVs predate tilebench/provenance.py)",
            "software": "PyTorch/Triton/cuTile columns from the paper campaign (cuda-tile 1.3.0 per developer_guide.md); TileLang 0.1.11",
            "tilelang_column": "direct TileLang runtime (PR #319), measured in a later campaign than the frozen PyTorch/Triton/cuTile columns",
            "raw_logs": "archive/raw-logs-2026-09-18",
        },
        "profiling": {
            "triton_cutile": "Nsight Compute 2026.1.1 (report session), --set full, application replay, --cache-control none; captured 2026-08 on dgx003 from tilebench_env; profiling source commit and cuda-tile version not recorded in the reports",
            "tilelang": "HF bcui2/NCU_report PR #2 (commit 21037737): driver 595.58.03, PyTorch 2.10.0+cu130, Triton 3.6.0, cuda-tile 1.3.0, TileLang 0.1.11, CUDA compiler 13.0.88, Nsight Compute 2026.1.1; 97 full, 5 targeted, 8 reduced (kernel replay)",
            "metadata_archive": "archive/raw-logs-2026-09-18 outputs/profiling/B200/{ncu_catalogue,kernel_counts}.json",
        },
    },
    "GH200": {
        "benchmark_campaign": {
            "csv_dir": "results/GH200/csv",
            "timing": "Proton, CUDA graph",
            "final_measurement_warmup_repeat": "1/3 (--warmup 1 --repeat 3)",
            "l2_eviction": "120 MiB (2x 60 MiB L2)",
            "benchmark_source": "c882fe50 (89 CSVs), 3c5eccbf (batched_matmul_autotune.csv)",
            "software": "torch 2.10.0+cu130, Triton 3.6.0, cuda-tile 1.5.0 (pip tileiras 13.4.92), TileLang 0.1.11",
            "tilelang_column": "measured in the same run as the other backends",
            "raw_logs": "archive/tilebenchpp-2026-10 66046918",
        },
        "profiling": {
            "all": "Nsight Compute 2025.4.0, --set full --import-source on --replay-mode application --cache-control none --app-replay-mode strict; 3 warmups + 120 MiB eviction outside the range, one impl.run() inside (PROVENANCE.md)",
            "sources": "Triton 108 + cuTile 110: 638ea849; Triton bitonic_sort fp16/fp32: 5610f18f; TileLang 110: d6ddb622",
            "metadata_archive": "archive/tilebenchpp-2026-10 outputs/profiling/GH200/{ncu_catalogue,kernel_counts}.json, ncu_sweep/{PROVENANCE.md,report_manifest.json}",
            "tilelang_hopper_path": "9 operators with a Blackwell TMEM path (1d/2d/3d_conv, batched_matmul, block_sparse_attention, flash_attention, matmul_fp32_fp16_fp8, matmul_int8, streamk_matmul) build a fragment-accumulator body on sm_90 (supports_tmem() dispatch)",
        },
    },
}

TRITON_HOST_BT = {"matmul_fp32_fp16_fp8", "matmul_int8", "batched_matmul", "streamk_matmul"}
TL_TMEM_OPS = {"1d_conv", "2d_conv", "3d_conv", "batched_matmul", "block_sparse_attention", "flash_attention",
               "matmul_fp32_fp16_fp8", "matmul_int8", "streamk_matmul"}


def read_csv(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class Dev:
    """Lookup helpers over one device's normalized tables."""

    def __init__(self, root, dev):
        d = Path(root) / dev
        self.dev = dev
        self.prof = {r["profile_id"]: r for r in read_csv(d / "profile_index.csv")}
        self.bench = {}
        for r in read_csv(d / "benchmark_cases.csv"):
            if r["mode"] == "autotune":
                self.bench[(r["dsl"], r["case_id"])] = r
        self.m = collections.defaultdict(list)          # pid -> metric rows
        for r in read_csv(d / "kernel_metrics_long.csv.gz"):
            if r["status"] == "collected":
                self.m[r["profile_id"]].append(r)
        self.mix = collections.defaultdict(list)
        for r in read_csv(d / "instruction_mix.csv"):
            self.mix[r["profile_id"]].append(r)
        self.paths = collections.defaultdict(list)
        for r in read_csv(d / "execution_paths.csv"):
            self.paths[r["profile_id"]].append(r)

    def pid(self, dsl, op, dt):
        return f"{self.dev}/{dsl}/{op}/{dt}"

    def total(self, pid, metric, stage=None):
        vals = {}
        for r in self.m[pid]:
            if r["raw_metric_name"] == metric and (stage is None or r["stage"] == stage):
                vals[r["launch_id"]] = fnum(r["value"])
        vals = [v for v in vals.values() if v is not None]
        return sum(vals) if vals else None

    def dom(self, pid, metric):
        dur = {r["launch_id"]: fnum(r["value"]) for r in self.m[pid] if r["raw_metric_name"] == "gpu__time_duration.sum"}
        best = None
        for r in self.m[pid]:
            if r["raw_metric_name"] == metric:
                d = dur.get(r["launch_id"]) or 0
                if best is None or d > best[0]:
                    best = (d, fnum(r["value"]) if fnum(r["value"]) is not None else r["value"])
        return best[1] if best else None

    def fam(self, pid, family):
        rows = [r for r in self.mix[pid] if r["count_kind"] == "dynamic_warp_inst_executed_family_total"]
        if not rows:
            return None
        return sum(int(r["count"]) for r in rows if r["instruction_family"] == family)

    def opc(self, pid, prefix, static=False):
        kind = "static_sass_instruction_count" if static else "dynamic_warp_inst_executed_with_modifier"
        rows = [r for r in self.mix[pid] if r["count_kind"] == kind]
        if not rows:
            return None
        return sum(int(r["count"]) for r in rows if r["instruction_name"] == prefix or r["instruction_name"].startswith(prefix + "."))

    def lat(self, pid):
        p = self.prof.get(pid)
        if not p:
            return None
        b = self.bench.get((p["dsl"], p["case_id"]))
        return fnum(b["dsl_ms"]) if b else None

    def matrix_path(self, pid):
        return " / ".join(f"{r['stage']}:{r['matrix_path']}" for r in self.paths[pid])

    def access_path(self, pid):
        return " / ".join(f"{r['stage']}:{r['access_path']}" for r in self.paths[pid])


def g4(v):
    if v is None:
        return "NA"
    if isinstance(v, str):
        return v
    if abs(v) >= 1e6:
        return f"{v / 1e6:.4g}M"
    return f"{v:.4g}"


def evaluate(D, term, dev, dsl, op, dt):
    d = D[dev]
    pid = d.pid(dsl, op, dt)
    kind, _, arg = term.partition(":")
    if kind == "total":
        return d.total(pid, arg)
    if kind == "dom":
        return d.dom(pid, arg)
    if kind == "fam":
        return d.fam(pid, arg)
    if kind == "opc":
        return d.opc(pid, arg)
    if kind == "sopc":
        return d.opc(pid, arg, static=True)
    if kind == "lat":
        return d.lat(pid)
    if kind == "ratio":
        a, b = arg.split("/")
        x, y = d.total(pid, a), d.total(pid, b)
        return x / y if x is not None and y else None
    if kind == "stage":
        st, m = arg.split("@")
        return d.total(pid, m, stage=st)
    if kind == "path":
        return d.matrix_path(pid)
    raise ValueError(term)


def metric_name(term):
    kind, _, arg = term.partition(":")
    return {"fam": f"sass__inst_executed_per_opcode[family={arg}]", "opc": f"sass__inst_executed_per_opcode_with_modifier_all[{arg}*]",
            "sopc": f"static_sass[{arg}*]", "lat": "benchmark_cases.dsl_ms", "path": "execution_paths.matrix_path",
            "ratio": arg, "stage": arg.replace("@", ":")}.get(kind, arg)


# ------------------------------------------------------------------------------------------ mechanism specs
# Each spec: device(s) and DSLs compared, the terms to report, and the interpretation. Observations are
# generated from the extracted data; nothing below is a measured number.
def S(device, dsls, op, dt, mid, terms, hyp, alt, conf, quality, sources, scope=None):
    return dict(device=device, dsls=dsls, operator=op, dtype=dt, mechanism_id=mid, terms=terms, hypothesis=hyp,
                alternative=alt, confounders=conf, quality=quality, sources=sources, scope=scope)


SPECS = [
    # ---------------- B200, Triton vs cuTile (paper Appendix D.3)
    S("B200", ("cutile", "triton"), "weight_dequant", "bf16", "index_gather_instruction_expansion",
      ["lat", "total:smsp__inst_executed.sum", "fam:integer_address", "fam:predicate_select", "total:sass__inst_executed_global_loads",
       "dom:launch__registers_per_thread", "dom:sm__warps_active.avg.pct_of_peak_sustained_active"],
      "cuTile lowers the scale gather (ct.gather) and its div/mod index chain into a longer integer/predicate sequence with high register use; Triton keeps a short affine address path; matched global loads",
      "different autotuned tile sizes change per-thread work", "cuda-tile version not recorded for the B200 profiles", "high",
      "impl_triton.py/impl_cutile.py (weight_dequant); paper Appendix D.3"),
    S("B200", ("cutile", "triton"), "destindex", "int8", "scatter_elementwise_lowering",
      ["lat", "total:smsp__inst_executed.sum", "total:sass__inst_executed_global_stores",
       "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum/l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
       "dom:sm__warps_active.avg.pct_of_peak_sustained_active"],
      "ct.scatter is lowered element-wise (1 sector/request) while Triton keeps the contiguous feature dimension in 128-bit stores",
      "occupancy differences", "two launches (nope/rope regions) summed", "high", "paper Appendix D.3"),
    S("B200", ("cutile", "triton"), "1d_conv", "fp16", "implicit_gemm_index_expansion",
      ["lat", "total:smsp__inst_executed.sum", "fam:integer_address", "fam:predicate_select", "fam:legacy_mma", "fam:tcgen05",
       "dom:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"],
      "cuTile gather-based im2col expands address/predicate work; tensor pipe mostly idle in both", "MMA family differs with the winner tile (cuTile winner uses legacy HMMA here)",
      "winner configs differ", "high", "paper Appendix D.3"),
    S("B200", ("cutile", "triton"), "cross_entropy", "fp16", "reduction_granularity",
      ["lat", "total:smsp__inst_executed.sum", "total:sass__inst_executed_shared_loads", "total:sass__inst_executed_shared_stores",
       "fam:synchronization", "dom:launch__block_size", "dom:sm__warps_active.avg.pct_of_peak_sustained_active"],
      "Triton maps a short row to one warp (no shared-memory reduction); cuTile uses a 128-thread CTA with a shared-memory tree",
      "higher cuTile occupancy partly compensates", "", "high", "paper Appendix D.3"),
    S("B200", ("cutile", "triton", "tilelang"), "moe_topk_gating", "fp16", "reduction_granularity",
      ["lat", "total:smsp__inst_executed.sum", "total:sass__inst_executed_shared_loads", "total:sass__inst_executed_shared_stores",
       "dom:launch__block_size"],
      "Triton and TileLang select single-warp CTAs and avoid shared-memory reductions; cuTile's 128-thread CTA adds shared-memory traffic and instructions",
      "", "TileLang latency measured in a separate campaign", "high", "paper Appendix D.3; impl_tilelang.py (threads search space {32,64,128})"),
    S("B200", ("cutile", "triton", "tilelang"), "flash_decode", "fp32", "runtime_loop_pipelining",
      ["lat", "total:smsp__inst_executed.sum", "dom:smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
       "dom:launch__grid_size"],
      "16-CTA grid; Triton and TileLang pipeline the runtime sequence loop (num_stages / T.Pipelined), cuTile's loop shows a longer dependency chain and more instructions",
      "", "TileLang latency measured in a separate campaign", "medium", "paper Appendix D.3; impl_tilelang.py T.Pipelined"),
    S("B200", ("triton", "cutile", "tilelang"), "histogramming", "int32", "privatization_algorithm",
      ["lat", "opc:ATOMG", "opc:ATOMS", "opc:MEMBAR", "stage:histogram_partial_kernel@gpu__time_duration.sum",
       "dom:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"],
      "TileLang privatizes in shared memory (ATOMS, one fence per CTA); Triton/cuTile use global atomics into partial rows with a fence per atomic (algorithmic difference, not compiler lowering)",
      "", "different algorithms; stage duration is NCU time (attribution only)", "high", "impl_*.py (histogramming); paper Appendix D.3"),
    S("B200", ("cutile", "triton", "tilelang"), "matmul_fp32_fp16_fp8", "fp32", "matrix_path_and_tile_capacity",
      ["lat", "path", "total:l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum", "dom:launch__shared_mem_per_block",
       "dom:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "opc:HMMA", "opc:UTCHMMA", "opc:LDGSTS"],
      "Triton and cuTile use TMA + tcgen05 (cuTile with a larger tile and half the TMA bytes); TileLang's FP32 implementation keeps a register accumulator and lowers to legacy HMMA.1688.F32.TF32 fed by LDGSTS",
      "TileLang FP32 path is an implementation choice (use_tmem disabled for TF32)", "Triton pre-transposes B on the host (cached, outside timing); TileLang separate campaign", "high",
      "impl_*.py (matmul_fp32_fp16_fp8); paper Appendix D.3"),
    S("B200", ("triton", "cutile", "tilelang"), "matmul_fp32_fp16_fp8", "fp16", "operand_delivery_without_tma",
      ["lat", "path", "total:smsp__inst_executed.sum", "dom:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
       "sopc:LDGSTS", "sopc:UTMALDG"],
      "All three issue tcgen05 MMAs; TileLang delivers operands with cp.async (LDGSTS) instead of TMA and keeps the tensor pipe less busy",
      "TileLang profile is a reduced kernel-replay collection (no dynamic opcode counts)", "reduced collection; TileLang separate campaign", "medium",
      "HF PR #2 (reduced profile); static SASS"),
    S("B200", ("triton", "cutile", "tilelang"), "flash_attention", "fp16", "operand_staging_and_pipelining",
      ["lat", "path", "total:smsp__inst_executed.sum", "total:sass__inst_executed_global_loads", "total:sass__inst_executed_shared_stores",
       "dom:launch__shared_mem_per_block", "dom:sm__warps_active.avg.pct_of_peak_sustained_active"],
      "cuTile's 128x128 tile halves the K/V loop vs Triton; TileLang issues the same tcgen05 MMA count as cuTile but stages K/V synchronously (LDG->STS, no T.Pipelined with manual TMEM)",
      "", "TileLang source comment: T.Pipelined breaks the manual TMEM path; separate campaign", "high", "impl_tilelang.py (flash_attention); paper Appendix D.3"),
    S("B200", ("tilelang", "triton", "cutile"), "1d_conv", "fp16", "gathered_staging_thread_mapping",
      ["lat", "total:sass__inst_executed_global_loads",
       "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum/l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
       "dom:smsp__issue_active.avg.pct_of_peak_sustained_active", "dom:sm__maximum_warps_per_active_cycle_pct",
       "dom:sm__warps_active.avg.pct_of_peak_sustained_active", "dom:smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
       "opc:UTCHMMA"],
      "TileLang issues the same gathers and MMAs as Triton but with uncoalesced lane mapping and one resident 4-warp CTA per SM, leaving issue slots empty",
      "cause of the one-CTA residency is not exposed by the profile", "TileLang separate campaign", "high", "impl_tilelang.py/impl_triton.py (1d_conv)"),
    S("B200", ("tilelang", "triton"), "2d_max_pooling", "fp32", "shape_specialized_indexing",
      ["lat", "total:smsp__inst_executed.sum", "total:sass__inst_executed_global_loads", "opc:ISETP", "opc:IMAD",
       "dom:smsp__issue_active.avg.pct_of_peak_sustained_active"],
      "TileLang compiles the concrete shape (constants folded into immediates), Triton keeps H/W as runtime arguments and evaluates per-tap bounds predicates; matched loads and tile",
      "", "TileLang separate campaign; JIT specialization recompiles per shape", "high", "impl_tilelang.py/impl_triton.py (2d_max_pooling)"),
    S("B200", ("tilelang", "triton"), "reverse_array", "int8", "per_loop_vectorization",
      ["lat", "total:sass__inst_executed_global_stores",
       "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum/l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
       "dom:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"],
      "TileLang emits byte stores on the contiguous output side; Triton packs bytes and issues one 64-bit store per 8 elements",
      "", "TileLang separate campaign", "medium", "static/dynamic SASS (STG.E.U8 vs STG.E.64)"),
    S("B200", ("cutile", "triton"), "streamk_matmul", "fp16", "first_wave_runtime_loop",
      ["lat", "stage:first_wave@gpu__time_duration.sum", "stage:full_tiles@gpu__time_duration.sum", "fam:synchronization"],
      "same stream-K decomposition; cuTile's first_wave runtime loop is slower while full_tiles is faster", "", "stage durations are NCU time; Triton pre-transposes B on the host", "medium",
      "paper Appendix D.3"),
    S("B200", ("cutile", "triton"), "linear_self_attention", "fp32", "low_parallelism_stage",
      ["lat", "stage:z_kernel@gpu__time_duration.sum", "stage:kv_gemm_kernel@gpu__time_duration.sum", "path"],
      "8-CTA column-reduction stage dominates both; cuTile's runtime-K KV loop is unpipelined", "", "stage durations are NCU time", "medium", "paper Appendix D.3"),
    # ---------------- GH200
    S("GH200", ("cutile", "triton"), "matmul_fp32_fp16_fp8", "fp32", "wgmma_operand_relayout",
      ["lat", "path", "opc:HGMMA", "opc:STS", "opc:LDSM", "fam:integer_address", "dom:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"],
      "both issue the same TF32 WGMMA count; cuTile adds shared-memory re-layout (STS+LDSM per WGMMA) because its B tile is N-major while Hopper TF32 WGMMA needs K-major smem operands",
      "Triton pre-transposes B on the host (cached, outside the timed region), so the comparison mixes layout preprocessing with compiler quality", "Triton host-side B transpose; cuda-tile 1.5.0", "high",
      "impl_triton.py _bt_cache; impl_cutile.py (matmul_fp32_fp16_fp8)"),
    S("GH200", ("cutile", "triton"), "matmul_fp32_fp16_fp8", "fp8_e4m3fn", "wgmma_operand_relayout",
      ["lat", "path", "opc:QGMMA", "opc:STS", "opc:PRMT", "dom:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"],
      "same QGMMA count; cuTile adds shared-memory stores and permutes for FP8 operand layout", "Triton host transpose", "Triton host-side B transpose; cuda-tile 1.5.0", "high",
      "impl_triton.py _bt_cache"),
    S("GH200", ("cutile", "triton"), "destindex", "int8", "scatter_elementwise_lowering",
      ["lat", "total:smsp__inst_executed.sum",
       "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum/l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"],
      "the element-wise scatter lowering seen on B200 persists on Hopper", "", "cuda-tile 1.5.0 vs B200 profile version unrecorded", "high", "paper Appendix D.3 (B200 counterpart)"),
    S("GH200", ("cutile", "triton", "tilelang"), "moe_topk_gating", "fp16", "reduction_granularity",
      ["lat", "total:smsp__inst_executed.sum", "total:sass__inst_executed_shared_stores", "dom:launch__block_size"],
      "TileLang (32-thread CTA) and Triton (64-thread winner on GH200, with some shared stores) execute far fewer instructions than cuTile's 128-thread CTA with a shared-memory tree; CTA granularity remains the discriminating factor",
      "", "Triton winner differs from B200 (64 vs 32 threads)", "high", "paper Appendix D.3"),
    S("GH200", ("cutile", "triton", "tilelang"), "flash_decode", "fp32", "runtime_loop_pipelining",
      ["lat", "total:smsp__inst_executed.sum", "dom:smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
      "cuTile executes several times more instructions than Triton/TileLang in the same 16-CTA runtime loop; unlike B200, its long-scoreboard stall ratio is not higher than Triton's, so only the instruction-expansion part of the B200 explanation is reproduced on GH200",
      "dependency-chain explanation not established on GH200", "", "medium", "paper Appendix D.3"),
    S("GH200", ("tilelang", "triton"), "matmul_fp32_fp16_fp8", "fp8_e4m3fn", "tilelang_hopper_legacy_mma",
      ["lat", "path", "opc:HMMA", "opc:QGMMA", "total:smsp__inst_executed.sum"],
      "TileLang's Hopper fragment-accumulator body lowers FP8 to legacy MMA while Triton uses QGMMA", "", "TileLang Hopper path is a compatibility body (supports_tmem dispatch)", "medium",
      "developer_guide.md (GH200 TileLang dispatch)"),
    S("GH200", ("tilelang", "triton"), "1d_conv", "fp16", "gathered_staging_thread_mapping",
      ["lat", "total:smsp__inst_executed.sum",
       "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum/l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
       "dom:sm__warps_active.avg.pct_of_peak_sustained_active", "dom:smsp__issue_active.avg.pct_of_peak_sustained_active"],
      "TileLang convolution remains latency bound on Hopper with low issue activity", "Hopper body differs from the Blackwell TMEM body", "TileLang Hopper compatibility body", "medium", "impl_tilelang.py (1d_conv)"),
]

# cross-device specs: same DSL, both devices
XSPECS = [
    ("triton", "weight_dequant", "bf16", "config_dependent_instruction_count",
     ["lat", "total:smsp__inst_executed.sum", "fam:integer_address", "dom:launch__block_size"],
     "Triton executes more instructions on GH200 than on B200 for the same source; the autotuned winners differ (see winner_config_json), so this is not attributed to the architecture",
     "a controlled recompile in the prior GH200 analysis attributed the inflation to the winner config; not reproduced here", "medium"),
    ("cutile", "matmul_fp32_fp16_fp8", "fp32", "cross_arch_matrix_path",
     ["lat", "path", "opc:STS", "opc:LDSM", "total:smsp__inst_executed.sum"],
     "cuTile FP32 GEMM uses tcgen05 on B200 and WGMMA plus shared-memory re-layout on GH200; latency rank versus Triton reverses",
     "cuda-tile version differs between campaigns; Triton host-side B transpose", "medium"),
    ("tilelang", "matmul_fp32_fp16_fp8", "fp16", "cross_arch_matrix_path",
     ["lat", "path", "total:smsp__inst_executed.sum"],
     "TileLang FP16 GEMM issues tcgen05 on B200 (reduced profile) and WGMMA on GH200", "B200 profile is reduced; TileLang Hopper body", "medium"),
    ("cutile", "flash_decode", "fp32", "cross_arch_runtime_loop",
     ["lat", "total:smsp__inst_executed.sum", "dom:smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
     "cuTile flash_decode latency at the profiled case is similar on both devices, while its instruction count and long-scoreboard ratio differ between them; the stall composition is architecture dependent",
     "different benchmark protocols and cuda-tile versions", "medium"),
    ("tilelang", "1d_conv", "fp16", "cross_arch_gather_mapping",
     ["lat", "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum/l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
      "dom:sm__warps_active.avg.pct_of_peak_sustained_active", "path"],
     "TileLang convolution keeps the uncoalesced gather signature (sectors/request well above Triton) on both devices; the one-resident-CTA occupancy pattern appears only in the B200 tcgen05 body",
     "Blackwell TMEM body vs Hopper fragment body", "medium"),
]


def confounders_for(dsl, op, b, g):
    c = ["architecture sm_100 vs sm_90",
         "benchmark protocol: B200 frozen columns (20/100 config, 64 MB eviction for 41 ops) vs GH200 1/3 with 120 MiB eviction",
         "NCU 2026.1.1 (B200) vs 2025.4.0 (GH200)"]
    if dsl == "cutile":
        c.append("cuda-tile: B200 paper campaign 1.3.0 per developer_guide, B200 profile capture version unrecorded; GH200 1.5.0 + tileiras 13.4.92")
    if dsl == "tilelang":
        c.append("B200 TileLang profiled in a separate environment (HF PR #2) and benchmarked in a separate campaign")
        if op in TL_TMEM_OPS:
            c.append("TileLang uses the TMEM body on B200 and a fragment-accumulator body on GH200 (supports_tmem dispatch)")
    if dsl == "triton" and op in TRITON_HOST_BT:
        c.append("Triton pre-transposes B on the host and caches it (_bt_cache): outside the timed region and the NCU range")
    if b and g:
        if b.get("winner_config_json") != g.get("winner_config_json"):
            c.append("autotuned winner differs")
        if b.get("kernel_names") != g.get("kernel_names"):
            c.append("kernel set/names differ")
        if b.get("collection_level") != g.get("collection_level"):
            c.append(f"collection level differs ({b.get('collection_level')} vs {g.get('collection_level')})")
    return c


PAIR_FAMILIES = ["integer_address", "predicate_select", "floating_point", "conversion_packing", "global_load_store",
                 "async_copy_ldgsts", "shared_memory", "local_memory", "tma", "legacy_mma", "wgmma", "tcgen05",
                 "synchronization", "atomic", "control_flow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--inventory-dir", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    D = {d: Dev(root, d) for d in DEVS}

    # ---------------- environment.json
    for dev in DEVS:
        inv = json.load(open(Path(a.inventory_dir) / f"inventory_{dev}.json"))
        P = list(D[dev].prof.values())
        env = {
            "device": dev, "architecture": C.DEVICES[dev]["architecture"], "compute_capability": C.DEVICES[dev]["compute_capability"],
            "hf_dataset": {"repo_id": inv["hf_repo"], "repo_type": "dataset", "revision": inv["hf_revision"],
                           "folder": C.DEVICES[dev]["hf_folder"], "n_reports": len(inv["reports"]),
                           "n_sha256_match": sum(r["hf_match"] for r in inv["reports"])},
            "report_sessions": {
                "ncu_versions": dict(collections.Counter(p["ncu_version"] for p in P)),
                "profiler_cuda_versions": dict(collections.Counter(p["profiler_cuda_version"] for p in P)),
                "report_created_range": [min(p["report_created"] for p in P), max(p["report_created"] for p in P)],
                "collection_levels": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in collections.Counter((p["dsl"], p["collection_level"], p["replay_mode"]) for p in P).items()},
            },
            **SOFTWARE[dev],
            "repo_head_used_for_configs": C.git_rev(a.repo, "HEAD"),
            "archive_refs": {r: C.git_rev(a.repo, r) for r in ("origin/archive/raw-logs-2026-09-18", "origin/archive/tilebenchpp-2026-10")},
            "notes": ["NCU kernel durations are profiler measurements; formal latency comes only from benchmark_cases.csv (dsl_ms)",
                      "B200 and GH200 counters are not normalized by any architecture conversion factor"],
        }
        json.dump(env, open(root / dev / "environment.json", "w"), indent=1)

    # ---------------- cross_device_pairs.csv
    cols = ["operator", "category", "dtype", "dsl", "case_id_B200", "case_id_GH200", "case_match", "params_json",
            "winner_B200", "winner_GH200", "winner_equal", "ncu_version_B200", "ncu_version_GH200",
            "collection_B200", "collection_GH200", "launch_count_B200", "launch_count_GH200", "kernel_names_B200", "kernel_names_GH200",
            "matrix_path_B200", "matrix_path_GH200", "access_path_B200", "access_path_GH200",
            "dsl_ms_B200", "dsl_ms_GH200", "latency_ratio_GH200_over_B200",
            "inst_executed_B200", "inst_executed_GH200"] + \
           [f"{f}_{d}" for f in PAIR_FAMILIES for d in DEVS] + \
           ["dram_bytes_read_B200", "dram_bytes_read_GH200", "regs_dominant_B200", "regs_dominant_GH200",
            "smem_per_block_dominant_B200", "smem_per_block_dominant_GH200", "achieved_occ_dominant_B200", "achieved_occ_dominant_GH200",
            "missing_metrics", "confounders"]
    pairs = []
    for pid_b, b in sorted(D["B200"].prof.items()):
        dsl, op, dt = b["dsl"], b["operator"], b["dtype"]
        pid_g = D["GH200"].pid(dsl, op, dt)
        g = D["GH200"].prof.get(pid_g)
        if g is None:
            continue
        r = {"operator": op, "category": C.CATEGORY[op], "dtype": dt, "dsl": dsl, "case_id_B200": b["case_id"], "case_id_GH200": g["case_id"],
             "case_match": b["case_id"] == g["case_id"], "params_json": b["params_json"],
             "winner_B200": b["winner_config_json"], "winner_GH200": g["winner_config_json"],
             "winner_equal": b["winner_config_json"] == g["winner_config_json"],
             "ncu_version_B200": b["ncu_version"], "ncu_version_GH200": g["ncu_version"],
             "collection_B200": f"{b['collection_level']}/{b['replay_mode']}", "collection_GH200": f"{g['collection_level']}/{g['replay_mode']}",
             "launch_count_B200": b["launch_count"], "launch_count_GH200": g["launch_count"],
             "kernel_names_B200": b["kernel_names"], "kernel_names_GH200": g["kernel_names"],
             "matrix_path_B200": D["B200"].matrix_path(pid_b), "matrix_path_GH200": D["GH200"].matrix_path(pid_g),
             "access_path_B200": D["B200"].access_path(pid_b), "access_path_GH200": D["GH200"].access_path(pid_g)}
        lb, lg = D["B200"].lat(pid_b), D["GH200"].lat(pid_g)
        r.update({"dsl_ms_B200": b["benchmark_dsl_ms"], "dsl_ms_GH200": g["benchmark_dsl_ms"],
                  "latency_ratio_GH200_over_B200": f"{lg / lb:.6g}" if lb and lg else ""})
        missing = []
        for dev, pid in (("B200", pid_b), ("GH200", pid_g)):
            d = D[dev]
            v = d.total(pid, "smsp__inst_executed.sum")
            r[f"inst_executed_{dev}"] = "" if v is None else int(v)
            fams_avail = d.fam(pid, "integer_address") is not None
            if not fams_avail:
                missing.append(f"{dev}:dynamic_opcode_counts")
            for f in PAIR_FAMILIES:
                v = d.fam(pid, f)
                r[f"{f}_{dev}"] = "" if v is None else v
            v = d.total(pid, "dram__bytes_read.sum")
            r[f"dram_bytes_read_{dev}"] = "" if v is None else int(v)
            if v is None:
                missing.append(f"{dev}:dram__bytes_read.sum")
            for key, m in (("regs_dominant", "launch__registers_per_thread"), ("smem_per_block_dominant", "launch__shared_mem_per_block"),
                           ("achieved_occ_dominant", "sm__warps_active.avg.pct_of_peak_sustained_active")):
                v = d.dom(pid, m)
                r[f"{key}_{dev}"] = "" if v is None else v
                if v is None:
                    missing.append(f"{dev}:{m}")
        r["missing_metrics"] = ";".join(missing)
        r["confounders"] = " | ".join(confounders_for(dsl, op, b, g))
        pairs.append(r)
    with open(root / "cross_device_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(pairs)

    # ---------------- diagnosis_evidence.csv
    dcols = ["device", "dsl", "operator", "dtype", "comparison_scope", "mechanism_id", "observation", "mechanism_hypothesis",
             "supporting_metric_names", "supporting_profile_ids", "alternative_explanation", "confounders", "evidence_quality",
             "review_status", "source_paths"]
    per_dev = {d: [] for d in DEVS}
    for s in SPECS:
        dev = s["device"]
        parts = []
        for t in s["terms"]:
            vals = ", ".join(f"{dsl}={g4(evaluate(D, t, dev, dsl, s['operator'], s['dtype']))}" for dsl in s["dsls"])
            parts.append(f"{metric_name(t)}: {vals}")
        pids = [D[dev].pid(d, s["operator"], s["dtype"]) for d in s["dsls"]]
        per_dev[dev].append({
            "device": dev, "dsl": "|".join(s["dsls"]), "operator": s["operator"], "dtype": s["dtype"],
            "comparison_scope": f"cross_dsl:{dev}:" + "_vs_".join(s["dsls"]), "mechanism_id": s["mechanism_id"],
            "observation": " ; ".join(parts) + " (latency = formal CSV dsl_ms at the profiled case, ms)",
            "mechanism_hypothesis": s["hypothesis"], "supporting_metric_names": ";".join(metric_name(t) for t in s["terms"]),
            "supporting_profile_ids": ";".join(pids), "alternative_explanation": s["alternative"], "confounders": s["confounders"],
            "evidence_quality": s["quality"], "review_status": "machine_generated_observation; interpretation pending author review",
            "source_paths": s["sources"]})
    for dsl, op, dt, mid, terms, hyp, alt, q in XSPECS:
        parts = []
        for t in terms:
            vals = ", ".join(f"{dev}={g4(evaluate(D, t, dev, dsl, op, dt))}" for dev in DEVS)
            parts.append(f"{metric_name(t)}: {vals}")
        b, g = D["B200"].prof.get(D["B200"].pid(dsl, op, dt)), D["GH200"].prof.get(D["GH200"].pid(dsl, op, dt))
        for dev in DEVS:
            per_dev[dev].append({
                "device": dev, "dsl": dsl, "operator": op, "dtype": dt, "comparison_scope": f"cross_device:{dsl}:B200_vs_GH200",
                "mechanism_id": mid, "observation": " ; ".join(parts) + " (latency = formal CSV dsl_ms at the profiled case, ms)",
                "mechanism_hypothesis": hyp, "supporting_metric_names": ";".join(metric_name(t) for t in terms),
                "supporting_profile_ids": ";".join(D[d].pid(dsl, op, dt) for d in DEVS), "alternative_explanation": alt,
                "confounders": " | ".join(confounders_for(dsl, op, b, g)), "evidence_quality": q,
                "review_status": "machine_generated_observation; interpretation pending author review",
                "source_paths": "cross_device_pairs.csv"})
    for dev in DEVS:
        with open(root / dev / "diagnosis_evidence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=dcols, lineterminator="\n")
            w.writeheader()
            w.writerows(per_dev[dev])
    print("pairs", len(pairs), {d: len(v) for d, v in per_dev.items()})


if __name__ == "__main__":
    main()
