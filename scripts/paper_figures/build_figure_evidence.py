"""Select device-native profiling evidence for the figures and write it to the combined layer.

Inputs: the normalized device packages only (nvidia/{B200,GH200}, amd/MI300X) and combined/ tables; no raw report.
Outputs (artifacts/paper_figures/combined/):
  figure_evidence.csv        one row per evidence value: id, figure, case, device, DSL, metric, value, unit,
                             measurement_kind (dynamic counter / static SASS|ISA / launch config / PC sampling /
                             hardware counter / diagnostic experiment), scope, profile id, source, confidence, note
  execution_path_matrix.csv  categorical execution-path classes per representative operator/dtype x device/DSL
  rq2_case_selection.json    machine-readable selection of the Figure-3 case studies with justification
Usage: PYTHONPATH=.:scripts/paper_figures python scripts/paper_figures/build_figure_evidence.py --repo .
"""
import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nvidia_evidence import Dev  # noqa: E402  (NVIDIA package lookups)

csv.field_size_limit(1 << 30)

RQ2_CASES = [
    {"case": "matmul_fp32_fp16_fp8/fp32", "operator": "matmul_fp32_fp16_fp8", "dtype": "fp32", "dsls": ["triton", "cutile"],
     "mechanism": "matrix operand delivery",
     "why": "the cuTile/Triton ranking reverses between B200 and GH200; dynamic TMA bytes and shared-memory store counts differ by device; "
            "MI300X lowers the same TensorDescriptor source to 32-bit pointer loads (static ISA) with a diagnostic pointer-load variant",
     "caveats": ["Triton reads a transposed B copy prepared outside the timed region; cuTile loads B as [K, N]",
                 "cuTile winner tile differs (256x256x64 on B200, 128x128x32 on GH200) and cuda-tile differs (1.3.0 vs 1.5.0)"]},
    {"case": "destindex/int8", "operator": "destindex", "dtype": "int8", "dsls": ["triton", "cutile"],
     "mechanism": "indexing overhead",
     "why": "identical store sectors but 16x more (byte-wide) store instructions and ~17x more instructions for cuTile on both NVIDIA devices; "
            "Triton keeps 128-bit stores on NVIDIA but compiles to per-lane byte stores on MI300X (static ISA)",
     "caveats": ["the PyTorch baseline (ATen index_copy) differs between vendors"]},
    {"case": "1d_conv/fp16", "operator": "1d_conv", "dtype": "fp16", "dsls": ["triton", "tilelang"],
     "mechanism": "memory access and latency hiding",
     "why": "TileLang touches 7-10x more L1 load sectors than Triton with the same 16-bit load width and DRAM traffic, at lower occupancy and issue "
            "activity; on MI300X Triton's speedup rises because the PyTorch baseline runs MIOpen implicit GEMM plus layout transposes",
     "caveats": ["TileLang uses a different kernel body on Hopper", "the MI300X change is a baseline (PyTorch path) effect"]},
]
NOT_SELECTED = {
    "weight_dequant/bf16": "GH200 Triton instruction inflation is attributable to a different autotuned winner (config effect, not architecture); kept for Figure A5 as a confounder example",
    "block_sparse_attention/fp16": "moved to the appendix (A3 execution paths); evidence kept as 'supporting' rows",
    "flash_decode/fp32": "moved to the appendix (A5 instruction expansion); evidence kept as 'supporting' rows",
    "vector_add/fp32": "moved to the appendix (A5 MI300X load policy); evidence kept as 'supporting' rows",
}
SUPPORTING = ("block_sparse_attention/fp16", "flash_decode/fp32", "vector_add/fp32")

A3_ROWS = [("matmul_fp32_fp16_fp8", "fp32"), ("matmul_fp32_fp16_fp8", "fp16"), ("matmul_fp32_fp16_fp8", "fp8_e4m3fn"),
           ("matmul_int8", "int8"), ("batched_matmul", "fp32"), ("streamk_matmul", "fp16"), ("flash_attention", "fp16"),
           ("block_sparse_attention", "fp16"), ("linear_self_attention", "fp32"), ("1d_conv", "fp16"), ("3d_conv", "fp32"),
           ("destindex", "int8"), ("histogramming", "int32"), ("2d_max_pooling", "fp32"), ("vector_add", "fp32")]
COLS = [("B200", "triton"), ("B200", "cutile"), ("B200", "tilelang"), ("GH200", "triton"), ("GH200", "cutile"),
        ("GH200", "tilelang"), ("MI300X", "triton")]


def read(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


class AMD:
    def __init__(self, root):
        d = root / "amd" / "MI300X"
        self.paths = defaultdict(list)
        for r in read(d / "execution_paths.csv"):
            self.paths[r["profile_id"]].append(r)
        self.m = defaultdict(dict)
        for r in read(d / "kernel_metrics_long.csv.gz"):
            if r["status"] in ("collected", "derived") and r["value"] != "":
                self.m[r["profile_id"]].setdefault(r["raw_metric_name"], (r["value"], r["unit"], r["scope"], r["counter_kind"]))
        self.diag = read(d / "diagnosis_evidence.csv")
        self.exp = read(d / "diagnostic_experiments.csv")

    def pid(self, op, dt):
        return f"MI300X.triton.{op}.{dt}.rocprof_compute"

    def metric(self, op, dt, name):
        """Exact raw-metric name first; a name ending in '*' is a prefix match."""
        m = self.m[self.pid(op, dt)]
        if not name.endswith("*"):
            return (name, m[name]) if name in m else (None, None)
        for k, v in m.items():
            if k.startswith(name[:-1]):
                return k, v
        return None, None


def nv_row(ev, fig, case, dev, dsl, metric, value, unit, kind, scope, pid, src, conf="high", note=""):
    ev.append({"evidence_id": f"{fig}:{case}:{dev}:{dsl}:{metric}", "figure": fig, "case": case, "device": dev, "dsl": dsl,
               "metric_name": metric, "value": "" if value is None else value, "unit": unit, "measurement_kind": kind, "scope": scope,
               "profile_id": pid, "source_file": src, "confidence": conf, "note": note})


def build_evidence(root, NV, A):
    ev = []
    srcN = "artifacts/paper_figures/nvidia/{dev}/{f}"
    srcA = "artifacts/paper_figures/amd/MI300X/{f}"

    def nvm(fig, case, dev, dsl, op, dt, metric, kind="ncu_counter", mode="total", unit="", note=""):
        d = NV[dev]
        pid = d.pid(dsl, op, dt)
        if metric.startswith("fam:"):
            v = d.fam(pid, metric[4:])
            name, k, f = f"sass__inst_executed_per_opcode[family={metric[4:]}]", "dynamic_sass_count", "instruction_mix.csv"
        elif metric.startswith("opc:"):
            v = d.opc(pid, metric[4:])
            name, k, f = f"sass__inst_executed_per_opcode_with_modifier_all[{metric[4:]}*]", "dynamic_sass_count", "instruction_mix.csv"
        elif metric.startswith("ratio:"):
            a, b = metric[6:].split("/")
            x, y = d.total(pid, a), d.total(pid, b)
            v = x / y if x is not None and y else None
            name, k, f = metric[6:], kind, "kernel_metrics_long.csv.gz"
        else:
            v = d.total(pid, metric) if mode == "total" else d.dom(pid, metric)
            name, k, f = metric, kind, "kernel_metrics_long.csv.gz"
        nv_row(ev, fig, case, dev, dsl, name, v, unit, k, "sum over the profiled run() launches" if mode == "total" else "longest launch",
               pid, srcN.format(dev=dev.replace("", ""), f=f).replace("{dev}", dev), "high" if v is not None else "missing", note)
        return v

    def mix(fig, case, dev, dsl, op, dt, prefix):
        """Dynamic warp-level opcode-with-modifier counts of one opcode (e.g. STG -> 'STG.E.128:128000'), as text."""
        d = NV[dev]
        pid = d.pid(dsl, op, dt)
        c = defaultdict(int)
        for r in d.mix.get(pid, []):
            n = r["instruction_name"]
            if r["count_kind"] == "dynamic_warp_inst_executed_with_modifier" and (n == prefix or n.startswith(prefix + ".")):
                c[n] += int(float(r["count"]))
        nv_row(ev, fig, case, dev, dsl, f"opcode_mix[{prefix}]", ";".join(f"{k}:{v}" for k, v in sorted(c.items())) or None, "text",
               "dynamic_sass_count", "sum over the profiled run() launches", pid, srcN.format(dev=dev, f="instruction_mix.csv"),
               "high" if c else "missing")

    # ---------------- Figure 3 / A5 NVIDIA evidence
    for dev in ("B200", "GH200"):
        F = "rq2"
        c = "matmul_fp32_fp16_fp8/fp32"
        for dsl in ("triton", "cutile"):
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum", unit="byte")
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "opc:STS")
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "opc:LDSM")
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "fam:wgmma")
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "fam:tcgen05")
            nvm(F, c, dev, dsl, "matmul_fp32_fp16_fp8", "fp32", "launch__shared_mem_per_block", mode="dom", unit="byte/block", kind="launch_config")
        c = "destindex/int8"
        for dsl in ("triton", "cutile", "tilelang"):
            nvm(F, c, dev, dsl, "destindex", "int8", "smsp__inst_executed.sum", unit="warp inst")
            nvm(F, c, dev, dsl, "destindex", "int8", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", unit="requests")
            nvm(F, c, dev, dsl, "destindex", "int8", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", unit="sectors")
            nvm(F, c, dev, dsl, "destindex", "int8", "opc:STG")
            mix(F, c, dev, dsl, "destindex", "int8", "STG")
        c = "1d_conv/fp16"
        for dsl in ("triton", "tilelang", "cutile"):
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", unit="requests")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", unit="sectors")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "dram__bytes_read.sum", unit="byte")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "l1tex__t_sector_hit_rate.pct", mode="dom", unit="%")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "sm__warps_active.avg.pct_of_peak_sustained_active", mode="dom", unit="%")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "sm__maximum_warps_per_active_cycle_pct", mode="dom", unit="%", kind="launch_config")
            nvm(F, c, dev, dsl, "1d_conv", "fp16", "smsp__issue_active.avg.pct_of_peak_sustained_active", mode="dom", unit="%")
            mix(F, c, dev, dsl, "1d_conv", "fp16", "LDG")
        F = "supporting"
        c = "block_sparse_attention/fp16"
        for dsl in ("triton", "cutile"):
            nvm(F, c, dev, dsl, "block_sparse_attention", "fp16", "launch__shared_mem_per_block", mode="dom", unit="byte/block", kind="launch_config")
            nvm(F, c, dev, dsl, "block_sparse_attention", "fp16", "smsp__inst_executed.sum")
            nvm(F, c, dev, dsl, "block_sparse_attention", "fp16", "fam:legacy_mma")
        c = "flash_decode/fp32"
        for dsl in ("triton", "cutile"):
            nvm(F, c, dev, dsl, "flash_decode", "fp32", "smsp__inst_executed.sum", unit="warp inst")
            nvm(F, c, dev, dsl, "flash_decode", "fp32", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", mode="dom",
                unit="cycles/issue", kind="ncu_stall_ratio")
            nvm(F, c, dev, dsl, "flash_decode", "fp32", "launch__grid_size", mode="dom", unit="CTAs", kind="launch_config")
        c = "vector_add/fp32"
        nvm(F, c, dev, "triton", "vector_add", "fp32", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", mode="dom", unit="%")
        nvm(F, c, dev, "triton", "vector_add", "fp32", "ratio:l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum/l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            unit="sectors/request")
        # ---------------- A5 extra NVIDIA evidence (within-device paired counters)
        F = "a5"
        for op, dt in (("weight_dequant", "bf16"), ("destindex", "int8"), ("cross_entropy", "fp16"), ("moe_topk_gating", "fp16")):
            for dsl in ("triton", "cutile", "tilelang"):
                nvm(F, f"{op}/{dt}", dev, dsl, op, dt, "smsp__inst_executed.sum", unit="warp inst")
                nvm(F, f"{op}/{dt}", dev, dsl, op, dt, "sass__inst_executed_global_loads", unit="warp inst")
        for dt in ("fp32", "fp16", "fp8_e4m3fn"):
            for dsl in ("triton", "cutile", "tilelang"):
                nvm(F, f"matmul_fp32_fp16_fp8/{dt}", dev, dsl, "matmul_fp32_fp16_fp8", dt, "opc:STS")
                nvm(F, f"matmul_fp32_fp16_fp8/{dt}", dev, dsl, "matmul_fp32_fp16_fp8", dt, "fam:wgmma")
                nvm(F, f"matmul_fp32_fp16_fp8/{dt}", dev, dsl, "matmul_fp32_fp16_fp8", dt, "fam:tcgen05")
                nvm(F, f"matmul_fp32_fp16_fp8/{dt}", dev, dsl, "matmul_fp32_fp16_fp8", dt, "fam:legacy_mma")
                nvm(F, f"matmul_fp32_fp16_fp8/{dt}", dev, dsl, "matmul_fp32_fp16_fp8", dt,
                    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum", unit="byte")
        for dsl in ("triton", "cutile", "tilelang"):
            nvm(F, "flash_decode/fp32", dev, dsl, "flash_decode", "fp32", "smsp__inst_executed.sum", unit="warp inst")
            nvm(F, "flash_decode/fp32", dev, dsl, "flash_decode", "fp32", "launch__grid_size", mode="dom", unit="CTAs", kind="launch_config")
        for dsl in ("triton", "cutile", "tilelang"):
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", unit="requests")
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", unit="sectors")
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "dram__bytes_read.sum", unit="byte")
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "smsp__issue_active.avg.pct_of_peak_sustained_active", mode="dom", unit="%")
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "sm__warps_active.avg.pct_of_peak_sustained_active", mode="dom", unit="%")
            nvm(F, "1d_conv/fp16", dev, dsl, "1d_conv", "fp16", "sm__maximum_warps_per_active_cycle_pct", mode="dom", unit="%", kind="launch_config")

    # ---------------- MI300X native evidence
    def amd_metric(fig, case, op, dt, prefix, note=""):
        k, v = A.metric(op, dt, prefix)
        nv_row(ev, fig, case, "MI300X", "triton", k or prefix, v[0] if v else None, v[1] if v else "",
               (v[3] if v else "missing"), v[2] if v else "", A.pid(op, dt), srcA.format(f="kernel_metrics_long.csv.gz"),
               "high" if v else "missing", note)
        return float(v[0]) if v else None

    for fig in ("rq2", "a5"):
        amd_metric(fig, "matmul_fp32_fp16_fp8/fp32", "matmul_fp32_fp16_fp8", "fp32", "2.1.10 | MFMA Utilization | Avg*")
        amd_metric(fig, "1d_conv/fp16", "1d_conv", "fp16", "SQ_INSTS_VALU_INT32")
        amd_metric(fig, "1d_conv/fp16", "1d_conv", "fp16", "SQ_INSTS_VALU")
        amd_metric(fig, "destindex/int8", "destindex", "int8", "2.1.15 | Wavefront Occupancy | Avg*")
    amd_metric("supporting", "block_sparse_attention/fp16", "block_sparse_attention", "fp16", "2.1.10 | MFMA Utilization | Avg*")
    amd_metric("supporting", "block_sparse_attention/fp16", "block_sparse_attention", "fp16", "7.1.9 | Scratch Allocation | Avg*")
    amd_metric("supporting", "block_sparse_attention/fp16", "block_sparse_attention", "fp16", "dispatch.Arch_VGPR")
    amd_metric("a5", "vector_add/fp32", "vector_add", "fp32", "17.1.5 | HBM Bandwidth | Avg*")
    # static ISA facts from execution_paths (text, static)
    for op, dt in (("matmul_fp32_fp16_fp8", "fp32"), ("block_sparse_attention", "fp16"), ("destindex", "int8"), ("flash_decode", "fp32"),
                   ("1d_conv", "fp16"), ("vector_add", "fp32")):
        fig = "supporting" if f"{op}/{dt}" in SUPPORTING else "rq2"
        for r in A.paths[A.pid(op, dt)]:
            nv_row(ev, fig, f"{op}/{dt}", "MI300X", "triton", f"execution_paths.access_path[{r['stage']}]", r["access_path"], "text",
                   "static_isa+source", "kernel binary", A.pid(op, dt), srcA.format(f="execution_paths.csv"), "high")
            nv_row(ev, fig, f"{op}/{dt}", "MI300X", "triton", f"execution_paths.resource_summary[{r['stage']}]", r["resource_summary"], "text",
                   "launch_record+static_isa", "kernel launch", A.pid(op, dt), srcA.format(f="execution_paths.csv"), "high")
    for op, dt in (("1d_conv", "fp16"), ("destindex", "int8"), ("matmul_fp32_fp16_fp8", "fp32")):
        pid = f"MI300X.torch.{op}.{dt}.kernel_trace"
        rows = A.paths.get(pid, [])
        nv_row(ev, "rq2", f"{op}/{dt}", "MI300X", "pytorch", "kernel_trace.kernels",
               ";".join(f"{r['stage']}={r['notes'].removeprefix('kernel ')[:60]}" for r in rows) or None, "text", "kernel_trace",
               "PyTorch reference run()", pid, srcA.format(f="execution_paths.csv"), "medium" if rows else "missing")
    # diagnostic experiments (not formal)
    for x in A.exp:
        if x["experiment_id"] in ("gemm_desc_vs_ptr_fp32", "gemm_desc_vs_ptr_fp16", "cache_modifier_ablation_fp32",
                                  "flush_protocol_variants_fp32") and x["latency_ms"]:
            nv_row(ev, "a5", f"{x['operator']}/{x['dtype']}", "MI300X", "triton" if "torch" not in x["variant"] else "pytorch",
                   f"diagnostic:{x['experiment_id']}:{x['variant']}", x["latency_ms"], "ms", "diagnostic_experiment",
                   x["measurement_protocol"][:200], "", srcA.format(f="diagnostic_experiments.csv"), "medium",
                   f"changed_factor={x['changed_factor']}; other_changes={x['other_configuration_changes']}; notes={x['notes']}")
    for r in A.diag:
        if r["operator"] not in {c["operator"] for c in RQ2_CASES} | {c.split("/")[0] for c in SUPPORTING}:
            continue
        for dt in r["dtype"].split(";"):              # MI300X records list several dtypes in one row
            nv_row(ev, "supporting" if f"{r['operator']}/{dt.strip()}" in SUPPORTING else "rq2", f"{r['operator']}/{dt.strip()}", "MI300X", "triton", f"diagnosis:{r['mechanism_id']}:{r.get('mechanism_role', '')}",
                   r["mechanism_hypothesis"], "text", "diagnosis_record", "operator-level diagnosis", r["supporting_profile_ids"][:120],
                   srcA.format(f="diagnosis_evidence.csv"), r.get("trace_check_status", ""), r["observation"][:300])
    return ev


# ---------------------------------------------------------------- execution-path classification
def nv_classes(dev, dsl, op, dt, NV):
    d = NV[dev]
    pid = d.pid(dsl, op, dt)
    rows = d.paths.get(pid, [])
    if not rows:
        return None
    mp = " ".join(r["matrix_path"] for r in rows)
    ap = " ".join(r["access_path"] for r in rows)
    buf = " ".join(r["buffer_location"] for r in rows)
    lay = " ".join(r["layout_operations"] for r in rows)
    atom = " ".join(r["atomic_path"] for r in rows)
    static = any(r["evidence_confidence"] != "high" for r in rows)
    matrix = "tcgen05" if "tcgen05[" in mp else "wgmma" if "wgmma[" in mp else "legacy_mma" if "legacy_mma[" in mp else "none"
    # operand LOAD paths observed in any stage (union); TMA only when a TMA load (UTMALDG) is present -- bulk-copy
    # stores (UBLKCP.G.S, UTMASTG) are not operand loads
    ops_ = [t for t, pat in (("TMA", r"UTMALDG"), ("cp.async", r"cp\.async\["), ("LDG", r"LDG\[")) if re.search(pat, ap)]
    operand = "+".join(ops_) or "unknown"
    vec = re.findall(r"LDG\[([^\]]*)\]", ap)
    flags = []
    if "TMEM" in buf:
        flags.append("TMEM")
    if "shared" in buf:
        flags.append("smem")
    if "local" in buf:
        flags.append("spill")
    # global atomics always; shared atomics only when executed at scale (tcgen05 kernels issue a few ATOMS for the
    # TMEM allocator bookkeeping, which is not an algorithmic atomic)
    if re.search(r"(ATOMG|REDG|RED)[.\w]*:\d", atom):
        flags.append("atom.global")
    n_atoms, n_ctas = 0, 0
    for r in rows:
        n_atoms += sum(int(x) for x in re.findall(r"ATOMS[.\w]*:(\d+)", r["atomic_path"]))
        g = re.search(r"launches=(\d+);grid=([\d.]+)", r["resource_summary"])
        if g:
            n_ctas += int(g.group(1)) * int(float(g.group(2)))
    if not static and n_ctas and n_atoms / n_ctas > 64:      # > 64 shared atomics per CTA: algorithmic, not TMEM bookkeeping
        flags.append("atom.shared")
    if re.search(r"LDSM:\d|STSM:\d", lay):
        flags.append("ldsm/stsm")
    return {"matrix_path": matrix, "operand_path": operand, "flags": ";".join(flags), "ldg_width_mix": ";".join(vec),
            "evidence_kind": "static_sass" if static else "dynamic_sass", "profile_id": pid, "raw_matrix_path": mp[:300]}


def amd_classes(op, dt, A):
    pid = A.pid(op, dt)
    rows = A.paths.get(pid, [])
    if not rows:
        return None
    mp = " ".join(r["matrix_path"] for r in rows)
    ap = " ".join(r["access_path"] for r in rows)
    buf = " ".join(r["buffer_location"] for r in rows)
    lay = " ".join(r["layout_operations"] for r in rows)
    atom = " ".join(r["atomic_path"] for r in rows)
    matrix = "mfma" if "MFMA:" in mp else "none" if "not_observed" in mp else "unknown"
    operand = "desc→ptr" if "TensorDescriptor in source" in ap else "pointer" if "pointer loads" in ap else "unknown"
    flags = []
    if "LDS-staged" in buf:
        flags.append("LDS")
    m = re.search(r"scratch (\d+) B", buf)
    if m and int(m.group(1)) > 0:
        flags.append("spill")
    if re.search(r"convert_layout x[1-9]", lay):
        flags.append("convert_layout")
    if re.search(r"(atomic|ATOM)[^;]*[1-9]", atom):
        flags.append("atomic")
    return {"matrix_path": matrix, "operand_path": operand, "flags": ";".join(flags), "ldg_width_mix": "",
            "evidence_kind": "static_isa", "profile_id": pid, "raw_matrix_path": mp[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", help="output directory (default: <repo>/artifacts/paper_figures/combined)")
    a = ap.parse_args()
    root = Path(a.repo) / "artifacts" / "paper_figures"
    NV = {d: Dev(root / "nvidia", d) for d in ("B200", "GH200")}
    A = AMD(root)
    ev = build_evidence(root, NV, A)
    out = Path(a.out) if a.out else root / "combined"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "figure_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(ev)
    mat = []
    for op, dt in A3_ROWS:
        for dev, dsl in COLS:
            c = amd_classes(op, dt, A) if dev == "MI300X" else nv_classes(dev, dsl, op, dt, NV)
            if c is None:
                status = "unsupported" if (dev, op, dt) == ("MI300X", "matmul_fp32_fp16_fp8", "fp8_e4m3fn") else "no_profile"
                c = {"matrix_path": "n/a" if status == "unsupported" else "unknown", "operand_path": "", "flags": "", "ldg_width_mix": "",
                     "evidence_kind": status, "profile_id": "", "raw_matrix_path": ""}
            mat.append({"operator": op, "dtype": dt, "device": dev, "dsl": dsl, **c})
    with open(out / "execution_path_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mat[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(mat)
    json.dump({"figure": "fig_rq2_cross_device_diagnosis", "max_cases": 3, "selected": RQ2_CASES, "not_selected": NOT_SELECTED,
               "selection_rules": ["three distinct mechanisms (matrix operand delivery, indexing overhead, memory access and latency hiding), "
                                   "not the three largest speedup gaps",
                                   "formal latency only (benchmark_cases_normalized) at the SAME case_id_v2 that the profiles captured",
                                   "full (not reduced) NCU reports for every NVIDIA DSL shown, plus MI300X static ISA or kernel trace",
                                   "each case shows a change across devices (RQ2), with its confounders stated",
                                   "no cross-vendor numeric counter axis"]},
              open(out / "rq2_case_selection.json", "w"), indent=1)
    print("evidence rows", len(ev), "missing", sum(1 for e in ev if e["confidence"] == "missing"), "path cells", len(mat))


if __name__ == "__main__":
    main()
