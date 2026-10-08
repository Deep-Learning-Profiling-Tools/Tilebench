"""Shared constants and helpers for the MI300X paper-figure extraction and validation.

Nothing here runs a profiler, a benchmark or a GPU kernel: every function reads existing files.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "paper_figures" / "amd" / "MI300X"
DIAGNOSIS_JSON = REPO / "scripts" / "paper_figures" / "data" / "mi300x_diagnosis_2026-10-05.json"

DEVICE = "MI300X"
ARCH = "gfx942"
N_CU = 304

# Workload categories exactly as in the TileBench paper operator table
# (bowencui123/TileBench_EMNLP_26, table_tex/benchmark_list.tex at commit 6a661c9, column "Category").
CATEGORY_SOURCE = "TileBench paper table_tex/benchmark_list.tex (TileBench_EMNLP_26 @ 6a661c9), column Category"
CATEGORY = {
    "flash_attention": "Matrix Mult./Attn.", "flash_decode": "Matrix Mult./Attn.", "block_sparse_attention": "Matrix Mult./Attn.",
    "softmax": "Reduction/Norm.", "rope": "Point-wise", "matmul_fp32_fp16_fp8": "Matrix Mult./Attn.",
    "matmul_int8": "Matrix Mult./Attn.", "streamk_matmul": "Matrix Mult./Attn.", "layernorm": "Reduction/Norm.",
    "rmsnorm": "Reduction/Norm.", "vector_add": "Point-wise", "mul2": "Point-wise", "relu": "Point-wise",
    "destindex": "Data Layout", "quantize_global": "Point-wise", "dequantize_rowwise": "Point-wise",
    "cross_entropy": "Reduction/Norm.", "matrix_transpose": "Data Layout", "dropout": "Point-wise", "swiglu": "Point-wise",
    "kl_divergence": "Reduction/Norm.", "fused_activation": "Point-wise", "mean_reduction": "Reduction/Norm.",
    "argmax": "Reduction/Norm.", "l2_norm": "Reduction/Norm.", "2d_conv": "Stencil/Conv.", "top_k_selection": "Data Layout",
    "histogramming": "Reduction/Norm.", "linear_self_attention": "Matrix Mult./Attn.", "jacobi_stencil_2d": "Stencil/Conv.",
    "bitonic_sort": "Data Layout", "radix_sort": "Data Layout", "batched_matmul": "Matrix Mult./Attn.",
    "batch_normalization": "Reduction/Norm.", "2d_max_pooling": "Stencil/Conv.", "gaussian_blur": "Stencil/Conv.",
    "1d_conv": "Stencil/Conv.", "3d_conv": "Stencil/Conv.", "matrix_copy": "Data Layout", "reverse_array": "Data Layout",
    "interleave": "Data Layout", "weight_dequant": "Point-wise", "moe_topk_gating": "Reduction/Norm.",
    "leaky_relu": "Point-wise", "sigmoid": "Point-wise",
}

EXPECTED_OPERATORS = 45
EXPECTED_PAIRS = 109
KNOWN_EXCLUSION = {"operator": "matmul_fp32_fp16_fp8", "dtype": "fp8_e4m3fn",
                   "reason": "FP8 E4M3FN is UNSUPPORTED_DTYPE in the MI300X formal run (gfx942 FP8 is E4M3FNUZ; the "
                             "PyTorch reference torch._scaled_mm rejects e4m3fn on gfx942). No formal CSV row, no autotune "
                             "winner, no profile. Not replaced by another dtype."}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(d) -> str:
    """Same algorithm as outputs/profiling/MI300X/rocprof_compute_sweep/report_manifest.json `tree_sha256`:
    sha256 over the sorted lines '<relpath>\\t<sha256(file)>' joined by '\\n'."""
    rows = []
    for dp, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(dp, f)
            rows.append(f"{os.path.relpath(p, d)}\t{sha256_file(p)}")
    rows.sort()
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def write_csv(path: Path, header, rows, gz: bool = False):
    """Deterministic CSV (\\n line ends, QUOTE_MINIMAL); gzip with mtime=0 and no embedded file name."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        if isinstance(r, dict):
            r = [r.get(h, "") for h in header]
        w.writerow(["" if v is None else v for v in r])
    data = buf.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as g:
                g.write(data)
    else:
        path.write_bytes(data)


def read_csv(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def num(v):
    """repr-exact text for a number computed or copied in this extraction (no rounding)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(v)
    return str(v)


# ---------------------------------------------------------------- AMDGCN static classification
# Family names follow the AMD instruction classes; first match wins.
ISA_FAMILIES = [
    ("MFMA", r"^v_mfma"), ("MFMA_ACCVGPR_MOVE", r"^v_accvgpr"),
    ("SCRATCH", r"^scratch_"),
    ("VMEM_LOAD", r"^(buffer|global)_load"), ("VMEM_STORE", r"^(buffer|global)_store"),
    ("VMEM_ATOMIC", r"^(buffer|global)_atomic"), ("FLAT", r"^flat_"),
    ("LDS_PERMUTE", r"^ds_(bpermute|permute|swizzle)"), ("LDS_READ", r"^ds_read"), ("LDS_WRITE", r"^ds_write"),
    ("LDS_ATOMIC_OTHER", r"^ds_"),
    ("SMEM", r"^s_(load|buffer_load|store|dcache)"), ("WAITCNT", r"^s_waitcnt"), ("BARRIER", r"^s_barrier"),
    ("BRANCH", r"^s_(cbranch|branch|setpc|swappc|getpc)"), ("SALU", r"^s_"),
    ("VALU_CONVERT", r"^v_cvt"), ("VALU_TRANSCENDENTAL", r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_"),
    ("VALU_INT_MUL", r"^v_(mul_hi|mul_lo|mad_u64|mad_i64|mul_u32|mul_i32|mad_u32|mad_i32)"),
    ("VALU_CMP_SELECT", r"^v_(cmp|cndmask)"),
    ("VALU_LANE", r"^v_(readfirstlane|readlane|writelane)"),
    ("VALU", r"^v_"),
]
_FAM_RX = [(n, re.compile(r)) for n, r in ISA_FAMILIES]
MEM_MODIFIERS = ("sc0", "sc1", "nt", "glc", "slc", "dlc")
_OPC_RX = re.compile(r"^(?:[sv]_|(?:global|buffer|ds|flat|scratch)_)")


def isa_family(opcode: str) -> str:
    for n, rx in _FAM_RX:
        if rx.match(opcode):
            return n
    return "OTHER"


def access_width_bits(opcode: str):
    m = re.match(r"^(?:buffer|global|flat|scratch)_(?:load|store|atomic)_(\w+)", opcode)
    if not m:
        return None
    w = m.group(1)
    for suffix, bits in (("dwordx4", 128), ("dwordx3", 96), ("dwordx2", 64), ("dword", 32),
                         ("ushort", 16), ("sshort", 16), ("short", 16), ("ubyte", 8), ("sbyte", 8), ("byte", 8)):
        if w.startswith(suffix) or w.endswith(suffix):
            return bits
    return None


def parse_amdgcn(text: str):
    """Static instruction list of the first kernel function in a Triton .amdgcn dump, plus metadata."""
    md = {}
    for k, rx in [("NumVgprs", r"; NumVgprs: (\d+)"), ("NumAgprs", r"; NumAgprs: (\d+)"),
                  ("TotalNumVgprs", r"; TotalNumVgprs: (\d+)"), ("TotalNumSgprs", r"; TotalNumSgprs: (\d+)"),
                  ("ScratchSize", r"; ScratchSize: (\d+)"), ("Occupancy", r"; Occupancy: (\d+)"),
                  ("codeLenInByte", r"; codeLenInByte = (\d+)"), ("LDSByteSize", r"; LDSByteSize: (\d+)"),
                  (".vgpr_spill_count", r"\.vgpr_spill_count:\s+(\d+)"), (".sgpr_spill_count", r"\.sgpr_spill_count:\s+(\d+)")]:
        m = re.search(rx, text)
        md[k] = int(m.group(1)) if m else None
    body = text.split(".Lfunc_end")[0]
    instrs = []
    for ln in body.splitlines():
        s = ln.split(";")[0].strip()
        if not s or s.startswith((".", "//")) or s.endswith(":"):
            continue
        op = s.split()[0]
        if not _OPC_RX.match(op):
            continue
        toks = set(re.split(r"[\s,]+", s)[1:])
        mods = tuple(m for m in MEM_MODIFIERS if m in toks)
        dpp = ("dpp" in s) or ("row_" in s) or ("quad_perm" in s)
        instrs.append({"opcode": op, "mods": mods, "dpp": dpp, "text": s})
    return md, instrs
