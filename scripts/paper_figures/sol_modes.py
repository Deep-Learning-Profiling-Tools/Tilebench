"""Algorithm-aware SOL compute-mode manifest (one row per operator x dtype) and the empirical peaks it is scored against.

The reference compute mode of a task is fixed by its frozen canonical algorithm and numerical contract
(tilebench/llm/v2/contracts/data/<op>/contract.md, sections "Algorithm family and structure" and "Precision and
accumulation"), never by the input dtype alone, the accumulator or output dtype, the compiled ISA, or a profiler
report. The entries below were audited against those contracts, against the matrix-multiply primitives present in
impl_{triton,cutile,tilelang}.py (source level), and against the approved declaration
tilebench/llm/v2/manifests/arithmetic_modes.yaml (revision 2); a disagreement with that declaration fails the build.

Peaks come only from the PR #323 empirical profiles tilebench/data/peak_performance/empirical/<device>.json (read from
the merged commit, never from the legacy peak_performance/<device>.json).

  python scripts/paper_figures/sol_modes.py      -> artifacts/paper_figures/sol/sol_mode_manifest.{json,csv}
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "paper_figures" / "sol"
PR323_MERGE = "72d7cec623e6238643ed5d2099d9aa9889c4ea87"     # PR #323 merge commit on main
CONTRACT_COMMIT = "72d7cec623e6238643ed5d2099d9aa9889c4ea87"  # contracts + arithmetic_modes.yaml read from here
OPERATOR_TREE_COMMIT = "ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe"  # contracts' source_sha; operators tree == this worktree's
DEVICES = ("B200", "GH200", "MI300X")

# Workload classes required by the audit.
CLASSES = {
    "mma": "1 MMA-eligible matrix computation",
    "vector": "2 non-MMA vector / elementwise arithmetic",
    "reduction": "3 reduction / normalisation with fp32 or mixed-precision accumulation",
    "movement": "4 memory movement / indexing / dtype conversion (no arithmetic on the data)",
    "integer": "5 integer arithmetic without MMA",
    "selection": "6 sorting / scanning / selection (comparison work, no floating-point compute peak represents it)",
    "multistage": "7 multistage operator with more than one arithmetic mode",
}

# Algorithm-level compute modes -> PR #323 key per device. None = that device has no calibrated value for the mode.
MODE_KEYS = {
    "fp16_mma": {d: "peak_tflops.fp16_mma" for d in DEVICES},
    "bf16_mma": {d: "peak_tflops.bf16_mma" for d in DEVICES},
    # TF32-class: fp32 operands rounded/truncated to a 10-bit mantissa, fp32 accumulation (the contracts' class).
    # NVIDIA realises it as TF32, CDNA3 as XF32 (decision D1).
    "tf32_class_mma": {"B200": "peak_tflops.tf32_mma", "GH200": "peak_tflops.tf32_mma", "MI300X": "peak_tflops.xf32_mma"},
    # NVIDIA float8_e4m3fn. MI300X only has e4m3fnuz, a different format: never substituted (no MI300X rows exist).
    "fp8_e4m3fn_mma": {"B200": "peak_tflops.fp8_e4m3fn_mma", "GH200": "peak_tflops.fp8_e4m3fn_mma", "MI300X": None},
    "int8_mma": {d: "peak_tops.int8_mma" for d in DEVICES},
    "fp32_vector": {d: "peak_tflops.fp32_vector" for d in DEVICES},
    "fp16_vector": {d: "peak_tflops.fp16_vector" for d in DEVICES},      # measured as packed FP16x2 FMA
    "bf16_vector": {d: "peak_tflops.bf16_vector" for d in DEVICES},      # measured as packed BF16x2 FMA; absent on MI300X
    "memory_only": {d: None for d in DEVICES},
}
# arithmetic_modes.yaml (rev 2) internal names -> algorithm-level mode, for the consistency check only.
REV2 = {"mma_fp16_f32acc": "fp16_mma", "mma_bf16_f32acc": "bf16_mma", "mma_tf32_f32acc": "tf32_class_mma",
        "mma_fp8_e4m3_f32acc": "fp8_e4m3fn_mma", "mma_int8_i32acc": "int8_mma", "fp32_fma_vector": "fp32_vector",
        "fp16x2_fma_vector": "fp16_vector", "bf16x2_fma_vector": "bf16_vector",
        "memory_only": "memory_only", "no_compute_term": "memory_only"}

# Decisions taken during the audit (any status other than "approved" blocks the SOL build).
DECISIONS = {
    "D1_tf32_class_on_cdna3": {
        "status": "approved",
        "decided_by": "study owner, 2026-10-09: map to xf32_mma (architecture-specific realisation of the TF32 class, "
                      "documented as not bit-identical to NVIDIA TF32)",
        "question": "Score the TF32-class contract (fp32 operands at 10-bit mantissa, fp32 accumulation) on MI300X "
                    "against the calibrated XF32 matrix rate (peak_tflops.xf32_mma)?",
        "evidence": [
            "contracts (batched_matmul, matmul_fp32_fp16_fp8, streamk_matmul, 1d/2d/3d_conv, linear_self_attention) "
            "define the fp32 class as operands rounded to TF32 (10-bit mantissa) with fp32 accumulation; narrower "
            "operand formats are forbidden",
            "Triton 3.6 AMD backend (backends/amd/compiler.py): 'Enable XF32 (TF32) for CDNA3 GPUs' -- gfx942 accepts "
            "input_precision='tf32' and lowers it to v_mfma_f32_*_xf32",
            "CDNA3 XF32: fp32 operands with the mantissa cut to 10 bits (low 13 bits truncated), fp32 accumulation",
            "not bit-identical to NVIDIA TF32 (operand rounding and internal accumulation details differ; on NVIDIA "
            "the DSLs also differ: Triton feeds fp32 registers to the tf32 MMA, cuTile casts to ct.tfloat32)",
        ],
        "affects": "MI300X fp32 cases of batched_matmul, matmul_fp32_fp16_fp8, streamk_matmul, 1d_conv, 2d_conv, "
                   "3d_conv, linear_self_attention (140 cases; compute-bound for the GEMM/conv cases, so material)",
        "if_rejected": "these MI300X cases get status missing_calibration (MI300X has no TF32 mode)",
    },
    "D2_mi300x_bf16_vector": {
        "status": "approved",
        "decided_by": "study owner, 2026-10-09: conditional memory-dominance assumption, not a measured BF16 vector peak; "
                      "only for the identified 100 MI300X bf16 cases after per-case verification of the critical "
                      "throughput P* = F * BW / Q; status 'bf16_vector_peak_unavailable; conditional_memory_dominance'; no "
                      "BF16 vector peak is fabricated and no FP16/FP32/BF16-MMA rate is substituted; RQ1 aggregates are "
                      "reported with and without these cases (appendix)",
        "status_label": "bf16_vector_peak_unavailable; conditional_memory_dominance",
        "max_critical_throughput_tflops": 3.99,     # verified per case in sol_data.py and validate_plots.py
        "question": "MI300X has no calibrated bf16_vector (gfx942 has no packed BF16 FMA). Exclude the bf16 cases that "
                    "need it, or approve a memory-term-dominance rule?",
        "evidence": [
            "affected: leaky_relu, mul2, vector_add, weight_dequant, jacobi_stencil_2d at bf16 on MI300X (100 cases)",
            "for every one of them the compute term F/P would reach the memory term Q/BW only if P <= 3.99 TFLOP/s; "
            "MI300X fp16_vector is 145.92 and fp32_vector 130.24 TFLOP/s",
        ],
        "default": "missing_calibration: excluded from every aggregate and reported as missing (no substitution)",
        "alternative": "dominance rule: T_SOL = Q/BW, flagged, valid for any bf16 vector peak above P* (P* <= 3.99)",
    },
    "D3_memory_only_M2": {
        "status": "approved",
        "decided_by": "study owner, 2026-10-09: adopt the approved v2 decision M2 for the paper, including int8 mul2/"
                      "vector_add; the memory-only subgroup is reported beside the overall aggregate",
        "question": "Adopt the approved v2 decision M2 (memory-only bound T_SOL = Q/BW when F is an element, "
                    "comparison, conversion or integer count without a calibrated arithmetic unit) for the paper?",
        "evidence": [
            "M2 approved 2026-10-06 (arithmetic_modes.yaml), with the condition that these tasks are not described "
            "as a full compute+memory roofline and the memory-only subgroup is reported beside the overall aggregate",
            "no-arithmetic operators (copy, transpose, reverse, interleave, destindex, quantize_global conversion) and "
            "comparison/selection operators (relu, argmax, 2d_max_pooling, bitonic_sort, top_k_selection, "
            "radix_sort, histogramming) have no FMA-type work for any calibrated peak to represent",
            "integer arithmetic without MMA: mul2/int8 and vector_add/int8 (int32_vector is not calibrated on any "
            "device); their compute term would reach the memory term only below 3.42 TOP/s",
        ],
        "if_rejected": "the int8 mul2/vector_add cases get status missing_calibration; no-arithmetic and "
                       "comparison operators stay memory-only (they have no arithmetic term to model)",
    },
}

# Approved decisions of arithmetic_modes.yaml that the paper reuses unchanged (overrides of the frozen F/Q).
OVERRIDES = {
    "flash_attention": {"decision": "M3_flash_attention_causal_F", "F": "2 * batch_size * n_heads * head_dim * seq_len * (seq_len + 1)"},
    "radix_sort": {"decision": "M4_radix_sort_Q", "Q": "2 * n * dtype_size"},
}

# Per-operator audit. modes: dtype -> algorithm-level mode. decisions: dtype -> decision ids the row depends on.
OPS = {
    # ---- 1 MMA-eligible matrix computation
    "batched_matmul": dict(cls="mma", algo="BATCH independent tiled GEMMs, full-K local accumulator per output, tensor-core chunk products",
        prec="fp16/bf16 native products; fp32 operands TF32-class; fp32 accumulation",
        modes={"fp16": "fp16_mma", "bf16": "bf16_mma", "fp32": "tf32_class_mma"},
        why="contract: each K chunk is a tensor-core matrix multiply; fp32 inputs run at the TF32 class"),
    "matmul_fp32_fp16_fp8": dict(cls="mma", algo="blocked dense GEMM with fp32 accumulation",
        prec="fp32 operands TF32-class (10-bit mantissa); fp16 native; fp8_e4m3fn native without scales; fp32 accumulation",
        modes={"fp16": "fp16_mma", "fp32": "tf32_class_mma", "fp8_e4m3fn": "fp8_e4m3fn_mma"},
        why="MMA-eligible GEMM for every dtype; fp32 accumulation does not make it a vector workload"),
    "streamk_matmul": dict(cls="mma", algo="hybrid Stream-K GEMM (fp32 atomic fixup) plus data-parallel tiles",
        prec="fp32 operands rounded to TF32; fp16/bf16 native; fp32 accumulation",
        modes={"fp16": "fp16_mma", "bf16": "bf16_mma", "fp32": "tf32_class_mma"},
        why="tensor-core GEMM; the atomic fixup is fp32 vector traffic not counted in F"),
    "matmul_int8": dict(cls="mma", algo="blocked int8 GEMM with in-mainloop decoding of packed B",
        prec="exact int8 x int8 products, int32 accumulation",
        modes={"int8": "int8_mma"},
        why="int8 matrix product is MMA-eligible; F counts 2 integer ops per MAC (TOP/s); the decode is not in F"),
    "1d_conv": dict(cls="mma", algo="implicit-GEMM grouped convolution, tensor-core chunk products, no im2col",
        prec="fp16 native products; fp32 operands TF32-class; fp32 accumulation",
        modes={"fp16": "fp16_mma", "fp32": "tf32_class_mma"},
        why="contract stage 2 multiplies gathered patch/weight tiles with a matrix-multiply primitive"),
    "2d_conv": dict(cls="mma", algo="implicit-GEMM grouped convolution, tensor-core chunk products, no im2col",
        prec="fp16 native products; fp32 operands TF32-class; fp32 accumulation",
        modes={"fp16": "fp16_mma", "fp32": "tf32_class_mma"}, why="implicit GEMM (contract)"),
    "3d_conv": dict(cls="mma", algo="implicit-GEMM grouped convolution, tensor-core chunk products, no im2col",
        prec="fp16 native products; fp32 operands TF32-class; fp32 accumulation",
        modes={"fp16": "fp16_mma", "fp32": "tf32_class_mma"}, why="implicit GEMM (contract)"),
    # ---- 7 multistage
    "flash_attention": dict(cls="multistage", algo="causal online-softmax attention: QK^T and PV tile matmuls + fp32 softmax",
        prec="fp16 operands into both matmuls, fp32 accumulation; softmax statistics fp32; P rounded to fp16",
        modes={"fp16": "fp16_mma"},
        why="F counts only the two causal GEMMs (approved M3) at fp16_mma; exp/rescale work has no calibrated mode "
            "and is not modelled (T_SOL is a lower bound for it)"),
    "block_sparse_attention": dict(cls="multistage", algo="CSR block-sparse online-softmax attention with tensor-core QK^T and PV",
        prec="fp16 tensor-core products, fp32 accumulation and softmax",
        modes={"fp16": "fp16_mma"}, why="as flash_attention: GEMM FLOPs at fp16_mma; softmax not modelled"),
    "linear_self_attention": dict(cls="multistage", algo="linear attention: two dense reductions (S, numerator) + column sum Z and denominator",
        prec="S and numerator TF32-class operands, fp32 accumulation; Z and denominator full fp32; phi exp fp32",
        modes={"fp32": "tf32_class_mma"},
        why="frozen F = 4MD^2 (TF32-class GEMMs) + 3MD (fp32 vector) at the TF32-class rate; per-mode decomposition "
            "(+3MD/P_fp32_vector) changes T_SOL in 0 of 60 cases (all memory-bound), so the frozen model is kept"),
    # ---- 2 non-MMA vector, input-dtype arithmetic is the reference
    "leaky_relu": dict(cls="vector", algo="single-pass elementwise select(x, 0.01x)",
        prec="input dtype (fp32 with one cast accepted)", modes={"fp16": "fp16_vector", "bf16": "bf16_vector", "fp32": "fp32_vector"},
        why="non-MMA; contract allows input-dtype arithmetic, so the reference is the dtype's vector peak",
        decisions={"bf16": ["D2_mi300x_bf16_vector"]}),
    "mul2": dict(cls="vector", algo="flat elementwise doubling",
        prec="one op in the element's dtype (int8: exact integer doubling)",
        modes={"fp16": "fp16_vector", "bf16": "bf16_vector", "fp32": "fp32_vector", "int8": "memory_only"},
        why="non-MMA elementwise; int8 is integer arithmetic without a calibrated integer vector mode (M2)",
        cls_by_dtype={"int8": "integer"}, decisions={"bf16": ["D2_mi300x_bf16_vector"], "int8": ["D3_memory_only_M2"]}),
    "vector_add": dict(cls="vector", algo="elementwise binary add",
        prec="add in the input dtype (int8 exact)",
        modes={"fp16": "fp16_vector", "bf16": "bf16_vector", "fp32": "fp32_vector", "int8": "memory_only"},
        why="Example A: elementwise, not MMA -> fp16_vector for fp16; int8 integer add without a calibrated mode (M2)",
        cls_by_dtype={"int8": "integer"}, decisions={"bf16": ["D2_mi300x_bf16_vector"], "int8": ["D3_memory_only_M2"]}),
    "weight_dequant": dict(cls="vector", algo="elementwise multiply by a block-constant broadcast scale",
        prec="multiply in the input dtype (fp32 with one cast accepted)",
        modes={"fp16": "fp16_vector", "bf16": "bf16_vector", "fp32": "fp32_vector"},
        why="non-MMA elementwise multiply in the input dtype", decisions={"bf16": ["D2_mi300x_bf16_vector"]}),
    "rope": dict(cls="vector", algo="elementwise 2x2 rotation per pair with shared cos/sin",
        prec="input dtype (reference); fp32 with one cast accepted", modes={"fp16": "fp16_vector", "fp32": "fp32_vector"},
        why="non-MMA; the reference evaluates in the input dtype"),
    "jacobi_stencil_2d": dict(cls="vector", algo="single-iteration out-of-place 5-point Jacobi stencil",
        prec="input dtype expected; fp32 sum with one cast permitted",
        modes={"fp16": "fp16_vector", "bf16": "bf16_vector", "fp32": "fp32_vector"},
        why="non-MMA stencil in the input dtype", decisions={"bf16": ["D2_mi300x_bf16_vector"]}),
    # ---- 2 non-MMA vector, contract mandates fp32 arithmetic
    "dequantize_rowwise": dict(cls="vector", algo="streaming int8 -> fp16 map with a per-row scale",
        prec="x and scale converted to fp32, both multiplies fp32", modes={"int8": "fp32_vector"},
        why="int8 input but fp32 arithmetic by contract: input dtype does not set the mode"),
    "dropout": dict(cls="vector", algo="streaming mask-and-rescale map", prec="fp32 for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="contract mandates fp32 arithmetic"),
    "fused_activation": dict(cls="vector", algo="streaming fused multiply-add + SiLU map", prec="fp32",
        modes={"fp32": "fp32_vector"}, why="fp32 arithmetic"),
    "sigmoid": dict(cls="vector", algo="elementwise sigmoid", prec="explicit upcast, fp32 for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="contract mandates fp32 arithmetic"),
    "swiglu": dict(cls="vector", algo="elementwise fused gate x*sigmoid(x)*y", prec="both operands upcast, fp32",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="contract mandates fp32 arithmetic"),
    "gaussian_blur": dict(cls="vector", algo="direct 2-D stencil (7x7 taps), per-output fp32 accumulation",
        prec="pixel and weight converted to fp32 before each multiply (approved C1), fp32 accumulation",
        modes={"fp16": "fp32_vector", "fp32": "fp32_vector"},
        why="single-channel direct stencil is not a GEMM; fp32 products by contract (C1)"),
    # ---- 3 reductions / normalisation
    "softmax": dict(cls="reduction", algo="online two-traversal row softmax", prec="fp32 statistics and exp for all dtypes",
        modes={"fp16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 arithmetic by contract"),
    "cross_entropy": dict(cls="reduction", algo="row-wise stable log-sum-exp plus gather", prec="fp32 regardless of input dtype",
        modes={"fp16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 arithmetic by contract"),
    "kl_divergence": dict(cls="reduction", algo="fused elementwise map plus row sum", prec="fp32",
        modes={"fp32": "fp32_vector"}, why="fp32 arithmetic"),
    "mean_reduction": dict(cls="reduction", algo="single-pass row sum and divide", prec="fp32 accumulation for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 accumulation by contract"),
    "l2_norm": dict(cls="reduction", algo="two-pass row normalisation", prec="fp32 for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 arithmetic by contract"),
    "layernorm": dict(cls="reduction", algo="two-pass row normalisation with affine epilogue", prec="fp32 for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 arithmetic by contract"),
    "rmsnorm": dict(cls="reduction", algo="two-pass sum-of-squares then scale", prec="fp32 for all dtypes",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"}, why="fp32 arithmetic by contract"),
    "batch_normalization": dict(cls="reduction", algo="two-pass statistics then apply, two-level per-channel reduction",
        prec="fp32 for all dtypes", modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"},
        why="fp32 arithmetic by contract"),
    "flash_decode": dict(cls="reduction", algo="log-sum-exp weighted combine of per-split partial outputs", prec="fp32",
        modes={"fp32": "fp32_vector"}, why="combine stage only: no matrix product; fp32 vector arithmetic"),
    "moe_topk_gating": dict(cls="selection", algo="k rounds of row max/argmax with exclusion masking + softmax over k values",
        prec="selection on fp32 after exact upcast; fp32 softmax",
        modes={"fp16": "fp32_vector", "bf16": "fp32_vector", "fp32": "fp32_vector"},
        why="approved rev-2 declaration charges the simplified op count (mostly comparisons) at fp32_vector; "
            "memory-bound in every case (compute/memory <= 0.38), so a memory-only bound gives the same T_SOL"),
    # ---- 6 selection / sort / comparison
    "argmax": dict(cls="selection", algo="chunked row scan keeping running best value/index", prec="comparisons only",
        modes={"fp16": "memory_only", "fp32": "memory_only"}, why="comparison-only (M2)",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "fp32")}),
    "top_k_selection": dict(cls="selection", algo="hierarchical tournament block top-K'", prec="comparisons only",
        modes={"fp32": "memory_only"}, why="comparison-only (M2)", decisions={"fp32": ["D3_memory_only_M2"]}),
    "bitonic_sort": dict(cls="selection", algo="in-place bitonic sorting network", prec="comparisons only",
        modes={"fp16": "memory_only", "fp32": "memory_only"}, why="comparison network (M2)",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "fp32")}),
    "radix_sort": dict(cls="selection", algo="LSD counting radix sort, stable passes", prec="exact int32",
        modes={"int32": "memory_only"}, why="integer sort (M2); Q = 2*n*dtype_size compulsory I/O (approved M4)",
        decisions={"int32": ["D3_memory_only_M2"]}),
    "2d_max_pooling": dict(cls="selection", algo="direct sliding-window max", prec="comparisons only",
        modes={"fp16": "memory_only", "bf16": "memory_only", "fp32": "memory_only"}, why="comparison stencil (M2)",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32")}),
    "relu": dict(cls="selection", algo="elementwise compare-and-select against zero", prec="no arithmetic",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")}, why="comparison/select only (M2)",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32", "int8")}),
    "histogramming": dict(cls="integer", algo="two-level privatised integer histogram", prec="exact int32 counting",
        modes={"int32": "memory_only"}, why="integer counting without a calibrated mode (M2)",
        decisions={"int32": ["D3_memory_only_M2"]}),
    # ---- 4 memory movement / indexing / conversion
    "matrix_copy": dict(cls="movement", algo="flat streaming copy", prec="no arithmetic",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")}, why="no arithmetic work",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32", "int8")}),
    "matrix_transpose": dict(cls="movement", algo="out-of-place tiled transpose", prec="no arithmetic (frozen F = 0)",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")}, why="frozen F = 0 (no compute term)"),
    "reverse_array": dict(cls="movement", algo="single-pass out-of-place reversal", prec="no arithmetic",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")}, why="no arithmetic work",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32", "int8")}),
    "interleave": dict(cls="movement", algo="elementwise zip of two streams", prec="no arithmetic",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")}, why="no arithmetic work",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32", "int8")}),
    "destindex": dict(cls="movement", algo="index-driven scatter copy (row permutation)", prec="exact copy; int32/int64 index arithmetic",
        modes={dt: "memory_only" for dt in ("fp16", "bf16", "fp32", "int8")},
        why="no arithmetic on the data; index arithmetic is address computation (M2)",
        decisions={dt: ["D3_memory_only_M2"] for dt in ("fp16", "bf16", "fp32", "int8")}),
    "quantize_global": dict(cls="movement", algo="elementwise fp32 -> fp16 conversion (no scaling)", prec="one RNE conversion",
        modes={"fp32": "memory_only"}, why="dtype conversion only (M2)", decisions={"fp32": ["D3_memory_only_M2"]}),
}


def git_show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


def load_peaks():
    """{device: (doc, sha256 of the file bytes, git blob id)} from the PR #323 merge commit."""
    out = {}
    for d in DEVICES:
        p = f"tilebench/data/peak_performance/empirical/{d}.json"
        raw = git_show(PR323_MERGE, p)
        blob = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{PR323_MERGE}:{p}"], capture_output=True, text=True,
                              check=True).stdout.strip()
        out[d] = (json.loads(raw), hashlib.sha256(raw).hexdigest(), blob, p)
    return out


def peak_value(doc, key):
    """(value in the file's unit, unit) for 'peak_tflops.<k>' / 'peak_tops.<k>', or (None, None) when absent."""
    if key is None:
        return None, None
    sec, k = key.split(".")
    v = (doc.get(sec) or {}).get(k)
    return (float(v), {"peak_tflops": "TFLOP/s", "peak_tops": "TOP/s"}[sec]) if v is not None else (None, None)


def build(data_dtypes):
    """data_dtypes: {op: sorted dtypes present in the formal data}. Returns (rows, problems)."""
    modes_doc = yaml.safe_load(git_show(CONTRACT_COMMIT, "tilebench/llm/v2/manifests/arithmetic_modes.yaml"))
    peaks = load_peaks()
    rows, problems = [], []
    for op in sorted(data_dtypes):
        e = OPS.get(op)
        if e is None:
            problems.append(f"{op}: no audit entry")
            continue
        cfg = yaml.safe_load((REPO / "tilebench/benchmarks/operators" / op / "config.yaml").read_text())["metrics"]
        rev2 = modes_doc["operators"][op]
        for dt in data_dtypes[op]:
            mode = e["modes"].get(dt)
            if mode is None:
                problems.append(f"{op}/{dt}: no audited mode")
                continue
            if REV2[rev2["modes"][dt]] != mode:
                problems.append(f"{op}/{dt}: audited {mode} != rev2 {rev2['modes'][dt]}")
            ov = OVERRIDES.get(op, {})
            keys = MODE_KEYS[mode]
            row = {
                "operator": op, "dtype": dt,
                "workload_class": CLASSES[(e.get("cls_by_dtype") or {}).get(dt, e["cls"])],
                "canonical_algorithm": e["algo"], "arithmetic_precision": e["prec"],
                "mma_eligible": e["cls"] in ("mma", "multistage"),
                "compute_mode": mode, "rev2_declared_mode": rev2["modes"][dt],
                "device_specific_peak_key": {d: keys[d] for d in DEVICES},
                "peak_value": {}, "peak_unit": {},
                "F_expression": ov.get("F", cfg.get("flops_expr")).strip() if mode != "memory_only" else None,
                "F_expression_frozen": str(cfg.get("flops_expr")).strip(),
                "Q_expression": ov.get("Q", cfg.get("bytes_expr")).strip(),
                "Q_expression_frozen": str(cfg.get("bytes_expr")).strip(),
                "f_kind": rev2["f_kind"], "q_kind": "compulsory_io" if op == "radix_sort" else rev2["q_kind"],
                "approved_overrides": [ov["decision"]] if ov else [],
                "source_file": {"contract": f"tilebench/llm/v2/contracts/data/{op}/contract.md",
                                "config": f"tilebench/benchmarks/operators/{op}/config.yaml",
                                "impl": [f"tilebench/benchmarks/operators/{op}/impl_{b}.py" for b in ("triton", "cutile", "tilelang")],
                                "declaration": "tilebench/llm/v2/manifests/arithmetic_modes.yaml"},
                "source_commit": {"contract_and_declaration": CONTRACT_COMMIT, "operators_tree": OPERATOR_TREE_COMMIT,
                                  "peaks": PR323_MERGE},
                "justification": e["why"],
                "decisions": list((e.get("decisions") or {}).get(dt, [])),
            }
            if mode == "tf32_class_mma":
                row["decisions"].append("D1_tf32_class_on_cdna3")
            for d in DEVICES:
                v, u = peak_value(peaks[d][0], keys[d])
                if keys[d] is not None and v is None:
                    row["peak_value"][d], row["peak_unit"][d] = None, None
                else:
                    row["peak_value"][d], row["peak_unit"][d] = v, u
            # Per-device target status. A mode without a calibrated value is never filled from another mode.
            row["target_status"] = {}
            for d in DEVICES:
                if mode == "memory_only":
                    row["target_status"][d] = "ok; memory_only"
                elif row["peak_value"][d] is not None:
                    row["target_status"][d] = "ok"
                elif mode == "bf16_vector" and d == "MI300X":
                    row["target_status"][d] = DECISIONS["D2_mi300x_bf16_vector"]["status_label"]
                elif mode == "fp8_e4m3fn_mma" and d == "MI300X":
                    row["target_status"][d] = "not_applicable: no MI300X data (UNSUPPORTED_DTYPE; gfx942 has e4m3fnuz only)"
                else:
                    row["target_status"][d] = "missing_calibration"
                    problems.append(f"{op}/{dt}/{d}: no calibrated {keys[d]} and no approved rule")
            pend = [x for x in row["decisions"] if DECISIONS[x]["status"] != "approved"]
            row["review_status"] = ("pending: " + ", ".join(pend)) if pend else \
                ("approved" + (" (" + ", ".join(row["decisions"]) + ")" if row["decisions"] else ""))
            rows.append(row)
    return rows, problems, peaks, modes_doc


def main(out=OUT):
    out = Path(out)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figure_data as FD
    dts = {}
    for r in FD.read_csv(FD.COMBINED / "benchmark_cases_normalized.csv.gz"):
        if r["mode"] == "autotune":
            dts.setdefault(r["operator"], set()).add(r["dtype"])
    rows, problems, peaks, _ = build({o: sorted(v) for o, v in dts.items()})
    if problems:
        raise SystemExit("mode audit failed:\n  " + "\n  ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "tilearena-sol-mode-manifest/1",
        "rule": "reference compute mode = f(frozen canonical algorithm, numerical contract); never input/accumulator/"
                "output dtype alone, compiled ISA or profiler counters; same mode for every DSL of an "
                "(operator, dtype, device)",
        "classes": CLASSES, "mode_keys": MODE_KEYS, "decisions": DECISIONS, "overrides": OVERRIDES,
        "peaks": {d: {"file": peaks[d][3], "commit": PR323_MERGE, "sha256": peaks[d][1], "git_blob": peaks[d][2],
                      "calibration_id": peaks[d][0]["calibration_id"], "values": peaks[d][0]} for d in DEVICES},
        "rows": rows,
    }
    (out / "sol_mode_manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
    cols = ["operator", "dtype", "workload_class", "canonical_algorithm", "arithmetic_precision", "mma_eligible",
            "compute_mode", "rev2_declared_mode", "peak_key_B200", "peak_key_GH200", "peak_key_MI300X", "F_expression",
            "Q_expression", "f_kind", "q_kind", "approved_overrides", "source_file", "source_commit", "justification",
            "decisions", "review_status"]
    with open(out / "sol_mode_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in cols if k in r},
                        **{f"peak_key_{d}": "none (memory-only)" if r["compute_mode"] == "memory_only"
                           else (r["device_specific_peak_key"][d] if r["peak_value"][d] is not None
                                 else f"{r['device_specific_peak_key'][d]} absent -> {r['target_status'][d]}")
                           for d in DEVICES},
                        "approved_overrides": ";".join(r["approved_overrides"]), "decisions": ";".join(r["decisions"]),
                        "source_file": r["source_file"]["contract"], "source_commit": CONTRACT_COMMIT[:12]})
    pend = sum(r["review_status"].startswith("pending") for r in rows)
    if pend:
        raise SystemExit(f"{pend} rows depend on pending decisions")
    print(f"{len(rows)} operator/dtype rows over {len({r['operator'] for r in rows})} operators; all decisions approved")


if __name__ == "__main__":
    main()
