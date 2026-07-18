#!/usr/bin/env python3
"""Measure persistent (software) peak bandwidth and FLOPS on the current GPU.

Uses inner-loop amortization to eliminate kernel launch overhead for small ops,
without CUDA Graphs (which can interfere with cuBLAS autotune).

Bandwidth:
  - HBM peak:   cudaMemcpy on large buffers (>= 64 MB)
  - L2 peak:    cudaMemcpy on small buffers (<= 8 MB, fits in L2 cache)

FLOPS:
  - FP32 / BF16 / FP16:  torch.matmul  (cuBLAS)
  - FP8  (E4M3):         torch._scaled_mm
  - INT8:                 torch._int_mm
  - FP4  (E2M1):         torch._scaled_mm  (blockwise 1x16 scaling)

Usage:
    python scripts/measure_peak.py [--json]
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
import torch

# Allow running as: python scripts/measure_peak.py (without PYTHONPATH=.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.gpu import gpu_label as _default_gpu_label  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 30
REPEAT = 100
TARGET_MS = 2.0  # target wall time per timed region (for calibrating inner_iters)

# Buffer sizes for HBM bandwidth (must exceed L2 ~50 MB)
HBM_BW_SIZES_MB = [64, 128, 256, 512, 1024]

# Buffer sizes for L2 bandwidth (must fit in L2 ~50 MB; src+dst together)
L2_BW_SIZES_MB = [1, 2, 4, 8, 16]

# GEMM sizes for FLOPS
GEMM_SIZES = [4096, 8192, 12288, 16384]

def _sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Calibrate inner_iters so each timed region takes ~TARGET_MS
# ---------------------------------------------------------------------------
def _calibrate(fn, target_ms=TARGET_MS):
    """Run fn() a few times, return inner_iters so that inner_iters * fn() >= target_ms."""
    # Warmup
    for _ in range(5):
        fn()
    _sync()

    # Time a single call
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    _sync()
    single_ms = start.elapsed_time(end)

    if single_ms <= 0:
        single_ms = 0.001  # guard against 0

    iters = max(1, int(target_ms / single_ms))
    return iters


# ---------------------------------------------------------------------------
# Core: inner-loop timed measurement
# ---------------------------------------------------------------------------
def _measure(fn, warmup=WARMUP, repeat=REPEAT):
    """Warmup, calibrate inner_iters, then return per-call median time in ms."""
    # Full warmup
    for _ in range(warmup):
        fn()
    _sync()

    inner_iters = _calibrate(fn)

    times_ms = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            fn()
        end.record()
        _sync()
        times_ms.append(start.elapsed_time(end) / inner_iters)

    return statistics.median(times_ms), inner_iters


# ---------------------------------------------------------------------------
# Bandwidth: D2D memcpy (copy_)
# ---------------------------------------------------------------------------
def measure_memcpy_bw(sizes_mb):
    device = torch.device("cuda")
    results = []
    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024
        src = torch.empty(numel, dtype=torch.uint8, device=device)
        dst = torch.empty(numel, dtype=torch.uint8, device=device)
        src.fill_(42)

        med_ms, iters = _measure(lambda: dst.copy_(src))
        bw = (numel * 2 / 1e9) / (med_ms / 1e3)  # read + write
        results.append((size_mb, med_ms, bw, iters))
        del src, dst
    return results


# ---------------------------------------------------------------------------
# Bandwidth: memset (zero_)
# ---------------------------------------------------------------------------
def measure_memset_bw(sizes_mb):
    device = torch.device("cuda")
    results = []
    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024
        buf = torch.empty(numel, dtype=torch.uint8, device=device)

        med_ms, iters = _measure(lambda: buf.zero_())
        bw = (numel / 1e9) / (med_ms / 1e3)
        results.append((size_mb, med_ms, bw, iters))
        del buf
    return results


# ---------------------------------------------------------------------------
# FLOPS: standard dtypes via torch.matmul (cuBLAS)
# ---------------------------------------------------------------------------
def measure_matmul_flops(dtype_name, dtype):
    device = torch.device("cuda")
    results = []
    for M in GEMM_SIZES:
        A = torch.randn(M, M, dtype=dtype, device=device)
        B = torch.randn(M, M, dtype=dtype, device=device)

        med_ms, iters = _measure(lambda: torch.matmul(A, B))
        flops = 2.0 * M * M * M
        tflops = (flops / 1e12) / (med_ms / 1e3)
        results.append((M, med_ms, tflops, iters))
        del A, B
    return results


# ---------------------------------------------------------------------------
# FLOPS: FP8 via torch._scaled_mm (tensorwise scaling)
# ---------------------------------------------------------------------------
def measure_fp8_flops():
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    results = []
    for M in GEMM_SIZES:
        A = torch.randn(M, M, device=device).to(fp8_dtype)
        # B must be column-major for _scaled_mm: create (N,K) row-major, then .t() → (K,N) col-major
        B = torch.randn(M, M, device=device).to(fp8_dtype).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)

        def run():
            torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b,
                             out_dtype=torch.float16)

        med_ms, iters = _measure(run)
        flops = 2.0 * M * M * M
        tflops = (flops / 1e12) / (med_ms / 1e3)
        results.append((M, med_ms, tflops, iters))
        del A, B
    return results


# ---------------------------------------------------------------------------
# FLOPS: INT8 via torch._int_mm
# ---------------------------------------------------------------------------
def measure_int8_flops():
    device = torch.device("cuda")
    results = []
    for M in GEMM_SIZES:
        A = torch.randint(-128, 127, (M, M), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (M, M), dtype=torch.int8, device=device)

        med_ms, iters = _measure(lambda: torch._int_mm(A, B))
        flops = 2.0 * M * M * M
        tflops = (flops / 1e12) / (med_ms / 1e3)
        results.append((M, med_ms, tflops, iters))
        del A, B
    return results


# ---------------------------------------------------------------------------
# FLOPS: FP4 via torch._scaled_mm (blockwise 1x16 scaling)
# ---------------------------------------------------------------------------
def measure_fp4_flops():
    device = torch.device("cuda")
    results = []
    for M in GEMM_SIZES:
        K, N = M, M
        nbytes_A = M * K // 2
        nbytes_B = K * N // 2
        A_raw = torch.randint(0, 256, (nbytes_A,), dtype=torch.uint8, device=device)
        A_fp4 = A_raw.view(torch.float4_e2m1fn_x2).reshape(M, K // 2)
        B_raw = torch.randint(0, 256, (nbytes_B,), dtype=torch.uint8, device=device)
        # Column-major B: create as (N, K//2) row-major then .t() → (K//2, N) col-major
        B_fp4 = B_raw.view(torch.float4_e2m1fn_x2).reshape(N, K // 2).t()
        sa = torch.ones(M * K // 16, dtype=torch.float8_e4m3fn, device=device)
        sb = torch.ones(K * N // 16, dtype=torch.float8_e4m3fn, device=device)

        def run():
            torch._scaled_mm(A_fp4, B_fp4, scale_a=sa, scale_b=sb,
                             out_dtype=torch.float16)

        med_ms, iters = _measure(run)
        flops = 2.0 * M * K * N
        tflops = (flops / 1e12) / (med_ms / 1e3)
        results.append((M, med_ms, tflops, iters))
        del A_raw, A_fp4, B_raw, B_fp4, sa, sb
    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _print_bw_table(results, label):
    print(f"  {'Size (MB)':>10}  {'Median (ms)':>12}  {'BW (GB/s)':>10}  {'inner_iters':>11}")
    print(f"  {'-'*51}")
    for size_mb, med_ms, bw, iters in results:
        print(f"  {size_mb:>10}  {med_ms:>12.6f}  {bw:>10.1f}  {iters:>11}")
    peak = max(r[2] for r in results)
    print(f"\n  Peak {label}: {peak:.1f} GB/s")
    return peak


def _print_flops_table(results, dtype_name):
    print(f"\n  dtype: {dtype_name}")
    print(f"  {'M':>8}  {'Median (ms)':>12}  {'TFLOPS':>10}  {'inner_iters':>11}")
    print(f"  {'-'*49}")
    for M, med_ms, tflops, iters in results:
        print(f"  {M:>8}  {med_ms:>12.4f}  {tflops:>10.1f}  {iters:>11}")
    best = max(r[2] for r in results)
    print(f"  Peak {dtype_name}: {best:.1f} TFLOPS")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Measure persistent peak BW & FLOPS")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Short GPU name (e.g. B200) for output filenames. "
                             "Auto-detected from CUDA device name if omitted.")
    parser.add_argument("--json", action="store_true", help="Print JSON config block")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_label = args.gpu if args.gpu else _default_gpu_label(gpu_name)
    print(f"GPU: {gpu_name}")
    print(f"GPU label: {gpu_label}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}, Target per-region: {TARGET_MS} ms")
    print(f"Timing: CUDA events + inner-loop amortization, median of {REPEAT} trials")
    print()

    # ===== HBM Bandwidth =====
    print("=" * 60)
    print("HBM Bandwidth — memset (zero_)")
    print("=" * 60)
    memset_hbm = measure_memset_bw(HBM_BW_SIZES_MB)
    peak_memset_hbm = _print_bw_table(memset_hbm, "memset HBM")
    print()

    print("=" * 60)
    print("HBM Bandwidth — D2D memcpy (copy_)")
    print("=" * 60)
    memcpy_hbm = measure_memcpy_bw(HBM_BW_SIZES_MB)
    peak_memcpy_hbm = _print_bw_table(memcpy_hbm, "memcpy HBM")
    print()

    peak_hbm_bw = max(peak_memset_hbm, peak_memcpy_hbm)
    print(f"  >>> Persistent peak HBM BW: {peak_hbm_bw:.1f} GB/s <<<")
    print()

    # ===== L2 Cache Bandwidth =====
    print("=" * 60)
    print("L2 Cache Bandwidth — D2D memcpy (copy_)  [small buffers]")
    print("=" * 60)
    memcpy_l2 = measure_memcpy_bw(L2_BW_SIZES_MB)
    peak_l2_memcpy = _print_bw_table(memcpy_l2, "memcpy L2")
    print()

    print("=" * 60)
    print("L2 Cache Bandwidth — memset (zero_)  [small buffers]")
    print("=" * 60)
    memset_l2 = measure_memset_bw(L2_BW_SIZES_MB)
    peak_l2_memset = _print_bw_table(memset_l2, "memset L2")
    print()

    peak_l2_bw = max(peak_l2_memcpy, peak_l2_memset)
    print(f"  >>> Persistent peak L2 BW: {peak_l2_bw:.1f} GB/s <<<")
    print()

    # ===== FLOPS =====
    print("=" * 60)
    print("Compute — cuBLAS GEMM")
    print("=" * 60)

    peak_tflops = {}

    fp32_results = measure_matmul_flops("fp32", torch.float32)
    peak_tflops["fp32"] = _print_flops_table(fp32_results, "fp32")

    bf16_results = measure_matmul_flops("bf16", torch.bfloat16)
    peak_tflops["bf16"] = _print_flops_table(bf16_results, "bf16")

    fp16_results = measure_matmul_flops("fp16", torch.float16)
    peak_tflops["fp16"] = _print_flops_table(fp16_results, "fp16")

    print("\n  --- FP8 (E4M3) via torch._scaled_mm ---")
    fp8_results = None
    try:
        fp8_results = measure_fp8_flops()
        peak_tflops["fp8"] = _print_flops_table(fp8_results, "fp8")
    except Exception as e:
        print(f"  FP8 failed: {e}")

    print("\n  --- INT8 via torch._int_mm ---")
    int8_results = None
    try:
        int8_results = measure_int8_flops()
        peak_tflops["int8"] = _print_flops_table(int8_results, "int8")
    except Exception as e:
        print(f"  INT8 failed: {e}")

    print("\n  --- FP4 (E2M1) via torch._scaled_mm ---")
    fp4_results = None
    try:
        fp4_results = measure_fp4_flops()
        peak_tflops["fp4"] = _print_flops_table(fp4_results, "fp4")
    except Exception as e:
        print(f"  FP4 failed: {e}")

    # ===== SUMMARY =====
    print()
    print("=" * 60)
    print("SUMMARY — Persistent Peak (for config.yaml)")
    print("=" * 60)
    print(f"  peak_hbm_bw_GBs: {peak_hbm_bw:.1f}")
    print(f"  peak_l2_bw_GBs:  {peak_l2_bw:.1f}")
    dtype_order = ["fp32", "bf16", "fp16", "fp8", "int8", "fp4"]
    for d in dtype_order:
        if d in peak_tflops:
            print(f"  peak_tflops.{d}: {peak_tflops[d]:.1f}")
    print()

    # ===== Save results to JSON =====
    def _bw_detail(rows):
        return [{"size_mb": s, "median_ms": round(m, 6), "bw_GBs": round(b, 1)}
                for s, m, b, _ in rows]

    def _flops_detail(rows):
        return [{"M": m, "median_ms": round(ms, 4), "tflops": round(t, 1)}
                for m, ms, t, _ in rows]

    full_results = {
        "gpu": gpu_name,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "warmup": WARMUP,
            "repeat": REPEAT,
            "target_ms": TARGET_MS,
        },
        "bandwidth": {
            "hbm": {
                "memset": _bw_detail(memset_hbm),
                "memcpy": _bw_detail(memcpy_hbm),
                "peak_memset_GBs": round(peak_memset_hbm, 1),
                "peak_memcpy_GBs": round(peak_memcpy_hbm, 1),
                "peak_GBs": round(peak_hbm_bw, 1),
            },
            "l2": {
                "memcpy": _bw_detail(memcpy_l2),
                "memset": _bw_detail(memset_l2),
                "peak_memcpy_GBs": round(peak_l2_memcpy, 1),
                "peak_memset_GBs": round(peak_l2_memset, 1),
                "peak_GBs": round(peak_l2_bw, 1),
            },
        },
        "compute": {},
        "summary": {
            "peak_hbm_bw_GBs": round(peak_hbm_bw, 1),
            "peak_l2_bw_GBs": round(peak_l2_bw, 1),
            "peak_tflops": {k: round(v, 1) for k, v in peak_tflops.items()},
        },
    }

    # Add per-dtype FLOPS detail
    all_flops = {"fp32": fp32_results, "bf16": bf16_results, "fp16": fp16_results}
    if "fp8" in peak_tflops:
        all_flops["fp8"] = fp8_results
    if "int8" in peak_tflops:
        all_flops["int8"] = int8_results
    if "fp4" in peak_tflops:
        all_flops["fp4"] = fp4_results

    for dname, rows in all_flops.items():
        full_results["compute"][dname] = {
            "detail": _flops_detail(rows),
            "peak_tflops": round(peak_tflops[dname], 1),
        }

    # Save detailed results to results/<GPU>/peak_performance/peak_performance.json
    detail_dir = os.path.join(os.path.dirname(__file__), "..", "results",
                              gpu_label, "peak_performance")
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, "peak_performance.json")
    with open(detail_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Detailed results saved to {os.path.abspath(detail_path)}")

    # Save summary to data/peak_performance/<GPU>.json
    # Keys match what core/metrics.py and scripts/visualize.py expect
    summary = {
        "gpu": gpu_name,
        "peak_bw_GBs": round(peak_hbm_bw, 1),
        "peak_tflops": {k: round(v, 1) for k, v in peak_tflops.items()},
    }
    summary_dir = os.path.join(os.path.dirname(__file__), "..", "data", "peak_performance")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{gpu_label}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {os.path.abspath(summary_path)}")

    if args.json:
        print("\nJSON config block:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
