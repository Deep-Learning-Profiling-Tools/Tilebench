"""Characterise the TileLang fp8 error: is it a bad accumulate, a bad epilogue
cast, a layout problem, or K-dependent drift?

Compares against two references:
  ref_exact  - fp64 matmul of the *dequantised* fp8 inputs (what the kernel
               should compute, modulo fp32 accumulation order)
  ref_fp8out - ref_exact rounded to e4m3, i.e. the best any kernel writing an
               fp8 output could possibly do
"""

import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_triton)
from tilelang_variants import build_matmul  # noqa: E402

DT = torch.float8_e4m3fn


def stats(name, out, ref_exact, ref_fp8out):
    o = out.float().double()
    d = (o - ref_exact).abs()
    dq = (o - ref_fp8out).abs()
    n = torch.isnan(o).sum().item()
    i = torch.isinf(o).sum().item()
    print(f"{name:<38} max|-exact| {d.max().item():9.2f}  "
          f"rel {(d.max() / ref_exact.abs().max()).item():8.2e}  "
          f"max|-fp8ref| {dq.max().item():9.2f}  nan {n}  inf {i}  "
          f"absmax {o.abs().max().item():8.2f}")


def main():
    torch.manual_seed(0)
    M = N = 512

    print("=== K sweep, 256x256x128 tile, ws=on ===")
    print(f"{'case':<38}")
    for K in (128, 256, 512, 1024, 4096):
        af = torch.randn(M, K, device="cuda", dtype=torch.float32)
        bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
        a, b = af.to(DT), bf.to(DT)
        # dequantised inputs: what the tensor core actually sees
        ad, bd = a.float().double(), b.float().double()
        ref_exact = ad @ bd
        ref_fp8out = ref_exact.float().to(DT).float().double()

        c = torch.empty(M, N, device="cuda", dtype=DT)
        kern = build_matmul(bm=256, bn=256, bk=128, gs=64, threads=256,
                            stages=3, warp_spec=True)
        kern(a, b, c, dtype="float8_e4m3fn")
        stats(f"K={K:<6} tilelang ws=on", c, ref_exact, ref_fp8out)
        stats(f"K={K:<6} triton", impl_triton.run(a, b), ref_exact, ref_fp8out)
        stats(f"K={K:<6} cutile", impl_cutile.run(a, b), ref_exact, ref_fp8out)
        print()

    # Is the error in the epilogue cast or in the accumulation? Rebuild the
    # same kernel writing a float16 output so the fp8 round-trip is removed.
    print("=== same GEMM, float16 output (removes the e4m3 epilogue cast) ===")
    import tilelang
    import tilelang.language as T

    def kern16(a, b, c):
        M_, K_, N_ = T.const("M_, K_, N_")
        a: T.Tensor((M_, K_), "float8_e4m3")
        b: T.Tensor((K_, N_), "float8_e4m3")
        c: T.Tensor((M_, N_), "float16")
        with T.Kernel(T.ceildiv(M_, 256), T.ceildiv(N_, 256), threads=256) as (pm, pn):
            sm, sn = pm * 256, pn * 256
            at = T.alloc_shared((256, 128), "float8_e4m3")
            bt = T.alloc_shared((128, 256), "float8_e4m3")
            acc = T.alloc_fragment((256, 256), "float32")
            acc_t = T.alloc_tmem((256, 256), "float32")
            mbar = T.alloc_barrier(1)
            for k in T.Pipelined(T.ceildiv(K_, 128), num_stages=3):
                T.copy(a[sm, k * 128], at)
                T.copy(b[k * 128, sn], bt)
                T.gemm(at, bt, acc_t, mbar=mbar, clear_accum=k == 0)
            T.copy(acc_t, acc)
            T.copy(acc, c[sm, sn])

    K = 1024
    af = torch.randn(M, K, device="cuda", dtype=torch.float32)
    bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a, b = af.to(DT), bf.to(DT)
    ref_exact = a.float().double() @ b.float().double()
    ref_fp8out = ref_exact.float().to(DT).float().double()
    c16 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    try:
        k16 = tilelang.jit()(kern16)
        k16(a, b, c16)
        stats(f"K={K:<6} tilelang -> fp16 out", c16, ref_exact, ref_fp8out)
    except Exception as e:
        msg = [l for l in str(e).strip().splitlines() if l.strip()]
        print("fp16-out variant FAILED:", (msg[-1] if msg else "")[:100])


if __name__ == "__main__":
    main()
