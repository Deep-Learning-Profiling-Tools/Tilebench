"""Launch one matmul kernel (one backend, one dtype) for NCU capture.

TileLang is available in two flavours:
  tilelang     - exactly as impl_tilelang.py ships (warp specialization disabled)
  tilelang_ws  - identical kernel with warp specialization left enabled, which is
                 the only configuration that produces correct fp16 output
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DTYPES = {
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
    "fp32": torch.float32,
}


def make_inputs(dtype, M, N, K):
    torch.manual_seed(0)
    a32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    return a32.to(dtype), b32.to(dtype)


def launch_tilelang(a, b, dtype, M, N, K, warp_spec: bool):
    import tilelang

    from tilelang_variants import build_matmul, DEFAULT_CFG

    cfg = DEFAULT_CFG[a.dtype]
    kern = build_matmul(warp_spec=warp_spec, **cfg)
    c = torch.empty((M, N), device="cuda", dtype=a.dtype)
    name = str(a.dtype).removeprefix("torch.")
    return lambda: kern(a, b, c, dtype=name), c


MATCHED = {
    torch.float16: dict(
        tl=dict(bm=256, bn=256, bk=64, gs=64, threads=256, stages=3),
        tr={"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        ct=dict(tm=256, tn=256, tk=64, group_size_m=8, occupancy=4),
    ),
    torch.float8_e4m3fn: dict(
        tl=dict(bm=256, bn=256, bk=128, gs=64, threads=256, stages=3),
        tr={"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
        ct=dict(tm=256, tn=256, tk=128, group_size_m=8, occupancy=4),
    ),
}


def apply_matched(dt):
    from types import SimpleNamespace

    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_cutile, impl_triton

    from tilelang_variants import DEFAULT_CFG

    m = MATCHED[dt]
    DEFAULT_CFG[dt] = dict(m["tl"])
    impl_triton._DEFAULT_CONFIGS[dt] = dict(m["tr"])
    impl_cutile._DEFAULT_CONFIGS[dt] = SimpleNamespace(**m["ct"])


def launch_triton(a, b, dtype, M, N, K, warp_spec=None):
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_triton

    out = {}

    def go():
        out["c"] = impl_triton.run(a, b)

    go()
    return go, out["c"]


def launch_cutile(a, b, dtype, M, N, K, warp_spec=None):
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_cutile

    out = {}

    def go():
        out["c"] = impl_cutile.run(a, b)

    go()
    return go, out["c"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=("tilelang", "tilelang_ws", "triton", "cutile"))
    p.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=20480)
    p.add_argument("--bm", type=int, default=None)
    p.add_argument("--bn", type=int, default=None)
    p.add_argument("--bk", type=int, default=None)
    p.add_argument("--stages", type=int, default=None)
    p.add_argument("--matched", action="store_true",
                   help="force all backends to the recorded tuned tile shape")
    args = p.parse_args()

    torch.cuda.set_device(0)
    dt = DTYPES[args.dtype]
    a, b = make_inputs(dt, args.M, args.N, args.K)

    if args.matched:
        apply_matched(dt)

    if args.backend.startswith("tilelang"):
        from tilelang_variants import DEFAULT_CFG
        for k_, v_ in (("bm", args.bm), ("bn", args.bn), ("bk", args.bk),
                       ("stages", args.stages)):
            if v_ is not None:
                DEFAULT_CFG[dt][k_] = v_
        fn, c = launch_tilelang(a, b, dt, args.M, args.N, args.K,
                                warp_spec=args.backend.endswith("_ws"))
    elif args.backend == "triton":
        fn, c = launch_triton(a, b, dt, args.M, args.N, args.K)
    else:
        fn, c = launch_cutile(a, b, dt, args.M, args.N, args.K)

    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    ref = a.float().double() @ b.float().double()
    err = (c.float().double() - ref).abs().max().item()
    print(f"backend={args.backend} dtype={args.dtype} "
          f"M={args.M} N={args.N} K={args.K} max_abs_err={err:.4f}")


if __name__ == "__main__":
    main()
