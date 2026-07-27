"""Accuracy of the *shipped* impl_tilelang kernel at its own recorded tuned
config, vs Triton/cuTile, vs torch. Checks whether the benchmark's reported
tilelang_ok=True is measuring what we think.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_tilelang, impl_triton)

TUNED_TL = {
    torch.float16: {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 64, "threads": 256, "num_stages": 3},
    torch.float8_e4m3fn: {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
                          "GROUP_SIZE_M": 64, "threads": 256, "num_stages": 3},
}


def main():
    M = N = 4096
    K = 8192
    torch.manual_seed(0)
    for dt, cfg in TUNED_TL.items():
        af = torch.randn(M, K, device="cuda", dtype=torch.float32)
        bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
        a, b = af.to(dt), bf.to(dt)
        ref = af.double() @ bf.double()
        # what the benchmark actually compares against: torch on the cast inputs
        tref = torch.matmul(a.to(torch.float16), b.to(torch.float16)).double()

        name = str(dt).removeprefix("torch.")
        print(f"\n=== {name}  M=N={M} K={K} ===")

        old = dict(impl_tilelang._DEFAULT_CONFIGS[dt])
        impl_tilelang._DEFAULT_CONFIGS[dt] = cfg
        tl = impl_tilelang.run(a, b)
        impl_tilelang._DEFAULT_CONFIGS[dt] = old
        tl_def = impl_tilelang.run(a, b)
        tr = impl_triton.run(a, b)
        ct = impl_cutile.run(a, b)

        for lbl, out in (("tilelang @ tuned 256x256", tl),
                         ("tilelang @ shipped default", tl_def),
                         ("triton  @ default", tr),
                         ("cutile  @ default", ct)):
            o = out.float().double()
            print(f"{lbl:<30} max|-fp64| {(o - ref).abs().max().item():10.3f}"
                  f"   max|-torch| {(o - tref).abs().max().item():10.3f}"
                  f"   rel {( (o-ref).abs().max()/ref.abs().max() ).item():.2e}")


if __name__ == "__main__":
    main()
