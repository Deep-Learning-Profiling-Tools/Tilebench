"""Verify the fp8 fix through the shipped impl: default path and autotune path,
for every dtype, against Triton/cuTile.
"""

import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_tilelang, impl_triton)


def main():
    M = N = 4096
    K = 8192
    flop = 2 * M * N * K
    torch.manual_seed(0)

    print("config space sizes from matmul_configs():")
    for name in ("float32", "float16", "float8_e4m3fn"):
        n = len(impl_tilelang.matmul_configs(dtype=name))
        bad = [c for c in impl_tilelang.matmul_configs(dtype=name)
               if c["BLOCK_SIZE_N"] == 256]
        print(f"  {name:<16} {n:3d} configs, {len(bad):3d} with BLOCK_SIZE_N=256")

    for dt in (torch.float16, torch.float8_e4m3fn):
        af = torch.randn(M, K, device="cuda", dtype=torch.float32)
        bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
        a, b = af.to(dt), bf.to(dt)
        ref = a.float().double() @ b.float().double()
        relmax = ref.abs().max()
        name = str(dt).removeprefix("torch.")
        print(f"\n=== {name}  M=N={M} K={K} ===")
        print(f"{'path':<34}{'TFLOP/s':>10}{'rel err':>12}   ok")

        def show(lbl, out, ms=None):
            e = ((out.float().double() - ref).abs().max() / relmax).item()
            tf = f"{flop / ms / 1e9:10.1f}" if ms else " " * 10
            print(f"{lbl:<34}{tf}{e:12.2e}   {'OK' if e < 0.2 else 'WRONG'}")

        def timed(fn):
            out = fn()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            st.record()
            for _ in range(10):
                fn()
            en.record()
            torch.cuda.synchronize()
            return out, st.elapsed_time(en) / 10

        out, ms = timed(lambda: impl_tilelang.run(a, b))
        show("tilelang default", out, ms)
        out, ms = timed(lambda: impl_triton.run(a, b))
        show("triton default", out, ms)
        out, ms = timed(lambda: impl_cutile.run(a, b))
        show("cutile default", out, ms)

        print(">>> autotuning tilelang (this is TileLang's own tuner)", flush=True)
        out = impl_tilelang.run(a, b, autotune=True)
        cfg = impl_tilelang.get_last_config()
        show(f"tilelang autotuned", out)
        print(f"     winner: {cfg}")


if __name__ == "__main__":
    main()
