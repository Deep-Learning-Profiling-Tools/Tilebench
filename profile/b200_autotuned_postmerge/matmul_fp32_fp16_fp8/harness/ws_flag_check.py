"""Effect of TL_DISABLE_WARP_SPECIALIZED on the shipped impl, measured on the
benchmark's own inputs (1/sqrt(K)-scaled) and its own verifier tolerances.

Toggles the flag by rebuilding the kernel both ways via the harness variant,
and separately confirms impl_tilelang.run (which now has the flag False) still
verifies for every dtype.
"""

import sys
from pathlib import Path

import torch
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_tilelang, impl_torch, impl_triton)
from core.verifier import verify  # noqa: E402
from data.tensors import get_generator  # noqa: E402
from tilelang_variants import build_matmul  # noqa: E402

CFG = yaml.safe_load(
    (ROOT / "benchmarks/operators/matmul_fp32_fp16_fp8/config.yaml").read_text())
ATOL, RTOL = float(CFG["verify"]["atol"]), float(CFG["verify"]["rtol"])

# recorded tuned winners (K=8192 row of b200_autotuned_postmerge)
TUNED = {
    torch.float16: dict(bm=256, bn=256, bk=64, gs=64, threads=256, stages=3),
    torch.float8_e4m3fn: dict(bm=256, bn=256, bk=128, gs=64, threads=256, stages=3),
}


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    gen = get_generator("matmul_fp32_fp16_fp8")
    M = N = 4096
    K = 8192
    flop = 2 * M * N * K
    print(f"benchmark inputs (1/sqrt(K)-scaled), verify atol={ATOL} rtol={RTOL}")

    for dt, cfg in TUNED.items():
        name = str(dt).removeprefix("torch.")
        a, b = gen(M=M, N=N, K=K, dtype=dt)
        ref = impl_torch.run(a, b)
        torch.cuda.synchronize()
        c = torch.empty(M, N, device=a.device, dtype=dt)

        print(f"\n=== {name}  M=N={M} K={K}  at tuned "
              f"{cfg['bm']}x{cfg['bn']}x{cfg['bk']} ===")
        print(f"{'variant':<34}{'TFLOP/s':>10}{'maxabs':>10}   verify")
        for ws in (False, True):
            kern = build_matmul(warp_spec=ws, **cfg)
            ms = timed(lambda: kern(a, b, c, dtype=name))
            torch.cuda.synchronize()
            ok, _ = verify(c, ref, atol=ATOL, rtol=RTOL)
            d = (c.float() - ref.float()).abs().max().item()
            print(f"{'warp_spec=' + ('on' if ws else 'off (was shipped)'):<34}"
                  f"{flop / ms / 1e9:10.1f}{d:10.4f}   {'PASS' if ok else 'FAIL'}")

    print("\n=== shipped impl_tilelang.run with the new flag, all dtypes ===")
    print(f"{'dtype':<16}{'TFLOP/s':>10}{'maxabs':>10}   verify")
    for dt in (torch.float32, torch.float16, torch.float8_e4m3fn):
        a, b = gen(M=M, N=N, K=K, dtype=dt)
        ref = impl_torch.run(a, b)
        torch.cuda.synchronize()
        out = impl_tilelang.run(a, b)
        ms = timed(lambda: impl_tilelang.run(a, b))
        ok, err = verify(out, ref, atol=ATOL, rtol=RTOL)
        d = (out.float() - ref.float()).abs().max().item()
        print(f"{str(dt).removeprefix('torch.'):<16}{flop / ms / 1e9:10.1f}"
              f"{d:10.4f}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("   ", err.strip().splitlines()[0][:90])


if __name__ == "__main__":
    main()
