"""Head-to-head with the warp-spec flag flipped: all three backends, benchmark
inputs, graph-timed, at each backend's own recorded tuned config.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_torch, impl_triton)
from core.verifier import verify  # noqa: E402
from data.tensors import get_generator  # noqa: E402
from tilelang_variants import build_matmul  # noqa: E402

TUNED = {
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


def gt(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    gen = get_generator("matmul_fp32_fp16_fp8")
    M = N = 4096
    print("benchmark inputs, CUDA-graph timed, each backend at its tuned config")
    for K in (2048, 8192, 20480):
        flop = 2 * M * N * K
        print(f"\n===== M=N={M}  K={K} =====")
        print(f"{'backend':<28}{'TFLOP/s':>10}{'vs TL':>8}{'maxabs':>10}   verify")
        for dt, spec in TUNED.items():
            name = str(dt).removeprefix("torch.")
            a, b = gen(M=M, N=N, K=K, dtype=dt)
            ref = impl_torch.run(a, b)
            torch.cuda.synchronize()
            impl_triton._DEFAULT_CONFIGS[dt] = spec["tr"]
            impl_cutile._DEFAULT_CONFIGS[dt] = SimpleNamespace(**spec["ct"])

            c = torch.empty(M, N, device=a.device, dtype=dt)
            kern = build_matmul(warp_spec=True, **spec["tl"])
            res = {}
            ms = gt(lambda: kern(a, b, c, dtype=name))
            ok, _ = verify(c, ref, atol=5.0, rtol=0.1)
            res["tilelang (ws=on)"] = (flop / ms / 1e9,
                                       (c.float() - ref.float()).abs().max().item(), ok)
            for lbl, mod in (("triton", impl_triton), ("cutile", impl_cutile)):
                out = mod.run(a, b)
                ms = gt(lambda: mod.run(a, b))
                ok, _ = verify(out, ref, atol=5.0, rtol=0.1)
                res[lbl] = (flop / ms / 1e9,
                            (out.float() - ref.float()).abs().max().item(), ok)
            ms = gt(lambda: impl_torch.run(a, b))
            res["torch"] = (flop / ms / 1e9, 0.0, True)

            base = res["tilelang (ws=on)"][0]
            print(f"  -- {name} --")
            for lbl, (tf, d, ok) in res.items():
                print(f"  {lbl:<26}{tf:10.1f}{tf / base:8.2f}{d:10.4f}   "
                      f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
