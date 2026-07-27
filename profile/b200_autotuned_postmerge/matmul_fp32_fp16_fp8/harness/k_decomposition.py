"""Where do TileLang's extra cycles live: the K-loop, or the prologue/epilogue?

At a fixed tile and fixed M,N the grid is constant, so sweeping K varies only
the K-loop trip count. Fit  time = intercept + slope * K:

    intercept -> prologue + epilogue (tmem->reg->global, dtype convert), O(1)
    slope     -> steady-state cost per unit of K, O(K)

If TileLang's slope is inflated, the defect is in the K-loop (descriptor
rebuild, MMA/copy overlap). If its intercept is inflated, the defect is the
epilogue. This decomposes the gap causally without touching the kernels.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.operators.matmul_fp32_fp16_fp8 import (  # noqa: E402
    impl_cutile, impl_triton)
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

KS = (1024, 2048, 4096, 8192)


def gt(fn, iters=15):
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
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def fit(ks, ts):
    """Least-squares slope/intercept of t = c + m*K."""
    n = len(ks)
    mk = sum(ks) / n
    mt = sum(ts) / n
    num = sum((k - mk) * (t - mt) for k, t in zip(ks, ts))
    den = sum((k - mk) ** 2 for k in ks)
    m = num / den
    c = mt - m * mk
    ss_tot = sum((t - mt) ** 2 for t in ts)
    ss_res = sum((t - (c + m * k)) ** 2 for k, t in zip(ks, ts))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return c, m, r2


def main():
    gen = get_generator("matmul_fp32_fp16_fp8")
    M = N = 4096
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for dt, spec in TUNED.items():
        if only and only not in str(dt):
            continue
        name = str(dt).removeprefix("torch.")
        impl_triton._DEFAULT_CONFIGS[dt] = spec["tr"]
        impl_cutile._DEFAULT_CONFIGS[dt] = SimpleNamespace(**spec["ct"])
        kern = build_matmul(warp_spec=True, **spec["tl"])

        print(f"\n===== {name}, matched tuned tile, M=N={M} =====", flush=True)
        print(f"{'K':>7}" + "".join(f"{b:>12}" for b in ("tilelang","triton","cutile")), flush=True)
        series = {"tilelang": [], "triton": [], "cutile": []}
        for K in KS:
            a, b = gen(M=M, N=N, K=K, dtype=dt)
            c = torch.empty(M, N, device=a.device, dtype=dt)
            row = []
            ms = gt(lambda: kern(a, b, c, dtype=name))
            series["tilelang"].append(ms)
            row.append(ms)
            for tag, mod in (("triton", impl_triton), ("cutile", impl_cutile)):
                mod.run(a, b)
                ms = gt(lambda: mod.run(a, b))
                series[tag].append(ms)
                row.append(ms)
            print(f"{K:>7}" + "".join(f"{v:12.4f}" for v in row), flush=True)
            del a, b, c
            torch.cuda.empty_cache()

        print(f"\n  fit  time_ms = intercept + slope*K")
        print(f"  {'backend':<12}{'intercept ms':>14}{'slope us/1k-K':>16}{'R^2':>8}")
        fits = {}
        for tag in ("tilelang", "triton", "cutile"):
            c, m, r2 = fit(KS, series[tag])
            fits[tag] = (c, m)
            print(f"  {tag:<12}{c:14.4f}{m * 1000 * 1000:16.2f}{r2:8.5f}")

        ctl, mtl = fits["tilelang"]
        for other in ("triton", "cutile"):
            co, mo = fits[other]
            print(f"\n  vs {other}:  intercept {ctl / co if co else float('nan'):5.2f}x   "
                  f"slope {mtl / mo:5.2f}x")
            K0 = 8192
            gap = (ctl + mtl * K0) - (co + mo * K0)
            print(f"    at K={K0}: total gap {gap:.4f} ms  =  "
                  f"{(ctl - co):.4f} from intercept ({100 * (ctl - co) / gap:.0f}%)  +  "
                  f"{(mtl - mo) * K0:.4f} from slope ({100 * (mtl - mo) * K0 / gap:.0f}%)")


if __name__ == "__main__":
    main()
