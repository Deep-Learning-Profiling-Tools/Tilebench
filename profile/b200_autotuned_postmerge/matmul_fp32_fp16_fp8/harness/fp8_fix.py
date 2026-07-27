"""Confirm the fp8 BLOCK_SIZE_N=256 bug and find the fastest *correct* config.

Findings this checks:
  1. fp8 is wrong iff bn==256; the upper 128 columns of each N tile are bad.
  2. fp16 at the same bn==256 is fine, so it is fp8-specific.
  3. What the wrong half actually contains.
  4. Best correct fp8 config vs the (wrong) 256x256x128 the autotuner picked.
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


def graph_time(fn, iters=20):
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
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    torch.manual_seed(0)

    # ---- 1/2/3: structure, at a small shape -------------------------------
    M = N = 512
    K = 1024
    af = torch.randn(M, K, device="cuda", dtype=torch.float32)
    bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a8, b8 = af.to(DT), bf.to(DT)
    ref8 = a8.float().double() @ b8.float().double()

    c = torch.empty(M, N, device="cuda", dtype=DT)
    kern = build_matmul(bm=256, bn=256, bk=128, gs=64, threads=256,
                        stages=3, warp_spec=True)
    kern(a8, b8, c, dtype="float8_e4m3fn")
    o = c.float().double()
    rq = ref8.float().to(DT).float().double()

    lo = (o[:, 0:128] - rq[:, 0:128]).abs().max().item()
    hi = (o[:, 128:256] - rq[:, 128:256]).abs().max().item()
    print(f"fp8 bn=256: cols   0-127 max err {lo:.4f}   cols 128-255 max err {hi:.4f}")

    bad = o[:, 128:256]
    print(f"  bad half: absmax {bad.abs().max().item():.2f}  "
          f"zeros {(bad == 0).float().mean().item() * 100:.1f}%  "
          f"mean {bad.mean().item():.3f}  (good half mean "
          f"{o[:, 0:128].mean().item():.3f})")
    # is the bad half a copy of the good half, i.e. the same B columns re-used?
    d_self = (bad - o[:, 0:128]).abs().max().item()
    print(f"  bad half vs good half of same tile: max diff {d_self:.4f}"
          f"   -> {'IDENTICAL (B columns re-read)' if d_self < 1e-9 else 'not a copy'}")

    # fp16 control at the same tile
    a16, b16 = af.to(torch.float16), bf.to(torch.float16)
    ref16 = a16.float().double() @ b16.float().double()
    c16 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    k16 = build_matmul(bm=256, bn=256, bk=64, gs=64, threads=256,
                       stages=3, warp_spec=True)
    k16(a16, b16, c16, dtype="float16")
    e16 = ((c16.float().double() - ref16).abs().max() / ref16.abs().max()).item()
    print(f"fp16 control at bn=256: rel err {e16:.2e}  "
          f"-> {'OK, fp8-specific' if e16 < 0.2 else 'also wrong'}")

    # ---- 4: perf of correct fp8 configs at the benchmark shape ------------
    M = N = 4096
    K = 8192
    flop = 2 * M * N * K
    af = torch.randn(M, K, device="cuda", dtype=torch.float32)
    bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a8, b8 = af.to(DT), bf.to(DT)
    ref8 = a8.float().double() @ b8.float().double()
    relmax = ref8.abs().max()
    c = torch.empty(M, N, device="cuda", dtype=DT)

    print(f"\n=== fp8 configs at M=N={M} K={K} (ws=on) ===")
    print(f"{'config':<34}{'ms':>9}{'TFLOP/s':>10}{'rel err':>12}   ok")
    cands = [
        (256, 256, 128, 256), (256, 256, 64, 256),   # what the tuner picked
        (256, 128, 128, 256), (256, 128, 64, 256),
        (128, 128, 128, 256), (128, 128, 64, 256),
        (256, 128, 128, 128), (128, 128, 128, 128),
    ]
    best = None
    for bm, bn, bk, thr in cands:
        try:
            k = build_matmul(bm=bm, bn=bn, bk=bk, gs=64, threads=thr,
                             stages=3, warp_spec=True)
            ms = graph_time(lambda: k(a8, b8, c, dtype="float8_e4m3fn"))
            e = ((c.float().double() - ref8).abs().max() / relmax).item()
            ok = e < 0.2
            tf = flop / ms / 1e9
            if ok and (best is None or tf > best[1]):
                best = (f"{bm}x{bn}x{bk} thr{thr}", tf)
            print(f"{f'{bm}x{bn}x{bk} thr{thr}':<34}{ms:9.3f}{tf:10.1f}"
                  f"{e:12.2e}   {'OK' if ok else 'WRONG'}")
        except Exception as ex:
            msg = [l for l in str(ex).strip().splitlines() if l.strip()]
            print(f"{f'{bm}x{bn}x{bk} thr{thr}':<34} FAILED "
                  f"{(msg[-1] if msg else '')[:44]}")

    for lbl, mod in (("Triton  (tuned default)", impl_triton),
                     ("cuTile  (tuned default)", impl_cutile)):
        out = mod.run(a8, b8)
        ms = graph_time(lambda: mod.run(a8, b8))
        e = ((out.float().double() - ref8).abs().max() / relmax).item()
        print(f"{lbl:<34}{ms:9.3f}{flop / ms / 1e9:10.1f}{e:12.2e}   OK")

    if best:
        print(f"\nfastest CORRECT tilelang fp8: {best[0]} at {best[1]:.0f} TFLOP/s")


if __name__ == "__main__":
    main()
