"""Does TileLang's fixed num_stages=3 explain the tensor-pipe starvation?

NCU shows TileLang holding the tcgen05 / tensor-memory pipe active only ~34% of
the time vs cuTile's ~69%, while allocating 96 KB of shared memory against
cuTile's 224 KB. 96 KB = 3 stages x 32 KB (A 128x64 + B 64x128, fp16). Sweep the
stage count and see where the pipe utilisation and runtime go.
"""

import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tilelang_variants import build_matmul  # noqa: E402


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
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_cutile, impl_triton

    M = N = 4096
    K = 8192
    flop = 2 * M * N * K
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda", dtype=torch.float32)).half()
    b = (torch.randn(K, N, device="cuda", dtype=torch.float32)).half()
    ref = a.float().double() @ b.float().double()
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    print(f"fp16  M=N={M}  K={K}\n")
    print(f"{'config':<44}{'time':>10}{'TFLOP/s':>10}{'smem/CTA':>11}  max_err")
    for stages in (3, 4, 5, 6, 7, 8):
        cfg = dict(bm=128, bn=128, bk=64, gs=8, threads=256, stages=stages)
        for ws in (True,):
            try:
                kern = build_matmul(warp_spec=ws, **cfg)
                ms = graph_time(lambda: kern(a, b, c, dtype="float16"))
                err = (c.float().double() - ref).abs().max().item()
                smem = 2 * 128 * 64 * 2 * stages  # A + B tiles, fp16
                tag = f"TileLang 128x128x64 stages={stages} ws=on"
                print(f"{tag:<44}{ms:9.3f}m{flop / ms / 1e9:10.1f}"
                      f"{smem // 1024:>9} KB  {err:.3f}")
            except Exception as e:
                msg = [l for l in str(e).strip().splitlines() if l.strip()]
                print(f"{'TileLang stages=' + str(stages):<44} FAILED: "
                      f"{(msg[-1] if msg else '')[:70]}")

    print()
    for tag, mod in (("cuTile (impl default)", impl_cutile),
                     ("Triton (impl default)", impl_triton)):
        out = mod.run(a, b)
        ms = graph_time(lambda: mod.run(a, b))
        err = (out.float().double() - ref).abs().max().item()
        print(f"{tag:<44}{ms:9.3f}m{flop / ms / 1e9:10.1f}{'':>12}  {err:.3f}")


if __name__ == "__main__":
    main()
