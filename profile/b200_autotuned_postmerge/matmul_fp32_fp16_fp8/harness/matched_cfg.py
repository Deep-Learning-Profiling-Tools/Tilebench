"""All three backends at their *recorded tuned* tile shape, which is the same
shape for all three (fp16 256x256x64, fp8 256x256x128). Isolates codegen from
autotune-search effects, and toggles TL_DISABLE_WARP_SPECIALIZED on TileLang.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

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


# tuned winners from results/b200_autotuned_postmerge (K=8192 row)
TUNED = {
    "fp16": dict(
        torch_dtype=torch.float16,
        tl=dict(bm=256, bn=256, bk=64, gs=64, threads=256, stages=3),
        tr=dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=3),
        ct=SimpleNamespace(tm=256, tn=256, tk=64, group_size_m=8, occupancy=4),
    ),
    "fp8": dict(
        torch_dtype=torch.float8_e4m3fn,
        tl=dict(bm=256, bn=256, bk=128, gs=64, threads=256, stages=3),
        tr=dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128,
                GROUP_SIZE_M=8, num_warps=4, num_stages=3),
        ct=SimpleNamespace(tm=256, tn=256, tk=128, group_size_m=8, occupancy=4),
    ),
}


def main():
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_cutile, impl_triton

    M = N = 4096
    K = 8192
    flop = 2 * M * N * K
    torch.manual_seed(0)

    for tag, spec in TUNED.items():
        dt = spec["torch_dtype"]
        af = torch.randn(M, K, device="cuda", dtype=torch.float32)
        bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
        a, b = af.to(dt), bf.to(dt)
        ref = af.double() @ bf.double()
        c = torch.empty(M, N, device="cuda", dtype=dt)
        name = str(dt).removeprefix("torch.")

        impl_triton._DEFAULT_CONFIGS[dt] = spec["tr"]
        impl_cutile._DEFAULT_CONFIGS[dt] = spec["ct"]

        print(f"\n===== {tag}  M=N={M} K={K}  (all at the recorded tuned tile) =====")
        print(f"{'backend / config':<50}{'ms':>9}{'TFLOP/s':>10}   max_err")

        for ws in (False, True):
            print(f">>> building tilelang ws={'on' if ws else 'off'}", flush=True)
            try:
                kern = build_matmul(warp_spec=ws, **spec["tl"])
                ms = graph_time(lambda: kern(a, b, c, dtype=name))
                err = (c.float().double() - ref).abs().max().item()
                cfg = spec["tl"]
                lbl = (f"TileLang {cfg['bm']}x{cfg['bn']}x{cfg['bk']} "
                       f"thr{cfg['threads']} st{cfg['stages']} "
                       f"ws={'on' if ws else 'off'}")
                print(f"{lbl:<50}{ms:9.3f}{flop / ms / 1e9:10.1f}   {err:.3f}")
            except Exception as e:
                msg = [l for l in str(e).strip().splitlines() if l.strip()]
                print(f"{'TileLang ws=' + str(ws):<50} FAILED: {(msg[-1] if msg else '')[:60]}")

        for lbl, mod in (("Triton", impl_triton), ("cuTile", impl_cutile)):
            print(f">>> running {lbl}", flush=True)
            out = mod.run(a, b)
            ms = graph_time(lambda: mod.run(a, b))
            err = (out.float().double() - ref).abs().max().item()
            print(f"{lbl + ' (forced to same tile)':<50}{ms:9.3f}"
                  f"{flop / ms / 1e9:10.1f}   {err:.3f}")


if __name__ == "__main__":
    main()
