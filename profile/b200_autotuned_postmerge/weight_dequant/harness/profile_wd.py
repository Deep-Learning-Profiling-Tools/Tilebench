#!/usr/bin/env python3
"""Single-launch weight_dequant harness for NCU, one backend per invocation."""
import argparse
import sys

import torch

sys.path.insert(0, "/home/arustagi/repos/Tilebench")
from benchmarks.operators.weight_dequant import (  # noqa: E402
    impl_tilelang as tl_wd, impl_triton as tr_wd, impl_cutile as ct_wd)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=("tilelang", "triton", "cutile"))
    p.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--TILE_SIZE", type=int, default=128)
    a = p.parse_args()

    dt = DTYPES[a.dtype]
    M = N = a.M
    ts = a.TILE_SIZE
    sr = (M + ts - 1) // ts
    sc = (N + ts - 1) // ts
    X = torch.randn(M, N, device="cuda", dtype=dt)
    S = torch.rand(sr, sc, device="cuda", dtype=dt) + 0.5

    mod = {"tilelang": tl_wd, "triton": tr_wd, "cutile": ct_wd}[a.backend]

    def once():
        return mod.run(X, S, M, N, ts, autotune=False)

    for _ in range(3):
        once()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    out = once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    # correctness against a plain torch expansion
    ref = X * S.repeat_interleave(ts, 0)[:M, None].squeeze(1) if False else None
    r = torch.arange(M, device="cuda") // ts
    c = torch.arange(N, device="cuda") // ts
    ref = X * S[r][:, c]
    err = (out.view(M, N).float() - ref.float()).abs().max().item()
    print(f"backend={a.backend} dtype={a.dtype} M={M} TILE_SIZE={ts} "
          f"max_abs_err={err:.6f}", flush=True)


if __name__ == "__main__":
    main()
