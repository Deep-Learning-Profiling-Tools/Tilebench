#!/usr/bin/env python3
"""Single-launch sigmoid harness for NCU, one backend per invocation."""
import argparse
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/arustagi/repos/Tilebench")
from benchmarks.operators.sigmoid import (  # noqa: E402
    impl_tilelang as tl_s, impl_triton as tr_s, impl_cutile as ct_s)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=("tilelang", "triton", "cutile"))
    p.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    p.add_argument("--N", type=int, default=50_000_000)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--warps", type=int, default=8)
    p.add_argument("--occupancy", type=int, default=16)
    a = p.parse_args()

    dt = DTYPES[a.dtype]
    X = torch.randn(a.N, device="cuda", dtype=dt)

    if a.backend == "tilelang":
        mod = tl_s
        mod._DEFAULT_CONFIG = {"BLOCK_SIZE": a.block, "threads": a.threads}
    elif a.backend == "triton":
        mod = tr_s
        mod._DEFAULT_CONFIG = {"BLOCK_SIZE": a.block, "num_warps": a.warps,
                               "num_stages": 1}
    else:
        mod = ct_s
        mod._DEFAULT_CONFIG = SimpleNamespace(tile=a.block,
                                              occupancy=a.occupancy)

    def once():
        return mod.run(X, a.N, autotune=False)

    for _ in range(3):
        once()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    out = once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    ref = torch.sigmoid(X.float())
    err = (out.float() - ref).abs().max().item()
    print(f"backend={a.backend} dtype={a.dtype} N={a.N} block={a.block} "
          f"max_abs_err={err:.6e}", flush=True)


if __name__ == "__main__":
    main()
