#!/usr/bin/env python3
"""Single-launch BMM harness for NCU, one backend per invocation.

Each backend runs at the tile the postmerge autotune recorded for it, so the
comparison is tuned-vs-tuned. Only the region between profiler.start/stop is
captured, so NCU sees exactly one kernel launch per backend.
"""
import argparse
import sys

import torch

sys.path.insert(0, "/home/arustagi/repos/Tilebench")
from benchmarks.operators.batched_matmul import (  # noqa: E402
    impl_tilelang as tl_bmm, impl_triton as tr_bmm, impl_cutile as ct_bmm)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Winners recorded in results/b200_autotuned_postmerge/operators/batched_matmul.json
TUNED_TL = {
    "fp16": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                 GROUPSIZE=1, threads=128, num_stages=4),
    "bf16": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                 GROUPSIZE=1, threads=128, num_stages=4),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=("tilelang", "triton", "cutile", "torch"))
    p.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    p.add_argument("--BATCH", type=int, default=32)
    p.add_argument("--M", type=int, default=640)
    p.add_argument("--bm", type=int, default=None)
    p.add_argument("--bn", type=int, default=None)
    p.add_argument("--bk", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--stages", type=int, default=None)
    a = p.parse_args()

    dt = DTYPES[a.dtype]
    B, M = a.BATCH, a.M
    N = K = M
    A = (torch.randn(B * M * K, device="cuda", dtype=dt) * 0.05)
    Bf = (torch.randn(B * K * N, device="cuda", dtype=dt) * 0.05)
    A3, B3 = A.view(B, M, K), Bf.view(B, K, N)
    C3 = torch.empty(B, M, N, device="cuda", dtype=dt)

    if a.backend == "tilelang":
        cfg = dict(TUNED_TL.get(a.dtype, TUNED_TL["fp16"]))
        for k, v in (("BLOCK_SIZE_M", a.bm), ("BLOCK_SIZE_N", a.bn),
                     ("BLOCK_SIZE_K", a.bk), ("threads", a.threads),
                     ("num_stages", a.stages)):
            if v is not None:
                cfg[k] = v
        kern = tl_bmm.bmm_kernel.compile(A3, B3, C3, dtype=str(dt).removeprefix("torch."), **cfg)

        def once():
            kern(A3, B3, C3)
        label = "x".join(str(cfg[k]) for k in
                         ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
        label += f" t{cfg['threads']} s{cfg['num_stages']}"
    elif a.backend == "torch":
        def once():
            torch.bmm(A3, B3)
        label = "cublas"
    else:
        mod = {"triton": tr_bmm, "cutile": ct_bmm}[a.backend]

        def once():
            mod.run(A, Bf, B, M, N, K, autotune=False)
        label = "default-cfg"

    for _ in range(3):
        once()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    ref = torch.bmm(A3.float(), B3.float())
    once()
    torch.cuda.synchronize()
    err = (C3.float() - ref).abs().max().item() if a.backend == "tilelang" else -1.0
    print(f"backend={a.backend} dtype={a.dtype} BATCH={B} M={M} cfg=[{label}] "
          f"max_abs_err={err:.4f}", flush=True)


if __name__ == "__main__":
    main()
