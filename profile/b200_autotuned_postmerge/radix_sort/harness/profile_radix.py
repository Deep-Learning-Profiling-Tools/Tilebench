#!/usr/bin/env python3
"""One full radix_sort run() under NCU; all kernels captured, attributed by name."""
import argparse
import sys

import torch

sys.path.insert(0, "/home/arustagi/repos/Tilebench")
from benchmarks.operators.radix_sort import (  # noqa: E402
    impl_tilelang as tl_r, impl_triton as tr_r, impl_cutile as ct_r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=("tilelang", "triton", "cutile"))
    p.add_argument("--N", type=int, default=1_000_000)
    a = p.parse_args()

    torch.manual_seed(0)
    X = torch.randint(0, 2**30, (a.N,), device="cuda", dtype=torch.int32)
    mod = {"tilelang": tl_r, "triton": tr_r, "cutile": ct_r}[a.backend]

    out = mod.run(X, a.N, autotune=False)   # warmup / compile
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    out = mod.run(X, a.N, autotune=False)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    ref = torch.sort(X).values
    ok = torch.equal(out.view(-1), ref)
    print(f"backend={a.backend} N={a.N} sorted_correct={ok}", flush=True)


if __name__ == "__main__":
    main()
