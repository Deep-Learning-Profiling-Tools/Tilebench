from __future__ import annotations
import torch
import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[-1])
def k(N, block=128):
    @T.prim_func
    def main(A: T.Tensor((N,), T.float32), B: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                B[bx * block + i] = A[bx * block + i] + 1
    return main
kern = k(256)
