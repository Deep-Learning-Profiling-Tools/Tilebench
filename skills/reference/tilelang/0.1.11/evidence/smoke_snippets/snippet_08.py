import torch
import tilelang
import tilelang.language as T
@tilelang.jit
def column_sums(M, N, BM=32, BN=64):
    @T.prim_func
    def main(X: T.Tensor((M, N), T.float32), Y: T.Tensor((N,), T.float32)):   # Y zero-initialized by caller
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            x_frag = T.alloc_fragment((BM, BN), T.float32)
            part = T.alloc_fragment((BN,), T.float32)
            T.copy(X[by * BM, bx * BN], x_frag)
            T.reduce_sum(x_frag, part, dim=0)
            for j in T.Parallel(BN):
                T.atomic_add(Y[bx * BN + j], part[j])
    return main
