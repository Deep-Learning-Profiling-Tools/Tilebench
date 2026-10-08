import torch
import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[1])
def row_cumsum(M, N):
    @T.prim_func
    def main(X: T.Tensor((M, N), T.float32), Y: T.Tensor((M, N), T.float32)):
        with T.Kernel(M, threads=128) as bm:
            row = T.alloc_shared((N,), T.float32)
            T.copy(X[bm, 0:N], row)
            T.cumsum(row, dim=0)                    # in place
            T.copy(row, Y[bm, 0:N])
    return main
