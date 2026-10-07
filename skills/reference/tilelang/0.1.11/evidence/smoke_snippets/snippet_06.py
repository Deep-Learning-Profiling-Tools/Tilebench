import torch
import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[-1])
def scale_add(block=256, dtype=T.float32):
    N = T.dynamic("N")

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype), alpha: T.float32,
             Z: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                Z[bx * block + i] = alpha * X[bx * block + i] + Y[bx * block + i]
    return main

_scale_add_kernel = scale_add()          # compiled once; N is bound at every call

def run(x, y, alpha=2.0):
    return _scale_add_kernel(x.contiguous(), y.contiguous(), alpha)
