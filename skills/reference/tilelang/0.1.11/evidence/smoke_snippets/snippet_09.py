import torch
import tilelang
import tilelang.language as T
@T.macro
def square(x):
    return x * x

@tilelang.jit(out_idx=[1])
def squares(N, block=128):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                Y[bx * block + i] = square(X[bx * block + i])
    return main
