import torch
import tilelang
import tilelang.language as T
a = torch.randn(4096, device="cuda")
@tilelang.jit(out_idx=[-1])                       # lazy
def add_one(N, block=128, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                B[bx * block + i] = A[bx * block + i] + 1
    return main

kernel = add_one(4096)                            # JITKernel
b = kernel(a)

@tilelang.jit                                     # eager
def add_one_eager(A, block=128, dtype=T.float32):
    N = T.const("N")                              # bound from A.shape[0]
    A: T.Tensor((N,), dtype)                      # or A: T.Tensor[[N], dtype]
    B = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
        for i in T.Parallel(block):
            B[bx * block + i] = A[bx * block + i] + 1
    return B

b = add_one_eager(a)                              # compiles if needed, runs, returns B
kernel = add_one_eager.compile(a)                 # or .compile(N=4096): JITKernel, not run
