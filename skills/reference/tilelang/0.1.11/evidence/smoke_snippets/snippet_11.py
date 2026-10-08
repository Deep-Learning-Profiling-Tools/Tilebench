import torch
import tilelang
import tilelang.language as T
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def vector_add(N, block=256, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                C[bx * block + i] = A[bx * block + i] + B[bx * block + i]
    return main

a = torch.randn(1000, device="cuda")
b = torch.randn(1000, device="cuda")
kernel = vector_add(1000)          # JITKernel; N = 1000 is not a multiple of 256 (tail stores are skipped)
c = kernel(a, b)                   # C allocated by the call and returned
