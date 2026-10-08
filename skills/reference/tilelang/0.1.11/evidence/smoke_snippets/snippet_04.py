import torch
import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[-1])
def matmul_tmem(M, N, K, block_M=128, block_N=128, block_K=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.bfloat16), B: T.Tensor((N, K), T.bfloat16),
             C: T.Tensor((M, N), T.bfloat16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            C_tmem = T.alloc_tmem([block_M, block_N], T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, clear_accum=k == 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    return main
