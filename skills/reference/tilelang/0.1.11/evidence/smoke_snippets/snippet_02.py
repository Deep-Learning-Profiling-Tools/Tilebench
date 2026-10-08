import torch
import tilelang
import tilelang.language as T
@tilelang.jit
def row_max(N, block_M=4, block_N=128, dtype=T.float32):
    M = T.dynamic("M")

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M,), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bm:
            x_frag = T.alloc_fragment((block_M, block_N), dtype)
            m_part = T.alloc_fragment((block_M,), dtype)
            m_acc = T.alloc_fragment((block_M,), dtype)
            T.fill(m_acc, -T.infinity(dtype))
            for kn in T.serial(T.ceildiv(N, block_N)):
                T.copy(X[bm * block_M, kn * block_N], x_frag)     # out-of-range columns read as 0
                for i, j in T.Parallel(block_M, block_N):
                    x_frag[i, j] = T.if_then_else(kn * block_N + j < N, x_frag[i, j], -T.infinity(dtype))
                T.reduce_max(x_frag, m_part, dim=1)
                for i in T.Parallel(block_M):
                    m_acc[i] = T.max(m_acc[i], m_part[i])
            T.copy(m_acc, Y[bm * block_M])
    return main

x = torch.randn(37, 1000, device="cuda")
y = torch.empty(37, device="cuda")
row_max(1000)(x, y)                # no out_idx: the kernel writes into y
