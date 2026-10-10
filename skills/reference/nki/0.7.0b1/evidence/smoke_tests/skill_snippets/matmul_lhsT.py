# example: matmul_lhsT
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_lhsT(lhsT, rhs, out_dtype):
    # out[M, N] = lhsT[K, M].T @ rhs[K, N]; K % 128 == 0, M <= 128, N <= 512
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)      # fp32 PSUM on trn2
    KT = 128
    for k in range(K // KT):
        s = nl.ndarray((KT, M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m = nl.ndarray((KT, N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=lhsT[k * KT:(k + 1) * KT, :])
        nisa.dma_copy(dst=m, src=rhs[k * KT:(k + 1) * KT, :])
        nisa.nc_matmul(dst=acc, stationary=s, moving=m, accumulate=(k > 0))
    res = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=acc)          # PSUM -> SBUF, cast to out_dtype
    nisa.dma_copy(dst=out, src=res)
    return out
