# example: colsum
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def colsum(x):                                 # x: [128, N] fp32, N <= 512
    N = x.shape[1]
    out = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    ones = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    acc = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=ones, moving=xt)
    r = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out
