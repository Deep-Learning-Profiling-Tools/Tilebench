# example: add_kernel
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def add_kernel(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = nl.tile_size.pmax                      # 128
    rows, cols = a.shape
    for i in range(rows // P):                 # Python range: unrolled at trace time
        a_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, cols), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=b_t, src=b[i * P:(i + 1) * P, :])
        c_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_t, data1=a_t, data2=b_t, op=nl.add)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=c_t)
    return out
