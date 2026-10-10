# example: fori_rows
import nki
import nki.language as nl
import nki.isa as nisa

P = 128


@nki.jit
def fori_rows(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    N = x.shape[1]
    T = x.shape[0] // P
    x3 = x.reshape((T, P, N))                  # HBM view: tile index on axis 0
    o3 = out.reshape((T, P, N))

    def body(i):                               # (a) i = 0, P, 2P, ... thanks to step=P
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[nl.ds(i, P), :])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=out[nl.ds(i, P), :], src=t)

    nl.fori_loop(0, x.shape[0], body, step=P)

    def body2(j):                              # (b) runtime index on a reshaped view
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=o3[j])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=2.0)
        nisa.dma_copy(dst=o3[j], src=t)

    nl.fori_loop(0, T, body2)

    def body3(j):                              # (c) explicit access pattern + scalar_offset
        src = o3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=j, indirect_dim=0)
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=src)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.subtract, operand0=1.0)
        nisa.dma_copy(dst=o3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=j, indirect_dim=0), src=t)

    nl.fori_loop(0, T, body3)
    return out                                 # = (x + 1) * 2 - 1
