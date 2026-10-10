# example: add_one_any_shape
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def add_one_any_shape(x):
    R, F = x.shape
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range((R + P - 1) // P):
        r0 = i * P
        rows = min(P, R - r0)                     # Python int at trace time
        t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:rows, :], src=x[r0:r0 + rows, :])
        nisa.tensor_scalar(dst=t[0:rows, :], data=t[0:rows, :], op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=out[r0:r0 + rows, :], src=t[0:rows, :])
    return out
