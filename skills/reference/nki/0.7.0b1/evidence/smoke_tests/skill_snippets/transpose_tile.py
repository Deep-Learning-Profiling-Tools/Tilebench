# example: transpose_tile
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def transpose_tile(x):                         # x: [128, 64] bf16 -> out: [64, 128]
    out = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    tp = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.psum)   # same dtype as input
    nisa.nc_transpose(dst=tp, data=xt, engine=nisa.engine.tensor)
    ts = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ts, src=tp)
    nisa.dma_copy(dst=out, src=ts)
    return out
