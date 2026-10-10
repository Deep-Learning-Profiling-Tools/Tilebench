"""(q) Broadcasting one row [1, F] to all 128 partitions (the partition-axis
counterpart of tensor_scalar's free-axis broadcast):
 - DMA with a zero partition stride: row.ap(pattern=[[0, 128], [1, F]], offset=0)
 - TensorE: ones[1, 128].T @ row[1, F] (contraction K = 1) -> PSUM [128, F]
 - nl.broadcast_to(tile[1, F], (128, F))"""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def bcast_dma(row):
    F = row.shape[1]
    out = nl.ndarray((128, F), dtype=row.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((128, F), dtype=row.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=row.ap(pattern=[[0, 128], [1, F]], offset=0))
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def bcast_matmul(row):
    F = row.shape[1]
    out = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.shared_hbm)
    r = nl.ndarray((1, F), dtype=row.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=row)
    ones = nl.ndarray((1, 128), dtype=row.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    p = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=ones, moving=r)
    s = nl.ndarray((128, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=p)
    nisa.dma_copy(dst=out, src=s)
    return out


@nki.jit
def bcast_nl(row):
    F = row.shape[1]
    out = nl.ndarray((128, F), dtype=row.dtype, buffer=nl.shared_hbm)
    r = nl.load(row)
    b = nl.broadcast_to(r, (128, F))
    nl.store(out, b)
    return out


torch.manual_seed(15)
row_c = torch.randn(1, 256, dtype=torch.float32)
row = C.to_dev(row_c)
ref = row_c.expand(128, 256)


def case(kern, exact):
    def f():
        out = kern(row)
        C.sync()
        if exact:
            ok = bool(torch.equal(out.cpu(), ref))
            return {"ok": ok, "exact": ok}
        ok, err = C.allclose(out, ref, 1e-3, 1e-3)
        return {"ok": ok, "max_abs_err": err}
    return f


C.run_case("dma_zero_partition_stride", "row.ap(pattern=[[0, 128], [1, F]], offset=0)", "one HBM row replicated into 128 partitions by DMA", case(bcast_dma, True))
C.run_case("ones_matmul_k1", "nisa.nc_matmul(stationary=ones[1,128], moving=row[1,F])", "K=1 outer product broadcast to PSUM (fp32 inputs)", case(bcast_matmul, False))
C.run_case("nl_broadcast_to", "nl.broadcast_to(tile[1, F], (128, F))", "high-level partition broadcast", case(bcast_nl, True))
C.emit()
