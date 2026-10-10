"""(h) nisa.dma_compute (DMA-engine add of several sources), nisa.core_barrier
(two physical cores of the LNC=2 logical core synchronise on a shared_hbm scratch
tensor), nisa.sendrecv and nisa.sendrecv_compute (SBUF tile exchange between the
two cores). All under the default LNC=2 launch."""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def dma_add_sbuf(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    at = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    bt = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a)
    nisa.dma_copy(dst=bt, src=b)
    r = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_compute(dst=r, srcs=[at, bt], reduce_op=nl.add)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def dma_add_hbm(a, b):
    # sources and destination all in HBM
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nisa.dma_compute(dst=out, srcs=[a, b], reduce_op=nl.add)
    return out


@nki.jit
def dma_mul_rejected(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nisa.dma_compute(dst=out, srcs=[a, b], reduce_op=nl.multiply)
    return out


@nki.jit
def barrier_exchange(dummy):
    # each core writes its own row of a shared_hbm scratch, barrier, then each core
    # reads BOTH rows and writes them into its own block of the output
    out = nl.ndarray((2, 2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    scratch = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    mine = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mine, value=float(pid + 1))
    nisa.dma_copy(dst=scratch[pid:pid + 1, :], src=mine)
    nisa.core_barrier(data=scratch, cores=(0, 1))
    both = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=both, src=scratch)
    nisa.dma_copy(dst=out[pid], src=both)
    return out


@nki.jit
def swap_tiles(dummy):
    out = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    other = 1 - pid
    send = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    recv = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=send, value=float(pid + 1))
    nisa.sendrecv(src=send, dst=recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)
    nisa.dma_copy(dst=out[pid:pid + 1, :], src=recv)
    return out


@nki.jit
def swap_add(dummy):
    out = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    other = 1 - pid
    send = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    res = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=send, value=float(pid + 1))
    nisa.sendrecv_compute(src=send, dst=res, send_to_rank=other, recv_from_rank=other, reduce_op=nl.add)
    nisa.dma_copy(dst=out[pid:pid + 1, :], src=res)
    return out


torch.manual_seed(8)
a_c = torch.randn(128, 256, dtype=torch.float32)
b_c = torch.randn(128, 256, dtype=torch.float32)
a, b = C.to_dev(a_c), C.to_dev(b_c)
dummy = C.to_dev(torch.zeros(1, 1))


def dma_sbuf():
    out = dma_add_sbuf(a, b)
    C.sync()
    ok, err = C.allclose(out, a_c + b_c, 1e-6, 1e-6)
    return {"ok": ok, "max_abs_err": err}


def dma_hbm():
    out = dma_add_hbm(a, b)
    C.sync()
    ok, err = C.allclose(out, a_c + b_c, 1e-6, 1e-6)
    return {"ok": ok, "max_abs_err": err}


def dma_mul():
    dma_mul_rejected(a, b)
    C.sync()


def barrier():
    out = barrier_exchange(dummy)
    C.sync()
    o = out.cpu()
    ok = bool((o[:, 0, :] == 1.0).all() and (o[:, 1, :] == 2.0).all())
    return {"ok": ok, "block0_rows": o[0, :, 0].tolist(), "block1_rows": o[1, :, 0].tolist()}


def swap():
    out = swap_tiles(dummy)
    C.sync()
    rows = out.cpu()[:, 0].tolist()
    return {"ok": rows == [2.0, 1.0], "rows (core0, core1)": rows}


def swap_sum():
    out = swap_add(dummy)
    C.sync()
    rows = out.cpu()[:, 0].tolist()
    return {"ok": rows == [3.0, 3.0], "rows (core0, core1)": rows}


C.run_case("dma_compute_add_sbuf", "nisa.dma_compute(dst, srcs=[a, b], reduce_op=nl.add)", "SBUF sources/destination", dma_sbuf)
C.run_case("dma_compute_add_hbm", "nisa.dma_compute HBM -> HBM", "HBM sources and destination", dma_hbm)
C.run_case("dma_compute_multiply", "nisa.dma_compute reduce_op=nl.multiply", "only nl.add is supported", dma_mul, expect_error=True)
C.run_case("core_barrier_shared_hbm", "nisa.core_barrier(data=shared_hbm_scratch, cores=(0, 1))", "each core writes one row, barrier, both cores read both rows", barrier)
C.run_case("sendrecv_swap", "nisa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id=0)", "SBUF tiles swapped between the two cores", swap)
C.run_case("sendrecv_compute_add", "nisa.sendrecv_compute(..., reduce_op=nl.add)", "both cores get the sum of the two tiles", swap_sum)
C.emit()
