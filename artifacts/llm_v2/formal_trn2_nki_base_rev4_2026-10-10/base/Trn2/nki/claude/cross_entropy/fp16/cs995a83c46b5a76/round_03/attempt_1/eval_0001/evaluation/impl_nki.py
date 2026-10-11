import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
NUM_PROGRAMS = 2          # a bare kernel call runs with LNC=2 on this host
MAX_CHUNK_ROWS = 8        # max rows per partition per chunk (ramp 1,2,4,8,8,...)
GPSIMD_FOLD = 1           # first max-fold level (512 -> 256) on GpSimd for chunks >= 1
LOAD_GROUPS = 4           # each chunk load split into this many partition groups

_LAST_CONFIG = {}


def _chunk_start(c, kmax):
    start = 0
    size = 1
    i = 0
    while i < c:
        start = start + size
        size = min(size * 2, kmax)
        i = i + 1
    return start


def _chunk_len(c, n, kmax):
    start = 0
    size = 1
    i = 0
    while i < c:
        start = start + size
        size = min(size * 2, kmax)
        i = i + 1
    return min(size, n - start)


def _num_chunks(n, kmax):
    cnt = 0
    start = 0
    size = 1
    while start < n:
        start = start + size
        size = min(size * 2, kmax)
        cnt = cnt + 1
    return cnt


def _load_chunk(lg3, p0, c, n, kmax, C, dt, ngrp):
    j0 = _chunk_start(c, kmax)
    k = _chunk_len(c, n, kmax)
    t = nl.ndarray((128, k, C), dtype=dt, buffer=nl.sbuf)
    gp = 128 // ngrp
    for g in range(ngrp):
        q0 = g * gp
        nisa.dma_copy(dst=t[q0:q0 + gp, :, :],
                      src=lg3[p0 + q0:p0 + q0 + gp, j0:j0 + k, :])
    return t


def _chunk_negmax(x3, nmx, c, n, kmax, C, dt, use_fold):
    j0 = _chunk_start(c, kmax)
    k = _chunk_len(c, n, kmax)
    mxc = nl.ndarray((128, k), dtype=dt, buffer=nl.sbuf)   # max is exact in fp16
    if use_fold == 1:
        h = C // 2
        h1 = nl.ndarray((128, k, h), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h1, data1=x3[:, :, 0:h], data2=x3[:, :, h:C],
                           op=nl.maximum, engine=nisa.engine.gpsimd)
        nisa.tensor_reduce(dst=mxc, op=nl.maximum, data=h1, axis=2)
    else:
        nisa.tensor_reduce(dst=mxc, op=nl.maximum, data=x3, axis=2)
    # -max in fp32 (exp bias)
    nisa.tensor_scalar(dst=nmx[:, j0:j0 + k], data=mxc, op0=nl.multiply,
                       operand0=-1.0, engine=nisa.engine.vector)


def _chunk_exp_gather(x3, nmx, ssum, xt, tgt_f, c, n, kmax, C, dt):
    j0 = _chunk_start(c, kmax)
    k = _chunk_len(c, n, kmax)
    esc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    sel = nl.ndarray((128, C), dtype=dt, buffer=nl.sbuf)
    for j in range(k):
        r = j0 + j
        # sum_j exp(x - max) in fp32 on ScalarE (fused row sum)
        nisa.activation(dst=esc, op=nl.exp, data=x3[:, j, :],
                        bias=nmx[:, r:r + 1], scale=1.0,
                        reduce_op=nl.add, reduce_res=ssum[:, r:r + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # target logit extracted from the loaded row
        nisa.range_select(dst=sel, on_true_tile=x3[:, j, :],
                          comp_op0=nl.greater_equal, comp_op1=nl.less_equal,
                          bound0=tgt_f[:, r:r + 1], bound1=tgt_f[:, r:r + 1],
                          reduce_cmd=nisa.reduce_cmd.reset_reduce,
                          reduce_res=xt[:, r:r + 1], reduce_op=nl.maximum)


@nki.jit
def _ce_kernel(logits, targets, kmax, fold, ngrp):
    B = logits.shape[0]
    C = logits.shape[1]
    P = 128
    NP = nl.num_programs()
    pid = nl.program_id(0)
    n = B // (NP * P)          # rows per partition handled by this program
    p0 = pid * P
    dt = logits.dtype
    nch = _num_chunks(n, kmax)
    use_fold = 0
    if fold == 1:
        if C % 2 == 0:
            use_fold = 1

    out = nl.ndarray((B,), dtype=dt, buffer=nl.shared_hbm)

    # row index = pid*(P*n) + p*n + j  -> contiguous rows per partition
    lg3 = logits.reshape((NP * P, n, C))
    tg2 = targets.reshape((NP * P, n))
    out2 = out.reshape((NP * P, n))

    # early Exp on ScalarE so its table load overlaps the first DMA;
    # result ~1e-13 is absorbed (exactly) when added to sum >= 1 in fp32
    warm_in = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=warm_in, value=-30.0)
    warm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=warm, op=nl.exp, data=warm_in)

    # first chunk load (critical path)
    xa = _load_chunk(lg3, p0, 0, n, kmax, C, dt, ngrp)

    # targets -> fp32 (exact for indices < 2^24)
    tgt_i = nl.ndarray((P, n), dtype=targets.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tgt_i, src=tg2[p0:p0 + P, :])
    tgt_f = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tgt_f, src=tgt_i)

    xb = xa
    if nch > 1:
        xb = _load_chunk(lg3, p0, 1, n, kmax, C, dt, ngrp)

    # per-row statistics (fp32), kept on chip
    nmx = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    ssum = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    xt = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)

    _chunk_negmax(xa, nmx, 0, n, kmax, C, dt, 0)
    for ci in range(nch):
        xc = xb
        if ci + 2 < nch:
            xc = _load_chunk(lg3, p0, ci + 2, n, kmax, C, dt, ngrp)
        if ci + 1 < nch:
            _chunk_negmax(xb, nmx, ci + 1, n, kmax, C, dt, use_fold)
        _chunk_exp_gather(xa, nmx, ssum, xt, tgt_f, ci, n, kmax, C, dt)
        xa = xb
        xb = xc

    # loss = log(sum) + max - x_target (fp32), rounded once to output dtype
    lse = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=lse, op=nl.log, data=ssum, bias=warm, scale=1.0)
    t1 = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=lse, data2=nmx, op=nl.subtract)
    res = nl.ndarray((P, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=t1, data2=xt, op=nl.subtract)
    nisa.dma_copy(dst=out2[p0:p0 + P, :], src=res)
    return out


def _host_schedule(n, kmax):
    sizes = []
    start = 0
    size = 1
    while start < n:
        sizes.append(min(size, n - start))
        start = start + size
        size = min(size * 2, kmax)
    return sizes


def run(logits, targets):
    global _LAST_CONFIG
    if logits.dim() != 2:
        raise ValueError("logits must be a 2D tensor with shape [batch_size, num_classes].")
    if targets.dim() != 1:
        raise ValueError("targets must be a 1D tensor with shape [batch_size].")
    B = int(logits.shape[0])
    C = int(logits.shape[1])
    if int(targets.shape[0]) != B:
        raise ValueError("Batch size of logits and targets must match.")
    if B % (PMAX * NUM_PROGRAMS) != 0:
        raise ValueError("batch_size must be a multiple of 256 for this implementation.")

    out = _ce_kernel(logits, targets, MAX_CHUNK_ROWS, GPSIMD_FOLD, LOAD_GROUPS)

    n = B // (PMAX * NUM_PROGRAMS)
    sizes = _host_schedule(n, MAX_CHUNK_ROWS)
    _LAST_CONFIG = {
        "programs": NUM_PROGRAMS,
        "partitions": PMAX,
        "rows_per_partition_per_program": n,
        "max_chunk_rows": MAX_CHUNK_ROWS,
        "chunk_schedule": sizes,
        "num_chunks": len(sizes),
        "load_partition_groups": LOAD_GROUPS,
        "num_classes": C,
        "max_fold_gpsimd_levels": GPSIMD_FOLD if C % 2 == 0 else 0,
        "gather": "range_select_max_reduce",
        "max_engine": "gpsimd_fold+vector_tensor_reduce",
        "exp_engine": "scalar_fused_rowsum",
        "stats_lookahead": 1,
        "dma_lookahead": 2,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
