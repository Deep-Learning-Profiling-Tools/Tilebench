import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
NUM_PROGRAMS = 2          # a bare kernel call runs with LNC=2 on this host
MAX_CHUNK_ROWS = 4        # rows per partition per DMA chunk (upper bound)
GPSIMD_MAX_LEVELS = 2     # pairwise-max folds of the class axis done on GpSimd
MIN_FOLD_WIDTH = 64       # never fold below this width

_LAST_CONFIG = {}


@nki.jit
def _ce_kernel(logits, targets, chunk, nlev):
    B = logits.shape[0]
    C = logits.shape[1]
    P = 128
    NP = nl.num_programs()
    pid = nl.program_id(0)
    n = B // (NP * P)          # rows per partition handled by this program
    p0 = pid * P
    dt = logits.dtype

    out = nl.ndarray((B,), dtype=dt, buffer=nl.shared_hbm)

    # row index = pid*(P*n) + p*n + j  -> contiguous rows per partition
    lg3 = logits.reshape((NP * P, n, C))
    tg2 = targets.reshape((NP * P, n))
    out2 = out.reshape((NP * P, n))

    # ---- targets -> fp32 [P, n] (exact for indices < 2^24)
    tgt_i = nl.ndarray((P, n), dtype=targets.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tgt_i, src=tg2[p0:p0 + P, :])
    tgt_f = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tgt_f, src=tgt_i)

    # ---- per-row statistics (fp32), kept on chip
    mx = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    ssum = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    xt = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)

    nch = (n + chunk - 1) // chunk
    for ci in range(nch):
        j0 = ci * chunk
        k = min(chunk, n - j0)

        x3 = nl.ndarray((P, k, C), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=x3, src=lg3[p0:p0 + P, j0:j0 + k, :])

        # ---- row max: exact pairwise-max folds on GpSimd (fp16 max is exact),
        #      final reduction over the remaining columns on VectorE into fp32
        red = x3
        w = C
        for lv in range(nlev):
            hw = w // 2
            h = nl.ndarray((P, k, hw), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=h, data1=red[:, :, 0:hw], data2=red[:, :, hw:w],
                               op=nl.maximum, engine=nisa.engine.gpsimd)
            red = h
            w = hw
        nisa.tensor_reduce(dst=mx[:, j0:j0 + k], op=nl.maximum, data=red, axis=2)
        nmx = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nmx, data=mx[:, j0:j0 + k], op0=nl.multiply,
                           operand0=-1.0, engine=nisa.engine.vector)

        # ---- sum_j exp(x - max) on ScalarE with fused fp32 row-sum
        esc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(k):
            nisa.activation(dst=esc, op=nl.exp, data=x3[:, j, :],
                            bias=nmx[:, j:j + 1], scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=ssum[:, j0 + j:j0 + j + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)

        # ---- target gather from the loaded row: select column == target, max-reduce
        sel = nl.ndarray((P, C), dtype=dt, buffer=nl.sbuf)
        for j in range(k):
            nisa.range_select(dst=sel,
                              on_true_tile=x3[:, j, :],
                              comp_op0=nl.greater_equal,
                              comp_op1=nl.less_equal,
                              bound0=tgt_f[:, j0 + j:j0 + j + 1],
                              bound1=tgt_f[:, j0 + j:j0 + j + 1],
                              reduce_cmd=nisa.reduce_cmd.reset_reduce,
                              reduce_res=xt[:, j0 + j:j0 + j + 1],
                              reduce_op=nl.maximum)

    # ---- loss = log(sum) + max - x_target  (fp32), rounded once to output dtype
    lse = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=lse, op=nl.log, data=ssum)
    t1 = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=lse, data2=mx, op=nl.add)
    res = nl.ndarray((P, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=t1, data2=xt, op=nl.subtract)
    nisa.dma_copy(dst=out2[p0:p0 + P, :], src=res)
    return out


def _fold_levels(C, max_levels, min_w):
    lv = 0
    w = C
    while lv < max_levels and w % 2 == 0 and (w // 2) >= min_w:
        w = w // 2
        lv += 1
    return lv


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

    n = B // (PMAX * NUM_PROGRAMS)
    chunk = max(1, min(MAX_CHUNK_ROWS, n // 2))
    nlev = _fold_levels(C, GPSIMD_MAX_LEVELS, MIN_FOLD_WIDTH)
    out = _ce_kernel(logits, targets, chunk, nlev)

    _LAST_CONFIG = {
        "programs": NUM_PROGRAMS,
        "partitions": PMAX,
        "rows_per_partition_per_program": n,
        "chunk_rows": chunk,
        "num_chunks": (n + chunk - 1) // chunk,
        "num_classes": C,
        "gpsimd_max_fold_levels": nlev,
        "vector_reduce_width": C >> nlev,
        "gather": "range_select_max_reduce",
        "max_engine": "gpsimd_pairwise_fold+vector_tensor_reduce",
        "exp_engine": "scalar_fused_rowsum",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
