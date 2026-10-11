import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
NUM_PROGRAMS = 2          # a bare kernel call runs with LNC=2 on this host
CHUNK_ELEMS = 2048        # free elements per partition per row chunk -> chunk rows = CHUNK_ELEMS // C

_LAST_CONFIG = {}


@nki.jit
def _ce_kernel(logits, targets, chunk):
    B = logits.shape[0]
    C = logits.shape[1]
    P = 128
    NP = nl.num_programs()
    pid = nl.program_id(0)
    n = B // (NP * P)          # rows per partition handled by this program
    p0 = pid * P

    out = nl.ndarray((B,), dtype=logits.dtype, buffer=nl.shared_hbm)

    # row index = pid*(P*n) + p*n + j  -> contiguous per partition
    lg3 = logits.reshape((NP * P, n, C))
    tg2 = targets.reshape((NP * P, n))
    out2 = out.reshape((NP * P, n))

    # ---- targets -> fp32 [P, n]
    tgt_i = nl.ndarray((P, n), dtype=targets.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tgt_i, src=tg2[p0:p0 + P, :])
    tgt_f = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tgt_f, src=tgt_i)

    # ---- class index row (0..C-1) on every partition
    cols_i = nl.ndarray((P, C), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=cols_i, pattern=[[1, C]], offset=0, channel_multiplier=0)
    cdt = nl.float32
    if C <= 2048:
        cdt = nl.float16       # integers up to 2048 are exact in fp16
    cols = nl.ndarray((P, C), dtype=cdt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cols, src=cols_i)

    # ---- per-row statistics (fp32), kept on chip
    mx = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    ssum = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    xt = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)

    nch = (n + chunk - 1) // chunk
    prev_p = None
    prev_j0 = 0
    prev_k = 0
    for ci in range(nch):
        j0 = ci * chunk
        k = min(chunk, n - j0)

        x3 = nl.ndarray((P, k, C), dtype=logits.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x3, src=lg3[p0:p0 + P, j0:j0 + k, :])

        # row max (fp32) and its negation for the exp bias
        nisa.tensor_reduce(dst=mx[:, j0:j0 + k], op=nl.maximum, data=x3, axis=2)
        nmx = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nmx, data=mx[:, j0:j0 + k], op0=nl.multiply,
                           operand0=-1.0, engine=nisa.engine.vector)

        # sum_j exp(x - max) on ScalarE, fp32 accumulation
        esc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(k):
            nisa.activation(dst=esc, op=nl.exp, data=x3[:, j, :],
                            bias=nmx[:, j:j + 1], scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=ssum[:, j0 + j:j0 + j + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)

        # target gather: one-hot mask (cols == target), product with the loaded row
        m3 = nl.ndarray((P, k, C), dtype=logits.dtype, buffer=nl.sbuf)
        for j in range(k):
            nisa.tensor_scalar(dst=m3[:, j, :], data=cols, op0=nl.equal,
                               operand0=tgt_f[:, j0 + j:j0 + j + 1],
                               engine=nisa.engine.vector)
        p3 = nl.ndarray((P, k, C), dtype=logits.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=p3, data1=x3, data2=m3, op=nl.multiply,
                           engine=nisa.engine.gpsimd)

        # lagged reduction of the previous chunk's product (software pipelining)
        if ci > 0:
            nisa.tensor_reduce(dst=xt[:, prev_j0:prev_j0 + prev_k], op=nl.add,
                               data=prev_p, axis=2)
        prev_p = p3
        prev_j0 = j0
        prev_k = k

    nisa.tensor_reduce(dst=xt[:, prev_j0:prev_j0 + prev_k], op=nl.add,
                       data=prev_p, axis=2)

    # ---- loss = log(sum) + max - x_target  (fp32), rounded once to output dtype
    lse = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=lse, op=nl.log, data=ssum)
    t1 = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=lse, data2=mx, op=nl.add)
    res = nl.ndarray((P, n), dtype=logits.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=t1, data2=xt, op=nl.subtract)
    nisa.dma_copy(dst=out2[p0:p0 + P, :], src=res)
    return out


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

    chunk = max(1, CHUNK_ELEMS // C)
    n = B // (PMAX * NUM_PROGRAMS)
    out = _ce_kernel(logits, targets, chunk)

    _LAST_CONFIG = {
        "programs": NUM_PROGRAMS,
        "partitions": PMAX,
        "rows_per_partition_per_program": n,
        "chunk_rows": chunk,
        "num_chunks": (n + chunk - 1) // chunk,
        "num_classes": C,
        "gather": "iota_onehot_mask_mul_reduce",
        "mask_mul_engine": "gpsimd",
        "exp_engine": "scalar_fused_rowsum",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
