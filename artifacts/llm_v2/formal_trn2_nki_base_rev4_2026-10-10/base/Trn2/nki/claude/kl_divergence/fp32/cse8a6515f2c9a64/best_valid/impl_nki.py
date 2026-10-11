import torch  # only used for type hints / metadata
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_LOG_GUARD = 1.0e-30      # added inside the ScalarE log: y == 0 -> finite log -> 0 * finite = 0
_FT_MAX = 4096            # max free elements per chunk (per partition, per input)
_MIN_CHUNKS = 8           # aim for at least this many chunks per core (pipelining)
_MIN_FREE = 512           # minimum chunk free size
_N_PROG = 2               # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


@nki.jit
def _kl_div_kernel(log_y_pred, y_true, rows_per_group, col_chunks):
    rows = y_true.shape[0]
    cols = y_true.shape[1]
    out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    K = rows // (n_prog * _P)          # rows handled by each partition on this core
    seg = K * cols                     # contiguous elements per partition per input

    # partition p of program a owns rows a*P*K + p*K + [0, K)  (contiguous in memory)
    yv = y_true.reshape((n_prog, _P, seg))
    lv = log_y_pred.reshape((n_prog, _P, seg))
    ov = out.reshape((n_prog, _P, K))

    nC = col_chunks
    m = rows_per_group
    if nC > 1:
        m = 1

    bias_t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=bias_t, value=_LOG_GUARD)

    partials = nl.ndarray((_P, K * nC), dtype=nl.float32, buffer=nl.sbuf)

    n_groups = (K + m - 1) // m
    for g in range(n_groups):
        k0 = g * m
        mm = min(m, K - k0)
        for j in range(nC):
            c0 = ((j * cols) // nC) // 64 * 64
            c1 = cols
            if j + 1 < nC:
                c1 = (((j + 1) * cols) // nC) // 64 * 64
            W = c1 - c0
            F = mm * W                 # if mm > 1 then W == cols: whole rows, contiguous
            s0 = k0 * cols + c0

            yt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
            lt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=yv[pid, :, s0:s0 + F])
            nisa.dma_copy(dst=lt, src=lv[pid, :, s0:s0 + F])

            wt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
            # L = log(y + guard)   (guard keeps y == 0 finite; no effect on normal y)
            nisa.activation(dst=wt, op=nl.log, data=yt, bias=bias_t, scale=1.0)
            # d = L - log_p
            nisa.tensor_tensor(dst=wt, data1=wt, data2=lt, op=nl.subtract,
                               engine=nisa.engine.vector)
            # t = y * d
            nisa.tensor_tensor(dst=wt, data1=wt, data2=yt, op=nl.multiply,
                               engine=nisa.engine.vector)
            # per-row fp32 sum of the chunk (ScalarE fused reduction)
            for r in range(mm):
                slot = (k0 + r) * nC + j
                nisa.activation(dst=lt[:, r * W:(r + 1) * W], op=nl.copy,
                                data=wt[:, r * W:(r + 1) * W],
                                reduce_op=nl.add,
                                reduce_res=partials[:, slot:slot + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)

    if nC > 1:
        res = nl.ndarray((_P, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=res, op=nl.add,
                           data=partials.reshape((_P, K, nC)), axis=2)
        nisa.dma_copy(dst=ov[pid], src=res)
    else:
        nisa.dma_copy(dst=ov[pid], src=partials)
    return out


def _make_config(rows, cols):
    K = max(1, rows // (_N_PROG * _P))
    total = K * cols
    ft = _FT_MAX
    if total // ft < _MIN_CHUNKS:
        ft = max(_MIN_FREE, total // _MIN_CHUNKS)
    if cols <= ft:
        m = max(1, min(K, ft // cols))
        nC = 1
    else:
        m = 1
        nC = -(-cols // ft)
        nC = max(1, min(nC, cols // _MIN_FREE))
    return {
        "lnc_programs": _N_PROG,
        "rows_per_partition": K,
        "chunk_free_target": ft,
        "rows_per_group": m,
        "col_chunks": nC,
        "log_guard_bias": _LOG_GUARD,
        "log_engine": "scalar",
        "elementwise_engine": "vector",
        "row_reduce_engine": "scalar(activation fused reduce)",
    }


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows = int(y_true.shape[0])
    cols = int(y_true.shape[1])
    cfg = _make_config(rows, cols)
    out = _kl_div_kernel(log_y_pred, y_true, cfg["rows_per_group"], cfg["col_chunks"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
