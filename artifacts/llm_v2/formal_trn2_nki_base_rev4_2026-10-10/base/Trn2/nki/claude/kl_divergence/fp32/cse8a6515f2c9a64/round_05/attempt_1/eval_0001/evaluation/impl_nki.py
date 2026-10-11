import torch  # metadata only (shapes); no torch compute on the data
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_LOG_GUARD = 1.0e-30      # inside the ScalarE log: y == 0 -> finite log -> 0 * finite = 0
_SMAX = 8192              # max chunk free size (fp32 elements per partition per input)
_SB = 2048                # max size of a multiply/reduce unit
_GRAN = 512               # chunk boundary granularity (elements)
_TAPER_MIN = 3072         # split the final chunk if it is at least this large
_N_PROG = 2               # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


def _bound(i, T, n_main, tail_b):
    # boundary i of the per-partition chunk sequence (flat position in the K*cols stream)
    r = ((i * T) // n_main) // _GRAN * _GRAN
    if i >= n_main:
        r = T
        if tail_b > 0:
            if i == n_main:
                r = T - tail_b
    return r


@nki.jit
def _kl_div_kernel(log_y_pred, y_true, n_main, tail_b, n_slots):
    rows = y_true.shape[0]
    cols = y_true.shape[1]
    out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    K = rows // (n_prog * _P)          # rows per partition on this core
    T = K * cols                       # contiguous elements per partition per input

    # partition p of program a owns rows a*P*K + p*K + [0, K)  (contiguous in memory)
    yv = y_true.reshape((n_prog, _P, T))
    lv = log_y_pred.reshape((n_prog, _P, T))
    ov = out.reshape((n_prog, _P, K))

    S = n_slots

    bias_t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=bias_t, value=_LOG_GUARD)

    partials = nl.ndarray((_P, K * S), dtype=nl.float32, buffer=nl.sbuf)
    if S > 1:
        nisa.memset(dst=partials, value=0.0)

    n_pieces = n_main
    if tail_b > 0:
        n_pieces = n_main + 1

    cur_row = -1
    cur_slot = 0
    for i in range(n_pieces):
        a = _bound(i, T, n_main, tail_b)
        b = _bound(i + 1, T, n_main, tail_b)
        F = b - a

        yt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        lt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=yt, src=yv[pid, :, a:b])
        nisa.dma_copy(dst=lt, src=lv[pid, :, a:b])

        wt = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        # L = log(y + guard)  (only needs y; overlaps the log_p load)
        nisa.activation(dst=wt, op=nl.log, data=yt, bias=bias_t, scale=1.0)
        # d = L - log_p
        nisa.tensor_tensor(dst=wt, data1=wt, data2=lt, op=nl.subtract,
                           engine=nisa.engine.vector)

        k_first = a // cols
        k_last = (b - 1) // cols
        for k in range(k_first, k_last + 1):
            s0 = max(a, k * cols)
            s1 = min(b, (k + 1) * cols)
            L = s1 - s0
            nu = (L + _SB - 1) // _SB
            if k != cur_row:
                cur_row = k
                cur_slot = 0
            for u in range(nu):
                u0 = s0 + ((u * L) // nu) // 64 * 64
                u1 = s1
                if u + 1 < nu:
                    u1 = s0 + (((u + 1) * L) // nu) // 64 * 64
                W = u1 - u0
                tt = nl.ndarray((_P, W), dtype=nl.float32, buffer=nl.sbuf)
                # t = y * d
                nisa.tensor_tensor(dst=tt, data1=wt[:, u0 - a:u1 - a],
                                   data2=yt[:, u0 - a:u1 - a], op=nl.multiply,
                                   engine=nisa.engine.vector)
                slot = k * S + cur_slot
                # fp32 sum of this row segment (ScalarE fused reduction)
                nisa.activation(dst=tt, op=nl.copy, data=tt,
                                reduce_op=nl.add,
                                reduce_res=partials[:, slot:slot + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                cur_slot = cur_slot + 1

    if S > 1:
        res = nl.ndarray((_P, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=res, op=nl.add,
                           data=partials.reshape((_P, K, S)), axis=2)
        nisa.dma_copy(dst=ov[pid], src=res)
    else:
        nisa.dma_copy(dst=ov[pid], src=partials)
    return out


def _count_slots(K, cols, T, n_main, tail_b):
    counts = [0] * K
    n_pieces = n_main + (1 if tail_b > 0 else 0)
    for i in range(n_pieces):
        a = _bound(i, T, n_main, tail_b)
        b = _bound(i + 1, T, n_main, tail_b)
        for k in range(a // cols, (b - 1) // cols + 1):
            s0 = max(a, k * cols)
            s1 = min(b, (k + 1) * cols)
            counts[k] += (s1 - s0 + _SB - 1) // _SB
    return max(counts)


def _make_config(rows, cols):
    K = max(1, rows // (_N_PROG * _P))
    T = K * cols
    n_main = max(2, -(-T // _SMAX))
    s_last = T - (((n_main - 1) * T) // n_main) // _GRAN * _GRAN
    tail_b = 0
    if s_last >= _TAPER_MIN:
        tail_b = ((s_last + 640) // 1280) * _GRAN   # ~40% of the last chunk
    n_slots = _count_slots(K, cols, T, n_main, tail_b)
    return {
        "lnc_programs": _N_PROG,
        "rows_per_partition": K,
        "elems_per_partition": T,
        "chunk_max": _SMAX,
        "n_main_chunks": n_main,
        "tail_split": tail_b,
        "n_pieces": n_main + (1 if tail_b > 0 else 0),
        "unit_max": _SB,
        "slots_per_row": n_slots,
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
    out = _kl_div_kernel(log_y_pred, y_true, cfg["n_main_chunks"], cfg["tail_split"],
                         cfg["slots_per_row"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
