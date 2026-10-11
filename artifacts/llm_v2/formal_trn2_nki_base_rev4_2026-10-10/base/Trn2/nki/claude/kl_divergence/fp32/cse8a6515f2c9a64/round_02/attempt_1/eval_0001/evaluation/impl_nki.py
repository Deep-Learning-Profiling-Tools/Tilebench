import torch  # only used for type hints / metadata
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_UNIT = 512               # chunk granule (elements per partition); cols is a multiple of 1024
_LOG_GUARD = 1.0e-30      # added inside the ScalarE log: y == 0 -> finite log -> 0 * finite = 0
_N_PROG = 2               # LNC=2: two physical cores per logical core
_FB_MAX = 8               # max steady chunk size in units (8 * 512 = 4096 elements / partition)
_MIN_BODY_CHUNKS = 6      # shrink the steady chunk until there are at least this many
_PREFETCH = 2             # chunks of loads issued ahead of the chunk being finished

_LAST_CONFIG = {}


def _nsplit(F):
    if F >= 2048:
        return 2
    return 1


def _issue_load(yt, lt, yv, lv, pid, s0, F):
    ns = _nsplit(F)
    pp = _P // ns
    for q in range(ns):
        p0 = q * pp
        nisa.dma_copy(dst=yt[p0:p0 + pp, :], src=yv[pid, p0:p0 + pp, s0:s0 + F])
        nisa.dma_copy(dst=lt[p0:p0 + pp, :], src=lv[pid, p0:p0 + pp, s0:s0 + F])


def _finish_chunk(wt, lt, yt, partials, s0, F, cols):
    # wt holds log(y + guard); d = L - log_p ; t = y * d   (fp32, VectorE)
    nisa.tensor_tensor(dst=wt, data1=wt, data2=lt, op=nl.subtract,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=wt, data1=wt, data2=yt, op=nl.multiply,
                       engine=nisa.engine.vector)
    # per-row pieces of this chunk: fp32 sum on ScalarE (fused reduction)
    e = s0 + F
    a = s0
    while a < e:
        b = min(e, (a // cols + 1) * cols)
        slot = a // _UNIT
        nisa.activation(dst=lt[:, a - s0:b - s0], op=nl.copy,
                        data=wt[:, a - s0:b - s0],
                        reduce_op=nl.add,
                        reduce_res=partials[:, slot:slot + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        a = b


@nki.jit
def _kl_div_kernel(log_y_pred, y_true, starts, sizes):
    rows = y_true.shape[0]
    cols = y_true.shape[1]
    out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    K = rows // (n_prog * _P)          # rows per partition on this core
    seg = K * cols                     # contiguous elements per partition per input
    NU = seg // _UNIT
    upr = cols // _UNIT                # units per row
    n = len(sizes)
    assert starts[n - 1] + sizes[n - 1] == NU, "schedule does not match shape"

    # partition p of program a owns rows a*P*K + p*K + [0, K)  (contiguous in memory)
    yv = y_true.reshape((n_prog, _P, seg))
    lv = log_y_pred.reshape((n_prog, _P, seg))
    ov = out.reshape((n_prog, _P, K))

    bias_t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=bias_t, value=_LOG_GUARD)

    # one partial-sum slot per 512-element unit; only the first unit of each piece is
    # written, the rest stay 0 -> final per-row sum over the row's slots
    partials = nl.ndarray((_P, NU), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=partials, value=0.0)

    # ---- prologue: loads for chunks 0 and 1, log for chunk 0 ----
    F0 = sizes[0] * _UNIT
    ya = nl.ndarray((_P, F0), dtype=nl.float32, buffer=nl.sbuf)
    la = nl.ndarray((_P, F0), dtype=nl.float32, buffer=nl.sbuf)
    _issue_load(ya, la, yv, lv, pid, starts[0] * _UNIT, F0)
    yb = ya
    lb = la
    if n > 1:
        F1 = sizes[1] * _UNIT
        yb = nl.ndarray((_P, F1), dtype=nl.float32, buffer=nl.sbuf)
        lb = nl.ndarray((_P, F1), dtype=nl.float32, buffer=nl.sbuf)
        _issue_load(yb, lb, yv, lv, pid, starts[1] * _UNIT, F1)
    wa = nl.ndarray((_P, F0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=wa, op=nl.log, data=ya, bias=bias_t, scale=1.0)
    wb = wa
    yc = ya
    lc = la

    # ---- steady state: loads(j+2) | log(j+1) | sub,mul,reduce(j) ----
    for j in range(n):
        if j + 2 < n:
            F2 = sizes[j + 2] * _UNIT
            yc = nl.ndarray((_P, F2), dtype=nl.float32, buffer=nl.sbuf)
            lc = nl.ndarray((_P, F2), dtype=nl.float32, buffer=nl.sbuf)
            _issue_load(yc, lc, yv, lv, pid, starts[j + 2] * _UNIT, F2)
        if j + 1 < n:
            F1n = sizes[j + 1] * _UNIT
            wb = nl.ndarray((_P, F1n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=wb, op=nl.log, data=yb, bias=bias_t, scale=1.0)
        _finish_chunk(wa, la, ya, partials, starts[j] * _UNIT, sizes[j] * _UNIT, cols)
        if j + 1 < n:
            ya = yb
            la = lb
            wa = wb
        if j + 2 < n:
            yb = yc
            lb = lc

    res = nl.ndarray((_P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=res, op=nl.add,
                       data=partials.reshape((_P, K, upr)), axis=2)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _make_schedule(rows, cols):
    K = max(1, rows // (_N_PROG * _P))
    NU = (K * cols) // _UNIT
    FB = _FB_MAX
    while FB > 1 and NU < _MIN_BODY_CHUNKS * FB:
        FB //= 2
    head = []
    t = 1
    while t < FB:
        head.append(t)
        t *= 2
    tail = []
    t = FB // 2
    while t >= 1:
        tail.append(t)
        t //= 2
    tail.append(1)
    body = NU - sum(head) - sum(tail)
    sizes = list(head) + [FB] * (body // FB)
    if body % FB:
        sizes.append(body % FB)
    sizes = sizes + tail
    starts = []
    acc = 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return K, NU, FB, tuple(starts), tuple(sizes)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows = int(y_true.shape[0])
    cols = int(y_true.shape[1])
    K, NU, FB, starts, sizes = _make_schedule(rows, cols)
    out = _kl_div_kernel(log_y_pred, y_true, starts, sizes)
    _LAST_CONFIG = {
        "lnc_programs": _N_PROG,
        "rows_per_partition": K,
        "unit_elems": _UNIT,
        "units_per_partition": NU,
        "steady_chunk_units": FB,
        "n_chunks": len(sizes),
        "chunk_sizes_units": list(sizes),
        "prefetch_chunks": _PREFETCH,
        "dma_split_rule": "2 if chunk>=2048 elems else 1",
        "log_guard_bias": _LOG_GUARD,
        "log_engine": "scalar",
        "elementwise_engine": "vector",
        "row_reduce_engine": "scalar(activation fused reduce) + vector final",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
