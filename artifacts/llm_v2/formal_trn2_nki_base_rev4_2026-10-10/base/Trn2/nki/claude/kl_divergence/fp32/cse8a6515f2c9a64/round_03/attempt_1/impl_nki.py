import torch  # metadata only (shapes); no torch compute on the data
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_UNIT = 512               # granule (elements / partition); cols is a multiple of 1024
_LOG_GUARD = 1.0e-30      # inside the ScalarE log: y == 0 -> finite log -> 0 * finite = 0
_N_PROG = 2               # LNC=2: two physical cores per logical core
_FD_MAX_UNITS = 12        # max DMA chunk: 12 * 512 elems / partition (3 MiB per DMA per core)
_FD_MIN_UNITS = 2
_PIECE_UNITS = 4          # compute piece: <= 2048 elems / partition
_PREFETCH = 2             # DMA chunks issued ahead of the chunk being computed (3 buffer sets)

_LAST_CONFIG = {}


def _sel(s, a, b, c):
    if s == 0:
        return a
    if s == 1:
        return b
    return c


def _issue_load(yt, lt, yv, lv, pid, s0, F):
    nisa.dma_copy(dst=yt, src=yv[pid, :, s0:s0 + F])
    nisa.dma_copy(dst=lt, src=lv[pid, :, s0:s0 + F])


def _log_piece(w, yt, a, F, bias_t):
    # w = log(y + guard)   (fp32, ScalarE)
    nisa.activation(dst=w, op=nl.log, data=yt[:, a:a + F], bias=bias_t, scale=1.0)


def _finish_piece(w, yt, lt, a, F, g0, partials, cols):
    # d = L - log_p ; t = y * d   (fp32, VectorE, in place on the piece tile)
    nisa.tensor_tensor(dst=w, data1=w, data2=lt[:, a:a + F], op=nl.subtract,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=w, data1=w, data2=yt[:, a:a + F], op=nl.multiply,
                       engine=nisa.engine.vector)
    # per-row fp32 sums (ScalarE fused reduction), split at row boundaries;
    # each row segment writes the slot of its first 512-element unit
    e = g0 + F
    x = g0
    while x < e:
        z = min(e, (x // cols + 1) * cols)
        slot = x // _UNIT
        nisa.activation(dst=w[:, x - g0:z - g0], op=nl.copy,
                        data=w[:, x - g0:z - g0],
                        reduce_op=nl.add,
                        reduce_res=partials[:, slot:slot + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        x = z


@nki.jit
def _kl_div_kernel(log_y_pred, y_true, ch_start, ch_size, pc_chunk, pc_start, pc_size, pf):
    rows = y_true.shape[0]
    cols = y_true.shape[1]
    out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    K = rows // (n_prog * _P)          # rows per partition on this core
    seg = K * cols                     # contiguous elements per partition per input
    NU = seg // _UNIT
    upr = cols // _UNIT                # units per row
    nch = len(ch_size)
    npc = len(pc_size)
    assert ch_start[nch - 1] + ch_size[nch - 1] == NU, "schedule does not match shape"

    # partition p of program a owns rows a*P*K + p*K + [0, K)  (contiguous in memory)
    yv = y_true.reshape((n_prog, _P, seg))
    lv = log_y_pred.reshape((n_prog, _P, seg))
    ov = out.reshape((n_prog, _P, K))

    bias_t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=bias_t, value=_LOG_GUARD)

    # one partial-sum slot per 512-element unit; unused slots stay 0
    partials = nl.ndarray((_P, NU), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=partials, value=0.0)

    y0 = None
    y1 = None
    y2 = None
    l0 = None
    l1 = None
    l2 = None

    # ---- prologue: DMA chunks 0..pf ----
    n_pro = min(nch, pf + 1)
    for k in range(n_pro):
        Fk = ch_size[k] * _UNIT
        yt = nl.ndarray((_P, Fk), dtype=nl.float32, buffer=nl.sbuf)
        lt = nl.ndarray((_P, Fk), dtype=nl.float32, buffer=nl.sbuf)
        _issue_load(yt, lt, yv, lv, pid, ch_start[k] * _UNIT, Fk)
        s = k % 3
        if s == 0:
            y0 = yt
            l0 = lt
        elif s == 1:
            y1 = yt
            l1 = lt
        else:
            y2 = yt
            l2 = lt

    # log of the first piece
    c0 = pc_chunk[0]
    Fp0 = pc_size[0] * _UNIT
    a0 = (pc_start[0] - ch_start[c0]) * _UNIT
    w_next = nl.ndarray((_P, Fp0), dtype=nl.float32, buffer=nl.sbuf)
    _log_piece(w_next, _sel(c0 % 3, y0, y1, y2), a0, Fp0, bias_t)

    # ---- steady state: [DMA chunk c+pf at chunk start] | log(piece i+1) | sub,mul,reduce(piece i)
    for i in range(npc):
        c = pc_chunk[i]
        if i > 0:
            if pc_chunk[i - 1] != c:
                k = c + pf
                if k < nch:
                    Fk = ch_size[k] * _UNIT
                    yt = nl.ndarray((_P, Fk), dtype=nl.float32, buffer=nl.sbuf)
                    lt = nl.ndarray((_P, Fk), dtype=nl.float32, buffer=nl.sbuf)
                    _issue_load(yt, lt, yv, lv, pid, ch_start[k] * _UNIT, Fk)
                    s = k % 3
                    if s == 0:
                        y0 = yt
                        l0 = lt
                    elif s == 1:
                        y1 = yt
                        l1 = lt
                    else:
                        y2 = yt
                        l2 = lt
        w = w_next
        if i + 1 < npc:
            c1 = pc_chunk[i + 1]
            Fp1 = pc_size[i + 1] * _UNIT
            a1 = (pc_start[i + 1] - ch_start[c1]) * _UNIT
            w_next = nl.ndarray((_P, Fp1), dtype=nl.float32, buffer=nl.sbuf)
            _log_piece(w_next, _sel(c1 % 3, y0, y1, y2), a1, Fp1, bias_t)
        Fp = pc_size[i] * _UNIT
        a = (pc_start[i] - ch_start[c]) * _UNIT
        _finish_piece(w, _sel(c % 3, y0, y1, y2), _sel(c % 3, l0, l1, l2),
                      a, Fp, pc_start[i] * _UNIT, partials, cols)

    res = nl.ndarray((_P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=res, op=nl.add,
                       data=partials.reshape((_P, K, upr)), axis=2)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _even_split(total, nparts):
    base = total // nparts
    extra = total % nparts
    parts = []
    for i in range(nparts):
        if i < extra:
            parts.append(base + 1)
        else:
            parts.append(base)
    return parts


def _split_tail(fd):
    if fd >= 4:
        a = fd // 2
        b = (fd - a) // 2
        c = fd - a - b
        t = [a, b, c]
        t.sort(reverse=True)
        return t
    if fd >= 2:
        return [fd - fd // 2, fd // 2]
    return [fd]


def _make_schedule(rows, cols):
    K = max(1, rows // (_N_PROG * _P))
    NU = (K * cols) // _UNIT
    fd = max(_FD_MIN_UNITS, min(_FD_MAX_UNITS, NU // 4))
    fd = max(1, min(fd, NU))
    head = []
    if fd >= 8 and NU >= 3 * fd:
        head = [fd // 2]
    tail = _split_tail(fd)
    body_total = NU - sum(head) - sum(tail)
    body = []
    if body_total > 0:
        nb = -(-body_total // fd)
        body = _even_split(body_total, nb)
    sizes = head + body + tail

    starts = []
    acc = 0
    for s in sizes:
        starts.append(acc)
        acc += s

    pc_chunk = []
    pc_start = []
    pc_size = []
    for ci in range(len(sizes)):
        u = sizes[ci]
        npcs = -(-u // _PIECE_UNITS)
        parts = _even_split(u, npcs)
        off = starts[ci]
        for p in parts:
            pc_chunk.append(ci)
            pc_start.append(off)
            pc_size.append(p)
            off += p

    return {
        "K": K,
        "NU": NU,
        "fd": fd,
        "ch_start": tuple(starts),
        "ch_size": tuple(sizes),
        "pc_chunk": tuple(pc_chunk),
        "pc_start": tuple(pc_start),
        "pc_size": tuple(pc_size),
        "pf": _PREFETCH,
    }


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows = int(y_true.shape[0])
    cols = int(y_true.shape[1])
    sc = _make_schedule(rows, cols)
    out = _kl_div_kernel(log_y_pred, y_true, sc["ch_start"], sc["ch_size"],
                         sc["pc_chunk"], sc["pc_start"], sc["pc_size"], sc["pf"])
    _LAST_CONFIG = {
        "lnc_programs": _N_PROG,
        "rows_per_partition": sc["K"],
        "unit_elems": _UNIT,
        "units_per_partition": sc["NU"],
        "dma_chunk_units": sc["fd"],
        "dma_chunk_sizes_units": list(sc["ch_size"]),
        "n_dma_chunks": len(sc["ch_size"]),
        "compute_piece_max_units": _PIECE_UNITS,
        "n_pieces": len(sc["pc_size"]),
        "prefetch_chunks": sc["pf"],
        "log_lookahead_pieces": 1,
        "log_guard_bias": _LOG_GUARD,
        "log_engine": "scalar",
        "elementwise_engine": "vector",
        "row_reduce_engine": "scalar(activation fused reduce, in place) + vector final",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
