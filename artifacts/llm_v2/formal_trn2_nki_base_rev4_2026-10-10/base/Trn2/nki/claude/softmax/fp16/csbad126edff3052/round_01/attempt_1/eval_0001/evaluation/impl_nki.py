import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals / deterministic functions of the shape)
# ---------------------------------------------------------------------------
P = 128                 # rows per tile (partition dim)
NBX = 4                 # input/output SBUF slots (fp16, normalised in place)
NBE = 2                 # fp32 exponential slots
LOOKAHEAD = 2           # tiles loaded ahead of the one being computed
COL_UNIT = 512          # column granularity for chunk boundaries
UNITS_PER_CHUNK = 4     # ~2048 columns per compute/DMA chunk

_LAST_CONFIG = {}


def _cbound(c, F, C, unit):
    nb = F // unit
    return ((c * nb) // C) * unit


def _pick4(i, a, b, c, d):
    r = a
    if i == 1:
        r = b
    elif i == 2:
        r = c
    elif i == 3:
        r = d
    return r


def _pick2(i, a, b):
    r = a
    if i == 1:
        r = b
    return r


def _load_tile(x, xt, r0, rows, F, C, unit, PG):
    pgs = P // PG
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        for g in range(PG):
            p0 = g * pgs
            p1 = min(rows, p0 + pgs)
            if p1 > p0:
                nisa.dma_copy(dst=xt[p0:p1, c0:c1], src=x[r0 + p0:r0 + p1, c0:c1])


def _row_max(xt, st, rows, F, C, unit):
    # st columns: [0, C) partial max, C full max, C+1 negated max
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        nisa.tensor_reduce(dst=st[0:rows, c:c + 1], op=nl.maximum,
                           data=xt[0:rows, c0:c1], axis=1)
    mcol = 0
    if C > 1:
        nisa.tensor_reduce(dst=st[0:rows, C:C + 1], op=nl.maximum,
                           data=st[0:rows, 0:C], axis=1)
        mcol = C
    ng = C + 1
    nisa.tensor_scalar(dst=st[0:rows, ng:ng + 1], data=st[0:rows, mcol:mcol + 1],
                       op0=nl.multiply, operand0=-1.0, engine=nisa.engine.vector)


def _row_exp(xt, et, st, rows, F, C, unit):
    # exp(x - m) in fp32, with fused per-chunk fp32 row sums in st[:, C+2 : 2C+2]
    ng = C + 1
    ps = C + 2
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        nisa.activation(dst=et[0:rows, c0:c1], op=nl.exp, data=xt[0:rows, c0:c1],
                        bias=st[0:rows, ng:ng + 1], scale=1.0,
                        reduce_op=nl.add, reduce_res=st[0:rows, ps + c:ps + c + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)


def _finish_tile(out, xt, et, st, r0, rows, F, C, unit, PG):
    ps = C + 2
    ss = 2 * C + 2
    rc = 2 * C + 3
    scol = ps
    if C > 1:
        nisa.tensor_reduce(dst=st[0:rows, ss:ss + 1], op=nl.add,
                           data=st[0:rows, ps:ps + C], axis=1)
        scol = ss
    nisa.reciprocal(dst=st[0:rows, rc:rc + 1], data=st[0:rows, scol:scol + 1])
    pgs = P // PG
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        # fp32 exp * fp32 (1/l), single downcast to fp16 into the input slot
        nisa.tensor_scalar(dst=xt[0:rows, c0:c1], data=et[0:rows, c0:c1],
                           op0=nl.multiply, operand0=st[0:rows, rc:rc + 1],
                           engine=nisa.engine.vector)
        for g in range(PG):
            p0 = g * pgs
            p1 = min(rows, p0 + pgs)
            if p1 > p0:
                nisa.dma_copy(dst=out[r0 + p0:r0 + p1, c0:c1], src=xt[p0:p1, c0:c1])


@nki.jit
def softmax_kernel(x, n_chunks, n_pgroups, col_unit):
    n_rows = x.shape[0]
    F = x.shape[1]
    out = nl.ndarray((n_rows, F), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = (n_rows + n_prog - 1) // n_prog
    row_lo = min(n_rows, pid * rpp)
    row_hi = min(n_rows, row_lo + rpp)
    T = (row_hi - row_lo + P - 1) // P

    C = n_chunks
    nst = 2 * C + 4

    xa = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xc = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xd = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    ea = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    eb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    sa = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    sb = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    sc = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    sd = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)

    # prologue: issue the first loads, then statistics (max) of tile 0
    n_pre = min(LOOKAHEAD, T)
    for t in range(n_pre):
        s = t % NBX
        r0 = row_lo + t * P
        rows = min(P, row_hi - r0)
        _load_tile(x, _pick4(s, xa, xb, xc, xd), r0, rows, F, C, col_unit, n_pgroups)
    if T > 0:
        rows0 = min(P, row_hi - row_lo)
        _row_max(xa, sa, rows0, F, C, col_unit)

    for t in range(T):
        tn = t + LOOKAHEAD
        if tn < T:
            sn = tn % NBX
            rn0 = row_lo + tn * P
            rowsn = min(P, row_hi - rn0)
            _load_tile(x, _pick4(sn, xa, xb, xc, xd), rn0, rowsn, F, C, col_unit, n_pgroups)

        s = t % NBX
        xt = _pick4(s, xa, xb, xc, xd)
        st = _pick4(s, sa, sb, sc, sd)
        et = _pick2(t % NBE, ea, eb)
        r0 = row_lo + t * P
        rows = min(P, row_hi - r0)

        # ScalarE: exponentials + row sums of tile t
        _row_exp(xt, et, st, rows, F, C, col_unit)

        # VectorE: max of the next tile before normalising this one
        t1 = t + 1
        if t1 < T:
            s1 = t1 % NBX
            r10 = row_lo + t1 * P
            rows1 = min(P, row_hi - r10)
            _row_max(_pick4(s1, xa, xb, xc, xd), _pick4(s1, sa, sb, sc, sd),
                     rows1, F, C, col_unit)

        # VectorE: normalise tile t, then store it
        _finish_tile(out, xt, et, st, r0, rows, F, C, col_unit, n_pgroups)

    return out


def _make_config(n_rows, n_cols):
    if n_cols % COL_UNIT == 0:
        unit = COL_UNIT
    else:
        unit = n_cols
    nb = n_cols // unit
    n_chunks = (nb + UNITS_PER_CHUNK - 1) // UNITS_PER_CHUNK
    if n_chunks < 1:
        n_chunks = 1
    n_pgroups = 8 // n_chunks
    if n_pgroups < 2:
        n_pgroups = 2
    if n_pgroups > 8:
        n_pgroups = 8
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "rows_per_tile": P,
        "col_unit": int(unit),
        "n_chunks": int(n_chunks),
        "n_pgroups": int(n_pgroups),
        "x_slots": NBX,
        "exp_slots": NBE,
        "lookahead": LOOKAHEAD,
        "lnc": 2,
        "path": "row_on_chip_single_load",
    }


def run(x):
    global _LAST_CONFIG
    n_rows = int(x.shape[0])
    n_cols = int(x.shape[1])
    cfg = _make_config(n_rows, n_cols)
    out = softmax_kernel(x, cfg["n_chunks"], cfg["n_pgroups"], cfg["col_unit"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
