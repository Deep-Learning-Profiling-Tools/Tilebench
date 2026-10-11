import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals / deterministic functions of the shape)
# ---------------------------------------------------------------------------
P = 128                 # rows per tile (partition dim)
NBE = 2                 # fp32 exponential slots
COL_UNIT = 512          # column granularity for load/max/exp chunk boundaries
UNITS_PER_CHUNK = 4     # ~2048 columns per load/max/exp chunk
NORM_PIECE = 2048       # max columns per normalisation instruction / store piece
SPLIT_NUM = 3           # fraction of the normalisation done on ScalarE = 3/5
SPLIT_DEN = 5
ALIGN = 128             # column alignment of the ScalarE/VectorE split
SBUF_BYTES = 212984
SBUF_MARGIN = 16384

_LAST_CONFIG = {}


def _cbound(c, F, C, unit):
    nb = F // unit
    return ((c * nb) // C) * unit


def _pbound(k, lo, hi, n):
    r = hi
    if k < n:
        nu = (hi - lo) // ALIGN
        r = lo + ((k * nu) // n) * ALIGN
    return r


def _pick5(i, a, b, c, d, e):
    r = a
    if i == 1:
        r = b
    elif i == 2:
        r = c
    elif i == 3:
        r = d
    elif i == 4:
        r = e
    return r


def _pick2(i, a, b):
    r = a
    if i == 1:
        r = b
    return r


def _load_tile(x, xt, r0, rows, F, C, unit, PG):
    pgs = (P + PG - 1) // PG
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        for g in range(PG):
            p0 = g * pgs
            p1 = min(rows, p0 + pgs)
            if p1 > p0:
                nisa.dma_copy(dst=xt[p0:p1, c0:c1], src=x[r0 + p0:r0 + p1, c0:c1])


def _store_piece(out, xt, r0, rows, c0, c1, PG):
    pgs = (P + PG - 1) // PG
    for g in range(PG):
        p0 = g * pgs
        p1 = min(rows, p0 + pgs)
        if p1 > p0:
            nisa.dma_copy(dst=out[r0 + p0:r0 + p1, c0:c1], src=xt[p0:p1, c0:c1])


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
    # exp(x - m) in fp32 (ScalarE), fused per-chunk fp32 row sums in st[:, C+2 : 2C+2]
    ng = C + 1
    ps = C + 2
    for c in range(C):
        c0 = _cbound(c, F, C, unit)
        c1 = _cbound(c + 1, F, C, unit)
        nisa.activation(dst=et[0:rows, c0:c1], op=nl.exp, data=xt[0:rows, c0:c1],
                        bias=st[0:rows, ng:ng + 1], scale=1.0,
                        reduce_op=nl.add, reduce_res=st[0:rows, ps + c:ps + c + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)


def _finish_tile(out, xt, et, st, r0, rows, F, C, S, piece, PGS):
    ps = C + 2
    ss = 2 * C + 2
    rc = 2 * C + 3
    scol = ps
    if C > 1:
        nisa.tensor_reduce(dst=st[0:rows, ss:ss + 1], op=nl.add,
                           data=st[0:rows, ps:ps + C], axis=1)
        scol = ss
    nisa.reciprocal(dst=st[0:rows, rc:rc + 1], data=st[0:rows, scol:scol + 1])
    ns = 0
    if S > 0:
        ns = (S + piece - 1) // piece
    nv = 0
    if F - S > 0:
        nv = (F - S + piece - 1) // piece
    nk = max(ns, nv)
    for k in range(nk):
        if k < ns:
            a0 = _pbound(k, 0, S, ns)
            a1 = _pbound(k + 1, 0, S, ns)
            # ScalarE: fp32 exp * fp32 (1/l), single downcast to fp16 into the input slot
            nisa.tensor_scalar(dst=xt[0:rows, a0:a1], data=et[0:rows, a0:a1],
                               op0=nl.multiply, operand0=st[0:rows, rc:rc + 1],
                               engine=nisa.engine.scalar)
            _store_piece(out, xt, r0, rows, a0, a1, PGS)
        if k < nv:
            b0 = _pbound(k, S, F, nv)
            b1 = _pbound(k + 1, S, F, nv)
            # VectorE: fp32 exp * fp32 (1/l), single downcast to fp16 into the input slot
            nisa.tensor_scalar(dst=xt[0:rows, b0:b1], data=et[0:rows, b0:b1],
                               op0=nl.multiply, operand0=st[0:rows, rc:rc + 1],
                               engine=nisa.engine.vector)
            _store_piece(out, xt, r0, rows, b0, b1, PGS)


@nki.jit
def softmax_kernel(x, n_chunks, ld_pgroups, st_pgroups, col_unit, s_split,
                   n_xslots, lookahead, piece):
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
    NX = n_xslots
    L = lookahead

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
    if NX > 4:
        xe = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        se = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        xe = xa
        se = sa

    # prologue: first loads, then the row max of tile 0
    n_pre = min(L, T)
    for t in range(n_pre):
        r0 = row_lo + t * P
        rows = min(P, row_hi - r0)
        _load_tile(x, _pick5(t % NX, xa, xb, xc, xd, xe), r0, rows, F, C,
                   col_unit, ld_pgroups)
    if T > 0:
        rows0 = min(P, row_hi - row_lo)
        _row_max(xa, sa, rows0, F, C, col_unit)

    # software pipeline:
    #   VectorE: finish(t-1) [sum, 1/l, its normalisation share], max(t+1)
    #   ScalarE: its normalisation share of t-1, exp(t)
    for it in range(T + 1):
        tn = it + L
        if tn < T:
            rn0 = row_lo + tn * P
            rowsn = min(P, row_hi - rn0)
            _load_tile(x, _pick5(tn % NX, xa, xb, xc, xd, xe), rn0, rowsn, F, C,
                       col_unit, ld_pgroups)

        tp = it - 1
        if tp >= 0:
            sp = tp % NX
            rp0 = row_lo + tp * P
            rowsp = min(P, row_hi - rp0)
            _finish_tile(out, _pick5(sp, xa, xb, xc, xd, xe), _pick2(tp % NBE, ea, eb),
                         _pick5(sp, sa, sb, sc, sd, se), rp0, rowsp, F, C,
                         s_split, piece, st_pgroups)

        t1 = it + 1
        if t1 < T:
            s1 = t1 % NX
            r10 = row_lo + t1 * P
            rows1 = min(P, row_hi - r10)
            _row_max(_pick5(s1, xa, xb, xc, xd, xe), _pick5(s1, sa, sb, sc, sd, se),
                     rows1, F, C, col_unit)

        if it < T:
            s0 = it % NX
            rc0 = row_lo + it * P
            rowsc = min(P, row_hi - rc0)
            _row_exp(_pick5(s0, xa, xb, xc, xd, xe), _pick2(it % NBE, ea, eb),
                     _pick5(s0, sa, sb, sc, sd, se), rowsc, F, C, col_unit)

    return out


def _make_config(n_rows, n_cols):
    F = n_cols
    if F % COL_UNIT == 0:
        unit = COL_UNIT
    else:
        unit = F
    nb = F // unit
    C = (nb + UNITS_PER_CHUNK - 1) // UNITS_PER_CHUNK
    if C < 1:
        C = 1
    ld_pg = 8 // C
    if ld_pg < 2:
        ld_pg = 2
    if ld_pg > 8:
        ld_pg = 8
    # split of the normalisation between ScalarE (cols [0, s)) and VectorE ([s, F))
    if F >= 2 * ALIGN:
        s = ((F * SPLIT_NUM * 2 + SPLIT_DEN * ALIGN) // (SPLIT_DEN * ALIGN * 2)) * ALIGN
        if s < ALIGN:
            s = ALIGN
        if s > F - ALIGN:
            s = F - ALIGN
    else:
        s = 0
    ns = (s + NORM_PIECE - 1) // NORM_PIECE
    nv = (F - s + NORM_PIECE - 1) // NORM_PIECE
    npieces = ns + nv
    if npieces < 1:
        npieces = 1
    st_pg = (C * ld_pg + npieces // 2) // npieces
    if st_pg < 1:
        st_pg = 1
    if st_pg > 8:
        st_pg = 8
    if 18 * F + SBUF_MARGIN <= SBUF_BYTES:
        nx = 5
    else:
        nx = 4
    lookahead = nx - 2
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "rows_per_tile": P,
        "col_unit": int(unit),
        "n_chunks": int(C),
        "ld_pgroups": int(ld_pg),
        "st_pgroups": int(st_pg),
        "s_split": int(s),
        "norm_piece": int(NORM_PIECE),
        "x_slots": int(nx),
        "exp_slots": NBE,
        "lookahead": int(lookahead),
        "lnc": 2,
        "path": "row_on_chip_single_load_act_dve_split_norm",
    }


def run(x):
    global _LAST_CONFIG
    n_rows = int(x.shape[0])
    n_cols = int(x.shape[1])
    cfg = _make_config(n_rows, n_cols)
    out = softmax_kernel(x, cfg["n_chunks"], cfg["ld_pgroups"], cfg["st_pgroups"],
                         cfg["col_unit"], cfg["s_split"], cfg["x_slots"],
                         cfg["lookahead"], cfg["norm_piece"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
