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
SBUF_MARGIN = 14336
MAX_XSLOTS = 8
LNC = 2

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


def _pick8(i, a0, a1, a2, a3, a4, a5, a6, a7):
    r = a0
    if i == 1:
        r = a1
    elif i == 2:
        r = a2
    elif i == 3:
        r = a3
    elif i == 4:
        r = a4
    elif i == 5:
        r = a5
    elif i == 6:
        r = a6
    elif i == 7:
        r = a7
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


def _prep_tile(st, rows, C):
    # fp32 row sum l (from the fused per-chunk sums) and its reciprocal (VectorE)
    ps = C + 2
    ss = 2 * C + 2
    rc = 2 * C + 3
    scol = ps
    if C > 1:
        nisa.tensor_reduce(dst=st[0:rows, ss:ss + 1], op=nl.add,
                           data=st[0:rows, ps:ps + C], axis=1)
        scol = ss
    nisa.reciprocal(dst=st[0:rows, rc:rc + 1], data=st[0:rows, scol:scol + 1])


def _norm_tile(out, xt, et, st, r0, rows, F, C, S, piece, PGS):
    rc = 2 * C + 3
    ns = 0
    if S > 0:
        ns = (S + piece - 1) // piece
    nv = 0
    if F - S > 0:
        nv = (F - S + piece - 1) // piece
    # VectorE share first: it runs while ScalarE computes the next tile's exp
    for k in range(nv):
        b0 = _pbound(k, S, F, nv)
        b1 = _pbound(k + 1, S, F, nv)
        # fp32 exp * fp32 (1/l), single downcast to fp16 into the input slot
        nisa.tensor_scalar(dst=xt[0:rows, b0:b1], data=et[0:rows, b0:b1],
                           op0=nl.multiply, operand0=st[0:rows, rc:rc + 1],
                           engine=nisa.engine.vector)
        _store_piece(out, xt, r0, rows, b0, b1, PGS)
    # ScalarE share (issued after the next tile's exp on ScalarE)
    for k in range(ns):
        a0 = _pbound(k, 0, S, ns)
        a1 = _pbound(k + 1, 0, S, ns)
        nisa.tensor_scalar(dst=xt[0:rows, a0:a1], data=et[0:rows, a0:a1],
                           op0=nl.multiply, operand0=st[0:rows, rc:rc + 1],
                           engine=nisa.engine.scalar)
        _store_piece(out, xt, r0, rows, a0, a1, PGS)


@nki.jit
def softmax_kernel(x, n_chunks, ld_pgroups, st_pgroups, col_unit, s_split,
                   n_xslots, piece):
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
    NXe = n_xslots
    if NXe > T:
        NXe = T
    if NXe < 1:
        NXe = 1
    if NXe >= T:
        L = T
    else:
        L = NXe - 2
        if L < 1:
            L = 1

    x0 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    s0 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    if NXe > 1:
        x1 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s1 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x1 = x0
        s1 = s0
    if NXe > 2:
        x2 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s2 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x2 = x0
        s2 = s0
    if NXe > 3:
        x3 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s3 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x3 = x0
        s3 = s0
    if NXe > 4:
        x4 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s4 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x4 = x0
        s4 = s0
    if NXe > 5:
        x5 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s5 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x5 = x0
        s5 = s0
    if NXe > 6:
        x6 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s6 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x6 = x0
        s6 = s0
    if NXe > 7:
        x7 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        s7 = nl.ndarray((P, nst), dtype=nl.float32, buffer=nl.sbuf)
    else:
        x7 = x0
        s7 = s0
    ea = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    eb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)

    # prologue: first loads (all of them when every tile has its own slot),
    # then the row max of tile 0
    n_pre = min(L, T)
    for t in range(n_pre):
        r0 = row_lo + t * P
        rows = min(P, row_hi - r0)
        _load_tile(x, _pick8(t % NXe, x0, x1, x2, x3, x4, x5, x6, x7), r0, rows,
                   F, C, col_unit, ld_pgroups)
    if T > 0:
        rows0 = min(P, row_hi - row_lo)
        _row_max(x0, s0, rows0, F, C, col_unit)

    # software pipeline, iteration it:
    #   VectorE: 1/l(it-1), its normalisation share of it-1, max(it+1)
    #   ScalarE: exp(it), then its normalisation share of it-1
    for it in range(T + 1):
        tn = it + L
        if tn < T:
            rn0 = row_lo + tn * P
            rowsn = min(P, row_hi - rn0)
            _load_tile(x, _pick8(tn % NXe, x0, x1, x2, x3, x4, x5, x6, x7), rn0,
                       rowsn, F, C, col_unit, ld_pgroups)

        tp = it - 1
        if tp >= 0:
            spp = tp % NXe
            rp0 = row_lo + tp * P
            rowsp = min(P, row_hi - rp0)
            _prep_tile(_pick8(spp, s0, s1, s2, s3, s4, s5, s6, s7), rowsp, C)

        if it < T:
            sc0 = it % NXe
            rc0 = row_lo + it * P
            rowsc = min(P, row_hi - rc0)
            _row_exp(_pick8(sc0, x0, x1, x2, x3, x4, x5, x6, x7),
                     _pick2(it % NBE, ea, eb),
                     _pick8(sc0, s0, s1, s2, s3, s4, s5, s6, s7),
                     rowsc, F, C, col_unit)

        if tp >= 0:
            spp2 = tp % NXe
            rq0 = row_lo + tp * P
            rowsq = min(P, row_hi - rq0)
            _norm_tile(out, _pick8(spp2, x0, x1, x2, x3, x4, x5, x6, x7),
                       _pick2(tp % NBE, ea, eb),
                       _pick8(spp2, s0, s1, s2, s3, s4, s5, s6, s7),
                       rq0, rowsq, F, C, s_split, piece, st_pgroups)

        t1 = it + 1
        if t1 < T:
            s1i = t1 % NXe
            r10 = row_lo + t1 * P
            rows1 = min(P, row_hi - r10)
            _row_max(_pick8(s1i, x0, x1, x2, x3, x4, x5, x6, x7),
                     _pick8(s1i, s0, s1, s2, s3, s4, s5, s6, s7),
                     rows1, F, C, col_unit)

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
    # number of fp16 input slots: x slots 2F bytes each, exp slots 4F bytes each
    budget = SBUF_BYTES - SBUF_MARGIN
    nx = (budget // F - 4 * NBE) // 2
    if nx > MAX_XSLOTS:
        nx = MAX_XSLOTS
    if nx < 3:
        nx = 3
    rpp = (n_rows + LNC - 1) // LNC
    T = (rpp + P - 1) // P
    nx_eff = nx
    if nx_eff > T:
        nx_eff = T
    if nx_eff < 1:
        nx_eff = 1
    if nx_eff >= T:
        lookahead = T
    else:
        lookahead = max(1, nx_eff - 2)
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
        "x_slots_eff": int(nx_eff),
        "exp_slots": NBE,
        "lookahead": int(lookahead),
        "lnc": LNC,
        "path": "row_on_chip_single_load_exp_before_norm_split",
    }


def run(x):
    global _LAST_CONFIG
    n_rows = int(x.shape[0])
    n_cols = int(x.shape[1])
    cfg = _make_config(n_rows, n_cols)
    out = softmax_kernel(x, cfg["n_chunks"], cfg["ld_pgroups"], cfg["st_pgroups"],
                         cfg["col_unit"], cfg["s_split"], cfg["x_slots"],
                         cfg["norm_piece"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
