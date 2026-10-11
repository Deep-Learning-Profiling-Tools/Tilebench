import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
PSUM_F32_COLS = 512     # one PSUM bank of fp32
PSUM_F16_COLS = 1024    # one PSUM bank of fp16
SBUF_BUDGET = 176128    # bytes per partition we allow for the big buffers
MAXBUF = 16
LMAX = 8                # max prefetch distance (chunks)
CHUNK_TARGET = 98304    # bytes per operand per chunk below which batches are grouped

_LAST_CONFIG = {}


# ----------------------------------------------------------------------------
# trace-time buffer selection helpers
# ----------------------------------------------------------------------------
def _sel2(j, t0, t1):
    t = t0
    if j % 2 == 1:
        t = t1
    return t


def _sel4(j, t0, t1, t2, t3):
    t = t0
    r = j % 4
    if r == 1:
        t = t1
    if r == 2:
        t = t2
    if r == 3:
        t = t3
    return t


def _pick(j, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15):
    t = t0
    if j == 1:
        t = t1
    if j == 2:
        t = t2
    if j == 3:
        t = t3
    if j == 4:
        t = t4
    if j == 5:
        t = t5
    if j == 6:
        t = t6
    if j == 7:
        t = t7
    if j == 8:
        t = t8
    if j == 9:
        t = t9
    if j == 10:
        t = t10
    if j == 11:
        t = t11
    if j == 12:
        t = t12
    if j == 13:
        t = t13
    if j == 14:
        t = t14
    if j == 15:
        t = t15
    return t


def _alloc_sb(idx, n, shape, dt):
    t = None
    if idx < n:
        t = nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)
    return t


# ----------------------------------------------------------------------------
# packed path: S <= 128, several batches per chunk
# ----------------------------------------------------------------------------
def _pk_load(A, B, a_t, x_t, c0, S, cb, G):
    # a_t[p, g, k] = A2[c0*S + g*128 + p, k]
    nisa.dma_copy(dst=a_t[0:P, 0:G, 0:S],
                  src=A.ap(pattern=[[S, P], [P * S, G], [1, S]], offset=c0 * S * S))
    # x_t[k, i, n] = B[c0 + i, k, n]
    nisa.dma_copy(dst=x_t[0:S, 0:cb, 0:S],
                  src=B.ap(pattern=[[S, S], [S * S, cb], [1, S]], offset=c0 * S * S))


def _pk_transpose(a_t, aT_t, pt, S, G):
    for g in range(G):
        nisa.nc_transpose(dst=pt[0:S, g * P:(g + 1) * P],
                          data=a_t[0:P, g, 0:S],
                          engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=aT_t[0:S, 0:G * P], src=pt[0:S, 0:G * P],
                     engine=nisa.engine.vector)


def _pk_compute(aT_t, x_t, o_t, q, out, c0, S, cb):
    for i in range(cb):
        nisa.nc_matmul(dst=q[0:S, i, 0:S],
                       stationary=aT_t[0:S, i * S:(i + 1) * S],
                       moving=x_t[0:S, i, 0:S],
                       accumulate=False)
    nisa.tensor_copy(dst=o_t[0:S, 0:cb, 0:S], src=q[0:S, 0:cb, 0:S],
                     engine=nisa.engine.scalar)
    nisa.dma_copy(dst=out.ap(pattern=[[S, S], [S * S, cb], [1, S]], offset=c0 * S * S),
                  src=o_t[0:S, 0:cb, 0:S])


def _packed_path(A, B, out, S, b_start, nb, cb, dt):
    nck = nb // cb
    G = (cb * S) // P
    w = PSUM_F32_COLS // cb
    D = min(nck, 4)
    L = D - 1

    a0 = _alloc_sb(0, D, (P, G, S), dt)
    a1 = _alloc_sb(1, D, (P, G, S), dt)
    a2 = _alloc_sb(2, D, (P, G, S), dt)
    a3 = _alloc_sb(3, D, (P, G, S), dt)
    x0 = _alloc_sb(0, D, (S, cb, S), dt)
    x1 = _alloc_sb(1, D, (S, cb, S), dt)
    x2 = _alloc_sb(2, D, (S, cb, S), dt)
    x3 = _alloc_sb(3, D, (S, cb, S), dt)
    o0 = _alloc_sb(0, D, (S, cb, S), dt)
    o1 = _alloc_sb(1, D, (S, cb, S), dt)
    o2 = _alloc_sb(2, D, (S, cb, S), dt)
    o3 = _alloc_sb(3, D, (S, cb, S), dt)
    t0 = nl.ndarray((S, G * P), dtype=dt, buffer=nl.sbuf)
    t1 = nl.ndarray((S, G * P), dtype=dt, buffer=nl.sbuf)
    q0 = nl.ndarray((P, cb, w), dtype=nl.float32, buffer=nl.psum)
    q1 = nl.ndarray((P, cb, w), dtype=nl.float32, buffer=nl.psum)
    p0 = nl.ndarray((P, PSUM_F16_COLS), dtype=dt, buffer=nl.psum)
    p1 = nl.ndarray((P, PSUM_F16_COLS), dtype=dt, buffer=nl.psum)

    for c in range(L + 1):
        _pk_load(A, B, _sel4(c, a0, a1, a2, a3), _sel4(c, x0, x1, x2, x3),
                 b_start + c * cb, S, cb, G)
    _pk_transpose(a0, t0, p0, S, G)
    for ci in range(nck):
        if ci >= 1:
            cn = ci + L
            if cn < nck:
                sl = cn % D
                _pk_load(A, B, _sel4(sl, a0, a1, a2, a3), _sel4(sl, x0, x1, x2, x3),
                         b_start + cn * cb, S, cb, G)
        if ci + 1 < nck:
            sn = (ci + 1) % D
            _pk_transpose(_sel4(sn, a0, a1, a2, a3), _sel2(ci + 1, t0, t1),
                          _sel2(ci + 1, p0, p1), S, G)
        sc = ci % D
        _pk_compute(_sel2(ci, t0, t1), _sel4(sc, x0, x1, x2, x3),
                    _sel4(sc, o0, o1, o2, o3), _sel2(ci, q0, q1), out,
                    b_start + ci * cb, S, cb)


# ----------------------------------------------------------------------------
# general path: S > 128, T-row tiles (+ optional remainder tile)
# A / C full tiles use interleaved rows: m = p*full + j  (contiguous per partition)
# ----------------------------------------------------------------------------
def _grp_start(j, gsz, full):
    s = j * gsz
    if s > full:
        s = full
    return s


def _grp_size(j, gsz, full):
    s = j * gsz
    n = 1
    if s < full:
        n = min(gsz, full - s)
    return n


def _grp_rows(j, gsz, full, T, rem):
    r = rem
    if j * gsz < full:
        r = T
    return r


def _g_load_chunk(A, B, abuf, bbuf, b0, S, T, full, rem, cb, MT):
    # abuf[p, w, j*S + k] = A[b0+w, p*full + j, k]   (full tiles)
    # abuf[p, w, full*S + k] = A[b0+w, T*full + p, k] (remainder tile)
    if full > 0:
        nisa.dma_copy(dst=abuf[0:T, 0:cb, 0:full * S],
                      src=A.ap(pattern=[[full * S, T], [S * S, cb], [1, full * S]],
                               offset=b0 * S * S))
    if rem > 0:
        nisa.dma_copy(dst=abuf[0:rem, 0:cb, full * S:full * S + S],
                      src=A.ap(pattern=[[S, rem], [S * S, cb], [1, S]],
                               offset=(b0 * S + T * full) * S))
    # bbuf[p, w*MT + kt, n] = B[b0+w, kt*T + p, n]
    for w in range(cb):
        base = (b0 + w) * S
        if full > 0:
            nisa.dma_copy(dst=bbuf[0:T, w * MT:w * MT + full, 0:S],
                          src=B.ap(pattern=[[S, T], [T * S, full], [1, S]], offset=base * S))
        if rem > 0:
            nisa.dma_copy(dst=bbuf[0:rem, w * MT + full, 0:S],
                          src=B[base + T * full:base + S, 0:S])


def _g_store_group(out, obuf, b, w, mt0, gm, mm, S, T, full, nch, cw):
    # obuf[p, w, mt*nch + c, n] -> C[b, row, c*cw + n]
    #   row = p*full + mt (full tiles), row = T*full + p (remainder tile)
    ns = gm * nch
    s0 = mt0 * nch
    if mt0 < full:
        nisa.dma_copy(dst=out.ap(pattern=[[full * S, mm], [cw, ns], [1, cw]],
                                 offset=(b * S + mt0) * S),
                      src=obuf[0:mm, w, s0:s0 + ns, 0:cw])
    else:
        nisa.dma_copy(dst=out.ap(pattern=[[S, mm], [cw, ns], [1, cw]],
                                 offset=(b * S + T * full) * S),
                      src=obuf[0:mm, w, s0:s0 + ns, 0:cw])


def _g_tr_group(abuf, w, aT, pt, kt0, g, kk, S, T, full, rem, MT, use_scalar):
    # transposes of K-tiles kt0 .. kt0+g-1 (all with kk rows) into one PSUM bank,
    # then one copy to aT[k, kt, m]
    for gi in range(g):
        kt = kt0 + gi
        for mt in range(MT):
            mm = T
            if mt >= full:
                mm = rem
            c0 = mt * S + kt * T
            nisa.nc_transpose(dst=pt[0:kk, gi, mt * T:mt * T + mm],
                              data=abuf[0:mm, w, c0:c0 + kk],
                              engine=nisa.engine.tensor)
    if use_scalar:
        nisa.tensor_copy(dst=aT[0:kk, kt0:kt0 + g, 0:S], src=pt[0:kk, 0:g, 0:S],
                         engine=nisa.engine.scalar)
    else:
        nisa.tensor_copy(dst=aT[0:kk, kt0:kt0 + g, 0:S], src=pt[0:kk, 0:g, 0:S],
                         engine=nisa.engine.vector)


def _g_out_group(aT, bbuf, w, obuf, q, mt0, gm, mm, S, T, full, rem, MT, nch, cw):
    # gm M-tiles (each nch column chunks) accumulated over full K into
    # one multi-bank PSUM tile, then one PSUM->SBUF copy (cast)
    for gi in range(gm):
        mt = mt0 + gi
        for kt in range(MT):
            kk = T
            if kt >= full:
                kk = rem
            for c in range(nch):
                nisa.nc_matmul(dst=q[0:mm, gi * nch + c, 0:cw],
                               stationary=aT[0:kk, kt, mt * T:mt * T + mm],
                               moving=bbuf[0:kk, w * MT + kt, c * cw:c * cw + cw],
                               accumulate=(kt > 0))
    ns = gm * nch
    s0 = mt0 * nch
    nisa.tensor_copy(dst=obuf[0:mm, w, s0:s0 + ns, 0:cw], src=q[0:mm, 0:ns, 0:cw],
                     engine=nisa.engine.scalar)


def _general_path(A, B, out, S, b_start, nb, cb, D, nO, L, T, full, rem, nch, cw, gT, dt):
    MT = full
    if rem > 0:
        MT = full + 1
    nck = nb // cb
    D = min(D, nck)
    nO = min(nO, nck)
    L = min(L, D - 1)
    shA = (T, cb, MT * S)
    shB = (T, cb * MT, S)
    shO = (T, cb, MT * nch, cw)

    a0 = _alloc_sb(0, D, shA, dt)
    a1 = _alloc_sb(1, D, shA, dt)
    a2 = _alloc_sb(2, D, shA, dt)
    a3 = _alloc_sb(3, D, shA, dt)
    a4 = _alloc_sb(4, D, shA, dt)
    a5 = _alloc_sb(5, D, shA, dt)
    a6 = _alloc_sb(6, D, shA, dt)
    a7 = _alloc_sb(7, D, shA, dt)
    a8 = _alloc_sb(8, D, shA, dt)
    a9 = _alloc_sb(9, D, shA, dt)
    a10 = _alloc_sb(10, D, shA, dt)
    a11 = _alloc_sb(11, D, shA, dt)
    a12 = _alloc_sb(12, D, shA, dt)
    a13 = _alloc_sb(13, D, shA, dt)
    a14 = _alloc_sb(14, D, shA, dt)
    a15 = _alloc_sb(15, D, shA, dt)

    x0 = _alloc_sb(0, D, shB, dt)
    x1 = _alloc_sb(1, D, shB, dt)
    x2 = _alloc_sb(2, D, shB, dt)
    x3 = _alloc_sb(3, D, shB, dt)
    x4 = _alloc_sb(4, D, shB, dt)
    x5 = _alloc_sb(5, D, shB, dt)
    x6 = _alloc_sb(6, D, shB, dt)
    x7 = _alloc_sb(7, D, shB, dt)
    x8 = _alloc_sb(8, D, shB, dt)
    x9 = _alloc_sb(9, D, shB, dt)
    x10 = _alloc_sb(10, D, shB, dt)
    x11 = _alloc_sb(11, D, shB, dt)
    x12 = _alloc_sb(12, D, shB, dt)
    x13 = _alloc_sb(13, D, shB, dt)
    x14 = _alloc_sb(14, D, shB, dt)
    x15 = _alloc_sb(15, D, shB, dt)

    o0 = _alloc_sb(0, nO, shO, dt)
    o1 = _alloc_sb(1, nO, shO, dt)
    o2 = _alloc_sb(2, nO, shO, dt)
    o3 = _alloc_sb(3, nO, shO, dt)
    o4 = _alloc_sb(4, nO, shO, dt)
    o5 = _alloc_sb(5, nO, shO, dt)
    o6 = _alloc_sb(6, nO, shO, dt)
    o7 = _alloc_sb(7, nO, shO, dt)
    o8 = _alloc_sb(8, nO, shO, dt)
    o9 = _alloc_sb(9, nO, shO, dt)
    o10 = _alloc_sb(10, nO, shO, dt)
    o11 = _alloc_sb(11, nO, shO, dt)
    o12 = _alloc_sb(12, nO, shO, dt)
    o13 = _alloc_sb(13, nO, shO, dt)
    o14 = _alloc_sb(14, nO, shO, dt)
    o15 = _alloc_sb(15, nO, shO, dt)

    t0 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    t1 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    QS = 2
    if nch > 2:
        QS = nch
    q0 = nl.ndarray((P, QS, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)
    q1 = nl.ndarray((P, QS, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)
    p0 = nl.ndarray((P, gT, S), dtype=dt, buffer=nl.psum)
    p1 = nl.ndarray((P, gT, S), dtype=dt, buffer=nl.psum)

    gO = 1
    if nch == 1:
        gO = 2
    nTG = (full + gT - 1) // gT
    nOG = (full + gO - 1) // gO
    if rem > 0:
        nTG = nTG + 1
        nOG = nOG + 1
    nJ = max(nTG, nOG)

    # prologue: prefetch the first L+1 chunks
    for c in range(L + 1):
        sl = c % D
        _g_load_chunk(A, B,
                      _pick(sl, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15),
                      _pick(sl, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15),
                      b_start + c * cb, S, T, full, rem, cb, MT)
    # transposes of batch 0 (copies split between vector and scalar)
    for j in range(nTG):
        _g_tr_group(a0, 0, t0, _sel2(j, p0, p1),
                    _grp_start(j, gT, full), _grp_size(j, gT, full),
                    _grp_rows(j, gT, full, T, rem), S, T, full, rem, MT, j % 2 == 1)

    for i in range(nb):
        c = i // cb
        w = i % cb
        if w == 0:
            if c >= 1:
                cn = c + L
                if cn < nck:
                    sl = cn % D
                    _g_load_chunk(A, B,
                                  _pick(sl, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15),
                                  _pick(sl, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15),
                                  b_start + cn * cb, S, T, full, rem, cb, MT)
        sc = c % D
        so = c % nO
        aTc = _sel2(i, t0, t1)
        bc = _pick(sc, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
        oc = _pick(so, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
        i1 = i + 1
        has_next = i1 < nb
        sa = (i1 // cb) % D
        an = _pick(sa, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)
        wn = i1 % cb
        aTn = _sel2(i1, t0, t1)
        bg = b_start + i
        is_last_chunk = c == nck - 1
        for j in range(nJ):
            if has_next:
                if j < nTG:
                    _g_tr_group(an, wn, aTn, _sel2(i1 * nTG + j, p0, p1),
                                _grp_start(j, gT, full), _grp_size(j, gT, full),
                                _grp_rows(j, gT, full, T, rem), S, T, full, rem, MT, False)
            if j < nOG:
                mt0 = _grp_start(j, gO, full)
                gm = _grp_size(j, gO, full)
                mm = _grp_rows(j, gO, full, T, rem)
                _g_out_group(aTc, bc, w, oc, _sel2(i * nOG + j, q0, q1),
                             mt0, gm, mm, S, T, full, rem, MT, nch, cw)
                if is_last_chunk:
                    _g_store_group(out, oc, bg, w, mt0, gm, mm, S, T, full, nch, cw)
        if c < nck - 1:
            if full > 0:
                _g_store_group(out, oc, bg, w, 0, full, T, S, T, full, nch, cw)
            if rem > 0:
                _g_store_group(out, oc, bg, w, full, 1, rem, S, T, full, nch, cw)


# ----------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------
@nki.jit
def _bmm_kernel(A, B, BATCH, S, mode, cb, D, nO, L, T, full, rem, nch, cw, gT):
    dt = A.dtype
    out = nl.ndarray((BATCH * S, S), dtype=dt, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    b_start = (pid * BATCH) // nprog
    b_end = ((pid + 1) * BATCH) // nprog
    nb = b_end - b_start
    if mode == 0:
        _packed_path(A, B, out, S, b_start, nb, cb, dt)
    else:
        _general_path(A, B, out, S, b_start, nb, cb, D, nO, L, T, full, rem, nch, cw, gT, dt)
    return out


# ----------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------
def _make_config(BATCH, M, N, K):
    S = int(M)
    nprog = 2
    nb_lo = BATCH // nprog
    nb_hi = BATCH - nb_lo
    mode = 1
    cb = 1
    if S <= 128 and BATCH % nprog == 0:
        c = min(nb_lo, 512 // S)
        found = 0
        while c >= 1:
            if nb_lo % c == 0 and (c * S) % 128 == 0:
                found = c
                break
            c -= 1
        if found > 0:
            mode = 0
            cb = found
    MT = (S + 127) // 128
    if S % MT == 0:
        T = S // MT
        full = MT
        rem = 0
    else:
        T = 128
        full = S // 128
        rem = S % 128
    nch = (S + 511) // 512
    while S % nch != 0:
        nch += 1
    cw = S // nch
    gT = 1
    if mode == 1:
        gT = max(1, min(max(full, 1), PSUM_F16_COLS // S))
    D = 1
    nO = 1
    L = 0
    if mode == 0:
        nck = nb_lo // cb
        D = min(nck, 4)
        nO = D
        L = D - 1
    else:
        cb = 1
        if BATCH % nprog == 0:
            while (cb * 2 <= nb_lo and nb_lo % (cb * 2) == 0
                   and cb * S * S * 2 < CHUNK_TARGET):
                cb = cb * 2
        nck = max(1, nb_hi // cb)
        unit = cb * MT * S * 2
        avail = SBUF_BUDGET - 2 * MT * S * 2
        nslots = max(avail // unit, 0)
        if nslots >= 3 * nck and nck <= MAXBUF:
            D = nck
            nO = nck
        else:
            nO = max(3, min(nck, nslots // 5))
            D = min(nck, MAXBUF, (nslots - nO) // 2)
            nO = min(nck, MAXBUF, nslots - 2 * D)
        D = max(D, min(2, nck))
        nO = max(nO, 1)
        L = min(D - 1, LMAX)
    return {
        "path": "packed" if mode == 0 else "tiled",
        "mode": mode,
        "cb": cb,
        "D": D,
        "nO": nO,
        "L": L,
        "T": T,
        "full": full,
        "rem": rem,
        "nch": nch,
        "cw": cw,
        "gT": gT,
        "lnc_split": "batch",
        "row_layout": "interleaved" if mode == 1 else "natural",
        "copy_split": "aT:vector,out:scalar",
    }


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    global _LAST_CONFIG
    BATCH = int(BATCH)
    M = int(M)
    N = int(N)
    K = int(K)
    cfg = _make_config(BATCH, M, N, K)
    A2 = A.view(BATCH * M, K)
    B2 = B.view(BATCH * K, N)
    out = _bmm_kernel(A2, B2, BATCH, M, cfg["mode"], cfg["cb"], cfg["D"], cfg["nO"],
                      cfg["L"], cfg["T"], cfg["full"], cfg["rem"], cfg["nch"], cfg["cw"],
                      cfg["gT"])
    _LAST_CONFIG = cfg
    return out.view(-1)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
