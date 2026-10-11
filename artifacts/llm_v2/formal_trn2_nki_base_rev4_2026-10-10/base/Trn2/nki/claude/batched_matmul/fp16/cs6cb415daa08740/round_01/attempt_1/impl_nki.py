import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
PSUM_F32_COLS = 512    # one PSUM bank of fp32
PSUM_F16_COLS = 1024   # one PSUM bank of fp16

_LAST_CONFIG = {}


# ----------------------------------------------------------------------------
# small helpers (trace-time buffer selection)
# ----------------------------------------------------------------------------
def _sel2(j, t0, t1):
    t = t0
    if j % 2 == 1:
        t = t1
    return t


def _sel3(j, t0, t1, t2):
    t = t0
    r = j % 3
    if r == 1:
        t = t1
    if r == 2:
        t = t2
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


# ----------------------------------------------------------------------------
# packed path: S <= 128, several batches per chunk
# ----------------------------------------------------------------------------
def _small_load(A, B, a_t, b_t, c0, S, cb, G):
    # a_t[p, g, k] = A2[c0*S + g*128 + p, k]
    nisa.dma_copy(dst=a_t[0:P, 0:G, 0:S],
                  src=A.ap(pattern=[[S, P], [P * S, G], [1, S]], offset=c0 * S * S))
    # b_t[k, i, n] = B[c0 + i, k, n]
    nisa.dma_copy(dst=b_t[0:S, 0:cb, 0:S],
                  src=B.ap(pattern=[[S, S], [S * S, cb], [1, S]], offset=c0 * S * S))


def _small_transpose(a_t, aT_t, S, G, dt):
    pt = nl.ndarray((P, PSUM_F16_COLS), dtype=dt, buffer=nl.psum)
    for g in range(G):
        nisa.nc_transpose(dst=pt[0:S, g * P:(g + 1) * P],
                          data=a_t[0:P, g, 0:S],
                          engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=aT_t[0:S, 0:G * P], src=pt[0:S, 0:G * P],
                     engine=nisa.engine.vector)


def _small_step(aT_c, b_c, o_c, a_n, aT_n, has_next, out, c0, S, cb, G, q, dt):
    for i in range(cb):
        nisa.nc_matmul(dst=q[0:S, i, 0:S],
                       stationary=aT_c[0:S, i * S:(i + 1) * S],
                       moving=b_c[0:S, i, 0:S],
                       accumulate=False)
    nisa.tensor_copy(dst=o_c[0:S, 0:cb, 0:S], src=q[0:S, 0:cb, 0:S],
                     engine=nisa.engine.vector)
    nisa.dma_copy(dst=out.ap(pattern=[[S, S], [S * S, cb], [1, S]], offset=c0 * S * S),
                  src=o_c[0:S, 0:cb, 0:S])
    if has_next:
        _small_transpose(a_n, aT_n, S, G, dt)


def _small_path(A, B, out, S, b_start, b_end, cb, dt):
    nb = b_end - b_start
    nck = nb // cb
    G = (cb * S) // P
    w = PSUM_F32_COLS // cb

    a0 = nl.ndarray((P, G, S), dtype=dt, buffer=nl.sbuf)
    a1 = nl.ndarray((P, G, S), dtype=dt, buffer=nl.sbuf)
    a2 = nl.ndarray((P, G, S), dtype=dt, buffer=nl.sbuf)
    x0 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    x1 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    x2 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    t0 = nl.ndarray((S, G * P), dtype=dt, buffer=nl.sbuf)
    t1 = nl.ndarray((S, G * P), dtype=dt, buffer=nl.sbuf)
    o0 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    o1 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    o2 = nl.ndarray((S, cb, S), dtype=dt, buffer=nl.sbuf)
    q0 = nl.ndarray((P, cb, w), dtype=nl.float32, buffer=nl.psum)
    q1 = nl.ndarray((P, cb, w), dtype=nl.float32, buffer=nl.psum)

    _small_load(A, B, a0, x0, b_start, S, cb, G)
    if nck > 1:
        _small_load(A, B, a1, x1, b_start + cb, S, cb, G)
    _small_transpose(a0, t0, S, G, dt)
    for ci in range(nck):
        if ci + 2 < nck:
            _small_load(A, B, _sel3(ci + 2, a0, a1, a2), _sel3(ci + 2, x0, x1, x2),
                        b_start + (ci + 2) * cb, S, cb, G)
        _small_step(_sel2(ci, t0, t1), _sel3(ci, x0, x1, x2), _sel3(ci, o0, o1, o2),
                    _sel3(ci + 1, a0, a1, a2), _sel2(ci + 1, t0, t1), ci + 1 < nck,
                    out, b_start + ci * cb, S, cb, G, _sel2(ci, q0, q1), dt)


# ----------------------------------------------------------------------------
# tiled path: one batch per step, T-row tiles (+ optional remainder tile)
# ----------------------------------------------------------------------------
def _large_load(X, x_t, b, S, T, full, rem):
    base = b * S
    if full > 0:
        nisa.dma_copy(dst=x_t[0:T, 0:full, 0:S],
                      src=X.ap(pattern=[[S, T], [T * S, full], [1, S]], offset=base * S))
    if rem > 0:
        nisa.dma_copy(dst=x_t[0:rem, full, 0:S], src=X[base + full * T:base + S, 0:S])


def _large_store(out, o_t, b, S, T, full, rem):
    base = b * S
    if full > 0:
        nisa.dma_copy(dst=out.ap(pattern=[[S, T], [T * S, full], [1, S]], offset=base * S),
                      src=o_t[0:T, 0:full, 0:S])
    if rem > 0:
        nisa.dma_copy(dst=out[base + full * T:base + S, 0:S], src=o_t[0:rem, full, 0:S])


def _large_transpose_kt(a_t, aT_t, kt, S, T, full, rem, MT, dt):
    kk = T
    if kt >= full:
        kk = rem
    pt = nl.ndarray((P, PSUM_F16_COLS), dtype=dt, buffer=nl.psum)
    for mt in range(MT):
        mm = T
        if mt >= full:
            mm = rem
        nisa.nc_transpose(dst=pt[0:kk, mt * T:mt * T + mm],
                          data=a_t[0:mm, mt, kt * T:kt * T + kk],
                          engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=aT_t[0:kk, kt, 0:S], src=pt[0:kk, 0:S],
                     engine=nisa.engine.vector)


def _large_step(aT_c, b_c, o_c, a_n, aT_n, has_next, out, bidx, li,
                S, T, full, rem, MT, nch, cw, q0, q1, q2, q3, dt):
    for mt in range(MT):
        mm = T
        if mt >= full:
            mm = rem
        for c in range(nch):
            n0 = c * cw
            nn = min(cw, S - n0)
            gidx = (li * MT + mt) * nch + c
            q = _sel4(gidx, q0, q1, q2, q3)
            for kt in range(MT):
                kk = T
                if kt >= full:
                    kk = rem
                nisa.nc_matmul(dst=q[0:mm, 0:nn],
                               stationary=aT_c[0:kk, kt, mt * T:mt * T + mm],
                               moving=b_c[0:kk, kt, n0:n0 + nn],
                               accumulate=(kt > 0))
            if gidx % 2 == 0:
                nisa.tensor_copy(dst=o_c[0:mm, mt, n0:n0 + nn], src=q[0:mm, 0:nn],
                                 engine=nisa.engine.scalar)
            else:
                nisa.tensor_copy(dst=o_c[0:mm, mt, n0:n0 + nn], src=q[0:mm, 0:nn],
                                 engine=nisa.engine.vector)
        if has_next:
            _large_transpose_kt(a_n, aT_n, mt, S, T, full, rem, MT, dt)
    _large_store(out, o_c, bidx, S, T, full, rem)


def _large_path(A, B, out, S, b_start, b_end, T, full, rem, nch, cw, dt):
    MT = full
    if rem > 0:
        MT = full + 1
    nb = b_end - b_start

    a0 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    a1 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    a2 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    x0 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    x1 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    x2 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    t0 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    t1 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    o0 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    o1 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    o2 = nl.ndarray((T, MT, S), dtype=dt, buffer=nl.sbuf)
    q0 = nl.ndarray((P, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)
    q1 = nl.ndarray((P, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)
    q2 = nl.ndarray((P, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)
    q3 = nl.ndarray((P, PSUM_F32_COLS), dtype=nl.float32, buffer=nl.psum)

    _large_load(A, a0, b_start, S, T, full, rem)
    _large_load(B, x0, b_start, S, T, full, rem)
    if nb > 1:
        _large_load(A, a1, b_start + 1, S, T, full, rem)
        _large_load(B, x1, b_start + 1, S, T, full, rem)
    for kt in range(MT):
        _large_transpose_kt(a0, t0, kt, S, T, full, rem, MT, dt)
    for i in range(nb):
        if i + 2 < nb:
            _large_load(A, _sel3(i + 2, a0, a1, a2), b_start + i + 2, S, T, full, rem)
            _large_load(B, _sel3(i + 2, x0, x1, x2), b_start + i + 2, S, T, full, rem)
        _large_step(_sel2(i, t0, t1), _sel3(i, x0, x1, x2), _sel3(i, o0, o1, o2),
                    _sel3(i + 1, a0, a1, a2), _sel2(i + 1, t0, t1), i + 1 < nb,
                    out, b_start + i, i, S, T, full, rem, MT, nch, cw,
                    q0, q1, q2, q3, dt)


# ----------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------
@nki.jit
def _bmm_kernel(A, B, BATCH, S, mode, cb, T, full, rem, nch, cw):
    dt = A.dtype
    out = nl.ndarray((BATCH * S, S), dtype=dt, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    b_start = (pid * BATCH) // nprog
    b_end = ((pid + 1) * BATCH) // nprog
    if mode == 0:
        _small_path(A, B, out, S, b_start, b_end, cb, dt)
    else:
        _large_path(A, B, out, S, b_start, b_end, T, full, rem, nch, cw, dt)
    return out


# ----------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------
def _make_config(BATCH, M, N, K):
    S = int(M)
    nprog = 2
    mode = 1
    cb = 1
    if S <= 128 and BATCH % nprog == 0:
        nb = BATCH // nprog
        c = min(nb, 512 // S)
        found = 0
        while c >= 1:
            if nb % c == 0 and (c * S) % 128 == 0:
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
    cw = (S + nch - 1) // nch
    return {
        "path": "packed" if mode == 0 else "tiled",
        "mode": mode,
        "cb": cb,
        "T": T,
        "full": full,
        "rem": rem,
        "nch": nch,
        "cw": cw,
        "lnc_split": "batch",
        "sbuf_pools": "a3/b3/aT2/o3",
        "psum_out_pool": 4 if mode == 1 else 2,
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
    out = _bmm_kernel(A2, B2, BATCH, M, cfg["mode"], cfg["cb"], cfg["T"],
                      cfg["full"], cfg["rem"], cfg["nch"], cfg["cw"])
    _LAST_CONFIG = cfg
    return out.view(-1)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
