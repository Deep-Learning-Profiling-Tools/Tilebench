import torch

try:
    import torch_neuronx  # noqa: F401  (registers the "neuron" device if not autoloaded)
except Exception:
    pass

import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Fixed configuration (literals; derived quantities are computed from shapes)
# ---------------------------------------------------------------------------
ROW_TILE_MAX = 126   # max output rows per row tile (contraction K = M + 2 <= 128)
BLK = 512            # matmul moving free size (one PSUM bank of fp32)
GROUP = 1024         # columns per PSUM tile / evacuation / output store
LOAD_CHUNK = 2048    # columns per input-load DMA
B_PERIOD_STT = 3     # every 3rd interior group: 3 matmuls + ScalarE copy
B_PERIOD_FALLBACK = 5
PREFETCH = 2         # row tiles loaded ahead


def _detect_stt_mode():
    # Host-side check of the installed API: is scalar_tensor_tensor usable as
    # (data op0 operand0) op1 operand1 ?
    try:
        import inspect
        names = list(inspect.signature(nisa.scalar_tensor_tensor).parameters.keys())
    except Exception:
        return 0
    for n in ("dst", "data", "op0", "operand0", "op1", "operand1"):
        if n not in names:
            return 0
    if "engine" in names:
        return 2
    return 1


_STT_MODE = _detect_stt_mode()
_LAST_CONFIG = {}


def _tile_m(t, base, rem):
    m = base
    if t < rem:
        m = base + 1
    return m


def _tile_b(t, row0, base, rem):
    return row0 + t * base + min(t, rem)


def _tile_kind(b, m, rows):
    kind = 0
    if b == 0:
        kind = 1
    if b + m == rows:
        kind = 2
    return kind


def _load_tile(x, xt, b, m, kind, cols, chunk):
    # partitions 0..nbody-1 = rows b..b+nbody-1 ; partition nbody = row b-1 (halo above)
    nbody = m + 1
    if kind == 2:
        nbody = m
    nch = (cols + chunk - 1) // chunk
    for ci in range(nch):
        c0 = ci * chunk
        w = min(chunk, cols - c0)
        nisa.dma_copy(dst=xt[0:nbody, c0:c0 + w], src=x[b:b + nbody, c0:c0 + w])
    if kind != 1:
        nisa.dma_copy(dst=xt[nbody:nbody + 1, 0:cols], src=x[b - 1:b, 0:cols])


def _ind(dst, fm, c):
    # dst = (fm == c) ? 1 : 0
    nisa.tensor_scalar(dst=dst, data=fm, op0=nl.equal, operand0=float(c),
                       engine=nisa.engine.vector)


def _scale(dst, src, s):
    # dst = src * s   (cast to dst dtype)
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=float(s),
                       engine=nisa.engine.vector)


def _interior_sv(dst, fm, band, c):
    # dst = 0.25 * (band + (fm == c))
    e = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    _ind(e, fm, c)
    s = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=band, data2=e, op=nl.add)
    _scale(dst, s, 0.25)


def _mm_pass(ps, stat, xt, m, kk, g0, nb, lo, hi, shift, acc):
    for bi in range(nb):
        c0 = g0 + bi * BLK
        cl = max(c0, lo)
        ch = min(c0 + BLK, hi)
        if cl < ch:
            nisa.nc_matmul(dst=ps[0:m, cl - g0:ch - g0],
                           stationary=stat[0:kk, 0:m],
                           moving=xt[0:kk, cl + shift:ch + shift],
                           accumulate=acc)


@nki.jit
def _jacobi_kernel(x, mode):
    rows = x.shape[0]
    cols = x.shape[1]
    dt = x.dtype
    out = nl.ndarray((rows, cols), dtype=dt, buffer=nl.shared_hbm)

    P = 128
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R2 = rows // n_prog
    row0 = pid * R2
    nt = (R2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    base = R2 // nt
    rem = R2 % nt

    pg = GROUP
    if cols < pg:
        pg = cols
    ngroups = (cols + pg - 1) // pg
    lchunk = LOAD_CHUNK
    if cols < lchunk:
        lchunk = cols

    bper = B_PERIOD_STT
    if mode == 0:
        bper = B_PERIOD_FALLBACK

    # ---------------- prefetch the first two row tiles ----------------
    xq0 = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
    m0 = _tile_m(0, base, rem)
    b0 = _tile_b(0, row0, base, rem)
    k0 = _tile_kind(b0, m0, rows)
    _load_tile(x, xq0, b0, m0, k0, cols, lchunk)
    xq1 = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
    if nt > 1:
        m1 = _tile_m(1, base, rem)
        b1 = _tile_b(1, row0, base, rem)
        k1 = _tile_kind(b1, m1, rows)
        _load_tile(x, xq1, b1, m1, k1, cols, lchunk)

    # ---------------- which coefficient matrices are needed ----------------
    has_first = False
    has_last = False
    need_hi = False
    need_lo = False
    m_last = base
    for t in range(nt):
        mt = _tile_m(t, base, rem)
        bt = _tile_b(t, row0, base, rem)
        kt = _tile_kind(bt, mt, rows)
        if kt == 1:
            has_first = True
        if kt == 2:
            has_last = True
            m_last = mt
        if kt == 0:
            if mt == base + 1:
                need_hi = True
            else:
                need_lo = True

    # ---------------- coefficient matrices: F[k, j] = j - k ----------------
    kio = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=kio, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    jio = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=jio, pattern=[[1, P]], offset=0, channel_multiplier=0)
    kf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kf, src=kio)
    jf = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=jf, src=jio)
    fm = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=fm, data=jf, op0=nl.subtract, operand0=kf)

    upi = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(upi, fm, 1)
    dni = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(dni, fm, -1)
    band = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=band, data1=upi, data2=dni, op=nl.add)
    cen = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(cen, fm, 0)
    shm = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(shm, cen, 0.25)

    sv_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_hi = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_lo = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)

    if has_first:
        # grid row 0 is boundary: exact centre copy, no horizontal terms
        _scale(sv_f, band, 0.25)
        nisa.tensor_copy(dst=sv_f[:, 0:1], src=cen[:, 0:1])
        nisa.tensor_copy(dst=sh_f, src=shm)
        nisa.memset(dst=sh_f[:, 0:1], value=0.0)
    if need_hi:
        _interior_sv(sv_hi, fm, band, -(base + 2))
    if need_lo:
        _interior_sv(sv_lo, fm, band, -(base + 1))
    if has_last:
        # halo above at partition m_last; grid row rows-1 is boundary
        _interior_sv(sv_l, fm, band, -m_last)
        nisa.tensor_copy(dst=sv_l[:, m_last - 1:m_last], src=cen[:, m_last - 1:m_last])
        nisa.tensor_copy(dst=sh_l, src=shm)
        nisa.memset(dst=sh_l[:, m_last - 1:m_last], value=0.0)

    # ---------------- main sweep ----------------
    icount = 0
    for t in range(nt):
        xt = xq0
        tn = t + PREFETCH
        if tn < nt:
            mn = _tile_m(tn, base, rem)
            bn = _tile_b(tn, row0, base, rem)
            kn = _tile_kind(bn, mn, rows)
            xn = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
            _load_tile(x, xn, bn, mn, kn, cols, lchunk)
            xq0 = xq1
            xq1 = xn
        else:
            xq0 = xq1

        m = _tile_m(t, base, rem)
        b = _tile_b(t, row0, base, rem)
        kind = _tile_kind(b, m, rows)

        kk = m + 2
        if kind != 0:
            kk = m + 1

        sv = sv_lo
        sh = shm
        if kind == 0:
            if m == base + 1:
                sv = sv_hi
        if kind == 1:
            sv = sv_f
            sh = sh_f
        if kind == 2:
            sv = sv_l
            sh = sh_l

        for g in range(ngroups):
            g0 = g * pg
            gw = min(pg, cols - g0)
            lo = max(g0, 1)
            hi = min(g0 + gw, cols - 1)
            nb = (gw + BLK - 1) // BLK

            path_a = False
            if kind == 0:
                if icount % bper != bper - 1:
                    path_a = True
                icount += 1

            ps = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)
            o = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)

            # vertical neighbours (+ boundary-row centre copy)
            _mm_pass(ps, sv, xt, m, kk, g0, nb, lo, hi, 0, False)
            # left neighbour
            _mm_pass(ps, sh, xt, m, kk, g0, nb, lo, hi, -1, True)

            if path_a:
                if mode == 0:
                    r4 = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=r4[0:m, lo - g0:hi - g0], op=nl.copy,
                                    data=xt[0:m, lo + 1:hi + 1], scale=0.25)
                    nisa.tensor_tensor(dst=o[0:m, lo - g0:hi - g0],
                                       data1=ps[0:m, lo - g0:hi - g0],
                                       data2=r4[0:m, lo - g0:hi - g0],
                                       op=nl.add, engine=nisa.engine.vector)
                elif mode == 1:
                    nisa.scalar_tensor_tensor(dst=o[0:m, lo - g0:hi - g0],
                                              data=xt[0:m, lo + 1:hi + 1],
                                              op0=nl.multiply, operand0=0.25,
                                              op1=nl.add,
                                              operand1=ps[0:m, lo - g0:hi - g0])
                else:
                    nisa.scalar_tensor_tensor(dst=o[0:m, lo - g0:hi - g0],
                                              data=xt[0:m, lo + 1:hi + 1],
                                              op0=nl.multiply, operand0=0.25,
                                              op1=nl.add,
                                              operand1=ps[0:m, lo - g0:hi - g0],
                                              engine=nisa.engine.vector)
            else:
                # right neighbour on TensorE, then plain evacuation on ScalarE
                _mm_pass(ps, sh, xt, m, kk, g0, nb, lo, hi, 1, True)
                nisa.tensor_copy(dst=o[0:m, lo - g0:hi - g0],
                                 src=ps[0:m, lo - g0:hi - g0],
                                 engine=nisa.engine.scalar)

            # boundary columns: exact centre copy
            if g0 == 0:
                nisa.tensor_copy(dst=o[0:m, 0:1], src=xt[0:m, 0:1],
                                 engine=nisa.engine.gpsimd)
            if g0 + gw == cols:
                nisa.tensor_copy(dst=o[0:m, gw - 1:gw], src=xt[0:m, cols - 1:cols],
                                 engine=nisa.engine.gpsimd)

            nisa.dma_copy(dst=out[b:b + m, g0:g0 + gw], src=o[0:m, 0:gw])

    return out


def _config_for(rows, cols, mode):
    n_prog = 2
    r2 = rows // n_prog
    nt = (r2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    pg = GROUP if cols >= GROUP else cols
    lc = LOAD_CHUNK if cols >= LOAD_CHUNK else cols
    bper = B_PERIOD_STT if mode != 0 else B_PERIOD_FALLBACK
    return {
        "programs": n_prog,
        "row_tile_max": ROW_TILE_MAX,
        "row_tiles_per_program": nt,
        "rows_per_tile_base": r2 // nt,
        "matmul_block_cols": BLK,
        "group_cols": pg,
        "load_chunk_cols": lc,
        "b_period": bper,
        "prefetch_tiles": PREFETCH,
        "stt_mode": mode,
    }


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG
    r = int(input.shape[0])
    c = int(input.shape[1])
    mode = int(_STT_MODE)
    out = _jacobi_kernel(input, mode)
    _LAST_CONFIG = _config_for(r, c, mode)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
