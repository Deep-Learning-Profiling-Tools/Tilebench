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
PSUM_GROUP = 1024    # columns per PSUM tile / evacuation / output store
LOAD_CHUNK = 2048    # columns per input-load DMA
HYB_PERIOD = 5       # every HYB_PERIOD-th interior group uses the 3-matmul path

_LAST_CONFIG = {}


def _ind(dst, fm, c):
    # dst = (fm == c) ? 1 : 0
    nisa.tensor_scalar(dst=dst, data=fm, op0=nl.equal, operand0=float(c),
                       engine=nisa.engine.vector)


def _scale(dst, src, s):
    # dst = src * s   (cast to dst dtype)
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=float(s),
                       engine=nisa.engine.vector)


@nki.jit
def _jacobi_kernel(x):
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
    m_hi = base + 1
    m_lo = base
    m_last = base
    if nt - 1 < rem:
        m_last = base + 1

    pg = PSUM_GROUP
    if cols < pg:
        pg = cols
    ngroups = (cols + pg - 1) // pg
    nchunks = (cols + LOAD_CHUNK - 1) // LOAD_CHUNK

    # ------------------------------------------------------------------
    # Coefficient (stationary) matrices, F[k, m] = m - k
    # Layout of an input tile: partitions 0..M-1 = output rows b..b+M-1
    # (centre aligned with output partition), then the row below (b+M),
    # then the row above (b-1).
    # ------------------------------------------------------------------
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

    cen = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(cen, fm, 0)
    upi = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(upi, fm, 1)
    dni = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(dni, fm, -1)
    band = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=band, data1=upi, data2=dni, op=nl.add)

    # centre selector (1.0) and horizontal coefficient (0.25 at centre)
    sc = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sc, src=cen)
    shm = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(shm, cen, 0.25)

    # first tile of the grid (row 0 is boundary; no row above)
    sv_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(sv_f, band, 0.25)
    nisa.tensor_copy(dst=sv_f[:, 0:1], src=sc[:, 0:1])
    sh_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sh_f, src=shm)
    nisa.memset(dst=sh_f[:, 0:1], value=0.0)

    # last tile of the grid (row rows-1 is boundary; row above at partition M)
    hl = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(hl, fm, -m_last)
    sl = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sl, data1=band, data2=hl, op=nl.add)
    sv_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(sv_l, sl, 0.25)
    nisa.tensor_copy(dst=sv_l[:, m_last - 1:m_last], src=sc[:, m_last - 1:m_last])
    sh_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sh_l, src=shm)
    nisa.memset(dst=sh_l[:, m_last - 1:m_last], value=0.0)

    # interior tiles: row above lives at partition M+1
    hh = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(hh, fm, -(m_hi + 1))
    shh = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=shh, data1=band, data2=hh, op=nl.add)
    sv_hi = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(sv_hi, shh, 0.25)

    hlo = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _ind(hlo, fm, -(m_lo + 1))
    slo = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=slo, data1=band, data2=hlo, op=nl.add)
    sv_lo = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    _scale(sv_lo, slo, 0.25)

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------
    gcount = 0
    for t in range(nt):
        m = m_lo
        if t < rem:
            m = m_hi
        b = row0 + t * base + min(t, rem)
        kind = 0
        if b == 0:
            kind = 1
        if b + m == rows:
            kind = 2

        # ---- load input tile ----
        xt = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
        nbody = m + 1
        if kind == 2:
            nbody = m
        for ci in range(nchunks):
            c0 = ci * LOAD_CHUNK
            w = min(LOAD_CHUNK, cols - c0)
            nisa.dma_copy(dst=xt[0:nbody, c0:c0 + w], src=x[b:b + nbody, c0:c0 + w])
        kk = nbody
        if kind != 1:
            nisa.dma_copy(dst=xt[nbody:nbody + 1, 0:cols], src=x[b - 1:b, 0:cols])
            kk = nbody + 1

        sv = sv_lo
        sh = shm
        if kind == 1:
            sv = sv_f
            sh = sh_f
        if kind == 2:
            assert m == m_last, "last tile size mismatch"
            sv = sv_l
            sh = sh_l
        if kind == 0:
            if m == m_hi:
                sv = sv_hi

        for g in range(ngroups):
            g0 = g * pg
            gw = min(pg, cols - g0)
            nb = gw // BLK
            use_b = False
            if kind == 0:
                if gcount % HYB_PERIOD != HYB_PERIOD - 1:
                    use_b = True
                gcount += 1

            ps = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)
            o = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)

            # vertical neighbours (+ boundary-row centre copy)
            for bi in range(nb):
                c0 = g0 + bi * BLK
                lo = max(c0, 1)
                hi = min(c0 + BLK, cols - 1)
                if lo < hi:
                    nisa.nc_matmul(dst=ps[0:m, lo - g0:hi - g0],
                                   stationary=sv[0:kk, 0:m],
                                   moving=xt[0:kk, lo:hi],
                                   accumulate=False)
            # left neighbour
            for bi in range(nb):
                c0 = g0 + bi * BLK
                lo = max(c0, 1)
                hi = min(c0 + BLK, cols - 1)
                if lo < hi:
                    nisa.nc_matmul(dst=ps[0:m, lo - g0:hi - g0],
                                   stationary=sh[0:kk, 0:m],
                                   moving=xt[0:kk, lo - 1:hi - 1],
                                   accumulate=True)
            # right neighbour (3-matmul path)
            if use_b == False:
                for bi in range(nb):
                    c0 = g0 + bi * BLK
                    lo = max(c0, 1)
                    hi = min(c0 + BLK, cols - 1)
                    if lo < hi:
                        nisa.nc_matmul(dst=ps[0:m, lo - g0:hi - g0],
                                       stationary=sh[0:kk, 0:m],
                                       moving=xt[0:kk, lo + 1:hi + 1],
                                       accumulate=True)
            # boundary columns: exact centre copy
            if g0 == 0:
                nisa.nc_matmul(dst=ps[0:m, 0:1],
                               stationary=sc[0:kk, 0:m],
                               moving=xt[0:kk, 0:1],
                               accumulate=False)
            if g0 + gw == cols:
                nisa.nc_matmul(dst=ps[0:m, gw - 1:gw],
                               stationary=sc[0:kk, 0:m],
                               moving=xt[0:kk, cols - 1:cols],
                               accumulate=False)

            if use_b:
                # r4 = 0.25 * right neighbour (0 on boundary columns)
                r4 = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
                lo_g = max(g0, 1)
                hi_g = min(g0 + gw, cols - 1)
                nisa.activation(dst=r4[0:m, lo_g - g0:hi_g - g0], op=nl.copy,
                                data=xt[0:m, lo_g + 1:hi_g + 1], scale=0.25)
                if g0 == 0:
                    nisa.memset(dst=r4[0:m, 0:1], value=0.0)
                if g0 + gw == cols:
                    nisa.memset(dst=r4[0:m, gw - 1:gw], value=0.0)
                nisa.tensor_tensor(dst=o[0:m, 0:gw], data1=ps[0:m, 0:gw],
                                   data2=r4[0:m, 0:gw], op=nl.add,
                                   engine=nisa.engine.vector)
            else:
                nisa.tensor_copy(dst=o[0:m, 0:gw], src=ps[0:m, 0:gw],
                                 engine=nisa.engine.scalar)

            nisa.dma_copy(dst=out[b:b + m, g0:g0 + gw], src=o[0:m, 0:gw])

    return out


def _config_for(rows, cols):
    n_prog = 2
    r2 = rows // n_prog
    nt = (r2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    pg = PSUM_GROUP if cols >= PSUM_GROUP else cols
    return {
        "programs": n_prog,
        "row_tile_max": ROW_TILE_MAX,
        "row_tiles_per_program": nt,
        "rows_per_tile_base": r2 // nt,
        "matmul_block_cols": BLK,
        "psum_group_cols": pg,
        "load_chunk_cols": LOAD_CHUNK,
        "hybrid_period": HYB_PERIOD,
    }


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG
    r = int(input.shape[0])
    c = int(input.shape[1])
    out = _jacobi_kernel(input)
    _LAST_CONFIG = _config_for(r, c)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
