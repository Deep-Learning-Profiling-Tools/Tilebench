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
GROUP = 1024         # columns per PSUM tile / evacuation / load chunk / store
B_PERIOD = 4         # every B_PERIOD-th interior group uses the 3-matmul + ScalarE path
PREFETCH = 2         # input row tiles loaded ahead (ring of 4 tile buffers)

_LAST_CONFIG = {}


def _load_tile(x, xt, b, m, kind, cols, chunk):
    # partitions 0..nbody-1 = rows b..b+nbody-1 ; partition nbody = row b-1 (halo above)
    nbody = m + 1
    if kind == 2:
        nbody = m
    if kind != 1:
        nisa.dma_copy(dst=xt[nbody:nbody + 1, 0:cols], src=x[b - 1:b, 0:cols])
    nch = (cols + chunk - 1) // chunk
    for ci in range(nch):
        c0 = ci * chunk
        w = min(chunk, cols - c0)
        nisa.dma_copy(dst=xt[0:nbody, c0:c0 + w], src=x[b:b + nbody, c0:c0 + w])


def _band_plus(dst, fm, band4, c):
    # dst = band4 + 0.25 * (fm == c)
    e = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e, data=fm, op0=nl.equal, operand0=float(c),
                       op1=nl.multiply, operand1=0.25, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=band4, data2=e, op=nl.add)


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

    pg = GROUP
    if cols < pg:
        pg = cols
    ngroups = (cols + pg - 1) // pg

    # ---------------- input tile ring ----------------
    xa = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
    xb = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
    xc = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)
    xd = nl.ndarray((P, cols), dtype=dt, buffer=nl.sbuf)

    # prefetch the first tiles before any setup work
    for t in range(PREFETCH):
        if t < nt:
            m = base
            if t < rem:
                m = base + 1
            b = row0 + t * base + min(t, rem)
            kind = 0
            if b == 0:
                kind = 1
            if b + m == rows:
                kind = 2
            xn = xa
            if t % 4 == 1:
                xn = xb
            if t % 4 == 2:
                xn = xc
            if t % 4 == 3:
                xn = xd
            _load_tile(x, xn, b, m, kind, cols, pg)

    # ---------------- which coefficient matrices are needed ----------------
    has_first = False
    if row0 == 0:
        has_first = True
    has_last = False
    if row0 + R2 == rows:
        has_last = True
    m_last = base
    if nt - 1 < rem:
        m_last = base + 1
    need_hi = False
    need_lo = False
    for t in range(nt):
        m = base
        if t < rem:
            m = base + 1
        b = row0 + t * base + min(t, rem)
        kind = 0
        if b == 0:
            kind = 1
        if b + m == rows:
            kind = 2
        if kind == 0:
            if t < rem:
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

    sq = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=fm, data2=fm, op=nl.multiply)
    band4 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=band4, data=sq, op0=nl.equal, operand0=1.0,
                       op1=nl.multiply, operand1=0.25, engine=nisa.engine.vector)
    shm = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shm, data=fm, op0=nl.equal, operand0=0.0,
                       op1=nl.multiply, operand1=0.25, engine=nisa.engine.vector)

    sv_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_hi = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_lo = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)

    if has_first:
        # first tile: output row 0 is boundary (exact centre copy), no row above
        nisa.tensor_copy(dst=sv_f, src=band4)
        nisa.tensor_scalar(dst=sv_f[:, 0:1], data=fm[:, 0:1], op0=nl.equal,
                           operand0=0.0, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=sh_f, src=shm)
        nisa.memset(dst=sh_f[:, 0:1], value=0.0)
    if need_hi:
        # interior tile with M = base+1 rows: halo-above at partition M+1
        _band_plus(sv_hi, fm, band4, -(base + 2))
    if need_lo:
        _band_plus(sv_lo, fm, band4, -(base + 1))
    if has_last:
        # last tile: halo-above at partition m_last, last row is boundary
        _band_plus(sv_l, fm, band4, -m_last)
        nisa.tensor_scalar(dst=sv_l[:, m_last - 1:m_last], data=fm[:, m_last - 1:m_last],
                           op0=nl.equal, operand0=0.0, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=sh_l, src=shm)
        nisa.memset(dst=sh_l[:, m_last - 1:m_last], value=0.0)

    # ---------------- PSUM and output staging rings ----------------
    pa = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)
    pb = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)
    pc = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)
    oa = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
    ob = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
    oc = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
    od = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)

    # ---------------- main sweep ----------------
    icount = 0
    gidx = 0
    for t in range(nt):
        # prefetch tile t + PREFETCH
        tn = t + PREFETCH
        if tn < nt:
            m1 = base
            if tn < rem:
                m1 = base + 1
            b1 = row0 + tn * base + min(tn, rem)
            kind1 = 0
            if b1 == 0:
                kind1 = 1
            if b1 + m1 == rows:
                kind1 = 2
            xn = xa
            if tn % 4 == 1:
                xn = xb
            if tn % 4 == 2:
                xn = xc
            if tn % 4 == 3:
                xn = xd
            _load_tile(x, xn, b1, m1, kind1, cols, pg)

        m = base
        if t < rem:
            m = base + 1
        b = row0 + t * base + min(t, rem)
        kind = 0
        if b == 0:
            kind = 1
        if b + m == rows:
            kind = 2

        xt = xa
        if t % 4 == 1:
            xt = xb
        if t % 4 == 2:
            xt = xc
        if t % 4 == 3:
            xt = xd

        kk = m + 2
        if kind != 0:
            kk = m + 1

        sv = sv_lo
        sh = shm
        if kind == 0:
            if t < rem:
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

            path_b = True
            if kind == 0:
                if icount % B_PERIOD != B_PERIOD - 1:
                    path_b = False
                icount += 1

            ps = pa
            if gidx % 3 == 1:
                ps = pb
            if gidx % 3 == 2:
                ps = pc
            o = oa
            if gidx % 4 == 1:
                o = ob
            if gidx % 4 == 2:
                o = oc
            if gidx % 4 == 3:
                o = od
            gidx += 1

            # vertical neighbours (+ boundary-row centre copy)
            for bi in range(nb):
                c0 = g0 + bi * BLK
                cl = max(c0, lo)
                ch = min(c0 + BLK, hi)
                if cl < ch:
                    nisa.nc_matmul(dst=ps[0:m, cl - g0:ch - g0],
                                   stationary=sv[0:kk, 0:m],
                                   moving=xt[0:kk, cl:ch],
                                   accumulate=False)
            # left neighbour
            for bi in range(nb):
                c0 = g0 + bi * BLK
                cl = max(c0, lo)
                ch = min(c0 + BLK, hi)
                if cl < ch:
                    nisa.nc_matmul(dst=ps[0:m, cl - g0:ch - g0],
                                   stationary=sh[0:kk, 0:m],
                                   moving=xt[0:kk, cl - 1:ch - 1],
                                   accumulate=True)
            if path_b:
                # right neighbour on TensorE
                for bi in range(nb):
                    c0 = g0 + bi * BLK
                    cl = max(c0, lo)
                    ch = min(c0 + BLK, hi)
                    if cl < ch:
                        nisa.nc_matmul(dst=ps[0:m, cl - g0:ch - g0],
                                       stationary=sh[0:kk, 0:m],
                                       moving=xt[0:kk, cl + 1:ch + 1],
                                       accumulate=True)
                nisa.tensor_copy(dst=o[0:m, lo - g0:hi - g0],
                                 src=ps[0:m, lo - g0:hi - g0],
                                 engine=nisa.engine.scalar)
            else:
                # out = 0.25 * right + psum   (single VectorE pass, fp32 internal)
                nisa.scalar_tensor_tensor(dst=o[0:m, lo - g0:hi - g0],
                                          data=xt[0:m, lo + 1:hi + 1],
                                          op0=nl.multiply, operand0=0.25,
                                          data1=ps[0:m, lo - g0:hi - g0],
                                          op1=nl.add)

            # boundary columns: exact centre copy
            if g0 == 0:
                nisa.tensor_copy(dst=o[0:m, 0:1], src=xt[0:m, 0:1],
                                 engine=nisa.engine.gpsimd)
            if g0 + gw == cols:
                nisa.tensor_copy(dst=o[0:m, gw - 1:gw], src=xt[0:m, cols - 1:cols],
                                 engine=nisa.engine.gpsimd)

            nisa.dma_copy(dst=out[b:b + m, g0:g0 + gw], src=o[0:m, 0:gw])

    return out


def _config_for(rows, cols):
    n_prog = 2
    r2 = rows // n_prog
    nt = (r2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    pg = GROUP if cols >= GROUP else cols
    return {
        "programs": n_prog,
        "row_tile_max": ROW_TILE_MAX,
        "row_tiles_per_program": nt,
        "rows_per_tile_base": r2 // nt,
        "matmul_block_cols": BLK,
        "group_cols": pg,
        "b_period": B_PERIOD,
        "prefetch_tiles": PREFETCH,
        "xt_ring": 4,
        "psum_ring": 3,
        "out_ring": 4,
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
