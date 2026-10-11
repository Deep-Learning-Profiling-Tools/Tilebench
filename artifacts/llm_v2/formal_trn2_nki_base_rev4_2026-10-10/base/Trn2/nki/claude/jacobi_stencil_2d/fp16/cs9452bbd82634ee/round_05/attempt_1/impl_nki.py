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
ROW_TILE_MAX = 126      # max output rows per row tile (contraction K = M + 2 <= 128)
BLK = 512               # matmul moving free size (one PSUM bank of fp32)
PG_MAX = 1024           # columns per PSUM tile / evacuation instruction
SG_MAX = 2048           # columns per output SBUF tile / store DMA
LOAD_MAX = 4096         # max columns per input-load DMA
A_PERIOD = 6            # every A_PERIOD-th interior group: 3 matmuls + ScalarE evacuation
A_PERIOD_FALLBACK = 5
PF_BUDGET_BYTES = 65536  # SBUF bytes per partition reserved for in-flight input tiles
PF_MAX = 4               # max tiles loaded ahead of the one being computed
ELEM_BYTES = 2           # fp16


def _detect_stt_mode():
    # Host-side check of the installed API signature of scalar_tensor_tensor.
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


def _load_tile(x, xt, b, m, kind, cols, lch, nld):
    # partitions 0..nbody-1 = rows b..b+nbody-1 ; partition nbody = row b-1 (halo above)
    nbody = m + 1
    if kind == 2:
        nbody = m
    if kind != 1:
        nisa.dma_copy(dst=xt[nbody:nbody + 1, 0:cols], src=x[b - 1:b, 0:cols])
    for ci in range(nld):
        c0 = ci * lch
        w = min(lch, cols - c0)
        if w > 0:
            nisa.dma_copy(dst=xt[0:nbody, c0:c0 + w], src=x[b:b + nbody, c0:c0 + w])


def _alloc_load(x, t, row0, base, rem, rows, cols, lch, nld, dt):
    m = _tile_m(t, base, rem)
    b = _tile_b(t, row0, base, rem)
    k = _tile_kind(b, m, rows)
    xt = nl.ndarray((128, cols), dtype=dt, buffer=nl.sbuf)
    _load_tile(x, xt, b, m, k, cols, lch, nld)
    return xt


def _coef(dst, src, c, s):
    # dst = (src == c) * s
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.equal, operand0=float(c),
                       op1=nl.multiply, operand1=float(s), engine=nisa.engine.vector)


def _copy(dst, src, on_act):
    if on_act == 1:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.scalar)
    else:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)


def _build_interior(dst, band, fm, m, on_act):
    # band + 0.25 at (k = m+1, i = 0): row above lives at partition m+1
    _copy(dst, band, on_act)
    e = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e, data=fm[:, 0:1], op0=nl.equal, operand0=float(-(m + 1)),
                       op1=nl.multiply, operand1=0.25, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst[:, 0:1], data1=band[:, 0:1], data2=e, op=nl.add,
                       engine=nisa.engine.vector)


def _build_first(svf, shf, band, sh, sc, on_act):
    # grid row 0 (output row 0 of this tile) is boundary: exact centre copy
    _copy(svf, band, on_act)
    nisa.tensor_copy(dst=svf[:, 0:1], src=sc[:, 0:1], engine=nisa.engine.vector)
    _copy(shf, sh, on_act)
    nisa.memset(dst=shf[:, 0:1], value=0.0)


def _build_last(svl, shl, band, sh, sc, fm, m, on_act):
    # row above at partition m; grid row rows-1 (output row m-1) is boundary
    _copy(svl, band, on_act)
    e = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e, data=fm[:, 0:1], op0=nl.equal, operand0=float(-m),
                       op1=nl.multiply, operand1=0.25, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=svl[:, 0:1], data1=band[:, 0:1], data2=e, op=nl.add,
                       engine=nisa.engine.vector)
    nisa.tensor_copy(dst=svl[:, m - 1:m], src=sc[:, m - 1:m], engine=nisa.engine.vector)
    _copy(shl, sh, on_act)
    nisa.memset(dst=shl[:, m - 1:m], value=0.0)


def _mm_pass(ps, stat, xt, m, kk, g0, gw, cols, shift, acc):
    nb = (gw + BLK - 1) // BLK
    for bi in range(nb):
        c0 = g0 + bi * BLK
        cl = max(c0, 1)
        ch = min(min(c0 + BLK, g0 + gw), cols - 1)
        if cl < ch:
            nisa.nc_matmul(dst=ps[0:m, cl - g0:ch - g0],
                           stationary=stat[0:kk, 0:m],
                           moving=xt[0:kk, cl + shift:ch + shift],
                           accumulate=acc)


@nki.jit
def _jacobi_kernel(x, mode, pfd):
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

    pg = PG_MAX
    if cols < pg:
        pg = cols
    sg = SG_MAX
    if cols < sg:
        sg = cols
    nsg = (cols + sg - 1) // sg
    nld = (cols + LOAD_MAX - 1) // LOAD_MAX
    lch = (cols + nld - 1) // nld
    lch = ((lch + BLK - 1) // BLK) * BLK

    aper = A_PERIOD
    if mode == 0:
        aper = A_PERIOD_FALLBACK

    D = pfd
    if D > nt - 1:
        D = nt - 1
    if D > PF_MAX:
        D = PF_MAX
    if D < 1:
        D = 1

    # ---------------- issue the first D tiles' loads before the setup ----------------
    m0 = _tile_m(0, base, rem)
    b0 = _tile_b(0, row0, base, rem)
    k0 = _tile_kind(b0, m0, rows)
    q0 = _alloc_load(x, 0, row0, base, rem, rows, cols, lch, nld, dt)
    q1 = q0
    q2 = q0
    q3 = q0
    if D >= 2:
        q1 = _alloc_load(x, 1, row0, base, rem, rows, cols, lch, nld, dt)
    if D >= 3:
        q2 = _alloc_load(x, 2, row0, base, rem, rows, cols, lch, nld, dt)
    if D >= 4:
        q3 = _alloc_load(x, 3, row0, base, rem, rows, cols, lch, nld, dt)

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

    # ---------------- coefficient matrices: fm[k, i] = i - k ----------------
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
    q = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q, data1=fm, data2=fm, op=nl.multiply)

    band = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)   # 0.25 at k = i +- 1
    _coef(band, q, 1.0, 0.25)
    sh = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)     # 0.25 at k = i
    _coef(sh, fm, 0.0, 0.25)
    sc = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)     # 1.0 at k = i
    _coef(sc, fm, 0.0, 1.0)

    sv_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_f = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sh_l = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_hi = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
    sv_lo = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)

    # matrices of the first tile first (VectorE), the rest afterwards (ScalarE copies)
    done_f = False
    done_l = False
    done_hi = False
    done_lo = False
    if k0 == 1:
        _build_first(sv_f, sh_f, band, sh, sc, 0)
        done_f = True
    if k0 == 2:
        _build_last(sv_l, sh_l, band, sh, sc, fm, m_last, 0)
        done_l = True
    if k0 == 0:
        if m0 == base + 1:
            _build_interior(sv_hi, band, fm, base + 1, 0)
            done_hi = True
        else:
            _build_interior(sv_lo, band, fm, base, 0)
            done_lo = True
    if need_hi:
        if done_hi == False:
            _build_interior(sv_hi, band, fm, base + 1, 1)
    if need_lo:
        if done_lo == False:
            _build_interior(sv_lo, band, fm, base, 1)
    if has_first:
        if done_f == False:
            _build_first(sv_f, sh_f, band, sh, sc, 1)
    if has_last:
        if done_l == False:
            _build_last(sv_l, sh_l, band, sh, sc, fm, m_last, 1)

    # ---------------- main sweep ----------------
    icount = 0
    for t in range(nt):
        xt = q0
        if D >= 2:
            q0 = q1
        if D >= 3:
            q1 = q2
        if D >= 4:
            q2 = q3
        if t + D < nt:
            xn = _alloc_load(x, t + D, row0, base, rem, rows, cols, lch, nld, dt)
            if D == 1:
                q0 = xn
            if D == 2:
                q1 = xn
            if D == 3:
                q2 = xn
            if D == 4:
                q3 = xn

        m = _tile_m(t, base, rem)
        b = _tile_b(t, row0, base, rem)
        kind = _tile_kind(b, m, rows)

        kk = m + 2
        if kind != 0:
            kk = m + 1

        sv = sv_lo
        shc = sh
        if kind == 0:
            if m == base + 1:
                sv = sv_hi
        if kind == 1:
            sv = sv_f
            shc = sh_f
        if kind == 2:
            sv = sv_l
            shc = sh_l

        for s in range(nsg):
            s0 = s * sg
            sw = min(sg, cols - s0)
            o = nl.ndarray((P, sg), dtype=dt, buffer=nl.sbuf)
            ngs = (sw + pg - 1) // pg
            for gi in range(ngs):
                g0 = s0 + gi * pg
                gw = min(pg, s0 + sw - g0)
                lo_g = max(g0, 1)
                hi_g = min(g0 + gw, cols - 1)

                path_b = False
                if kind == 0:
                    if icount % aper != aper - 1:
                        path_b = True
                    icount += 1

                ps = nl.ndarray((P, pg), dtype=nl.float32, buffer=nl.psum)

                # vertical neighbours (+ boundary-row centre copy)
                _mm_pass(ps, sv, xt, m, kk, g0, gw, cols, 0, False)
                # left neighbour
                _mm_pass(ps, shc, xt, m, kk, g0, gw, cols, -1, True)
                if path_b == False:
                    # right neighbour on TensorE
                    _mm_pass(ps, shc, xt, m, kk, g0, gw, cols, 1, True)
                # boundary columns: exact centre copy
                if g0 == 0:
                    nisa.nc_matmul(dst=ps[0:m, 0:1], stationary=sc[0:kk, 0:m],
                                   moving=xt[0:kk, 0:1], accumulate=False)
                if g0 + gw == cols:
                    nisa.nc_matmul(dst=ps[0:m, gw - 1:gw], stationary=sc[0:kk, 0:m],
                                   moving=xt[0:kk, cols - 1:cols], accumulate=False)

                if path_b:
                    if mode == 1:
                        nisa.scalar_tensor_tensor(dst=o[0:m, lo_g - s0:hi_g - s0],
                                                  data=xt[0:m, lo_g + 1:hi_g + 1],
                                                  op0=nl.multiply, operand0=0.25,
                                                  op1=nl.add,
                                                  operand1=ps[0:m, lo_g - g0:hi_g - g0])
                    elif mode == 2:
                        nisa.scalar_tensor_tensor(dst=o[0:m, lo_g - s0:hi_g - s0],
                                                  data=xt[0:m, lo_g + 1:hi_g + 1],
                                                  op0=nl.multiply, operand0=0.25,
                                                  op1=nl.add,
                                                  operand1=ps[0:m, lo_g - g0:hi_g - g0],
                                                  engine=nisa.engine.vector)
                    else:
                        r4 = nl.ndarray((P, pg), dtype=dt, buffer=nl.sbuf)
                        nisa.activation(dst=r4[0:m, lo_g - g0:hi_g - g0], op=nl.copy,
                                        data=xt[0:m, lo_g + 1:hi_g + 1], scale=0.25)
                        nisa.tensor_tensor(dst=o[0:m, lo_g - s0:hi_g - s0],
                                           data1=ps[0:m, lo_g - g0:hi_g - g0],
                                           data2=r4[0:m, lo_g - g0:hi_g - g0],
                                           op=nl.add, engine=nisa.engine.vector)
                    if g0 == 0:
                        nisa.tensor_copy(dst=o[0:m, 0:1], src=ps[0:m, 0:1],
                                         engine=nisa.engine.scalar)
                    if g0 + gw == cols:
                        nisa.tensor_copy(dst=o[0:m, cols - 1 - s0:cols - s0],
                                         src=ps[0:m, gw - 1:gw],
                                         engine=nisa.engine.scalar)
                else:
                    nisa.tensor_copy(dst=o[0:m, g0 - s0:g0 - s0 + gw],
                                     src=ps[0:m, 0:gw],
                                     engine=nisa.engine.scalar)

            nisa.dma_copy(dst=out[b:b + m, s0:s0 + sw], src=o[0:m, 0:sw])

    return out


def _prefetch_depth(rows, cols):
    n_prog = 2
    r2 = rows // n_prog
    nt = (r2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    nbuf = PF_BUDGET_BYTES // (cols * ELEM_BYTES)
    if nbuf > PF_MAX + 1:
        nbuf = PF_MAX + 1
    if nbuf < 2:
        nbuf = 2
    d = nbuf - 1
    if d > nt - 1:
        d = nt - 1
    if d < 1:
        d = 1
    return d


def _config_for(rows, cols, mode, pfd):
    n_prog = 2
    r2 = rows // n_prog
    nt = (r2 + ROW_TILE_MAX - 1) // ROW_TILE_MAX
    pg = PG_MAX if cols >= PG_MAX else cols
    sg = SG_MAX if cols >= SG_MAX else cols
    nld = (cols + LOAD_MAX - 1) // LOAD_MAX
    lch = (cols + nld - 1) // nld
    lch = ((lch + BLK - 1) // BLK) * BLK
    aper = A_PERIOD if mode != 0 else A_PERIOD_FALLBACK
    return {
        "programs": n_prog,
        "row_tile_max": ROW_TILE_MAX,
        "row_tiles_per_program": nt,
        "rows_per_tile_base": r2 // nt,
        "matmul_block_cols": BLK,
        "psum_group_cols": pg,
        "store_group_cols": sg,
        "load_chunk_cols": lch,
        "a_period": aper,
        "prefetch_tiles": pfd,
        "stt_mode": mode,
    }


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG
    r = int(input.shape[0])
    c = int(input.shape[1])
    mode = int(_STT_MODE)
    pfd = int(_prefetch_depth(r, c))
    out = _jacobi_kernel(input, mode, pfd)
    _LAST_CONFIG = _config_for(r, c, mode, pfd)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
