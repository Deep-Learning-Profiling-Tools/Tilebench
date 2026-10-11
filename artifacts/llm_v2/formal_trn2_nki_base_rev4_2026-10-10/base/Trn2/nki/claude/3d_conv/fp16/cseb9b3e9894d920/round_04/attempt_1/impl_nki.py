import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Implicit-GEMM 3-D convolution (fp16 in, fp32 PSUM accumulation, fp16 out).
# K = 128 = 2 stacked input depth planes x 64 channels,
# M = 128 = 2 output depths x 64 output channels,
# N <= 512 flattened padded spatial positions (garbage pad columns never stored).
# Input tile T_k holds planes (2k-1, 2k); pair p uses T_p and T_{p+1} -> each tile is
# loaded once per row block and shared by two depth pairs (blocks outer, pairs inner).
# Small widths: contiguous DMAs into SBUF staging + on-chip pad/compact copies.

_LAST_CONFIG = {}

_BLOCK_COLS_MAX_UNROLL = 4096
_BLOCK_COLS_MAX_LOOP = 3072
_BLOCK_OVERHEAD = 512
_MM_OVERHEAD = 64
_UNROLL_MAX_MM = 5000
_COMPACT_MAX_W = 128
_NT = 6


def _pick2(k, a, b):
    r = a
    if k == 1:
        r = b
    return r


def _pick3(k, a, b, c):
    r = a
    if k == 1:
        r = b
    if k == 2:
        r = c
    return r


def _pick6(k, a, b, c, d, e, f):
    r = a
    if k == 1:
        r = b
    if k == 2:
        r = c
    if k == 3:
        r = d
    if k == 4:
        r = e
    if k == 5:
        r = f
    return r


def _hap(t2, pattern, off, j, rt):
    if rt:
        r = t2.ap(pattern=pattern, offset=off, scalar_offset=j, indirect_dim=0)
    else:
        r = t2.ap(pattern=pattern, offset=off)
    return r


def _build_weights(wt, dt, tp):
    CO = wt.shape[0]
    C = wt.shape[1]
    wn = nl.ndarray((128, C * 27), dtype=dt, buffer=nl.sbuf)
    w2 = wt.reshape((CO, C * 27))
    nisa.dma_copy(dst=wn[0:64, :], src=w2)
    nisa.dma_copy(dst=wn[64:128, :], src=w2)
    wn3 = wn.reshape((128, C, 27))
    am = nl.ndarray((128, 2304), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=am, value=0.0)
    am3 = am.reshape((128, 256, 9))
    nisa.tensor_copy(dst=am3[0:64, 0:64, :], src=wn3[0:64, :, 0:9], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[0:64, 64:128, :], src=wn3[0:64, :, 9:18], engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=am3[64:128, 64:128, :], src=wn3[64:128, :, 0:9], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[0:64, 128:192, :], src=wn3[0:64, :, 18:27], engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=am3[64:128, 128:192, :], src=wn3[64:128, :, 9:18], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[64:128, 192:256, :], src=wn3[64:128, :, 18:27], engine=nisa.engine.scalar)
    wst = nl.ndarray((128, 2304), dtype=dt, buffer=nl.sbuf)
    for i in range(18):
        st0 = (i // 9) * 1152 + (i % 9)
        h = i % 2
        tps = tp[:, h * 128:(h + 1) * 128]
        nisa.nc_transpose(dst=tps, data=am[:, st0:st0 + 1144:9], engine=nisa.engine.tensor)
        if h == 0:
            nisa.tensor_copy(dst=wst[:, i * 128:(i + 1) * 128], src=tps, engine=nisa.engine.vector)
        else:
            nisa.tensor_copy(dst=wst[:, i * 128:(i + 1) * 128], src=tps, engine=nisa.engine.scalar)
    return wst


def _load_tile(inp_v, T, cp, b, ks, j, rt, compact, NP, H, W, D, Rb):
    # T (128, Rb+2, W+2): partitions 0..63 <- plane 2ks-1, 64..127 <- plane 2ks
    HW = H * W
    DHW = D * HW
    y0 = b * Rb
    rb = min(Rb, H - y0)
    ya = max(0, y0 - 1)
    yb = min(H, y0 + rb + 1)
    ra = ya - y0 + 1
    nr = yb - ya
    lo_ok = ks >= 1
    hi_ok = ks <= NP - 1
    if rt:
        lo_ok = True
        hi_ok = True
    if y0 == 0:
        nisa.memset(dst=T[:, 0:1, :], value=0.0)
    if y0 + rb == H:
        nisa.memset(dst=T[:, rb + 1:rb + 2, :], value=0.0)
    q0 = 64
    q1 = 64
    if lo_ok:
        q0 = 0
        off_lo = (2 * ks - 1) * HW + ya * W
        if compact:
            nisa.dma_copy(dst=cp[0:64, 0:nr * W],
                          src=_hap(inp_v, [[DHW, 64], [1, nr * W]], off_lo, j, rt))
        else:
            nisa.dma_copy(dst=T[0:64, ra:ra + nr, 1:W + 1],
                          src=_hap(inp_v, [[DHW, 64], [W, nr], [1, W]], off_lo, j, rt))
    else:
        nisa.memset(dst=T[0:64, 0:rb + 2, :], value=0.0)
    if hi_ok:
        q1 = 128
        off_hi = (2 * ks) * HW + ya * W
        if compact:
            nisa.dma_copy(dst=cp[64:128, 0:nr * W],
                          src=_hap(inp_v, [[DHW, 64], [1, nr * W]], off_hi, j, rt))
        else:
            nisa.dma_copy(dst=T[64:128, ra:ra + nr, 1:W + 1],
                          src=_hap(inp_v, [[DHW, 64], [W, nr], [1, W]], off_hi, j, rt))
    else:
        nisa.memset(dst=T[64:128, 0:rb + 2, :], value=0.0)
    if compact:
        cp3 = cp.reshape((128, Rb + 2, W))
        nisa.tensor_copy(dst=T[q0:q1, ra:ra + nr, 1:W + 1], src=cp3[q0:q1, 0:nr, :],
                         engine=nisa.engine.gpsimd)


def _compute_item(out_v, wst, ta, tb, sg, sc, p0, p1, p2, p3, b, pp, gc, j, rt, compact,
                  H, W, D, Rb):
    WP = W + 2
    HW = H * W
    DHW = D * HW
    y0 = b * Rb
    rb = min(Rb, H - y0)
    L = rb * WP - 2
    nc = (L + 511) // 512
    cs = (L + nc - 1) // nc
    cs = cs + (cs % 2)
    ng = (nc + 1) // 2
    ta2 = ta.reshape((128, (Rb + 2) * WP))
    tb2 = tb.reshape((128, (Rb + 2) * WP))
    sg2 = sg.reshape((128, Rb * WP))
    for g in range(ng):
        c0 = 2 * g
        gsz = min(2, nc - c0)
        par = gc % 2
        pa = _pick2(par, p0, p2)
        pb = _pick2(par, p1, p3)
        for i in range(18):
            t = i % 9
            kh = t // 3
            kw = t - 3 * kh
            sh = kh * WP + kw
            mv = _pick2(i // 9, ta2, tb2)
            stat = wst[:, i * 128:(i + 1) * 128]
            for q in range(gsz):
                s = (c0 + q) * cs
                e = min(L, s + cs)
                pt = _pick2(q, pa, pb)
                nisa.nc_matmul(dst=pt[:, 0:e - s], stationary=stat,
                               moving=mv[:, s + sh:e + sh], accumulate=(i > 0))
        for q in range(gsz):
            s = (c0 + q) * cs
            e = min(L, s + cs)
            pt = _pick2(q, pa, pb)
            if q == 0:
                nisa.tensor_copy(dst=sg2[:, s:e], src=pt[:, 0:e - s], engine=nisa.engine.vector)
            else:
                nisa.tensor_copy(dst=sg2[:, s:e], src=pt[:, 0:e - s], engine=nisa.engine.scalar)
        gc = gc + 1
    o0 = (2 * pp) * HW + y0 * W
    o1 = o0 + HW
    if compact:
        nisa.tensor_copy(dst=sc[:, 0:rb, :], src=sg[:, 0:rb, 0:W], engine=nisa.engine.vector)
        sc2 = sc.reshape((128, Rb * W))
        nisa.dma_copy(dst=_hap(out_v, [[DHW, 64], [1, rb * W]], o0, j, rt),
                      src=sc2[0:64, 0:rb * W])
        nisa.dma_copy(dst=_hap(out_v, [[DHW, 64], [1, rb * W]], o1, j, rt),
                      src=sc2[64:128, 0:rb * W])
    else:
        nisa.dma_copy(dst=_hap(out_v, [[DHW, 64], [W, rb], [1, W]], o0, j, rt),
                      src=sg[0:64, 0:rb, 0:W])
        nisa.dma_copy(dst=_hap(out_v, [[DHW, 64], [W, rb], [1, W]], o1, j, rt),
                      src=sg[64:128, 0:rb, 0:W])
    return gc


def _run_items(inp_v, out_v, wst, T0, T1, T2, T3, T4, T5, c0, c1, g0, g1, g2, s0, s1, s2,
               p0, p1, p2, p3, j, rt, compact, pbase, npairs, NP, H, W, D, Rb):
    NB = (H + Rb - 1) // Rb
    nt = npairs + 1
    NL = NB * nt
    NI = NB * npairs
    nld = 0
    gc = 0
    for it in range(NI):
        b = it // npairs
        pp = it - b * npairs
        u = b * nt + pp
        need = min(NL, u + _NT)
        while nld < need:
            bl = nld // nt
            kl = nld - bl * nt
            _load_tile(inp_v, _pick6(nld % _NT, T0, T1, T2, T3, T4, T5), _pick2(nld % 2, c0, c1),
                       bl, pbase + kl, j, rt, compact, NP, H, W, D, Rb)
            nld = nld + 1
        gc = _compute_item(out_v, wst, _pick6(u % _NT, T0, T1, T2, T3, T4, T5),
                           _pick6((u + 1) % _NT, T0, T1, T2, T3, T4, T5),
                           _pick3(it % 3, g0, g1, g2), _pick3(it % 3, s0, s1, s2),
                           p0, p1, p2, p3, b, pbase + pp, gc, j, rt, compact, H, W, D, Rb)


def _new_tile(Rb, WP, W, dt):
    T = nl.ndarray((128, Rb + 2, WP), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=T[:, :, 0:1], value=0.0)
    nisa.memset(dst=T[:, :, W + 1:W + 2], value=0.0)
    return T


def _ctx(inp_v, out_v, wt, dt, j, rt, compact, pbase, npairs, pextra, four, NP, H, W, D, Rb):
    WP = W + 2
    tp = nl.ndarray((128, 256), dtype=dt, buffer=nl.psum)
    wst = _build_weights(wt, dt, tp)
    T0 = _new_tile(Rb, WP, W, dt)
    T1 = _new_tile(Rb, WP, W, dt)
    T2 = _new_tile(Rb, WP, W, dt)
    T3 = _new_tile(Rb, WP, W, dt)
    T4 = _new_tile(Rb, WP, W, dt)
    T5 = _new_tile(Rb, WP, W, dt)
    g0 = nl.ndarray((128, Rb, WP), dtype=dt, buffer=nl.sbuf)
    g1 = nl.ndarray((128, Rb, WP), dtype=dt, buffer=nl.sbuf)
    g2 = nl.ndarray((128, Rb, WP), dtype=dt, buffer=nl.sbuf)
    c0 = T0
    c1 = T1
    s0 = g0
    s1 = g1
    s2 = g2
    if compact:
        c0 = nl.ndarray((128, (Rb + 2) * W), dtype=dt, buffer=nl.sbuf)
        c1 = nl.ndarray((128, (Rb + 2) * W), dtype=dt, buffer=nl.sbuf)
        s0 = nl.ndarray((128, Rb, W), dtype=dt, buffer=nl.sbuf)
        s1 = nl.ndarray((128, Rb, W), dtype=dt, buffer=nl.sbuf)
        s2 = nl.ndarray((128, Rb, W), dtype=dt, buffer=nl.sbuf)
    p0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    p1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    p2 = p0
    p3 = p1
    if four:
        p2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        p3 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    _run_items(inp_v, out_v, wst, T0, T1, T2, T3, T4, T5, c0, c1, g0, g1, g2, s0, s1, s2,
               p0, p1, p2, p3, j, rt, compact, pbase, npairs, NP, H, W, D, Rb)
    if pextra >= 0:
        _run_items(inp_v, out_v, wst, T0, T1, T2, T3, T4, T5, c0, c1, g0, g1, g2, s0, s1, s2,
                   p0, p1, p2, p3, j, rt, compact, pextra, 1, NP, H, W, D, Rb)


@nki.jit
def _conv3d_kernel(inp, wt, Rb, unroll, compact):
    B, C, D, H, W = inp.shape
    CO = wt.shape[0]
    dt = inp.dtype
    HW = H * W
    out = nl.ndarray((B, CO, D, H, W), dtype=dt, buffer=nl.shared_hbm)
    inp_v = inp.reshape((C * D // 2, 2 * HW))
    out_v = out.reshape((CO * D // 2, 2 * HW))

    NP = D // 2
    npg = nl.num_programs()
    pid = nl.program_id(0)
    per = (NP + npg - 1) // npg
    pa = min(NP, pid * per)
    pb = min(NP, pa + per)

    def body(j):
        # interior pair p = j + 1: tiles T_{j+1}, T_{j+2}; output planes 2j+2, 2j+3
        _ctx(inp_v, out_v, wt, dt, j, True, compact, 1, 1, -1, True, NP, H, W, D, Rb)

    if unroll:
        if pb > pa:
            _ctx(inp_v, out_v, wt, dt, None, False, compact, pa, pb - pa, -1, True,
                 NP, H, W, D, Rb)
    else:
        first = -1
        extra = -1
        if pa == 0 and pb > pa:
            first = 0
        if pb == NP and pb > pa:
            if first < 0:
                first = NP - 1
            else:
                extra = NP - 1
        if first >= 0:
            _ctx(inp_v, out_v, wt, dt, None, False, compact, first, 1, extra, False,
                 NP, H, W, D, Rb)
        jlo = max(pa, 1) - 1
        jhi = min(pb, NP - 1) - 1
        if jhi > jlo:
            nl.fori_loop(jlo, jhi, body)
    return out


def _block_cost(H, W, Rb):
    WP = W + 2
    tot = 0
    y0 = 0
    while y0 < H:
        rb = min(Rb, H - y0)
        L = rb * WP - 2
        nc = (L + 511) // 512
        tot += L + _MM_OVERHEAD * nc + _BLOCK_OVERHEAD
        y0 += rb
    return tot


def _choose_rb(H, W, cols_max):
    WP = W + 2
    rcap = max(1, min(H, (cols_max + 2) // WP))
    best_rb = 1
    best_cost = None
    for rb in range(1, rcap + 1):
        c = _block_cost(H, W, rb)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_rb = rb
    return best_rb


def _pair_chunks(H, W, Rb):
    WP = W + 2
    tot = 0
    y0 = 0
    while y0 < H:
        rb = min(Rb, H - y0)
        L = rb * WP - 2
        tot += (L + 511) // 512
        y0 += rb
    return tot


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, C, D, H, W = input.shape
    CO = weight.shape[0]
    if (stride != 1 or padding != 1 or groups != 1 or B != 1 or C != 64 or CO != 64
            or tuple(weight.shape[1:]) != (64, 3, 3, 3) or D % 2 != 0 or D < 6):
        raise NotImplementedError("configuration outside the supported task domain")
    NP = D // 2
    per_core = (NP + 1) // 2
    rb_u = _choose_rb(H, W, _BLOCK_COLS_MAX_UNROLL)
    mm_u = per_core * 18 * _pair_chunks(H, W, rb_u)
    unroll = bool(mm_u <= _UNROLL_MAX_MM)
    if unroll:
        Rb = rb_u
    else:
        Rb = _choose_rb(H, W, _BLOCK_COLS_MAX_LOOP)
    compact = bool(W <= _COMPACT_MAX_W)
    chunks = _pair_chunks(H, W, Rb)
    mm = per_core * 18 * chunks
    out = _conv3d_kernel(input, weight, Rb, unroll, compact)
    _LAST_CONFIG = {
        "rows_per_block": int(Rb),
        "chunks_per_pair": int(chunks),
        "matmuls_per_core": int(mm),
        "unrolled": unroll,
        "compact_io": compact,
        "path": "unrolled" if unroll else "boundary_pair_unrolled+fori_loop_selfcontained",
        "n_input_tiles": _NT,
        "plane_tile_reuse": True,
        "n_staging": 3,
        "n_psum_mm": 4,
        "chunk_group": 2,
        "chunk_max_cols": 512,
        "k_stack": "2 depth planes x 64 ch",
        "m_stack": "2 output depths x 64 ch",
        "lnc_split": "depth pairs",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
