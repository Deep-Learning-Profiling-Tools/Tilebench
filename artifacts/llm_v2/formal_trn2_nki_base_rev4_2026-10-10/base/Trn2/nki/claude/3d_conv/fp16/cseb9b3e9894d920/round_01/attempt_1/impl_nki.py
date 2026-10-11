import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Implicit-GEMM 3-D convolution (fp16 in, fp32 PSUM accumulation, fp16 out).
# K = 128 = 2 stacked input depth planes x 64 channels,
# M = 128 = 2 output depths x 64 output channels,
# N <= 512 flattened padded spatial positions (garbage pad columns never stored).

_LAST_CONFIG = {}

_BLOCK_COLS_MAX = 4096   # max flattened padded columns per row block
_BLOCK_OVERHEAD = 512    # cost-model penalty per row block (in PE columns)
_MM_OVERHEAD = 64        # cost-model penalty per matmul instruction (in PE columns)
_UNROLL_MAX_MM = 4000    # fully unroll when per-core matmul count is at most this


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


def _hap(t2, pattern, off, j, rt):
    if rt:
        r = t2.ap(pattern=pattern, offset=off, scalar_offset=j, indirect_dim=0)
    else:
        r = t2.ap(pattern=pattern, offset=off)
    return r


def _load_block(inp2, ta, tb, b, j, rt, zs, H, W, D, Rb, lo_ok, hi_ok):
    WP = W + 2
    HW = H * W
    DHW = D * HW
    RR = Rb + 2
    y0 = b * Rb
    rb = min(Rb, H - y0)
    ya = max(0, y0 - 1)
    yb = min(H, y0 + rb + 1)
    ra = ya - y0 + 1
    nr = yb - ya
    ta3 = ta.reshape((128, RR, WP))
    tb3 = tb.reshape((128, RR, WP))
    if y0 == 0:
        nisa.memset(dst=ta[:, 0:WP], value=0.0)
        nisa.memset(dst=tb[:, 0:WP], value=0.0)
    if y0 + rb == H:
        r1 = (rb + 1) * WP
        nisa.memset(dst=ta[:, r1:r1 + WP], value=0.0)
        nisa.memset(dst=tb[:, r1:r1 + WP], value=0.0)
    yo = ya * W
    if lo_ok:
        nisa.dma_copy(dst=ta3[0:64, ra:ra + nr, 1:W + 1],
                      src=_hap(inp2, [[DHW, 64], [W, nr], [1, W]], zs * HW + yo, j, rt))
    else:
        nisa.memset(dst=ta[0:64, 0:(rb + 2) * WP], value=0.0)
    nisa.dma_copy(dst=ta3[64:128, ra:ra + nr, 1:W + 1],
                  src=_hap(inp2, [[DHW, 64], [W, nr], [1, W]], (zs + 1) * HW + yo, j, rt))
    nisa.dma_copy(dst=tb3[0:64, ra:ra + nr, 1:W + 1],
                  src=_hap(inp2, [[DHW, 64], [W, nr], [1, W]], (zs + 2) * HW + yo, j, rt))
    if hi_ok:
        nisa.dma_copy(dst=tb3[64:128, ra:ra + nr, 1:W + 1],
                      src=_hap(inp2, [[DHW, 64], [W, nr], [1, W]], (zs + 3) * HW + yo, j, rt))
    else:
        nisa.memset(dst=tb[64:128, 0:(rb + 2) * WP], value=0.0)


def _compute_block(out2, wst, ta, tb, st, p0, p1, p2, p3, b, gc, j, rt, zs, H, W, D, Rb):
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
            mv = _pick2(i // 9, ta, tb)
            stat = wst[:, i * 128:(i + 1) * 128]
            for q in range(gsz):
                s = (c0 + q) * cs
                e = min(L, s + cs)
                ps = _pick2(q, pa, pb)
                nisa.nc_matmul(dst=ps[:, 0:e - s], stationary=stat,
                               moving=mv[:, s + sh:e + sh], accumulate=(i > 0))
        for q in range(gsz):
            s = (c0 + q) * cs
            e = min(L, s + cs)
            ps = _pick2(q, pa, pb)
            if q == 0:
                nisa.tensor_copy(dst=st[:, s:e], src=ps[:, 0:e - s], engine=nisa.engine.vector)
            else:
                nisa.tensor_copy(dst=st[:, s:e], src=ps[:, 0:e - s], engine=nisa.engine.scalar)
        gc = gc + 1
    st3 = st.reshape((128, Rb, WP))
    yo = y0 * W
    nisa.dma_copy(dst=_hap(out2, [[DHW, 64], [W, rb], [1, W]], (zs + 1) * HW + yo, j, rt),
                  src=st3[0:64, 0:rb, 0:W])
    nisa.dma_copy(dst=_hap(out2, [[DHW, 64], [W, rb], [1, W]], (zs + 2) * HW + yo, j, rt),
                  src=st3[64:128, 0:rb, 0:W])
    return gc


def _load_item(inp2, a0, a1, a2, b0, b1, b2, g, j, rt, pbase, NP, NB, H, W, D, Rb):
    p = pbase + g // NB
    b = g % NB
    k = g % 3
    zs = 2 * p - 1
    lo_ok = p > 0
    hi_ok = p < NP - 1
    if rt:
        zs = 0
        lo_ok = True
        hi_ok = True
    _load_block(inp2, _pick3(k, a0, a1, a2), _pick3(k, b0, b1, b2), b, j, rt, zs,
                H, W, D, Rb, lo_ok, hi_ok)


def _run_items(inp2, out2, wst, a0, a1, a2, b0, b1, b2, s0, s1, s2, p0, p1, p2, p3,
               j, rt, pbase, npairs, NP, H, W, D, Rb):
    NB = (H + Rb - 1) // Rb
    NI = npairs * NB
    npre = min(2, NI)
    for g in range(npre):
        _load_item(inp2, a0, a1, a2, b0, b1, b2, g, j, rt, pbase, NP, NB, H, W, D, Rb)
    gc = 0
    for g in range(NI):
        if g + 2 < NI:
            _load_item(inp2, a0, a1, a2, b0, b1, b2, g + 2, j, rt, pbase, NP, NB, H, W, D, Rb)
        p = pbase + g // NB
        b = g % NB
        k = g % 3
        zs = 2 * p - 1
        if rt:
            zs = 0
        gc = _compute_block(out2, wst, _pick3(k, a0, a1, a2), _pick3(k, b0, b1, b2),
                            _pick3(k, s0, s1, s2), p0, p1, p2, p3, b, gc, j, rt, zs,
                            H, W, D, Rb)


@nki.jit
def _conv3d_kernel(inp, wt, Rb, unroll):
    B, C, D, H, W = inp.shape
    CO = wt.shape[0]
    dt = inp.dtype
    WP = W + 2
    RR = Rb + 2
    HW = H * W
    out = nl.ndarray((B, CO, D, H, W), dtype=dt, buffer=nl.shared_hbm)
    inp2 = inp.reshape((C * D, HW))
    out2 = out.reshape((CO * D, HW))

    # ---------------- weights: build 18 stationaries [K=128, M=128] ----------------
    wn = nl.ndarray((128, C * 27), dtype=dt, buffer=nl.sbuf)
    w2 = wt.reshape((CO, C * 27))
    nisa.dma_copy(dst=wn[0:64, :], src=w2)
    nisa.dma_copy(dst=wn[64:128, :], src=w2)
    wn3 = wn.reshape((128, 64, 27))
    am = nl.ndarray((128, 2304), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=am, value=0.0)
    am3 = am.reshape((128, 256, 9))
    # am[m, typ*1152 + half*576 + ci*9 + t] = A_typ,t[m, half*64 + ci]  (A = stationary^T)
    nisa.tensor_copy(dst=am3[0:64, 0:64, :], src=wn3[0:64, :, 0:9], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[0:64, 64:128, :], src=wn3[0:64, :, 9:18], engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=am3[64:128, 64:128, :], src=wn3[64:128, :, 0:9], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[0:64, 128:192, :], src=wn3[0:64, :, 18:27], engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=am3[64:128, 128:192, :], src=wn3[64:128, :, 9:18], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=am3[64:128, 192:256, :], src=wn3[64:128, :, 18:27], engine=nisa.engine.scalar)
    wst = nl.ndarray((128, 2304), dtype=dt, buffer=nl.sbuf)
    tpa = nl.ndarray((128, 128), dtype=dt, buffer=nl.psum)
    tpb = nl.ndarray((128, 128), dtype=dt, buffer=nl.psum)
    for i in range(18):
        st0 = (i // 9) * 1152 + (i % 9)
        tp = _pick2(i % 2, tpa, tpb)
        nisa.nc_transpose(dst=tp, data=am[:, st0:st0 + 1144:9], engine=nisa.engine.tensor)
        if i % 2 == 0:
            nisa.tensor_copy(dst=wst[:, i * 128:(i + 1) * 128], src=tp, engine=nisa.engine.vector)
        else:
            nisa.tensor_copy(dst=wst[:, i * 128:(i + 1) * 128], src=tp, engine=nisa.engine.scalar)

    # ---------------- persistent rotating buffers ----------------
    a0 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    a1 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    a2 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    b0 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    b1 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    b2 = nl.ndarray((128, RR * WP), dtype=dt, buffer=nl.sbuf)
    for q in range(6):
        tq = a0
        if q == 1:
            tq = a1
        if q == 2:
            tq = a2
        if q == 3:
            tq = b0
        if q == 4:
            tq = b1
        if q == 5:
            tq = b2
        tq3 = tq.reshape((128, RR, WP))
        nisa.memset(dst=tq3[:, :, 0:1], value=0.0)
        nisa.memset(dst=tq3[:, :, W + 1:W + 2], value=0.0)
    s0 = nl.ndarray((128, Rb * WP), dtype=dt, buffer=nl.sbuf)
    s1 = nl.ndarray((128, Rb * WP), dtype=dt, buffer=nl.sbuf)
    s2 = nl.ndarray((128, Rb * WP), dtype=dt, buffer=nl.sbuf)
    p0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    p1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    p2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    p3 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)

    # ---------------- schedule over output depth pairs ----------------
    NP = D // 2
    npg = nl.num_programs()
    pid = nl.program_id(0)
    per = (NP + npg - 1) // npg
    pa = pid * per
    pb = min(NP, pa + per)

    def body(j):
        _run_items(inp2, out2, wst, a0, a1, a2, b0, b1, b2, s0, s1, s2, p0, p1, p2, p3,
                   j, True, 0, 1, NP, H, W, D, Rb)

    if unroll:
        if pb > pa:
            _run_items(inp2, out2, wst, a0, a1, a2, b0, b1, b2, s0, s1, s2, p0, p1, p2, p3,
                       None, False, pa, pb - pa, NP, H, W, D, Rb)
    else:
        lo = pa
        hi = pb
        if lo == 0 and lo < hi:
            _run_items(inp2, out2, wst, a0, a1, a2, b0, b1, b2, s0, s1, s2, p0, p1, p2, p3,
                       None, False, 0, 1, NP, H, W, D, Rb)
            lo = 1
        if hi == NP and lo < hi:
            _run_items(inp2, out2, wst, a0, a1, a2, b0, b1, b2, s0, s1, s2, p0, p1, p2, p3,
                       None, False, NP - 1, 1, NP, H, W, D, Rb)
            hi = NP - 1
        if lo < hi:
            nl.fori_loop(2 * lo - 1, 2 * hi - 1, body, step=2)
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


def _choose_rb(H, W):
    WP = W + 2
    rcap = max(1, min(H, (_BLOCK_COLS_MAX + 2) // WP))
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
            or tuple(weight.shape[2:]) != (3, 3, 3) or D % 2 != 0):
        raise NotImplementedError("configuration outside the supported task domain")
    Rb = _choose_rb(H, W)
    NP = D // 2
    per_core = (NP + 1) // 2
    chunks = _pair_chunks(H, W, Rb)
    mm = per_core * 18 * chunks
    unroll = bool(mm <= _UNROLL_MAX_MM)
    out = _conv3d_kernel(input, weight, Rb, unroll)
    _LAST_CONFIG = {
        "rows_per_block": int(Rb),
        "chunks_per_pair": int(chunks),
        "matmuls_per_core": int(mm),
        "unrolled": unroll,
        "n_input_sets": 3,
        "n_staging": 3,
        "n_psum": 4,
        "chunk_group": 2,
        "chunk_max_cols": 512,
        "k_stack": "2 depth planes x 64 ch",
        "m_stack": "2 output depths x 64 ch",
        "lnc_split": "depth pairs",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
