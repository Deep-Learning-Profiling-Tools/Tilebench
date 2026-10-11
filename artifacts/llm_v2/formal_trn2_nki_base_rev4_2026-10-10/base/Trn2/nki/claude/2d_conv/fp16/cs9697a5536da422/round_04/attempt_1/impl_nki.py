import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Static configuration (literals; the per-shape block plan is a fixed formula)
# ---------------------------------------------------------------------------
PSUM_COLS = 512   # fp32 columns per PSUM bank == max matmul moving free size
CPB = 8           # steady-state PSUM chunks per row block
NPC = 4           # partition pieces per DMA transfer

_LAST_CONFIG = {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _row_start(k, sched):
    r = 0
    i = 0
    while i < k:
        r = r + sched[i]
        i = i + 1
    return r


def _evac_copy(dst, src, which):
    if which == 0:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
    else:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.scalar)


def _load_block(x, bb, r0, nbr, H, C, RBA, W, npc):
    # Buffer row (1 + i) <-> image row (r0 - 1 + i), i = 0 .. nbr+1 ; row stride W (unpadded).
    xb = nl.ndarray((C, RBA, W), dtype=x.dtype, buffer=nl.sbuf)
    need = nbr + 2
    top = r0 - 1
    s_lo = max(0, top)
    s_hi = min(H, top + need)
    i_lo = s_lo - top
    i_hi = s_hi - top
    if i_lo > 0:
        nisa.memset(dst=xb[:, 1:1 + i_lo, :], value=0.0, engine=nisa.engine.gpsimd)
    if i_hi < need:
        nisa.memset(dst=xb[:, 1 + i_hi:1 + need, :], value=0.0, engine=nisa.engine.gpsimd)
    pg = _cdiv(C, npc)
    p0 = 0
    while p0 < C:
        p1 = min(C, p0 + pg)
        nisa.dma_copy(dst=xb[p0:p1, 1 + i_lo:1 + i_hi, :],
                      src=x[bb, p0:p1, s_lo:s_hi, :])
        p0 = p1
    return xb


def _prep_block(xb, xl, xr, nbr, W):
    # xl: copy with column W-1 zeroed (+ leading guard) -> used by kw=0 taps
    # xr: copy with column 0 zeroed (+ trailing guard)  -> used by kw=2 taps
    need = nbr + 2
    nisa.tensor_copy(dst=xl[:, 1:1 + need, :], src=xb[:, 1:1 + need, :],
                     engine=nisa.engine.gpsimd)
    nisa.memset(dst=xl[:, 0:need, W - 1:W], value=0.0, engine=nisa.engine.gpsimd)
    nisa.tensor_copy(dst=xr[:, 1:1 + need, :], src=xb[:, 1:1 + need, :],
                     engine=nisa.engine.vector)
    nisa.memset(dst=xr[:, 2:2 + need, 0:1], value=0.0, engine=nisa.engine.vector)


def _store_block(out, ob, bb, r0, nbr, Co, npc):
    pg = _cdiv(Co, npc)
    p0 = 0
    while p0 < Co:
        p1 = min(Co, p0 + pg)
        nisa.dma_copy(dst=out[bb, p0:p1, r0:r0 + nbr, :], src=ob[p0:p1, 0:nbr, :])
        p0 = p1


def _compute_block(xa, xla, xra, ob, wT, nbr, W, C, Co, RBA, nbmax, ev):
    # Output flat p = (oh - r0) * W + ow ; tap (kh, kw) reads buffer flat p + kh*W + kw + W - 1
    xbf = xa.reshape((C, RBA * W))
    xlf = xla.reshape((C, RBA * W))
    xrf = xra.reshape((C, RBA * W))
    obf = ob.reshape((Co, nbmax * W))
    L = nbr * W
    u = 16
    while L % u != 0:
        u = u // 2
    nch = _cdiv(L, PSUM_COLS)
    LU = L // u
    base = LU // nch
    extra = LU - base * nch
    s = 0
    for ci in range(nch):
        n = base
        if ci < extra:
            n = base + 1
        n = n * u
        ps = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
        idx = 0
        for q in range(3):
            kw = (q + 1) % 3
            src = xbf
            if kw == 0:
                src = xlf
            if kw == 2:
                src = xrf
            for kh in range(3):
                o = s + kh * W + kw + W - 1
                nisa.nc_matmul(dst=ps[:, 0:n], stationary=wT[:, kh * 3 + kw, :],
                               moving=src[:, o:o + n], accumulate=(idx > 0))
                idx = idx + 1
        _evac_copy(obf[:, s:s + n], ps[:, 0:n], ev % 2)
        ev = ev + 1
        s = s + n
    return ev


@nki.jit
def conv2d_flat_igemm(x, w, sched, split, nbmax, npc):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    Co = w.shape[0]
    Cg = w.shape[1]
    kH = w.shape[2]
    kW = w.shape[3]
    T = kH * kW
    RBA = nbmax + 4

    out = nl.ndarray((B, Co, H, W), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    nblk = len(sched)
    b_lo = 0
    b_hi = nblk
    if n_prog > 1:
        if pid == 0:
            b_hi = split
        else:
            b_lo = split
    nloc = b_hi - b_lo
    njobs = B * nloc

    if njobs > 0:
        # ---- weights: contiguous rows (Co, Cg*T) -> SBUF
        KWT = Cg * T
        w2 = w.reshape((Co, KWT))
        w_sb = nl.ndarray((Co, KWT), dtype=w.dtype, buffer=nl.sbuf)
        pgw = _cdiv(Co, npc)
        p0 = 0
        while p0 < Co:
            p1 = min(Co, p0 + pgw)
            nisa.dma_copy(dst=w_sb[p0:p1, :], src=w2[p0:p1, :])
            p0 = p1

        # ---- first input blocks (prefetch depth 2)
        k0 = b_lo
        nbr0 = sched[k0]
        xa = _load_block(x, 0, _row_start(k0, sched), nbr0, H, C, RBA, W, npc)
        xn = xa
        if njobs > 1:
            k1 = b_lo + (1 % nloc)
            xn = _load_block(x, 1 // nloc, _row_start(k1, sched), sched[k1], H, C, RBA,
                             W, npc)

        xla = nl.ndarray((C, RBA, W), dtype=x.dtype, buffer=nl.sbuf)
        xra = nl.ndarray((C, RBA, W), dtype=x.dtype, buffer=nl.sbuf)
        _prep_block(xa, xla, xra, nbr0, W)

        # ---- per-tap TensorE transposes -> wT[ic, t, oc]
        w3 = w_sb.reshape((Co, Cg, T))
        wT = nl.ndarray((Cg, T, Co), dtype=w.dtype, buffer=nl.sbuf)
        for kh in range(kH):
            tp = nl.ndarray((Cg, kW, Co), dtype=w.dtype, buffer=nl.psum)
            for kw in range(kW):
                nisa.nc_transpose(dst=tp[:, kw, :], data=w3[:, :, kh * kW + kw],
                                  engine=nisa.engine.tensor)
            _evac_copy(wT[:, kh * kW:(kh + 1) * kW, :], tp, (kh + 1) % 2)

        # ---- implicit GEMM over row blocks
        ev = 0
        for j in range(njobs):
            xq = xa
            if j + 2 < njobs:
                m = j + 2
                km = b_lo + (m % nloc)
                xq = _load_block(x, m // nloc, _row_start(km, sched), sched[km], H, C,
                                 RBA, W, npc)
            xln = xla
            xrn = xra
            if j + 1 < njobs:
                m1 = j + 1
                km1 = b_lo + (m1 % nloc)
                xln = nl.ndarray((C, RBA, W), dtype=x.dtype, buffer=nl.sbuf)
                xrn = nl.ndarray((C, RBA, W), dtype=x.dtype, buffer=nl.sbuf)
                _prep_block(xn, xln, xrn, sched[km1], W)
            bbj = j // nloc
            kj = b_lo + (j % nloc)
            r0 = _row_start(kj, sched)
            nbr = sched[kj]
            ob = nl.ndarray((Co, nbmax, W), dtype=x.dtype, buffer=nl.sbuf)
            ev = _compute_block(xa, xla, xra, ob, wT, nbr, W, C, Co, RBA, nbmax, ev)
            _store_block(out, ob, bbj, r0, nbr, Co, npc)
            xa = xn
            xla = xln
            xra = xrn
            xn = xq
    return out


# ---------------------------------------------------------------------------
# Host side: deterministic block plan (fixed formula of the shape)
# ---------------------------------------------------------------------------
def _balanced(total, parts):
    base = total // parts
    extra = total - base * parts
    res = []
    for i in range(parts):
        res.append(base + (1 if i < extra else 0))
    return res


def _core_sched(R, W):
    if R <= 0:
        return []
    U = _cdiv(R * W, PSUM_COLS)
    if U <= 2:
        return [R]
    front = []
    back = []
    rem = U
    c = 1
    while rem >= 2 * c:
        front.append(c)
        back.append(c)
        rem -= 2 * c
        c = min(2 * c, CPB)
    mid = []
    if rem > 0:
        mid = _balanced(rem, _cdiv(rem, CPB))
    chunks = front + mid + list(reversed(back))
    rows = []
    cum = 0
    prev = 0
    for ck in chunks:
        cum += ck
        b = (2 * cum * R + U) // (2 * U)
        if b > prev:
            rows.append(b - prev)
            prev = b
    if prev < R:
        rows.append(R - prev)
    return rows


def run(input, weight, stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, C, H, W = [int(s) for s in input.shape]
    Co, Cg, kH, kW = [int(s) for s in weight.shape]
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1 or padding != 1 or kH != 3 or kW != 3:
        raise NotImplementedError("only stride=1, padding=1, 3x3, groups=1 supported")
    if C > 128 or Co > 128 or Cg != C or W < 2:
        raise NotImplementedError("channels must be <= 128")

    OH = H
    R0 = (OH + 1) // 2
    R1 = OH - R0
    s0 = _core_sched(R0, W)
    s1 = _core_sched(R1, W)
    sched = tuple(s0 + s1)
    split = len(s0)
    nbmax = max(sched)

    out = conv2d_flat_igemm(input, weight, sched, split, nbmax, NPC)

    _LAST_CONFIG = {
        "algorithm": "flat_contiguous_implicit_gemm_masked_edge_copies",
        "block_rows_core0": list(s0),
        "block_rows_core1": list(s1),
        "max_block_rows": nbmax,
        "psum_cols": PSUM_COLS,
        "chunks_per_block_max": CPB,
        "prefetch_depth": 2,
        "dma_pieces": NPC,
        "tap_order_kw": [1, 2, 0],
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
