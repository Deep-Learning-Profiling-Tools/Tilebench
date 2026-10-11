import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Static configuration (literals; the per-shape plan is a fixed formula)
# ---------------------------------------------------------------------------
PSUM_COLS = 512          # fp32 columns per PSUM bank == max matmul moving free size
CPB = 8                  # max PSUM chunks per steady-state row block
FLAT_MIN_WP = 128        # padded row width from which PSUM chunks span row boundaries
DMA_MID_BYTES = 262144   # transfers >= this are split into 8 partition pieces, else 4

_LAST_CONFIG = {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _npieces(nbytes):
    if nbytes >= DMA_MID_BYTES:
        return 8
    return 4


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


def _load_block(x, bb, r0, nbr, H, W, Wp, pad, kH, C, RBA, esz):
    # Padded SBUF block: row i <-> image row (r0 - pad + i); column j <-> image col (j - pad).
    # Row stride Wp = W + pad: the pad columns at the start of row i+1 double as the right
    # padding of row i (flat-shift wrap-around reads zeros).
    xb = nl.ndarray((C, RBA, Wp), dtype=x.dtype, buffer=nl.sbuf)
    need = nbr + kH - 1
    top = r0 - pad
    s_lo = max(0, top)
    s_hi = min(H, top + need)
    i_lo = s_lo - top
    i_hi = s_hi - top
    if pad > 0:
        nisa.memset(dst=xb[:, 0:need + 1, 0:pad], value=0.0)
    if Wp > pad + W:
        nisa.memset(dst=xb[:, 0:need + 1, pad + W:Wp], value=0.0)
    if i_lo > 0:
        nisa.memset(dst=xb[:, 0:i_lo, pad:pad + W], value=0.0)
    if i_hi < need:
        nisa.memset(dst=xb[:, i_hi:need, pad:pad + W], value=0.0)
    if i_hi > i_lo:
        npc = _npieces(C * (i_hi - i_lo) * W * esz)
        pg = _cdiv(C, npc)
        p0 = 0
        while p0 < C:
            p1 = min(C, p0 + pg)
            nisa.dma_copy(dst=xb[p0:p1, i_lo:i_hi, pad:pad + W],
                          src=x[bb, p0:p1, s_lo:s_hi, :])
            p0 = p1
    return xb


def _store_block(out, ob, bb, r0, nbr, OW, Co, esz):
    npc = _npieces(Co * nbr * OW * esz)
    pg = _cdiv(Co, npc)
    p0 = 0
    while p0 < Co:
        p1 = min(Co, p0 + pg)
        nisa.dma_copy(dst=out[bb, p0:p1, r0:r0 + nbr, :], src=ob[p0:p1, 0:nbr, :])
        p0 = p1


def _compute_block(xb, ob, wT, nbr, flat, nr, C, Co, RBA, nbmax, Wp, OW, kW, T, ev):
    # flat-shift implicit GEMM: output flat p = oh*Wp + ow, tap (kh,kw) reads p + kh*Wp + kw
    xf = xb.reshape((C, RBA * Wp))
    if flat == 1:
        obf = ob.reshape((Co, nbmax * OW))
        L = (nbr - 1) * Wp + OW
        nch = _cdiv(L, PSUM_COLS)
        base = L // nch
        extra = L - base * nch
        s = 0
        for ci in range(nch):
            n = base
            if ci < extra:
                n = base + 1
            ps = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
            for t in range(T):
                kh = t // kW
                kw = t - kh * kW
                o = s + kh * Wp + kw
                nisa.nc_matmul(dst=ps[:, 0:n], stationary=wT[:, t, :],
                               moving=xf[:, o:o + n], accumulate=(t > 0))
            which = ev % 2
            e = s + n
            r = s // Wp
            c = s - r * Wp
            while r * Wp < e:
                rs = r * Wp
                c1 = min(OW, e - rs)
                if c1 > c:
                    _evac_copy(obf[:, r * OW + c:r * OW + c1],
                               ps[:, rs + c - s:rs + c1 - s], which)
                r = r + 1
                c = 0
            ev = ev + 1
            s = e
    else:
        nch = _cdiv(nbr, nr)
        base = nbr // nch
        extra = nbr - base * nch
        a = 0
        for ci in range(nch):
            n = base
            if ci < extra:
                n = base + 1
            L = (n - 1) * Wp + OW
            ps = nl.ndarray((Co, nr * Wp), dtype=nl.float32, buffer=nl.psum)
            for t in range(T):
                kh = t // kW
                kw = t - kh * kW
                o = (a + kh) * Wp + kw
                nisa.nc_matmul(dst=ps[:, 0:L], stationary=wT[:, t, :],
                               moving=xf[:, o:o + L], accumulate=(t > 0))
            psv = ps.reshape((Co, nr, Wp))
            _evac_copy(ob[:, a:a + n, :], psv[:, 0:n, 0:OW], ev % 2)
            ev = ev + 1
            a = a + n
    return ev


@nki.jit
def conv2d_igemm(x, w, pad, flat, nr, sched, split, nbmax, esz):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    Co = w.shape[0]
    Cg = w.shape[1]
    kH = w.shape[2]
    kW = w.shape[3]
    T = kH * kW
    OH = H + 2 * pad - kH + 1
    OW = W + 2 * pad - kW + 1
    Wp = max(W + pad, OW)
    RBA = nbmax + kH

    out = nl.ndarray((B, Co, OH, OW), dtype=x.dtype, buffer=nl.shared_hbm)

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
        npw = _npieces(Co * KWT * esz)
        pgw = _cdiv(Co, npw)
        p0 = 0
        while p0 < Co:
            p1 = min(Co, p0 + pgw)
            nisa.dma_copy(dst=w_sb[p0:p1, :], src=w2[p0:p1, :])
            p0 = p1

        # ---- first input blocks (prefetch depth 2)
        k0 = b_lo
        xa = _load_block(x, 0, _row_start(k0, sched), sched[k0], H, W, Wp, pad, kH, C,
                         RBA, esz)
        xn = xa
        if njobs > 1:
            k1 = b_lo + 1 % nloc
            xn = _load_block(x, 1 // nloc, _row_start(k1, sched), sched[k1], H, W, Wp, pad,
                             kH, C, RBA, esz)

        # ---- per-tap TensorE transposes -> wT[ic, t, oc]
        w3 = w_sb.reshape((Co, Cg, T))
        wT = nl.ndarray((Cg, T, Co), dtype=w.dtype, buffer=nl.sbuf)
        for kh in range(kH):
            tp = nl.ndarray((Cg, kW, Co), dtype=w.dtype, buffer=nl.psum)
            for kw in range(kW):
                nisa.nc_transpose(dst=tp[:, kw, :], data=w3[:, :, kh * kW + kw],
                                  engine=nisa.engine.tensor)
            _evac_copy(wT[:, kh * kW:(kh + 1) * kW, :], tp, kh % 2)

        # ---- implicit GEMM over row blocks
        ev = 0
        for j in range(njobs):
            xq = xa
            if j + 2 < njobs:
                m = j + 2
                km = b_lo + m % nloc
                xq = _load_block(x, m // nloc, _row_start(km, sched), sched[km], H, W, Wp,
                                 pad, kH, C, RBA, esz)
            bbj = j // nloc
            kj = b_lo + j % nloc
            r0 = _row_start(kj, sched)
            nbr = sched[kj]
            ob = nl.ndarray((Co, nbmax, OW), dtype=x.dtype, buffer=nl.sbuf)
            ev = _compute_block(xa, ob, wT, nbr, flat, nr, C, Co, RBA, nbmax, Wp, OW, kW,
                                T, ev)
            _store_block(out, ob, bbj, r0, nbr, OW, Co, esz)
            xa = xn
            xn = xq
    return out


# ---------------------------------------------------------------------------
# Host side: deterministic block plan (fixed formula of the shape)
# ---------------------------------------------------------------------------
def _cap_rows(k, flat, Wp, OW, nr):
    # max output rows of a block that needs at most k PSUM chunks
    if flat:
        return max(1, (PSUM_COLS * k - OW) // Wp + 1)
    return k * nr


def _balanced(total, parts):
    base = total // parts
    extra = total - base * parts
    res = []
    for i in range(parts):
        res.append(base + (1 if i < extra else 0))
    return res


def _core_sched(R, flat, Wp, OW, nr):
    if R <= 0:
        return []
    if flat:
        U = _cdiv((R - 1) * Wp + OW, PSUM_COLS)
    else:
        U = _cdiv(R, nr)
    if U <= 4 and not flat:
        return _balanced(R, U)
    front = []
    rem = U
    c = 1
    while c < CPB and rem >= 3 * c:
        front.append(c)
        rem -= 2 * c
        c *= 2
    back = list(reversed(front))
    fr = [_cap_rows(k, flat, Wp, OW, nr) for k in front]
    br = [_cap_rows(k, flat, Wp, OW, nr) for k in back]
    Rm = R - sum(fr) - sum(br)
    if Rm <= 0:
        nb = _cdiv(R, _cap_rows(CPB, flat, Wp, OW, nr))
        return _balanced(R, nb)
    if flat:
        nm = _cdiv(Rm, _cap_rows(CPB, flat, Wp, OW, nr))
        mid = _balanced(Rm, nm)
    else:
        Um = _cdiv(Rm, nr)
        nm = _cdiv(Um, CPB)
        cnts = _balanced(Um, nm)
        mid = []
        acc = 0
        for i in range(nm):
            if i == nm - 1:
                mid.append(Rm - acc)
            else:
                mid.append(cnts[i] * nr)
                acc += cnts[i] * nr
    return fr + mid + br


def run(input, weight, stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, C, H, W = [int(s) for s in input.shape]
    Co, Cg, kH, kW = [int(s) for s in weight.shape]
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1:
        raise NotImplementedError("only stride=1, groups=1 supported")
    if C > 128 or Co > 128 or Cg != C:
        raise NotImplementedError("channels must be <= 128")

    OH = H + 2 * padding - kH + 1
    OW = W + 2 * padding - kW + 1
    Wp = max(W + padding, OW)
    if Wp > PSUM_COLS and Wp < FLAT_MIN_WP:
        raise NotImplementedError("unsupported width")
    flat = 1 if Wp >= FLAT_MIN_WP else 0
    nr = max(1, PSUM_COLS // Wp)

    R0 = (OH + 1) // 2
    R1 = OH - R0
    s0 = _core_sched(R0, flat, Wp, OW, nr)
    s1 = _core_sched(R1, flat, Wp, OW, nr)
    sched = tuple(s0 + s1)
    split = len(s0)
    nbmax = max(sched)
    esz = int(input.element_size())

    out = conv2d_igemm(input, weight, padding, flat, nr, sched, split, nbmax, esz)

    _LAST_CONFIG = {
        "algorithm": "flat_shift_implicit_gemm_shared_pad",
        "mode": "flat_chunks" if flat else "row_chunks",
        "padded_width": Wp,
        "rows_per_chunk": nr if not flat else None,
        "block_rows_core0": list(s0),
        "block_rows_core1": list(s1),
        "max_block_rows": nbmax,
        "psum_cols": PSUM_COLS,
        "chunks_per_block_max": CPB,
        "prefetch_depth": 2,
        "dma_pieces_small_big": [4, 8],
        "dma_big_bytes": DMA_MID_BYTES,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
