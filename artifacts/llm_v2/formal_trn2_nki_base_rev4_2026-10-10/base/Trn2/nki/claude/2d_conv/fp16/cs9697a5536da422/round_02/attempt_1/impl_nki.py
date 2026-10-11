import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Static configuration (literals; the per-shape block plan is a fixed formula)
# ---------------------------------------------------------------------------
PSUM_COLS = 512          # fp32 columns per PSUM bank == max matmul moving free size
CMAX = 12                # steady-state PSUM chunks per row block
GROUP = 2                # chunks sharing one pass over the taps (stationary reuse)
DMA_PIECE_BYTES = 32768  # target bytes per DMA piece
MAX_DMA_PIECES = 8       # max partition groups per transfer
# ring sizes (fixed in the kernel code): PSUM 4, compact-in 3, padded-in 2,
# flat-out 2, compact-out 3

_LAST_CONFIG = {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _pick2(i, a, b):
    if i == 0:
        return a
    return b


def _pick3(i, a, b, c):
    if i == 0:
        return a
    if i == 1:
        return b
    return c


def _pick4(i, a, b, c, d):
    if i == 0:
        return a
    if i == 1:
        return b
    if i == 2:
        return c
    return d


def _pieces(nbytes):
    n = 2
    while n < MAX_DMA_PIECES and nbytes > n * DMA_PIECE_BYTES:
        n = n * 2
    return n


def _row_start(k, sched):
    r = 0
    i = 0
    while i < k:
        r = r + sched[i]
        i = i + 1
    return r


def _chunk_len(i, nch, base, extra, odd):
    n = 2 * base
    if i < extra:
        n = n + 2
    if i == nch - 1:
        n = n + odd
    return n


def _issue_load(x, xc, bb, r0, nbr, H, W, C, pad, kH, esz):
    # contiguous rows (per channel) -> compact SBUF staging
    top = r0 - pad
    s_lo = max(0, top)
    s_hi = min(H, top + nbr + kH - 1)
    nrows = s_hi - s_lo
    if nrows > 0:
        npc = _pieces(C * nrows * W * esz)
        pg = _cdiv(C, npc)
        p0 = 0
        while p0 < C:
            p1 = min(C, p0 + pg)
            nisa.dma_copy(dst=xc[p0:p1, 0:nrows, :], src=x[bb, p0:p1, s_lo:s_hi, :])
            p0 = p1


def _issue_pad(xp, xc, r0, nbr, H, W, Wp, pad, kH, RB, first):
    # compact staging -> zero-padded SBUF block (row i <-> image row r0-pad+i)
    top = r0 - pad
    need = nbr + kH - 1
    s_lo = max(0, top)
    s_hi = min(H, top + need)
    i_lo = s_lo - top
    i_hi = s_hi - top
    if first:
        if pad > 0:
            nisa.memset(dst=xp[:, 0:RB, 0:pad], value=0.0)
            nisa.memset(dst=xp[:, 0:RB, pad + W:Wp], value=0.0)
    if i_lo > 0:
        nisa.memset(dst=xp[:, 0:i_lo, :], value=0.0)
    if i_hi < need:
        nisa.memset(dst=xp[:, i_hi:need, :], value=0.0)
    if i_hi > i_lo:
        nisa.tensor_copy(dst=xp[:, i_lo:i_hi, pad:pad + W], src=xc[:, 0:i_hi - i_lo, :],
                         engine=nisa.engine.scalar)


def _compute(xp, of, wT, ps0, ps1, ps2, ps3, kk, nbr, C, Co, RB, rmax, Wp, kW, T):
    # flat-shift implicit GEMM: output flat p = oh*Wp + ow, tap reads p + kh*Wp + kw
    xf = xp.reshape((C, RB * Wp))
    ofl = of.reshape((Co, rmax * Wp))
    Lb = nbr * Wp - (kW - 1)
    nch = _cdiv(Lb, PSUM_COLS)
    half = Lb // 2
    odd = Lb - 2 * half
    base = half // nch
    extra = half - base * nch
    c = 0
    s = 0
    while c < nch:
        g = min(GROUP, nch - c)
        for t in range(T):
            kh = t // kW
            kw = t - kh * kW
            sc = s
            for q in range(g):
                n = _chunk_len(c + q, nch, base, extra, odd)
                ps = _pick4((kk + q) % 4, ps0, ps1, ps2, ps3)
                o = sc + kh * Wp + kw
                nisa.nc_matmul(dst=ps[:, 0:n], stationary=wT[:, t, :],
                               moving=xf[:, o:o + n], accumulate=(t > 0))
                sc = sc + n
        sc = s
        for q in range(g):
            n = _chunk_len(c + q, nch, base, extra, odd)
            ps = _pick4((kk + q) % 4, ps0, ps1, ps2, ps3)
            nisa.tensor_copy(dst=ofl[:, sc:sc + n], src=ps[:, 0:n], engine=nisa.engine.vector)
            sc = sc + n
        kk = kk + g
        c = c + g
        s = sc
    return kk


def _issue_store(out, of, oc, bb, r0, nbr, OW, Co, esz):
    nisa.tensor_copy(dst=oc[:, 0:nbr, :], src=of[:, 0:nbr, 0:OW], engine=nisa.engine.vector)
    npc = _pieces(Co * nbr * OW * esz)
    pg = _cdiv(Co, npc)
    p0 = 0
    while p0 < Co:
        p1 = min(Co, p0 + pg)
        nisa.dma_copy(dst=out[bb, p0:p1, r0:r0 + nbr, :], src=oc[p0:p1, 0:nbr, :])
        p0 = p1


@nki.jit
def conv2d_flat_igemm(x, w, pad, sched, split, esz):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    Co = w.shape[0]
    Cg = w.shape[1]
    kH = w.shape[2]
    kW = w.shape[3]
    T = kH * kW
    Wp = W + 2 * pad
    OH = H + 2 * pad - kH + 1
    OW = Wp - kW + 1

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
        rmax = 1
        tot_ch = 0
        k = b_lo
        while k < b_hi:
            nb_k = sched[k]
            rmax = max(rmax, nb_k)
            tot_ch = tot_ch + _cdiv(nb_k * Wp - (kW - 1), PSUM_COLS)
            k = k + 1
        tot_ch = tot_ch * B
        RB = rmax + kH - 1

        # ---- PSUM accumulator ring (one bank each)
        ps0 = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
        ps1 = ps0
        ps2 = ps0
        ps3 = ps0
        if tot_ch > 1:
            ps1 = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
        if tot_ch > 2:
            ps2 = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
        if tot_ch > 3:
            ps3 = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)

        # ---- weights: (Co, Cg*T) contiguous rows -> TensorE transposes -> wT[ic, t, oc]
        KWT = Cg * T
        w2 = w.reshape((Co, KWT))
        w_sb = nl.ndarray((Co, KWT), dtype=w.dtype, buffer=nl.sbuf)
        npw = _pieces(Co * KWT * esz)
        pgw = _cdiv(Co, npw)
        p0 = 0
        while p0 < Co:
            p1 = min(Co, p0 + pgw)
            nisa.dma_copy(dst=w_sb[p0:p1, :], src=w2[p0:p1, :])
            p0 = p1

        # ---- input staging rings
        xc0 = nl.ndarray((C, RB, W), dtype=x.dtype, buffer=nl.sbuf)
        xc1 = xc0
        xc2 = xc0
        if njobs > 1:
            xc1 = nl.ndarray((C, RB, W), dtype=x.dtype, buffer=nl.sbuf)
        if njobs > 2:
            xc2 = nl.ndarray((C, RB, W), dtype=x.dtype, buffer=nl.sbuf)
        xp0 = nl.ndarray((C, RB, Wp), dtype=x.dtype, buffer=nl.sbuf)
        xp1 = xp0
        if njobs > 1:
            xp1 = nl.ndarray((C, RB, Wp), dtype=x.dtype, buffer=nl.sbuf)

        # prologue loads (issued early, overlap the weight preparation)
        m = 0
        while m < njobs and m < 3:
            bbm = m // nloc
            km = b_lo + m % nloc
            _issue_load(x, _pick3(m % 3, xc0, xc1, xc2), bbm, _row_start(km, sched),
                        sched[km], H, W, C, pad, kH, esz)
            m = m + 1

        w3 = w_sb.reshape((Co, Cg, T))
        wT = nl.ndarray((Cg, T, Co), dtype=w.dtype, buffer=nl.sbuf)
        TA = _cdiv(T, 2)
        tpa = nl.ndarray((Cg, TA, Co), dtype=w.dtype, buffer=nl.psum)
        for t in range(TA):
            nisa.nc_transpose(dst=tpa[:, t, :], data=w3[:, :, t], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=wT[:, 0:TA, :], src=tpa, engine=nisa.engine.vector)
        TB = T - TA
        if TB > 0:
            tpb = nl.ndarray((Cg, TB, Co), dtype=w.dtype, buffer=nl.psum)
            for t in range(TB):
                nisa.nc_transpose(dst=tpb[:, t, :], data=w3[:, :, TA + t],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=wT[:, TA:T, :], src=tpb, engine=nisa.engine.vector)

        # ---- output staging rings
        of0 = nl.ndarray((Co, rmax, Wp), dtype=x.dtype, buffer=nl.sbuf)
        of1 = of0
        if njobs > 1:
            of1 = nl.ndarray((Co, rmax, Wp), dtype=x.dtype, buffer=nl.sbuf)
        oc0 = nl.ndarray((Co, rmax, OW), dtype=x.dtype, buffer=nl.sbuf)
        oc1 = oc0
        oc2 = oc0
        if njobs > 1:
            oc1 = nl.ndarray((Co, rmax, OW), dtype=x.dtype, buffer=nl.sbuf)
        if njobs > 2:
            oc2 = nl.ndarray((Co, rmax, OW), dtype=x.dtype, buffer=nl.sbuf)

        # pad job 0, then refill its compact slot
        k0 = b_lo
        _issue_pad(xp0, xc0, _row_start(k0, sched), sched[k0], H, W, Wp, pad, kH, RB, True)
        if njobs > 3:
            m = 3
            bbm = m // nloc
            km = b_lo + m % nloc
            _issue_load(x, xc0, bbm, _row_start(km, sched), sched[km], H, W, C, pad, kH, esz)

        kk = 0
        j = 0
        while j < njobs:
            if j + 1 < njobs:
                m = j + 1
                km = b_lo + m % nloc
                _issue_pad(_pick2(m % 2, xp0, xp1), _pick3(m % 3, xc0, xc1, xc2),
                           _row_start(km, sched), sched[km], H, W, Wp, pad, kH, RB, m < 2)
                if m + 3 < njobs:
                    m2 = m + 3
                    bb2 = m2 // nloc
                    k2 = b_lo + m2 % nloc
                    _issue_load(x, _pick3(m % 3, xc0, xc1, xc2), bb2, _row_start(k2, sched),
                                sched[k2], H, W, C, pad, kH, esz)
            bbj = j // nloc
            kj = b_lo + j % nloc
            r0 = _row_start(kj, sched)
            nbr = sched[kj]
            ofj = _pick2(j % 2, of0, of1)
            kk = _compute(_pick2(j % 2, xp0, xp1), ofj, wT, ps0, ps1, ps2, ps3, kk, nbr,
                          C, Co, RB, rmax, Wp, kW, T)
            _issue_store(out, ofj, _pick3(j % 3, oc0, oc1, oc2), bbj, r0, nbr, OW, Co, esz)
            j = j + 1
    return out


# ---------------------------------------------------------------------------
# Host side: deterministic block plan (fixed formula of the shape)
# ---------------------------------------------------------------------------
def _core_sched(R, Wp, kW):
    if R <= 0:
        return []
    L = R * Wp - (kW - 1)
    U = max(1, (L + PSUM_COLS - 1) // PSUM_COLS)
    if U <= 2:
        return [R]
    front = []
    rem = U
    c = 1
    while c < CMAX and rem >= 3 * c:
        front.append(c)
        rem -= 2 * c
        c *= 2
    targets = [float(v) for v in front]
    if rem > 0:
        nmid = (rem + CMAX - 1) // CMAX
        for _ in range(nmid):
            targets.append(rem / nmid)
    for v in reversed(front):
        targets.append(float(v))
    nb = len(targets)
    if nb > R:
        return [R]
    S = float(sum(targets))
    rows = []
    prev = 0
    acc = 0.0
    for i, tg in enumerate(targets):
        acc += tg
        if i == nb - 1:
            bnd = R
        else:
            bnd = int(round(R * acc / S))
            bnd = max(bnd, prev + 1)
            bnd = min(bnd, R - (nb - 1 - i))
        rows.append(bnd - prev)
        prev = bnd
    return rows


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
    Wp = W + 2 * padding
    R0 = (OH + 1) // 2
    R1 = OH - R0
    s0 = _core_sched(R0, Wp, kW)
    s1 = _core_sched(R1, Wp, kW)
    sched = tuple(s0 + s1)
    split = len(s0)
    esz = int(input.element_size())

    out = conv2d_flat_igemm(input, weight, padding, sched, split, esz)

    _LAST_CONFIG = {
        "algorithm": "flat_shift_implicit_gemm_ring_pipelined",
        "block_rows_core0": list(s0),
        "block_rows_core1": list(s1),
        "psum_cols": PSUM_COLS,
        "chunks_per_block_max": CMAX,
        "tap_group": GROUP,
        "psum_slots": 4,
        "input_compact_slots": 3,
        "input_padded_slots": 2,
        "output_slots": [2, 3],
        "dma_piece_bytes": DMA_PIECE_BYTES,
        "max_dma_pieces": MAX_DMA_PIECES,
        "padded_width": Wp,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
