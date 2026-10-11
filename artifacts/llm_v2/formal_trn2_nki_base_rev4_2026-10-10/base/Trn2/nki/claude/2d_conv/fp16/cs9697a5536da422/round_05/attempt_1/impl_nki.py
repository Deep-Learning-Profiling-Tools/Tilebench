import nki
import nki.isa as nisa
import nki.language as nl

# ---------------------------------------------------------------------------
# Static configuration (literals; the per-shape block plan is a fixed formula)
# ---------------------------------------------------------------------------
PSUM_COLS = 512            # fp32 columns per PSUM bank == max matmul moving free size
BLOCK_TARGET_ELEMS = 4096  # target (rows * padded width) per steady-state block
BLOCK_MIN_ELEMS = 512      # minimum work per block when splitting small problems
MIN_BLOCKS = 4             # aim for at least this many blocks per core when big enough
RAMP_MIN_WORK = 3000       # per-core rows*Wp above which head/tail ramp blocks are used
FLAT_MIN_W = 128           # use row-crossing (flat) PSUM chunks for W >= this
DMA_PART_GROUP = 32        # partitions per DMA piece for activations / outputs
W_PART_GROUP = 16          # partitions per DMA piece for the weight load
PREFETCH = 2               # input blocks loaded ahead of compute

_LAST_CONFIG = {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _evac_copy(dst, src, which):
    if which == 0:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
    else:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.scalar)


def _load_block(x, bb, r0, nbr, H, W, Wp, RB, C, pg):
    # Padded SBUF buffer: row i <-> image row (r0 - 1 + i), col j <-> image col (j - 1).
    xb = nl.ndarray((C, RB, Wp), dtype=x.dtype, buffer=nl.sbuf)
    need = nbr + 2
    top = r0 - 1
    s_lo = max(0, top)
    s_hi = min(H, top + need)
    i_lo = s_lo - top
    i_hi = s_hi - top
    if i_lo > 0:
        nisa.memset(dst=xb[:, 0:i_lo, :], value=0.0)
    if i_hi < need:
        nisa.memset(dst=xb[:, i_hi:need, :], value=0.0)
    nisa.memset(dst=xb[:, i_lo:i_hi, 0:1], value=0.0)
    nisa.memset(dst=xb[:, i_lo:i_hi, W + 1:Wp], value=0.0)
    p0 = 0
    while p0 < C:
        p1 = min(C, p0 + pg)
        nisa.dma_copy(dst=xb[p0:p1, i_lo:i_hi, 1:1 + W],
                      src=x[bb, p0:p1, s_lo:s_hi, :])
        p0 = p1
    return xb


def _compute_flat(xb, ob, wT, nbr, Wp, kW, T, C, Co, RB, nbmax, ev):
    # Output flat p = r*Wp + c (padded-width coordinates, 2 junk columns per row);
    # tap (kh, kw) reads buffer flat p + kh*Wp + kw.
    xf = xb.reshape((C, RB * Wp))
    obf = ob.reshape((Co, nbmax * Wp))
    L = nbr * Wp - (kW - 1)
    nch = _cdiv(L, PSUM_COLS)
    tgt = _cdiv(_cdiv(L, nch), 16) * 16
    s = 0
    for ci in range(nch):
        n = min(tgt, L - s)
        if n > 0:
            ps = nl.ndarray((Co, PSUM_COLS), dtype=nl.float32, buffer=nl.psum)
            for t in range(T):
                kh = t // kW
                kw = t - kh * kW
                off = s + kh * Wp + kw
                nisa.nc_matmul(dst=ps[:, 0:n], stationary=wT[:, t, :],
                               moving=xf[:, off:off + n], accumulate=(t > 0))
            _evac_copy(obf[:, s:s + n], ps[:, 0:n], ev % 2)
            ev = ev + 1
            s = s + n
    return ev


def _compute_rows(xb, ob, wT, nbr, nr, W, Wp, kW, T, C, Co, RB, ev):
    xf = xb.reshape((C, RB * Wp))
    nch = _cdiv(nbr, nr)
    base = nbr // nch
    extra = nbr - base * nch
    a = 0
    for ci in range(nch):
        n = base
        if ci < extra:
            n = base + 1
        L = n * Wp - (kW - 1)
        ps = nl.ndarray((Co, nr * Wp), dtype=nl.float32, buffer=nl.psum)
        for t in range(T):
            kh = t // kW
            kw = t - kh * kW
            off = (a + kh) * Wp + kw
            nisa.nc_matmul(dst=ps[:, 0:L], stationary=wT[:, t, :],
                           moving=xf[:, off:off + L], accumulate=(t > 0))
        psv = ps.reshape((Co, nr, Wp))
        _evac_copy(ob[:, a:a + n, :], psv[:, 0:n, 0:W], ev % 2)
        ev = ev + 1
        a = a + n
    return ev


def _store_block(out, ob, bb, r0, nbr, W, Co, pg):
    p0 = 0
    while p0 < Co:
        p1 = min(Co, p0 + pg)
        nisa.dma_copy(dst=out[bb, p0:p1, r0:r0 + nbr, :], src=ob[p0:p1, 0:nbr, 0:W])
        p0 = p1


@nki.jit
def conv2d_igemm(x, w, sched, starts, split, nbmax, flat, nr, pg, pgw):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    Co = w.shape[0]
    Cg = w.shape[1]
    kH = w.shape[2]
    kW = w.shape[3]
    T = kH * kW
    Wp = W + 2
    RB = nbmax + 2

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
        p0 = 0
        while p0 < Co:
            p1 = min(Co, p0 + pgw)
            nisa.dma_copy(dst=w_sb[p0:p1, :], src=w2[p0:p1, :])
            p0 = p1

        # ---- first input blocks
        k0 = b_lo
        xa = _load_block(x, 0, starts[k0], sched[k0], H, W, Wp, RB, C, pg)
        xn = xa
        if njobs > 1:
            k1 = b_lo + (1 % nloc)
            xn = _load_block(x, 1 // nloc, starts[k1], sched[k1], H, W, Wp, RB, C, pg)

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
                xq = _load_block(x, m // nloc, starts[km], sched[km], H, W, Wp, RB, C, pg)
            bbj = j // nloc
            kj = b_lo + (j % nloc)
            r0 = starts[kj]
            nbr = sched[kj]
            if flat == 1:
                ob = nl.ndarray((Co, nbmax, Wp), dtype=x.dtype, buffer=nl.sbuf)
                ev = _compute_flat(xa, ob, wT, nbr, Wp, kW, T, C, Co, RB, nbmax, ev)
                _store_block(out, ob, bbj, r0, nbr, W, Co, pg)
            else:
                ob = nl.ndarray((Co, nbmax, W), dtype=x.dtype, buffer=nl.sbuf)
                ev = _compute_rows(xa, ob, wT, nbr, nr, W, Wp, kW, T, C, Co, RB, ev)
                _store_block(out, ob, bbj, r0, nbr, W, Co, pg)
            xa = xn
            xn = xq
    return out


# ---------------------------------------------------------------------------
# Host side: deterministic block plan (fixed formula of the shape)
# ---------------------------------------------------------------------------
def _plain_blocks(R, nb):
    res = []
    r = 0
    while r < R:
        n = min(nb, R - r)
        res.append(n)
        r += n
    return res


def _balanced(total, parts):
    base = total // parts
    extra = total - base * parts
    res = []
    for i in range(parts):
        res.append(base + (1 if i < extra else 0))
    return res


def _core_blocks(R, Wp):
    if R <= 0:
        return []
    work = R * Wp
    nblk = max(_cdiv(work, BLOCK_TARGET_ELEMS),
               min(MIN_BLOCKS, _cdiv(work, BLOCK_MIN_ELEMS)))
    nblk = max(1, min(nblk, R))
    nb = _cdiv(R, nblk)
    if work >= RAMP_MIN_WORK:
        h1 = max(1, _cdiv(nb, 4))
        h2 = max(1, _cdiv(nb, 2))
        M = R - 2 * (h1 + h2)
        if M >= h2:
            mid = _balanced(M, _cdiv(M, nb))
            return [h1, h2] + mid + [h2, h1]
    return _plain_blocks(R, nb)


def run(input, weight, stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, C, H, W = [int(s) for s in input.shape]
    Co, Cg, kH, kW = [int(s) for s in weight.shape]
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1 or padding != 1 or kH != 3 or kW != 3:
        raise NotImplementedError("only stride=1, padding=1, 3x3, groups=1 supported")
    if C > 128 or Co > 128 or Cg != C:
        raise NotImplementedError("channels must be <= 128")

    Wp = W + 2
    OH = H
    R0 = (OH + 1) // 2
    R1 = OH - R0
    s0 = _core_blocks(R0, Wp)
    s1 = _core_blocks(R1, Wp)
    sched_l = s0 + s1
    starts_l = []
    acc = 0
    for v in sched_l:
        starts_l.append(acc)
        acc += v
    sched = tuple(sched_l)
    starts = tuple(starts_l)
    split = len(s0)
    nbmax = max(sched_l)
    flat = 1 if W >= FLAT_MIN_W else 0
    if flat == 0 and Wp > PSUM_COLS:
        raise NotImplementedError("padded width too large for row mode")
    nr = max(1, min(PSUM_COLS // Wp, nbmax))

    out = conv2d_igemm(input, weight, sched, starts, split, nbmax, flat, nr,
                       DMA_PART_GROUP, W_PART_GROUP)

    _LAST_CONFIG = {
        "algorithm": "implicit_gemm_padded_flat_shift",
        "chunk_mode": "flat_512" if flat == 1 else "row_aligned",
        "block_rows_core0": list(s0),
        "block_rows_core1": list(s1),
        "max_block_rows": nbmax,
        "rows_per_psum_chunk_rowmode": nr,
        "psum_cols": PSUM_COLS,
        "prefetch_depth": PREFETCH,
        "dma_partition_group": DMA_PART_GROUP,
        "weight_dma_partition_group": W_PART_GROUP,
        "padded_width": Wp,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
