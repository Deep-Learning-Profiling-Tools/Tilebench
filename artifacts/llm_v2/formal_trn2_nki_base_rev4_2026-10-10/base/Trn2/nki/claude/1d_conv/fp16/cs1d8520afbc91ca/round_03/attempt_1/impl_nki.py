"""
Implicit-GEMM grouped 1-D convolution (fp16) for Trn2 / NKI 0.7.0b1.

Mapping (groups == 1, stride == 1, out_L == L):
  out[b, oc, p] = sum_k  sum_ic  W[oc, ic, k] * x[b, ic, p + k - pad]
  * contraction (ic) on the 128 SBUF partitions, one nc_matmul per tap k:
        psum[oc, n] += WT_k[ic, oc]^T @ xwin[ic, n + k]
  * every output element: one fp32 PSUM accumulator over all 3*Cin terms,
    cast to fp16 once on PSUM->SBUF evacuation, then DMA'd to HBM.
  * input tiles are staged with a 64-byte-aligned halo (zero-filled only
    outside [0, L)); nothing intermediate goes to HBM.
  * LNC=2: each physical core handles one half of the length axis.
  * graded edge tiles (T/4, T/4, T/2, T..., T/2, T/4, T/4) for fast
    pipeline fill/drain; 5 input staging buffers (prefetch 4 tiles),
    ring of 8 output chunk buffers (4096 cols each, stored as soon as the
    chunk is evacuated), 3 PSUM groups of 2 banks; evacuation alternates
    between VectorE and ScalarE; DMAs split into 4 partition groups.
"""
import torch

try:
    import torch_neuronx  # noqa: F401  (registers the "neuron" device)
except Exception:  # pragma: no cover
    torch_neuronx = None

import nki
import nki.language as nl
import nki.isa as nisa

# ---------------- static configuration ----------------
_TILE = 8192      # columns per (full) staged input tile
_GROUP = 1024     # columns per PSUM group (2 banks of 512 fp32)
_NMM = 512        # columns per nc_matmul (moving free max)
_NX = 5           # input staging buffers (prefetch distance _NX - 1)
_NOC = 8          # output chunk buffers
_OCH = 4096       # columns per output chunk (multiple of _GROUP)
_NPS = 3          # PSUM group buffers
_ALIGN = 32       # halo granularity in elements (64 bytes for fp16)
_DS = 4           # DMA partition split

_LAST_CONFIG = {}


def _pick3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _pick5(i, a, b, c, d, e):
    r = e
    if i == 0:
        r = a
    elif i == 1:
        r = b
    elif i == 2:
        r = c
    elif i == 3:
        r = d
    return r


def _pick8(i, a, b, c, d, e, f, g, h):
    r = h
    if i == 0:
        r = a
    elif i == 1:
        r = b
    elif i == 2:
        r = c
    elif i == 3:
        r = d
    elif i == 4:
        r = e
    elif i == 5:
        r = f
    elif i == 6:
        r = g
    return r


def _is_graded(Lh, T):
    g = False
    if T % 128 == 0:
        if Lh % T == 0:
            if Lh >= 2 * T:
                g = True
    return g


def _num_tiles(Lh, T, graded):
    n = (Lh + T - 1) // T
    if graded:
        n = Lh // T + 4
    return n


def _tile_lo(i, n, Lh, T, graded):
    r = i * T
    if graded:
        q = T // 4
        r = T + (i - 3) * T
        if i == 0:
            r = 0
        elif i == 1:
            r = q
        elif i == 2:
            r = 2 * q
        elif i == n - 3:
            r = Lh - T
        elif i == n - 2:
            r = Lh - 2 * q
        elif i == n - 1:
            r = Lh - q
    return r


def _tile_hi(i, n, Lh, T, graded):
    r = min((i + 1) * T, Lh)
    if graded:
        r = Lh
        if i < n - 1:
            r = _tile_lo(i + 1, n, Lh, T, graded)
    return r


def _load_tile(inp, xb, b, s, e, L, HLX, HRX, CI, DS):
    # stage x[b, :, s-HLX : e+HRX] into xb[:, 0 : e-s+HLX+HRX], zero outside [0, L)
    lo = s - HLX
    hi = e + HRX
    clo = max(lo, 0)
    chi = min(hi, L)
    if clo > lo:
        nisa.memset(dst=xb[:, 0:clo - lo], value=0.0)
    if chi < hi:
        nisa.memset(dst=xb[:, chi - lo:hi - lo], value=0.0)
    pp = CI // DS
    for h in range(DS):
        nisa.dma_copy(dst=xb[h * pp:(h + 1) * pp, clo - lo:chi - lo],
                      src=inp[b, h * pp:(h + 1) * pp, clo:chi])


@nki.jit
def _conv1d_kernel(inp, w, pad, T, G, HLX, HRX, DS, OCH):
    B, CI, L = inp.shape
    CO, CIW, K = w.shape
    out = nl.ndarray((B, CO, L), dtype=inp.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    Lh = L // nprog
    c0 = pid * Lh
    HL = pad
    XW = HLX + T + HRX

    graded = _is_graded(Lh, T)
    ntc = _num_tiles(Lh, T, graded)
    NT = B * ntc

    DSI = 1
    if CI % DS == 0:
        DSI = DS
    DSO = 1
    if CO % DS == 0:
        DSO = DS
    po = CO // DSO

    # ---------- weights: wT[:, k*CO:(k+1)*CO][ic, oc] = w[oc, ic, k] ----------
    w_sb = nl.ndarray((CO, CIW * K), dtype=w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=w.reshape((CO, CIW * K)))
    pT = nl.ndarray((CIW, K * CO), dtype=w.dtype, buffer=nl.psum)
    for k in range(K):
        wk = nl.ndarray((CO, CIW), dtype=w.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wk, src=w_sb[:, k:CIW * K:K], engine=nisa.engine.vector)
        nisa.nc_transpose(dst=pT[:, k * CO:(k + 1) * CO], data=wk, engine=nisa.engine.tensor)
    wT = nl.ndarray((CIW, K * CO), dtype=w.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wT, src=pT, engine=nisa.engine.vector)

    # ---------- rotating buffers ----------
    xb0 = nl.ndarray((CI, XW), dtype=inp.dtype, buffer=nl.sbuf)
    xb1 = nl.ndarray((CI, XW), dtype=inp.dtype, buffer=nl.sbuf)
    xb2 = nl.ndarray((CI, XW), dtype=inp.dtype, buffer=nl.sbuf)
    xb3 = nl.ndarray((CI, XW), dtype=inp.dtype, buffer=nl.sbuf)
    xb4 = nl.ndarray((CI, XW), dtype=inp.dtype, buffer=nl.sbuf)
    ob0 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob1 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob2 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob3 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob4 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob5 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob6 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ob7 = nl.ndarray((CO, OCH), dtype=inp.dtype, buffer=nl.sbuf)
    ps0 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)
    ps1 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)
    ps2 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)

    # ---------- prologue: prefetch first tiles ----------
    npre = min(_NX - 1, NT)
    for t in range(npre):
        bq = t // ntc
        tq = t - bq * ntc
        sq = c0 + _tile_lo(tq, ntc, Lh, T, graded)
        eq = c0 + _tile_hi(tq, ntc, Lh, T, graded)
        _load_tile(inp, _pick5(t % _NX, xb0, xb1, xb2, xb3, xb4), bq, sq, eq, L,
                   HLX, HRX, CI, DSI)

    # ---------- main pipeline ----------
    gc = 0
    oc = 0
    for t in range(NT):
        tn = t + _NX - 1
        if tn < NT:
            bq = tn // ntc
            tq = tn - bq * ntc
            sq = c0 + _tile_lo(tq, ntc, Lh, T, graded)
            eq = c0 + _tile_hi(tq, ntc, Lh, T, graded)
            _load_tile(inp, _pick5(tn % _NX, xb0, xb1, xb2, xb3, xb4), bq, sq, eq, L,
                       HLX, HRX, CI, DSI)

        bc = t // ntc
        tcur = t - bc * ntc
        s = c0 + _tile_lo(tcur, ntc, Lh, T, graded)
        e = c0 + _tile_hi(tcur, ntc, Lh, T, graded)
        W = e - s
        xb = _pick5(t % _NX, xb0, xb1, xb2, xb3, xb4)
        ng = (W + G - 1) // G
        cs = 0
        ob = _pick8(oc % _NOC, ob0, ob1, ob2, ob3, ob4, ob5, ob6, ob7)
        for gi in range(ng):
            g0 = gi * G
            gw = min(G, W - g0)
            ps = _pick3(gc % _NPS, ps0, ps1, ps2)
            nb = (gw + _NMM - 1) // _NMM
            for k in range(K):
                for bi in range(nb):
                    m0 = bi * _NMM
                    mw = min(_NMM, gw - m0)
                    cc = HLX - HL + g0 + m0 + k
                    nisa.nc_matmul(dst=ps[:, m0:m0 + mw],
                                   stationary=wT[:, k * CO:(k + 1) * CO],
                                   moving=xb[:, cc:cc + mw],
                                   accumulate=(k > 0))
            d0 = g0 - cs
            if gc % 2 == 0:
                nisa.tensor_copy(dst=ob[:, d0:d0 + gw], src=ps[:, 0:gw],
                                 engine=nisa.engine.vector)
            else:
                nisa.activation(dst=ob[:, d0:d0 + gw], op=nl.copy, data=ps[:, 0:gw])
            gc = gc + 1
            gend = g0 + gw
            flush = False
            if gend - cs >= OCH:
                flush = True
            if gend == W:
                flush = True
            if flush:
                cw = gend - cs
                for h in range(DSO):
                    nisa.dma_copy(dst=out[bc, h * po:(h + 1) * po, s + cs:s + gend],
                                  src=ob[h * po:(h + 1) * po, 0:cw])
                cs = gend
                oc = oc + 1
                ob = _pick8(oc % _NOC, ob0, ob1, ob2, ob3, ob4, ob5, ob6, ob7)
    return out


def _graded_host(Lh, T):
    return (T % 128 == 0) and (Lh % T == 0) and (Lh >= 2 * T)


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, CI, L = input.shape
    CO, CIg, K = weight.shape
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1 or CIg != CI or K != 2 * padding + 1:
        raise NotImplementedError("unsupported conv1d configuration")
    if CI > 128 or CO > 128 or L % 2 != 0:
        raise NotImplementedError("unsupported conv1d shape")
    if _OCH % _GROUP != 0:
        raise NotImplementedError("bad static configuration")

    HL = padding
    HR = K - 1 - padding
    HLX = ((HL + _ALIGN - 1) // _ALIGN) * _ALIGN
    HRX = ((HR + _ALIGN - 1) // _ALIGN) * _ALIGN
    ds = _DS if (CI % _DS == 0 and CO % _DS == 0) else 1

    out = _conv1d_kernel(input, weight, padding, _TILE, _GROUP, HLX, HRX, ds, _OCH)

    Lh = L // 2
    _LAST_CONFIG = {
        "tile_cols": _TILE,
        "psum_group_cols": _GROUP,
        "matmul_cols": _NMM,
        "x_buffers": _NX,
        "out_chunk_buffers": _NOC,
        "out_chunk_cols": _OCH,
        "psum_buffers": _NPS,
        "lnc_programs": 2,
        "cols_per_program": Lh,
        "graded_edge_tiles": bool(_graded_host(Lh, _TILE)),
        "halo_left": HLX,
        "halo_right": HRX,
        "dma_partition_split": ds,
        "kernel_size": int(K),
        "padding": padding,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
