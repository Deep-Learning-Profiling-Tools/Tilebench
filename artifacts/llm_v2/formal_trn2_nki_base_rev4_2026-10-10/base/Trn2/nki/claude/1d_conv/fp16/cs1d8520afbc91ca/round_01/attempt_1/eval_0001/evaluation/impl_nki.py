"""
Implicit-GEMM grouped 1-D convolution (fp16) for Trn2 / NKI 0.7.0b1.

Mapping (groups == 1, stride == 1, out_L == L):
  out[b, oc, p] = sum_k  sum_ic  W[oc, ic, k] * x[b, ic, p + k - pad]
  * contraction (ic) on the 128 SBUF partitions, one nc_matmul per tap k:
        psum[oc, n] += WT_k[ic, oc]^T @ xwin[ic, n + k]
    (stationary = WT_k, moving = shifted window of the staged input tile)
  * every output element: one fp32 PSUM accumulator over all 3*Cin terms,
    cast to fp16 once on PSUM->SBUF evacuation, then DMA'd to HBM.
  * input tiles are staged with a halo (zero-filled only outside [0, L));
    nothing intermediate goes to HBM.
  * LNC=2: each physical core handles one half of the length axis.
  * explicit rotating buffers: 4 input tiles (prefetch distance 3),
    3 output tiles, 3 PSUM groups of 2 banks; evacuation alternates
    between VectorE and ScalarE.
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
_TILE = 4096      # columns per staged DMA tile
_GROUP = 1024     # columns per PSUM group (2 banks of 512 fp32)
_NMM = 512        # columns per nc_matmul (moving free max)
_NX = 4           # input staging buffers (prefetch distance _NX - 1)
_NO = 3           # output staging buffers
_NPS = 3          # PSUM group buffers

_LAST_CONFIG = {}


def _pick4(i, a, b, c, d):
    r = d
    if i == 0:
        r = a
    elif i == 1:
        r = b
    elif i == 2:
        r = c
    return r


def _pick3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _tile_lo(tt, T, split):
    r = tt * T
    if split:
        r = 0
        if tt > 0:
            r = T // 2 + (tt - 1) * T
    return r


def _tile_hi(tt, Lh, T, split):
    r = min((tt + 1) * T, Lh)
    if split:
        r = min(T // 2 + tt * T, Lh)
    return r


def _load_tile(inp, xb, b, s, e, L, HL, HR):
    # stage x[b, :, s-HL : e+HR] into xb[:, 0 : e-s+HL+HR], zero outside [0, L)
    lo = s - HL
    hi = e + HR
    clo = max(lo, 0)
    chi = min(hi, L)
    if clo > lo:
        nisa.memset(dst=xb[:, 0:clo - lo], value=0.0)
    if chi < hi:
        nisa.memset(dst=xb[:, chi - lo:hi - lo], value=0.0)
    nisa.dma_copy(dst=xb[:, clo - lo:chi - lo], src=inp[b, :, clo:chi])


@nki.jit
def _conv1d_kernel(inp, w, pad, T, G):
    B, CI, L = inp.shape
    CO, CIW, K = w.shape
    out = nl.ndarray((B, CO, L), dtype=inp.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    Lh = L // nprog
    c0 = pid * Lh
    HL = pad
    HR = K - 1 - pad
    XW = T + K - 1

    split = False
    if Lh % T == 0:
        if Lh >= 2 * T:
            split = True
    ntc = (Lh + T - 1) // T
    if split:
        ntc = Lh // T + 1
    NT = B * ntc

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
    ob0 = nl.ndarray((CO, T), dtype=inp.dtype, buffer=nl.sbuf)
    ob1 = nl.ndarray((CO, T), dtype=inp.dtype, buffer=nl.sbuf)
    ob2 = nl.ndarray((CO, T), dtype=inp.dtype, buffer=nl.sbuf)
    ps0 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)
    ps1 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)
    ps2 = nl.ndarray((CO, G), dtype=nl.float32, buffer=nl.psum)

    # ---------- prologue: prefetch first tiles ----------
    npre = min(_NX - 1, NT)
    for t in range(npre):
        bq = t // ntc
        tq = t - bq * ntc
        sq = c0 + _tile_lo(tq, T, split)
        eq = c0 + _tile_hi(tq, Lh, T, split)
        _load_tile(inp, _pick4(t % _NX, xb0, xb1, xb2, xb3), bq, sq, eq, L, HL, HR)

    # ---------- main pipeline ----------
    gc = 0
    for t in range(NT):
        tn = t + _NX - 1
        if tn < NT:
            bq = tn // ntc
            tq = tn - bq * ntc
            sq = c0 + _tile_lo(tq, T, split)
            eq = c0 + _tile_hi(tq, Lh, T, split)
            _load_tile(inp, _pick4(tn % _NX, xb0, xb1, xb2, xb3), bq, sq, eq, L, HL, HR)

        bc = t // ntc
        tcur = t - bc * ntc
        s = c0 + _tile_lo(tcur, T, split)
        e = c0 + _tile_hi(tcur, Lh, T, split)
        W = e - s
        xb = _pick4(t % _NX, xb0, xb1, xb2, xb3)
        ob = _pick3(t % _NO, ob0, ob1, ob2)
        ng = (W + G - 1) // G
        for gi in range(ng):
            g0 = gi * G
            gw = min(G, W - g0)
            ps = _pick3(gc % _NPS, ps0, ps1, ps2)
            nb = (gw + _NMM - 1) // _NMM
            for k in range(K):
                for bi in range(nb):
                    m0 = bi * _NMM
                    mw = min(_NMM, gw - m0)
                    cc = g0 + m0 + k
                    nisa.nc_matmul(dst=ps[:, m0:m0 + mw],
                                   stationary=wT[:, k * CO:(k + 1) * CO],
                                   moving=xb[:, cc:cc + mw],
                                   accumulate=(k > 0))
            if gc % 2 == 0:
                nisa.tensor_copy(dst=ob[:, g0:g0 + gw], src=ps[:, 0:gw],
                                 engine=nisa.engine.vector)
            else:
                nisa.activation(dst=ob[:, g0:g0 + gw], op=nl.copy, data=ps[:, 0:gw])
            gc = gc + 1
        nisa.dma_copy(dst=out[bc, :, s:e], src=ob[:, 0:W])
    return out


def _split_flag(Lh, T):
    return (Lh % T == 0) and (Lh >= 2 * T)


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

    out = _conv1d_kernel(input, weight, padding, _TILE, _GROUP)

    Lh = L // 2
    _LAST_CONFIG = {
        "tile_cols": _TILE,
        "psum_group_cols": _GROUP,
        "matmul_cols": _NMM,
        "x_buffers": _NX,
        "out_buffers": _NO,
        "psum_buffers": _NPS,
        "lnc_programs": 2,
        "cols_per_program": Lh,
        "half_edge_tiles": bool(_split_flag(Lh, _TILE)),
        "kernel_size": int(K),
        "padding": padding,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
