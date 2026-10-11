"""
Implicit-GEMM grouped 1-D convolution (fp16) for Trn2 / NKI 0.7.0b1.

Mapping (groups == 1, stride == 1, K == 3, pad == 1, out_L == L):
  out[b, oc, p] = sum_k sum_ic W[oc, ic, k] * x[b, ic, p + k - 1]
  * contraction (ic) on the 128 SBUF partitions, one nc_matmul per tap,
    all three taps accumulated in one fp32 PSUM accumulator, cast to fp16
    once on PSUM->SBUF evacuation, then DMA'd to HBM (no HBM intermediates).
  * alignment-friendly block layout: every TensorE moving operand starts at
    an even fp16 column (4-byte aligned).  Per PSUM bank (block m of a tile
    starting at s, c0 = s - 1 + 510*m):
        tap 1: psum[0 : ne+2]   = W1^T x[c0-1 ...]      (overwrite)
        tap 0: psum[1 : ne+2]  += W0^T x[c0-1 ...]
        tap 2: psum[1 : ne+2]  += W2^T x[c0+1 ...]
    psum column 1+j holds output c0+j; columns [2, 2+ne) (outputs
    s+510m ...) are evacuated to staging column 510m (even -> aligned).
    Column 0 is a discarded partial, column 1 duplicates the previous
    block's last output and is not stored.
  * input tiles staged with a 64-byte-aligned halo (zero-filled only outside
    [0, L)); 3 input / 3 output staging buffers of 16384 columns, 3 PSUM
    groups of 2 banks, graded edge tiles; LNC=2 splits the length axis.
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
_TILE = 16384     # columns per (full) staged DMA tile
_NV = 510         # outputs evacuated per PSUM bank block (even, <= 510)
_BANK = 512       # fp32 columns per PSUM bank
_NX = 3           # input staging buffers (prefetch distance _NX - 1)
_NO = 3           # output staging buffers
_NPS = 3          # PSUM group buffers (2 banks each)
_ALIGN = 32       # halo granularity in elements (64 bytes for fp16)

_LAST_CONFIG = {}


def _pick3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _is_graded(Lh, T):
    g = False
    if T % 256 == 0:
        if Lh % T == 0:
            if Lh >= 2 * T:
                g = True
    return g


def _num_tiles(Lh, T, graded):
    n = (Lh + T - 1) // T
    if graded:
        n = Lh // T + 6
    return n


def _tile_lo(i, n, Lh, T, graded):
    r = i * T
    if graded:
        e = T // 8
        r = T + (i - 4) * T
        if i == 0:
            r = 0
        elif i == 1:
            r = e
        elif i == 2:
            r = 2 * e
        elif i == 3:
            r = 4 * e
        elif i == n - 4:
            r = Lh - T
        elif i == n - 3:
            r = Lh - 4 * e
        elif i == n - 2:
            r = Lh - 2 * e
        elif i == n - 1:
            r = Lh - e
    return r


def _tile_hi(i, n, Lh, T, graded):
    r = min((i + 1) * T, Lh)
    if graded:
        r = Lh
        if i < n - 1:
            r = _tile_lo(i + 1, n, Lh, T, graded)
    return r


def _load_tile(inp, xb, b, s, e, L, HLX, HRX, CI):
    # stage x[b, :, s-HLX : e+HRX] into xb[:, 0 : e-s+HLX+HRX], zero outside [0, L)
    lo = s - HLX
    hi = e + HRX
    clo = max(lo, 0)
    chi = min(hi, L)
    if clo > lo:
        nisa.memset(dst=xb[:, 0:clo - lo], value=0.0)
    if chi < hi:
        nisa.memset(dst=xb[:, chi - lo:hi - lo], value=0.0)
    nisa.dma_copy(dst=xb[0:CI, clo - lo:chi - lo], src=inp[b, 0:CI, clo:chi])


@nki.jit
def _conv1d_kernel(inp, w, T, NV, HLX, HRX):
    B, CI, L = inp.shape
    CO, CIW, K = w.shape
    out = nl.ndarray((B, CO, L), dtype=inp.dtype, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    Lh = L // nprog
    c0 = pid * Lh
    XW = HLX + T + HRX
    G = 2 * _BANK

    graded = _is_graded(Lh, T)
    ntc = _num_tiles(Lh, T, graded)
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
        sq = c0 + _tile_lo(tq, ntc, Lh, T, graded)
        eq = c0 + _tile_hi(tq, ntc, Lh, T, graded)
        _load_tile(inp, _pick3(t % _NX, xb0, xb1, xb2), bq, sq, eq, L,
                   HLX, HRX, CI)

    # ---------- main pipeline ----------
    gc = 0
    blk = 0
    for t in range(NT):
        tn = t + _NX - 1
        if tn < NT:
            bq = tn // ntc
            tq = tn - bq * ntc
            sq = c0 + _tile_lo(tq, ntc, Lh, T, graded)
            eq = c0 + _tile_hi(tq, ntc, Lh, T, graded)
            _load_tile(inp, _pick3(tn % _NX, xb0, xb1, xb2), bq, sq, eq, L,
                       HLX, HRX, CI)

        bc = t // ntc
        tcur = t - bc * ntc
        s = c0 + _tile_lo(tcur, ntc, Lh, T, graded)
        e = c0 + _tile_hi(tcur, ntc, Lh, T, graded)
        W = e - s
        xb = _pick3(t % _NX, xb0, xb1, xb2)
        ob = _pick3(t % _NO, ob0, ob1, ob2)
        nbk = (W + NV - 1) // NV
        ngr = (nbk + 1) // 2
        for gi in range(ngr):
            ps = _pick3(gc % _NPS, ps0, ps1, ps2)
            mb = 2 * gi
            nbg = min(2, nbk - mb)
            # tap 1 (centre): overwrite psum[pb : pb+ne+2] with W1^T x[c0-1 ...]
            for bi in range(nbg):
                m = mb + bi
                ne = min(NV, W - NV * m)
                pb = bi * _BANK
                a = NV * m + HLX - 2
                nisa.nc_matmul(dst=ps[:, pb:pb + ne + 2],
                               stationary=wT[:, CO:2 * CO],
                               moving=xb[:, a:a + ne + 2],
                               accumulate=False)
            # tap 0: psum[pb+1 : pb+ne+2] += W0^T x[c0-1 ...]
            for bi in range(nbg):
                m = mb + bi
                ne = min(NV, W - NV * m)
                pb = bi * _BANK
                a = NV * m + HLX - 2
                nisa.nc_matmul(dst=ps[:, pb + 1:pb + ne + 2],
                               stationary=wT[:, 0:CO],
                               moving=xb[:, a:a + ne + 1],
                               accumulate=True)
            # tap 2: psum[pb+1 : pb+ne+2] += W2^T x[c0+1 ...]
            for bi in range(nbg):
                m = mb + bi
                ne = min(NV, W - NV * m)
                pb = bi * _BANK
                a = NV * m + HLX - 2
                nisa.nc_matmul(dst=ps[:, pb + 1:pb + ne + 2],
                               stationary=wT[:, 2 * CO:3 * CO],
                               moving=xb[:, a + 2:a + ne + 3],
                               accumulate=True)
            # evacuate outputs s+NV*m ... s+NV*m+ne-1 (psum cols pb+2 ...)
            for bi in range(nbg):
                m = mb + bi
                ne = min(NV, W - NV * m)
                pb = bi * _BANK
                o0 = NV * m
                if blk % 3 == 2:
                    nisa.activation(dst=ob[:, o0:o0 + ne], op=nl.copy,
                                    data=ps[:, pb + 2:pb + 2 + ne])
                else:
                    nisa.tensor_copy(dst=ob[:, o0:o0 + ne],
                                     src=ps[:, pb + 2:pb + 2 + ne],
                                     engine=nisa.engine.vector)
                blk = blk + 1
            gc = gc + 1
        nisa.dma_copy(dst=out[bc, 0:CO, s:e], src=ob[0:CO, 0:W])
    return out


def _graded_host(Lh, T):
    return (T % 256 == 0) and (Lh % T == 0) and (Lh >= 2 * T)


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    global _LAST_CONFIG
    B, CI, L = input.shape
    CO, CIg, K = weight.shape
    stride = int(stride)
    padding = int(padding)
    groups = int(groups)
    if stride != 1 or groups != 1 or CIg != CI or K != 3 or padding != 1:
        raise NotImplementedError("unsupported conv1d configuration")
    if CI > 128 or CO > 128 or L % 2 != 0:
        raise NotImplementedError("unsupported conv1d shape")

    HL = padding
    HR = K - 1 - padding
    HLX = ((HL + _ALIGN - 1) // _ALIGN) * _ALIGN
    HRX = ((HR + _ALIGN - 1) // _ALIGN) * _ALIGN
    if HLX % 2 != 0 or HLX < 2 or HRX < 1:
        raise NotImplementedError("unsupported halo configuration")

    out = _conv1d_kernel(input, weight, _TILE, _NV, HLX, HRX)

    Lh = L // 2
    _LAST_CONFIG = {
        "tile_cols": _TILE,
        "block_outputs_per_bank": _NV,
        "psum_group_banks": 2,
        "x_buffers": _NX,
        "out_buffers": _NO,
        "psum_buffers": _NPS,
        "lnc_programs": 2,
        "cols_per_program": Lh,
        "graded_edge_tiles": bool(_graded_host(Lh, _TILE)),
        "edge_grading": "T/8,T/8,T/4,T/2,...,T/2,T/4,T/8,T/8",
        "halo_left": HLX,
        "halo_right": HRX,
        "tap_order": "1,0,2",
        "moving_alignment": "even fp16 column (4-byte)",
        "evac_split_vector_scalar": "2:1",
        "kernel_size": int(K),
        "padding": padding,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
