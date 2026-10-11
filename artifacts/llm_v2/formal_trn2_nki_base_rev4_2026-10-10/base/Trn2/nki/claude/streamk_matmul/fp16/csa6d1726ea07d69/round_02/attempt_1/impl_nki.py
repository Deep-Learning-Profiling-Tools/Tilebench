import os

import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Hybrid Stream-K GEMM on one logical NeuronCore (P = 2 physical cores = "SMs").
# Logical tile: TM=256 rows x TN=1024 cols, K-iteration = 128-deep slab (I = K/128).
# Tile id t -> program p = t % P, local l = t // P; within program p's row-block half the
# local order is column-major (grouped order).  Tiles 0..S-1 form the Stream-K region; the
# host checks that each program's contiguous K-iteration range is exactly one whole tile
# (always true here since T is even), whose fp32 partial is DMA-accumulated (dma_compute add)
# into the zero-filled fp32 Cacc.  Remaining tiles are data-parallel: full-K in PSUM (fp32)
# and plain stores into Cacc.  Final stage: Cacc (fp32) is read back and rounded once to fp16.
# A layout transform (A^T tiles, fp16) is rebuilt by device work on every call.

_TM = 256
_TN = 1024
_TK = 128
_LAST_CONFIG = {}


def _load_b(b, bb, n0, w, KT, TN):
    for kt in range(KT):
        nisa.dma_copy(dst=bb[:, kt * TN:kt * TN + w],
                      src=b[kt * 128:(kt + 1) * 128, n0:n0 + w])


def _subtile(at_src, cacc_v, out_v, bcur, w, kt_lo, kt_hi, TN, atomic, aT, ps, res, o16):
    # stationary A^T slabs for 128 output rows
    nisa.dma_copy(dst=aT, src=at_src)
    nq = (w + 511) // 512
    for kt in range(kt_lo, kt_hi):
        for q in range(nq):
            c0 = q * 512
            cw = min(512, w - c0)
            nisa.nc_matmul(dst=ps[:, c0:c0 + cw],
                           stationary=aT[:, kt * 128:(kt + 1) * 128],
                           moving=bcur[:, kt * TN + c0:kt * TN + c0 + cw],
                           accumulate=(kt > kt_lo))
    for q in range(nq):
        c0 = q * 512
        cw = min(512, w - c0)
        nisa.tensor_copy(dst=res[:, c0:c0 + cw], src=ps[:, c0:c0 + cw])
    if atomic:
        # Stream-K segment: fp32 add of the partial into the zero-filled Cacc (DMA compute)
        nisa.dma_compute(dst=cacc_v, srcs=[cacc_v, res[:, 0:w]], reduce_op=nl.add)
    else:
        # data-parallel tile: ordinary store
        nisa.dma_copy(dst=cacc_v, src=res[:, 0:w])
    # final cast stage for this (completed) tile: Cacc fp32 -> fp16 output, one rounding
    nisa.dma_copy(dst=res[:, 0:w], src=cacc_v)
    nisa.tensor_copy(dst=o16[:, 0:w], src=res[:, 0:w])
    nisa.dma_copy(dst=out_v, src=o16[:, 0:w])


@nki.jit
def _streamk_kernel(a, b, TN):
    M, K = a.shape
    N = b.shape[1]
    KT = K // 128
    MT = M // 128
    adt = a.dtype

    out = nl.ndarray((M, N), dtype=adt, buffer=nl.shared_hbm)
    cacc = nl.ndarray((MT, 128, N), dtype=nl.float32, buffer=nl.shared_hbm)
    at = nl.ndarray((MT, 128, K), dtype=adt, buffer=nl.shared_hbm)
    out3 = out.reshape((MT, 128, N))

    npg = nl.num_programs()
    pid = nl.program_id(0)
    RT = MT // npg
    t_lo = pid * RT
    NB = (N + TN - 1) // TN

    bb0 = nl.ndarray((128, KT * TN), dtype=adt, buffer=nl.sbuf)
    bb1 = nl.ndarray((128, KT * TN), dtype=adt, buffer=nl.sbuf)
    aX = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aY = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    tr0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    tr1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    resX = nl.ndarray((128, TN), dtype=nl.float32, buffer=nl.sbuf)
    resY = nl.ndarray((128, TN), dtype=nl.float32, buffer=nl.sbuf)
    oX = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    oY = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    zt = nl.ndarray((128, TN), dtype=nl.float32, buffer=nl.sbuf)
    tp0 = nl.ndarray((128, 512), dtype=adt, buffer=nl.psum)
    tp1 = nl.ndarray((128, 512), dtype=adt, buffer=nl.psum)
    psX = nl.ndarray((128, TN), dtype=nl.float32, buffer=nl.psum)
    psY = nl.ndarray((128, TN), dtype=nl.float32, buffer=nl.psum)

    # ---- operand layout transform (per call): at[t][p, kt*128+m] = A[t*128+m, kt*128+p]
    for r in range(RT):
        t = t_lo + r
        if r % 2 == 0:
            arow = aX
            atr = tr0
        else:
            arow = aY
            atr = tr1
        nisa.dma_copy(dst=arow, src=a[t * 128:(t + 1) * 128, :])
        for g in range(KT // 4):
            if g % 2 == 0:
                tp = tp0
            else:
                tp = tp1
            for jj in range(4):
                kt = g * 4 + jj
                nisa.nc_transpose(dst=tp[:, jj * 128:(jj + 1) * 128],
                                  data=arow[:, kt * 128:(kt + 1) * 128],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=atr[:, g * 512:(g + 1) * 512], src=tp)
        nisa.dma_copy(dst=at[t], src=atr)

    # ---- prefetch the first B column blocks
    _load_b(b, bb0, 0, min(TN, N), KT, TN)
    if NB > 1:
        _load_b(b, bb1, TN, min(TN, N - TN), KT, TN)

    # ---- stage 1: zero-fill this program's whole Cacc region (before any tile work)
    nisa.memset(dst=zt, value=0.0)
    for c in range(NB):
        zn0 = c * TN
        zw = min(TN, N - zn0)
        for r in range(RT):
            nisa.dma_copy(dst=cacc[t_lo + r, :, zn0:zn0 + zw], src=zt[:, 0:zw])

    # ---- stages 2+3 (one persistent launch) + per-tile final cast
    for c in range(NB):
        n0 = c * TN
        w = min(TN, N - n0)
        if c % 2 == 0:
            bcur = bb0
            bnext = bb1
        else:
            bcur = bb1
            bnext = bb0
        if c >= 1:
            if c + 1 < NB:
                n1 = n0 + TN
                _load_b(b, bnext, n1, min(TN, N - n1), KT, TN)
        t0 = t_lo
        if c == 0:
            # Stream-K tile of this program (global tile id = pid), full K range segment
            _subtile(at[t_lo], cacc[t_lo, :, n0:n0 + w],
                     out[t_lo * 128:(t_lo + 1) * 128, n0:n0 + w],
                     bcur, w, 0, KT, TN, True, aX, psX, resX, oX)
            _subtile(at[t_lo + 1], cacc[t_lo + 1, :, n0:n0 + w],
                     out[(t_lo + 1) * 128:(t_lo + 2) * 128, n0:n0 + w],
                     bcur, w, 0, KT, TN, True, aY, psY, resY, oY)
            t0 = t_lo + 2
        npairs = (t_lo + RT - t0) // 2

        def body(j):
            _subtile(at.ap(pattern=[[K, 128], [1, K]], offset=t0 * 128 * K,
                           scalar_offset=j, indirect_dim=0),
                     cacc.ap(pattern=[[N, 128], [1, w]], offset=t0 * 128 * N + n0,
                             scalar_offset=j, indirect_dim=0),
                     out3.ap(pattern=[[N, 128], [1, w]], offset=t0 * 128 * N + n0,
                             scalar_offset=j, indirect_dim=0),
                     bcur, w, 0, KT, TN, False, aX, psX, resX, oX)
            _subtile(at.ap(pattern=[[K, 128], [1, K]], offset=(t0 + 1) * 128 * K,
                           scalar_offset=j, indirect_dim=0),
                     cacc.ap(pattern=[[N, 128], [1, w]], offset=(t0 + 1) * 128 * N + n0,
                             scalar_offset=j, indirect_dim=0),
                     out3.ap(pattern=[[N, 128], [1, w]], offset=(t0 + 1) * 128 * N + n0,
                             scalar_offset=j, indirect_dim=0),
                     bcur, w, 0, KT, TN, False, aY, psY, resY, oY)

        if npairs > 0:
            nl.fori_loop(0, 2 * npairs, body, step=2)

    return out


def _num_programs():
    v = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    try:
        p = int(v)
    except Exception:
        p = 2
    return 2 if p >= 2 else 1


def _make_config(M, N, K, P):
    TM, TN, TK = _TM, _TN, _TK
    if K % 512 != 0 or M % (TM * P) != 0 or N % 128 != 0:
        raise ValueError("unsupported shape for this kernel")
    tiles_m = (M + TM - 1) // TM
    tiles_n = (N + TN - 1) // TN
    T = tiles_m * tiles_n
    I = (K + TK - 1) // TK
    S = T % P
    if T - S > P:
        S += P
    total = S * I
    base, rem = divmod(total, P)
    ranges = []
    s0 = 0
    for p in range(P):
        L = base + (1 if p < rem else 0)
        ranges.append([s0, s0 + L])
        s0 += L
    # kernel layout: program p's Stream-K range must be exactly global tile p (whole tile)
    for p in range(P):
        if ranges[p] != [p * I, (p + 1) * I]:
            raise ValueError("stream-K partition not supported by this kernel layout")
    return {
        "TM": TM, "TN": TN, "TK": TK, "sub_tile_rows": 128, "psum_cols": 512,
        "P": P, "T": T, "I": I, "streamk_tiles": S, "dp_tiles": T - S,
        "streamk_ranges": ranges, "swizzle": "t%P -> program; column-major within row-half",
        "b_double_buffer": True, "row_split": "halves per program",
    }


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    M, K = a.shape
    N = b.shape[1]
    P = _num_programs()
    cfg = _make_config(int(M), int(N), int(K), P)
    a_c = a.contiguous()
    b_c = b.contiguous()
    out = _streamk_kernel[P](a_c, b_c, cfg["TN"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
