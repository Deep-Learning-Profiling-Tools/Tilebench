import os

import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Hybrid Stream-K GEMM on one logical NeuronCore (P = 2 programs, one per physical core).
#
# Logical tiles 128 x 128, K-iteration = 128-deep slab, I = K/128.  T = (M/128)*(N/128) is
# even, so the Stream-K region is S = P = 2 tiles and the even split of its S*I iterations
# gives program p exactly the I iterations of region tile p (row block 0, first column tile
# of program p's column half).
#
# Stage 1: Cacc (fp32 M x N, second kernel output) zero-filled in-kernel, each region before
#          its first write.
# Stage 2: Stream-K tile: fp32 PSUM accumulation over its K range, fp32 add into the
#          zero-filled Cacc (read-add-write, single contributor), final cast from Cacc.
# Stage 3: data-parallel tiles: full-K fp32 PSUM accumulation, ordinary stores into Cacc.
# Stage 4: one fp32 -> fp16 rounding of each finished tile into the output.
#
# Program p owns columns [p*N/P, (p+1)*N/P).  The first column tile of row block 0 of every
# 1024-row super block is handled in a small unrolled prologue (super block 0's one is the
# Stream-K tile); everything else is handled by a uniform per-super-block body that runs in
# a rolled device loop whose only live-ins are the kernel's HBM inputs/outputs.  A^T is
# produced on chip with TensorE transposes; B streams through in 512-column chunks.

_TM = 128
_TN = 128
_TK = 128
_SB = 1024
_CH = 512
_LAST_CONFIG = {}


def _pick2(i, x0, x1):
    if i == 0:
        return x0
    return x1


def _pick3(i, x0, x1, x2):
    if i == 0:
        return x0
    if i == 1:
        return x1
    return x2


def _pick8(i, x0, x1, x2, x3, x4, x5, x6, x7):
    if i == 0:
        return x0
    if i == 1:
        return x1
    if i == 2:
        return x2
    if i == 3:
        return x3
    if i == 4:
        return x4
    if i == 5:
        return x5
    if i == 6:
        return x6
    return x7


def _win(t, t3, W, r, col0, w, j, dyn, SB):
    # [128, w] window of a row-major (rows, W) HBM tensor at rows j*SB + r .., cols col0 ..
    if dyn:
        v = t3.ap(pattern=[[W, 128], [1, w]], offset=r * W + col0,
                  scalar_offset=j, indirect_dim=0)
    else:
        r0 = j * SB + r
        v = t[r0:r0 + 128, col0:col0 + w]
    return v


def _load_bchunk(b, bc, c0, cw, KT):
    # B[:, c0:c0+cw] -> bc[k, kt*512 + c]
    for kt in range(KT):
        nisa.dma_copy(dst=bc[:, kt * 512:kt * 512 + cw],
                      src=b[kt * 128:(kt + 1) * 128, c0:c0 + cw])


def _transpose(ar, aT, tp0, tp1, KT):
    # aT[k, kt*128 + m] = ar[m, kt*128 + k]
    for h in range(KT // 8):
        tp = _pick2(h % 2, tp0, tp1)
        for q in range(8):
            kt = h * 8 + q
            nisa.nc_transpose(dst=tp[:, q * 128:(q + 1) * 128],
                              data=ar[:, kt * 128:(kt + 1) * 128],
                              engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=aT[:, h * 1024:(h + 1) * 1024], src=tp)


def _strip_tiles(a, b, cacc, out, cb, K, SB, NSB):
    # tiles (rows jj*SB .. +128, cols cb .. cb+128); jj == 0 is this program's Stream-K tile
    KT = K // 128
    adt = a.dtype
    f32 = nl.float32
    bs = nl.ndarray((128, KT * 128), dtype=adt, buffer=nl.sbuf)
    par = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    paT = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    zs = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    pf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    ph = nl.ndarray((128, 128), dtype=adt, buffer=nl.sbuf)
    old = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    skr = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    fin = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    skh = nl.ndarray((128, 128), dtype=adt, buffer=nl.sbuf)
    ptp = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    ppm = nl.ndarray((128, 128), dtype=f32, buffer=nl.psum)

    nisa.memset(dst=zs, value=0.0)
    for kt in range(KT):
        nisa.dma_copy(dst=bs[:, kt * 128:(kt + 1) * 128],
                      src=b[kt * 128:(kt + 1) * 128, cb:cb + 128])
    for jj in range(NSB):
        r = jj * SB
        nisa.dma_copy(dst=par, src=a[r:r + 128, :])
        _transpose(par, paT, ptp, ptp, KT)
        for kt in range(KT):
            nisa.nc_matmul(dst=ppm, stationary=paT[:, kt * 128:(kt + 1) * 128],
                           moving=bs[:, kt * 128:(kt + 1) * 128],
                           accumulate=(kt > 0))
        # stage 1 for this tile: zero-fill of Cacc
        nisa.dma_copy(dst=cacc[r:r + 128, cb:cb + 128], src=zs)
        nisa.tensor_copy(dst=pf, src=ppm, engine=nisa.engine.vector)
        if jj == 0:
            # Stream-K segment: fp32 add of the partial into the zero-filled Cacc,
            # then the final cast of the tile from Cacc
            nisa.dma_copy(dst=old, src=cacc[r:r + 128, cb:cb + 128])
            nisa.tensor_tensor(dst=skr, data1=old, data2=pf, op=nl.add)
            nisa.dma_copy(dst=cacc[r:r + 128, cb:cb + 128], src=skr)
            nisa.dma_copy(dst=fin, src=cacc[r:r + 128, cb:cb + 128])
            nisa.tensor_copy(dst=skh, src=fin, engine=nisa.engine.scalar)
            nisa.dma_copy(dst=out[r:r + 128, cb:cb + 128], src=skh)
        else:
            # data-parallel tile: ordinary store, one rounding into the output
            nisa.tensor_copy(dst=ph, src=pf, engine=nisa.engine.scalar)
            nisa.dma_copy(dst=cacc[r:r + 128, cb:cb + 128], src=pf)
            nisa.dma_copy(dst=out[r:r + 128, cb:cb + 128], src=ph)


def _super_block(a, b, a3, cacc, c3, out, o3, j, dyn, cb, Nc, K, N, SB):
    # data-parallel tiles of rows j*SB .. j*SB+SB, columns [cb, cb+Nc), except the strip
    # tile (row block 0, cols cb .. cb+128) handled in _strip_tiles
    KT = K // 128
    adt = a.dtype
    f32 = nl.float32
    NRB = SB // 128
    NCH = (Nc + 511) // 512

    aT0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT2 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT3 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT4 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT5 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT6 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    aT7 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    ar0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    ar1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    bc0 = nl.ndarray((128, KT * 512), dtype=adt, buffer=nl.sbuf)
    bc1 = nl.ndarray((128, KT * 512), dtype=adt, buffer=nl.sbuf)
    zt = nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf)
    sf0 = nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf)
    sf1 = nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf)
    sf2 = nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf)
    sh0 = nl.ndarray((128, 512), dtype=adt, buffer=nl.sbuf)
    sh1 = nl.ndarray((128, 512), dtype=adt, buffer=nl.sbuf)
    sh2 = nl.ndarray((128, 512), dtype=adt, buffer=nl.sbuf)
    tp0 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    tp1 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    pm0 = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
    pm1 = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
    pm2 = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)

    nisa.memset(dst=zt, value=0.0)
    # A rows of row blocks 0 and 1, A^T of row block 0, then the first B chunk
    nisa.dma_copy(dst=ar0, src=_win(a, a3, K, 0, 0, K, j, dyn, SB))
    if NRB > 1:
        nisa.dma_copy(dst=ar1, src=_win(a, a3, K, 128, 0, K, j, dyn, SB))
    _transpose(ar0, aT0, tp0, tp1, KT)
    _load_bchunk(b, bc0, cb, min(512, Nc), KT)

    for c in range(NCH):
        c0 = cb + c * 512
        cw = min(512, Nc - c * 512)
        bc = _pick2(c % 2, bc0, bc1)
        pf_rb = 0
        if c == 0:
            pf_rb = NRB // 2
        for rb in range(NRB):
            if rb == pf_rb:
                if c + 1 < NCH:
                    _load_bchunk(b, _pick2((c + 1) % 2, bc0, bc1), cb + (c + 1) * 512,
                                 min(512, Nc - (c + 1) * 512), KT)
            if c == 0:
                if rb + 1 < NRB:
                    _transpose(_pick2((rb + 1) % 2, ar0, ar1),
                               _pick8(rb + 1, aT0, aT1, aT2, aT3, aT4, aT5, aT6, aT7),
                               tp0, tp1, KT)
                if rb + 2 < NRB:
                    nisa.dma_copy(dst=_pick2(rb % 2, ar0, ar1),
                                  src=_win(a, a3, K, (rb + 2) * 128, 0, K, j, dyn, SB))
            aT = _pick8(rb, aT0, aT1, aT2, aT3, aT4, aT5, aT6, aT7)
            moff = 0
            if c == 0:
                if rb == 0:
                    moff = 128
            w = cw - moff
            idx = (c * NRB + rb) % 3
            pm = _pick3(idx, pm0, pm1, pm2)
            sf = _pick3(idx, sf0, sf1, sf2)
            sh = _pick3(idx, sh0, sh1, sh2)
            # stage 1 for this region: zero-fill of Cacc
            nisa.dma_copy(dst=_win(cacc, c3, N, rb * 128, c0 + moff, w, j, dyn, SB),
                          src=zt[:, 0:w])
            # local fp32 accumulator over the full K range
            for kt in range(KT):
                nisa.nc_matmul(dst=pm[:, 0:w], stationary=aT[:, kt * 128:(kt + 1) * 128],
                               moving=bc[:, kt * 512 + moff:kt * 512 + cw],
                               accumulate=(kt > 0))
            nisa.tensor_copy(dst=sf[:, 0:w], src=pm[:, 0:w], engine=nisa.engine.vector)
            nisa.tensor_copy(dst=sh[:, 0:w], src=sf[:, 0:w], engine=nisa.engine.scalar)
            # data-parallel tiles: ordinary store to Cacc, one rounding into the output
            nisa.dma_copy(dst=_win(cacc, c3, N, rb * 128, c0 + moff, w, j, dyn, SB),
                          src=sf[:, 0:w])
            nisa.dma_copy(dst=_win(out, o3, N, rb * 128, c0 + moff, w, j, dyn, SB),
                          src=sh[:, 0:w])


@nki.jit
def _streamk_kernel(a, b, SB):
    M, K = a.shape
    N = b.shape[1]
    adt = a.dtype
    out = nl.ndarray((M, N), dtype=adt, buffer=nl.shared_hbm)
    cacc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    npg = nl.num_programs()
    pid = nl.program_id(0)
    Nc = N // npg
    cb = pid * Nc
    NSB = M // SB

    a3 = a.reshape((NSB, SB, K))
    c3 = cacc.reshape((NSB, SB, N))
    o3 = out.reshape((NSB, SB, N))

    # Stream-K tile + strip tiles (unrolled)
    _strip_tiles(a, b, cacc, out, cb, K, SB, NSB)

    # remaining data-parallel tiles
    if NSB == 1:
        _super_block(a, b, a3, cacc, c3, out, o3, 0, False, cb, Nc, K, N, SB)
    else:
        def body(j):
            _super_block(a, b, a3, cacc, c3, out, o3, j, True, cb, Nc, K, N, SB)

        nl.fori_loop(0, NSB, body)

    return out, cacc


def _num_programs():
    v = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    try:
        p = int(v)
    except Exception:
        p = 2
    if p >= 2:
        return 2
    return 1


def _make_config(M, N, K, P):
    if M % _SB != 0 or K % 1024 != 0 or K > 4096 or N % (128 * P) != 0 or N // P < 512:
        raise ValueError("unsupported shape for this kernel")
    tiles_m = (M + _TM - 1) // _TM
    tiles_n = (N + _TN - 1) // _TN
    T = tiles_m * tiles_n
    I = (K + _TK - 1) // _TK
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
    ok = (S == P)
    for p in range(P):
        if ranges[p] != [p * I, (p + 1) * I]:
            ok = False
    if not ok:
        raise ValueError("stream-K partition not supported by this kernel layout")
    Nc = N // P
    NSB = M // _SB
    return {
        "TM": _TM, "TN": _TN, "TK": _TK, "P": P, "T": T, "I": I,
        "streamk_tiles": S, "dp_tiles": T - S, "streamk_ranges": ranges,
        "column_split": "program p owns columns [p*N/P, (p+1)*N/P)",
        "cols_per_program": Nc, "column_chunk": _CH,
        "chunks_per_program": (Nc + _CH - 1) // _CH,
        "super_block_rows": _SB, "super_blocks": NSB,
        "rolled_loop_trips": NSB if NSB > 1 else 0,
        "streamk_tile_of_program_p": "rows 0..127, first 128 columns of its half",
        "strip_tiles": "row block 0 of each super block x first column tile (unrolled)",
        "aT": "on-chip TensorE transposes per super block",
        "zero_fill": "in-kernel, every Cacc region before its first write",
        "final_cast": "one fp32->fp16 rounding per finished tile",
        "b_buffers": 2, "a_stage_buffers": 2, "psum_banks": 7,
    }


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    M, K = a.shape
    N = b.shape[1]
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("this implementation handles fp16/bf16 inputs")
    P = _num_programs()
    cfg = _make_config(int(M), int(N), int(K), P)
    a_c = a.contiguous()
    b_c = b.contiguous()
    res = _streamk_kernel[P](a_c, b_c, _SB)
    out = res[0]
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
