import os

import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Hybrid Stream-K GEMM on one logical NeuronCore (P = 2 programs, one per physical core).
#
# Logical tiles: 128 x 128 (tm x tn), K-iteration = 128-deep slab, I = K/128.
# T = (M/128)*(N/128) is even, so the Stream-K region is S = P = 2 tiles and the even split
# of its S*I K-iterations gives program p exactly the I iterations of region tile p
# (tile id t -> program t % P; local id t // P -> (column group, super block, row block,
# column tile), so tiles 0 and 1 are row block 0 / first column tile of each program's
# column half).
#
# Stage 1: Cacc (fp32, M x N, second kernel output) is zero-filled in the kernel; each
#          128 x cw region is filled before anything else writes it.
# Stage 2: Stream-K tile of program p: fp32 PSUM accumulation over its K range, then
#          read-add-write into the zero-filled Cacc (single contributor; NKI has no atomic
#          primitive), final cast taken from Cacc.
# Stage 3: data-parallel tiles: full-K fp32 PSUM accumulation, ordinary store into Cacc.
# Stage 4: one fp32 -> fp16 rounding of each finished tile into the output (for DP tiles
#          from the very fp32 value that was stored to Cacc).
#
# A^T (fp16, tiled per 128-row block) is rebuilt in an internal HBM scratch on every call
# with TensorE transposes (each core does half of the row blocks, then a core barrier).
# Columns are split in halves between the programs; each program walks column groups of
# CW columns with the B block resident in SBUF.  Super block 0 (rows 0..1023) is unrolled
# (B streams in while it computes); the remaining super blocks run in a rolled device loop
# that only touches kernel-allocated HBM tensors.

_TM = 128
_TN = 128
_TK = 128
_CW = 1024
_SB = 1024
_LAST_CONFIG = {}


def _pick2(i, x0, x1):
    if i == 0:
        return x0
    return x1


def _pick4(i, x0, x1, x2, x3):
    if i == 0:
        return x0
    if i == 1:
        return x1
    if i == 2:
        return x2
    return x3


def _rows(t, t3, W, r, col0, w, j):
    # [128, w] window of a row-major (rows, W) HBM tensor: rows r.. (+ j*SB when j is the
    # runtime super-block index of a rolled loop), columns col0..col0+w
    if j is None:
        v = t[r:r + 128, col0:col0 + w]
    else:
        v = t3.ap(pattern=[[W, 128], [1, w]], offset=r * W + col0,
                  scalar_offset=j, indirect_dim=0)
    return v


def _load_b(b, bt, c0, cw, KT, CW):
    # B[:, c0:c0+cw] -> bt[k, kt*CW + c]; first 512-column chunk first
    w0 = min(512, cw)
    for kt in range(KT):
        nisa.dma_copy(dst=bt[:, kt * CW:kt * CW + w0],
                      src=b[kt * 128:(kt + 1) * 128, c0:c0 + w0])
    if cw > 512:
        for kt in range(KT):
            nisa.dma_copy(dst=bt[:, kt * CW + 512:kt * CW + cw],
                          src=b[kt * 128:(kt + 1) * 128, c0 + 512:c0 + cw])


def _mm_chunk(aT, bt, ps, KT, CW, off, w):
    # local fp32 accumulator: sum over all K-slabs of one 128 x w chunk
    for kt in range(KT):
        nisa.nc_matmul(dst=ps[:, 0:w], stationary=aT[:, kt * 128:(kt + 1) * 128],
                       moving=bt[:, kt * CW + off:kt * CW + off + w],
                       accumulate=(kt > 0))


def _process_rb(aT, bt, psA, psB, sf, sh, zt, cacc, c3, out, o3, at, at3,
                N, K, KT, CW, r, c0, cw, nr, j, sk, old, skr, fin, skh):
    # stage 1 for these rows / columns: zero-fill of Cacc
    nisa.dma_copy(dst=_rows(cacc, c3, N, r, c0, cw, j), src=zt[:, 0:cw])
    w0 = min(512, cw)
    w1 = cw - w0
    _mm_chunk(aT, bt, psA, KT, CW, 0, w0)
    nisa.tensor_copy(dst=sf[:, 0:w0], src=psA[:, 0:w0], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=sh[:, 0:w0], src=psA[:, 0:w0], engine=nisa.engine.scalar)
    if w1 > 0:
        _mm_chunk(aT, bt, psB, KT, CW, 512, w1)
    if nr >= 0:
        # A^T tile of the row block 4 ahead, into the tile this row block just released
        nisa.dma_copy(dst=aT, src=_rows(at, at3, K, nr, 0, K, j))
    if w1 > 0:
        nisa.tensor_copy(dst=sf[:, 512:cw], src=psB[:, 0:w1], engine=nisa.engine.vector)
        nisa.tensor_copy(dst=sh[:, 512:cw], src=psB[:, 0:w1], engine=nisa.engine.scalar)
    if sk:
        # Stream-K segment: fp32 add of the partial into the zero-filled Cacc,
        # then the final cast of that tile from Cacc
        nisa.dma_copy(dst=old, src=cacc[r:r + 128, c0:c0 + 128])
        nisa.tensor_tensor(dst=skr, data1=old, data2=sf[:, 0:128], op=nl.add)
        nisa.dma_copy(dst=cacc[r:r + 128, c0:c0 + 128], src=skr)
        nisa.dma_copy(dst=fin, src=cacc[r:r + 128, c0:c0 + 128])
        nisa.tensor_copy(dst=skh, src=fin)
        nisa.dma_copy(dst=out[r:r + 128, c0:c0 + 128], src=skh)
        # data-parallel tiles of the same row block
        nisa.dma_copy(dst=cacc[r:r + 128, c0 + 128:c0 + cw], src=sf[:, 128:cw])
        nisa.dma_copy(dst=out[r:r + 128, c0 + 128:c0 + cw], src=sh[:, 128:cw])
    else:
        # data-parallel tiles: ordinary store to Cacc, one rounding into the output
        nisa.dma_copy(dst=_rows(cacc, c3, N, r, c0, cw, j), src=sf[:, 0:cw])
        nisa.dma_copy(dst=_rows(out, o3, N, r, c0, cw, j), src=sh[:, 0:cw])


@nki.jit
def _streamk_kernel(a, b, CW, SB):
    M, K = a.shape
    N = b.shape[1]
    KT = K // 128
    adt = a.dtype
    f32 = nl.float32
    out = nl.ndarray((M, N), dtype=adt, buffer=nl.shared_hbm)
    cacc = nl.ndarray((M, N), dtype=f32, buffer=nl.shared_hbm)
    atsc = nl.ndarray((M, K), dtype=adt, buffer=nl.shared_hbm)

    npg = nl.num_programs()
    pid = nl.program_id(0)
    Nc = N // npg
    cb = pid * Nc
    NG = (Nc + CW - 1) // CW
    NSB = M // SB
    RPS = SB // 128
    NRB = M // 128
    NRBP = NRB // npg

    c3 = cacc.reshape((NSB, SB, N))
    o3 = out.reshape((NSB, SB, N))
    at3 = atsc.reshape((NSB, SB, K))

    # persistent on-chip buffers for the unrolled parts
    bt = nl.ndarray((128, KT * CW), dtype=adt, buffer=nl.sbuf)
    zt = nl.ndarray((128, CW), dtype=f32, buffer=nl.sbuf)
    st0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    st1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    st2 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    st3 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    ssf0 = nl.ndarray((128, CW), dtype=f32, buffer=nl.sbuf)
    ssf1 = nl.ndarray((128, CW), dtype=f32, buffer=nl.sbuf)
    ssh0 = nl.ndarray((128, CW), dtype=adt, buffer=nl.sbuf)
    ssh1 = nl.ndarray((128, CW), dtype=adt, buffer=nl.sbuf)
    old = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    skr = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    fin = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    skh = nl.ndarray((128, 128), dtype=adt, buffer=nl.sbuf)
    tp0 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    tp1 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    pss0 = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
    pss1 = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)

    # B block of column group 0 (independent of everything else)
    _load_b(b, bt, cb, min(CW, Nc), KT, CW)
    nisa.memset(dst=zt, value=0.0)

    # A^T scratch: atsc[rb*128 + k, kt*128 + m] = A[rb*128 + m, kt*128 + k]
    for q in range(NRBP):
        rb = pid * NRBP + q
        ar = _pick2(q % 2, st0, st1)
        ats = _pick2(q % 2, st2, st3)
        nisa.dma_copy(dst=ar, src=a[rb * 128:(rb + 1) * 128, :])
        for h in range(KT // 8):
            tp = _pick2(h % 2, tp0, tp1)
            for jj in range(8):
                kt = h * 8 + jj
                nisa.nc_transpose(dst=tp[:, jj * 128:(jj + 1) * 128],
                                  data=ar[:, kt * 128:(kt + 1) * 128],
                                  engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=ats[:, h * 1024:(h + 1) * 1024], src=tp)
        nisa.dma_copy(dst=atsc[rb * 128:(rb + 1) * 128, :], src=ats)
    if npg > 1:
        nisa.core_barrier(data=atsc, cores=(0, 1))

    for g in range(NG):
        c0 = cb + g * CW
        cw = min(CW, Nc - g * CW)
        # super block 0 (unrolled): A^T tiles of row blocks 0..3, then this group's B block
        for q in range(4):
            nisa.dma_copy(dst=_pick4(q, st0, st1, st2, st3),
                          src=atsc[q * 128:(q + 1) * 128, :])
        if g > 0:
            _load_b(b, bt, c0, cw, KT, CW)
        for rb in range(RPS):
            nr = -1
            if rb + 4 < RPS:
                nr = (rb + 4) * 128
            sk = False
            if g == 0:
                if rb == 0:
                    sk = True
            _process_rb(_pick4(rb % 4, st0, st1, st2, st3), bt, pss0, pss1,
                        _pick2(rb % 2, ssf0, ssf1), _pick2(rb % 2, ssh0, ssh1), zt,
                        cacc, c3, out, o3, atsc, at3, N, K, KT, CW,
                        rb * 128, c0, cw, nr, None, sk, old, skr, fin, skh)

        # super blocks 1..NSB-1: rolled device loop (data-parallel tiles only)
        if NSB > 1:
            def body(j):
                t0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
                t1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
                t2 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
                t3 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
                sf0 = nl.ndarray((128, CW), dtype=f32, buffer=nl.sbuf)
                sf1 = nl.ndarray((128, CW), dtype=f32, buffer=nl.sbuf)
                sh0 = nl.ndarray((128, CW), dtype=adt, buffer=nl.sbuf)
                sh1 = nl.ndarray((128, CW), dtype=adt, buffer=nl.sbuf)
                pA = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                pB = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for q in range(4):
                    nisa.dma_copy(dst=_pick4(q, t0, t1, t2, t3),
                                  src=_rows(atsc, at3, K, q * 128, 0, K, j))
                for rb in range(RPS):
                    nr = -1
                    if rb + 4 < RPS:
                        nr = (rb + 4) * 128
                    _process_rb(_pick4(rb % 4, t0, t1, t2, t3), bt, pA, pB,
                                _pick2(rb % 2, sf0, sf1), _pick2(rb % 2, sh0, sh1), zt,
                                cacc, c3, out, o3, atsc, at3, N, K, KT, CW,
                                rb * 128, c0, cw, nr, j, False, None, None, None, None)

            nl.fori_loop(1, NSB, body)

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
    if M % _SB != 0 or K % 1024 != 0 or N % (128 * P) != 0 or N // P < 512:
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
    return {
        "TM": _TM, "TN": _TN, "TK": _TK, "P": P, "T": T, "I": I,
        "streamk_tiles": S, "dp_tiles": T - S, "streamk_ranges": ranges,
        "column_split": "program p owns columns [p*N/P, (p+1)*N/P)",
        "cols_per_program": Nc, "column_group": _CW,
        "groups_per_program": (Nc + _CW - 1) // _CW,
        "super_block_rows": _SB, "super_blocks": M // _SB,
        "unrolled_super_blocks": 1, "rolled_loop_trips": M // _SB - 1,
        "swizzle": "t%P -> program; t//P -> (col group, super block, row block, col tile)",
        "aT": "per-call TensorE transpose into HBM scratch (fp16), core barrier",
        "zero_fill": "in-kernel, every Cacc region before its first write",
        "final_cast": "one fp32->fp16 rounding per finished tile",
        "psum_banks": 6,
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
    res = _streamk_kernel[P](a_c, b_c, _CW, _SB)
    out = res[0]
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
