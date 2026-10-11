import os

import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Hybrid Stream-K GEMM on one logical NeuronCore: P = 2 programs (one per physical core).
#   logical tile = 128 rows x 512 cols, K-iteration = 128-deep slab (I = K/128).
#   Stage 1: Cacc (fp32, M x N) zero-filled on the device inside run() (torch.zeros).
#   Stage 2: Stream-K region = tiles 0..S-1 (S = 2 here); program p's contiguous K-iteration
#            range [p*I, (p+1)*I) is one segment = the whole of region tile p; its fp32 PSUM
#            partial is added into the zero-filled Cacc (read-add-write; the only program
#            touching that tile, NKI has no atomic primitive).
#   Stage 3: data-parallel tiles: full-K fp32 accumulation in PSUM, plain store to Cacc.
#   Stage 4: per tile, after its Cacc value is final: Cacc -> SBUF -> one rounding to fp16 -> out.
# Tile-id swizzle: id t -> program t % 2 (row half), local l = t // 2 walks row groups of
# up to 1024 rows, column blocks inside a group, 128-row blocks inside a column block
# (grouped order).  A^T is rebuilt every call with TensorE transposes (kept in SBUF).

_TM = 128
_TN = 512
_TK = 128
_RGMAX = 1024
_LAST_CONFIG = {}


def _hbm_rows(t, t2, N, r, col0, w, j):
    # [128, w] window of a row-major (rows, N) HBM tensor at rows r.., columns col0 (+ j*512)
    if j is None:
        v = t[r:r + 128, col0:col0 + w]
    else:
        v = t2.ap(pattern=[[N, 128], [1, w]], offset=r * N + col0,
                  scalar_offset=j, indirect_dim=0)
    return v


def _load_b_block(b, b2, buf, N, KT, TN, col0, w, j):
    for kt in range(KT):
        nisa.dma_copy(dst=buf[:, kt * TN:kt * TN + w],
                      src=_hbm_rows(b, b2, N, kt * 128, col0, w, j))


def _compute_block(aT, buf, cacc, c2, out, o2, old, N, KT, TN, r0, nrb, col0, w, j,
                   sk, ps0, ps1, rs0, rs1, rs2, rs3, hs0, hs1, hs2, hs3):
    for rb in range(nrb):
        if rb % 2 == 0:
            ps = ps0
        else:
            ps = ps1
        q = rb % 4
        if q == 0:
            rs = rs0
            hs = hs0
        elif q == 1:
            rs = rs1
            hs = hs1
        elif q == 2:
            rs = rs2
            hs = hs2
        else:
            rs = rs3
            hs = hs3
        # local fp32 accumulator over the K-slabs of this tile
        for kt in range(KT):
            s0 = (rb * KT + kt) * 128
            nisa.nc_matmul(dst=ps[:, 0:w], stationary=aT[:, s0:s0 + 128],
                           moving=buf[:, kt * TN:kt * TN + w], accumulate=(kt > 0))
        r = r0 + rb * 128
        is_sk = False
        if sk:
            if rb == 0:
                is_sk = True
        if is_sk:
            # Stream-K segment: fp32 add of the partial into the zero-filled Cacc
            nisa.dma_copy(dst=old[:, 0:w], src=_hbm_rows(cacc, c2, N, r, col0, w, j))
            nisa.tensor_tensor(dst=rs[:, 0:w], data1=old[:, 0:w], data2=ps[:, 0:w],
                               op=nl.add, engine=nisa.engine.vector)
        else:
            # data-parallel tile: ordinary store of the full-K result
            nisa.tensor_copy(dst=rs[:, 0:w], src=ps[:, 0:w], engine=nisa.engine.vector)
        nisa.dma_copy(dst=_hbm_rows(cacc, c2, N, r, col0, w, j), src=rs[:, 0:w])
        # final cast of this (complete) tile: Cacc fp32 -> output dtype, one rounding
        nisa.dma_copy(dst=rs[:, 0:w], src=_hbm_rows(cacc, c2, N, r, col0, w, j))
        nisa.tensor_copy(dst=hs[:, 0:w], src=rs[:, 0:w], engine=nisa.engine.scalar)
        nisa.dma_copy(dst=_hbm_rows(out, o2, N, r, col0, w, j), src=hs[:, 0:w])


@nki.jit
def _streamk_kernel(a, b, cacc, TN, RGMAX):
    M, K = a.shape
    N = b.shape[1]
    KT = K // 128
    adt = a.dtype
    f32 = nl.float32
    out = nl.ndarray((M, N), dtype=adt, buffer=nl.shared_hbm)

    npg = nl.num_programs()
    pid = nl.program_id(0)
    R = M // npg
    rbase = pid * R
    G = (R + RGMAX - 1) // RGMAX
    NF = N // TN
    TW = N - NF * TN
    Q = (NF - 2) // 2
    LAST = NF - 1
    if TW > 0:
        LAST = NF

    b2 = b.reshape((K * N // TN, TN))
    c2 = cacc.reshape((M * N // TN, TN))
    o2 = out.reshape((M * N // TN, TN))

    # persistent SBUF buffers
    aT = nl.ndarray((128, KT * RGMAX), dtype=adt, buffer=nl.sbuf)
    bA = nl.ndarray((128, KT * TN), dtype=adt, buffer=nl.sbuf)
    bB = nl.ndarray((128, KT * TN), dtype=adt, buffer=nl.sbuf)
    ast0 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    ast1 = nl.ndarray((128, K), dtype=adt, buffer=nl.sbuf)
    old = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
    rso0 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
    rso1 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
    rso2 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
    rso3 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
    hso0 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    hso1 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    hso2 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    hso3 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
    # PSUM used only outside the rolled loops
    pso0 = nl.ndarray((128, TN), dtype=f32, buffer=nl.psum)
    pso1 = nl.ndarray((128, TN), dtype=f32, buffer=nl.psum)
    tpo0 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)
    tpo1 = nl.ndarray((128, 1024), dtype=adt, buffer=nl.psum)

    Fb = bA
    Sb = bB
    _load_b_block(b, b2, Fb, N, KT, TN, 0, TN, None)

    for g in range(G):
        r0 = rbase + g * RGMAX
        rg = min(RGMAX, R - g * RGMAX)
        nrb = rg // 128

        # prefetch B column block 1 of this group
        _load_b_block(b, b2, Sb, N, KT, TN, TN, TN, None)

        # A^T for this row group: aT[k, (rb*KT + kt)*128 + m] = A[r0 + rb*128 + m, kt*128 + k]
        for rb in range(nrb):
            if rb % 2 == 0:
                ast = ast0
            else:
                ast = ast1
            ra = r0 + rb * 128
            nisa.dma_copy(dst=ast, src=a[ra:ra + 128, :])
            for h in range(KT // 8):
                if h % 2 == 0:
                    tp = tpo0
                else:
                    tp = tpo1
                for jj in range(8):
                    kt = h * 8 + jj
                    nisa.nc_transpose(dst=tp[:, jj * 128:(jj + 1) * 128],
                                      data=ast[:, kt * 128:(kt + 1) * 128],
                                      engine=nisa.engine.tensor)
                d0 = (rb * KT + h * 8) * 128
                nisa.tensor_copy(dst=aT[:, d0:d0 + 1024], src=tp)

        # column block 0 (holds this program's Stream-K tile in group 0)
        sk0 = False
        if g == 0:
            sk0 = True
        _compute_block(aT, Fb, cacc, c2, out, o2, old, N, KT, TN, r0, nrb, 0, TN, None,
                       sk0, pso0, pso1, rso0, rso1, rso2, rso3, hso0, hso1, hso2, hso3)

        # column blocks (j, j+1), j odd, as a rolled device loop with B double buffering
        if Q > 0:
            def body(j):
                pq0 = nl.ndarray((128, TN), dtype=f32, buffer=nl.psum)
                pq1 = nl.ndarray((128, TN), dtype=f32, buffer=nl.psum)
                rq0 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
                rq1 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
                rq2 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
                rq3 = nl.ndarray((128, TN), dtype=f32, buffer=nl.sbuf)
                hq0 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
                hq1 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
                hq2 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
                hq3 = nl.ndarray((128, TN), dtype=adt, buffer=nl.sbuf)
                _load_b_block(b, b2, Fb, N, KT, TN, TN, TN, j)
                _compute_block(aT, Sb, cacc, c2, out, o2, old, N, KT, TN, r0, nrb, 0, TN, j,
                               False, pq0, pq1, rq0, rq1, rq2, rq3, hq0, hq1, hq2, hq3)
                _load_b_block(b, b2, Sb, N, KT, TN, 2 * TN, TN, j)
                _compute_block(aT, Fb, cacc, c2, out, o2, old, N, KT, TN, r0, nrb, TN, TN, j,
                               False, pq0, pq1, rq0, rq1, rq2, rq3, hq0, hq1, hq2, hq3)

            nl.fori_loop(1, 1 + 2 * Q, body, step=2)

        # remaining full blocks, the tail block, and the prefetch of the next group's block 0
        for c in range(2 * Q + 1, LAST + 1):
            if c % 2 == 0:
                cur = Fb
                nxt = Sb
            else:
                cur = Sb
                nxt = Fb
            pf = 0
            if c + 1 < NF:
                pf = 1
            elif c + 1 == NF:
                if TW > 0:
                    pf = 2
            if pf == 0:
                if g + 1 < G:
                    pf = 3
            if pf == 1:
                _load_b_block(b, b2, nxt, N, KT, TN, (c + 1) * TN, TN, None)
            elif pf == 2:
                _load_b_block(b, b2, nxt, N, KT, TN, NF * TN, TW, None)
            elif pf == 3:
                _load_b_block(b, b2, nxt, N, KT, TN, 0, TN, None)
            w = TN
            if c == NF:
                w = TW
            _compute_block(aT, cur, cacc, c2, out, o2, old, N, KT, TN, r0, nrb, c * TN, w,
                           None, False, pso0, pso1, rso0, rso1, rso2, rso3,
                           hso0, hso1, hso2, hso3)

        # next group's block 0 sits in the buffer not used by this group's last block
        if LAST % 2 == 0:
            Fn = Sb
            Sn = Fb
        else:
            Fn = Fb
            Sn = Sb
        Fb = Fn
        Sb = Sn

    return out


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
    TM, TN, TK = _TM, _TN, _TK
    if K % 1024 != 0 or M % (128 * P) != 0 or N < 2 * TN or N % 128 != 0:
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
    if S != P:
        raise ValueError("stream-K region layout not supported by this kernel")
    for p in range(P):
        if ranges[p] != [p * I, (p + 1) * I]:
            raise ValueError("stream-K partition not supported by this kernel layout")
    R = M // P
    G = (R + _RGMAX - 1) // _RGMAX
    return {
        "TM": TM, "TN": TN, "TK": TK, "P": P, "T": T, "I": I,
        "streamk_tiles": S, "dp_tiles": T - S, "streamk_ranges": ranges,
        "row_group_max": _RGMAX, "row_groups_per_program": G,
        "swizzle": "t%P -> program row half; t//P -> (row group, col block, row block)",
        "b_double_buffer": True, "psum_banks_loop": 2,
        "zero_fill": "torch.zeros fp32 Cacc", "final_cast": "per tile from Cacc",
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
    # stage 1: zero-filled fp32 accumulation target (device fill, every call)
    cacc = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    out = _streamk_kernel[P](a_c, b_c, cacc, _TN, _RGMAX)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
