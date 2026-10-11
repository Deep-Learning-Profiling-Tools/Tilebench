import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fixed configuration (shape-independent literals; S is a fixed formula of K)
_P = 128          # partition / k-tile size
_MB = 512         # rows of A kept transposed (A^T) resident in SBUF per block (4 m-tiles)
_MT = 4           # m-tiles per block
_NB = 512         # output columns per PSUM accumulation tile
_KG = 4           # k-tiles (of 128) per B DMA group
_NACC = 6         # rotating fp32 PSUM accumulator banks
_NPT = 1          # fp16 PSUM bank for TensorE transposes
_SLOTS_MAX = 160  # max k-tile slots in the A^T ring (1 KiB / partition each)

_LAST_CONFIG = {}


def _pick6(i, t0, t1, t2, t3, t4, t5):
    r = t0
    if i == 1:
        r = t1
    if i == 2:
        r = t2
    if i == 3:
        r = t3
    if i == 4:
        r = t4
    if i == 5:
        r = t5
    return r


def _tr_pair_pe(a, aT, pt, m0, kt0, sl, K):
    # A[m0:m0+512, kt0*128 : kt0*128+256] -> A^T ring slots sl, sl+1 (TensorE)
    at = nl.ndarray((128, 1024), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at.reshape((128, 4, 256)),
                  src=a.ap(pattern=[[K, 128], [128 * K, 4], [1, 256]],
                           offset=m0 * K + kt0 * 128))
    for jj in range(2):
        for mt in range(4):
            nisa.nc_transpose(
                dst=pt[:, jj * 512 + mt * 128: jj * 512 + (mt + 1) * 128],
                data=at[:, mt * 256 + jj * 128: mt * 256 + (jj + 1) * 128],
                engine=nisa.engine.tensor)
    # ring layout: aT[k_in, slot*512 + mt*128 + m_in]
    nisa.tensor_copy(dst=aT[:, sl * 512: sl * 512 + 1024], src=pt,
                     engine=nisa.engine.vector)


def _tr_block_dma(a, aT, r0, kt, col0):
    # A[r0:r0+128, kt*128 : kt*128+128] -> aT[:, col0 : col0+128] via DMA transposes
    for i in range(8):
        nisa.dma_transpose(
            dst=aT[:, col0 + i * 16: col0 + i * 16 + 16],
            src=a[r0 + i * 16: r0 + i * 16 + 16, kt * 128: kt * 128 + 128],
            dge_mode=nisa.dge_mode.hwdge)


@nki.jit
def _mm_kernel(a, b, S):
    M, K = a.shape
    K2, N = b.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rows = M // n_prog
    KT = K // 128
    NG = KT // 4
    n_mblk = rows // 512
    n_nblk = N // 512
    D = S - KT              # k-tiles of the next block prefetched by DMA transposes
    NIT = n_nblk * NG       # (n-block, k-group) iterations per m-block
    NE = 4 * D              # 128x128 blocks prefetched per next m-block

    # resident transposed A ring buffer
    aT = nl.ndarray((128, S * 512), dtype=a.dtype, buffer=nl.sbuf)

    # explicit PSUM: 6 rotating accumulators + 1 transpose bank
    acc0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc3 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc4 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc5 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    pt = nl.ndarray((128, 1024), dtype=a.dtype, buffer=nl.psum)

    # prologue: first k-group of the first M-block (slots 0..3)
    m_first = pid * rows
    _tr_pair_pe(a, aT, pt, m_first, 0, 0, K)
    _tr_pair_pe(a, aT, pt, m_first, 2, 2, K)

    for mb in range(n_mblk):
        m0 = pid * rows + mb * 512
        has_next = 0
        if mb + 1 < n_mblk:
            has_next = 1
        s0 = (mb * KT) % S
        s1 = ((mb + 1) * KT) % S
        for nbi in range(n_nblk):
            n0 = nbi * 512
            q = mb * n_nblk + nbi
            first_nb = 0
            if mb == 0:
                if nbi == 0:
                    first_nb = 1
            last_nb = 0
            if nbi == n_nblk - 1:
                if has_next == 1:
                    last_nb = 1

            for kg in range(NG):
                it = nbi * NG + kg
                # B[kg*512 : kg*512+512, n0:n0+512] as [128 p, 4 j, 512 n]
                bt = nl.ndarray((128, 2048), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=bt.reshape((128, 4, 512)),
                              src=b.ap(pattern=[[N, 128], [128 * N, 4], [1, 512]],
                                       offset=kg * 512 * N + n0))

                # DMA-transpose prefetch of the next M-block's first D k-tiles
                if has_next == 1:
                    t_lo = (it * NE) // NIT
                    t_hi = ((it + 1) * NE) // NIT
                    for t in range(t_lo, t_hi):
                        ktn = t // 4
                        mtn = t % 4
                        sln = (s1 + ktn) % S
                        _tr_block_dma(a, aT, m0 + 512 + mtn * 128, ktn,
                                      sln * 512 + mtn * 128)

                for j in range(4):
                    kt = kg * 4 + j
                    sl = (s0 + kt) % S
                    for mt in range(4):
                        nisa.nc_matmul(
                            dst=_pick6((4 * q + mt) % 6, acc0, acc1, acc2, acc3, acc4, acc5),
                            stationary=aT[:, sl * 512 + mt * 128: sl * 512 + (mt + 1) * 128],
                            moving=bt[:, j * 512:(j + 1) * 512],
                            accumulate=(kt > 0))
                    if first_nb == 1:
                        if kg + 1 < NG:
                            if j == 0:
                                _tr_pair_pe(a, aT, pt, m0, 4 * (kg + 1), 4 * (kg + 1), K)
                            if j == 2:
                                _tr_pair_pe(a, aT, pt, m0, 4 * (kg + 1) + 2,
                                            4 * (kg + 1) + 2, K)
                    if last_nb == 1:
                        # remaining (late) k-tiles of the next block go into ring
                        # slots just consumed by this final N-block (TensorE)
                        if j % 2 == 1:
                            ktn2 = kt - 1 + D
                            if ktn2 + 1 < KT:
                                if kt + 1 < KT:
                                    _tr_pair_pe(a, aT, pt, m0 + 512, ktn2,
                                                (s1 + ktn2) % S, K)

            # single rounding fp32 -> output dtype, then store per m-tile
            for mt in range(4):
                src = _pick6((4 * q + mt) % 6, acc0, acc1, acc2, acc3, acc4, acc5)
                ot = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
                if mt % 2 == 0:
                    nisa.tensor_copy(dst=ot, src=src, engine=nisa.engine.vector)
                else:
                    nisa.tensor_copy(dst=ot, src=src, engine=nisa.engine.scalar)
                nisa.dma_copy(dst=c[m0 + mt * 128: m0 + (mt + 1) * 128, n0: n0 + 512],
                              src=ot)

            if last_nb == 1:
                if D == 0:
                    _tr_pair_pe(a, aT, pt, m0 + 512, KT - 2, (s1 + KT - 2) % S, K)
    return c


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % (2 * _MB) == 0 and N % _NB == 0 and K % (_KG * _P) == 0
    KT = K // _P
    S = min(_SLOTS_MAX, 2 * KT)
    D = S - KT
    cfg = {
        "lnc_split": "M",
        "MB": _MB,
        "m_tiles_per_block": _MT,
        "NB": _NB,
        "k_tile": _P,
        "k_tiles_per_group": _KG,
        "psum_acc_banks_rotating": _NACC,
        "psum_transpose_banks": _NPT,
        "aT_ring_slots": S,
        "dma_transposed_ktiles_per_next_block": D,
        "a_transpose": "block0_tensor_lookahead__next_blocks_hwdge_dma_first_D_rest_tensor",
        "dma_transpose_piece": [16, 128],
        "accum": "fp32_psum",
    }
    out = _mm_kernel(a, b, S)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
