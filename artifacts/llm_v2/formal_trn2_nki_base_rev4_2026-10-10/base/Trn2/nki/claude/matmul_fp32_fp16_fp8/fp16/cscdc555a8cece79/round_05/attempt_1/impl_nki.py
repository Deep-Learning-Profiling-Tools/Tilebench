import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fixed configuration (identical for every configured shape)
_P = 128      # partition / k-tile size
_MB = 512     # rows of A kept transposed (A^T) resident in SBUF per block (4 m-tiles)
_MT = 4       # m-tiles per block
_NB = 512     # output columns per PSUM accumulation tile (one PSUM bank per m-tile)
_KG = 4       # k-tiles (of 128) per B DMA group / transpose group
_NACC = 6     # rotating fp32 PSUM accumulator banks
_NPT = 1      # fp16 PSUM banks for TensorE transposes

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


def _tr_half(a, aT, pt, m0, g, h, K):
    # A[m0:m0+512, g*512 + h*256 : g*512 + h*256 + 256] -> aT k-tiles (4g+2h), (4g+2h+1)
    at = nl.ndarray((128, 1024), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at.reshape((128, 4, 256)),
                  src=a.ap(pattern=[[K, 128], [128 * K, 4], [1, 256]],
                           offset=m0 * K + g * 512 + h * 256))
    for jj in range(2):
        for mt in range(4):
            nisa.nc_transpose(
                dst=pt[:, jj * 512 + mt * 128: jj * 512 + (mt + 1) * 128],
                data=at[:, mt * 256 + jj * 128: mt * 256 + (jj + 1) * 128],
                engine=nisa.engine.tensor)
    # aT layout: aT[k_in_tile, kt*512 + m]
    base = (g * 4 + h * 2) * 512
    nisa.tensor_copy(dst=aT[:, base: base + 1024], src=pt,
                     engine=nisa.engine.vector)


@nki.jit
def _mm_kernel(a, b):
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

    # resident transposed A block
    aT = nl.ndarray((128, KT * 512), dtype=a.dtype, buffer=nl.sbuf)

    # explicit PSUM: 6 rotating accumulators + 1 transpose bank
    acc0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc3 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc4 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    acc5 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    pt = nl.ndarray((128, 1024), dtype=a.dtype, buffer=nl.psum)

    # prologue: group 0 of the first M-block; the first B group is split per
    # k-tile so the first matmuls can start as soon as their slice arrives
    m_first = pid * rows
    _tr_half(a, aT, pt, m_first, 0, 0, K)
    bt_first = nl.ndarray((128, 2048), dtype=b.dtype, buffer=nl.sbuf)
    for j0 in range(4):
        nisa.dma_copy(dst=bt_first[:, j0 * 512:(j0 + 1) * 512],
                      src=b.ap(pattern=[[N, 128], [1, 512]],
                               offset=j0 * 128 * N))
    _tr_half(a, aT, pt, m_first, 0, 1, K)

    for mb in range(n_mblk):
        m0 = pid * rows + mb * 512
        for nbi in range(n_nblk):
            n0 = nbi * 512
            q = mb * n_nblk + nbi
            first_nb = 0
            if mb == 0:
                if nbi == 0:
                    first_nb = 1
            last_nb = 0
            if nbi == n_nblk - 1:
                if mb + 1 < n_mblk:
                    last_nb = 1

            for kg in range(NG):
                use_first = 0
                if first_nb == 1:
                    if kg == 0:
                        use_first = 1
                if use_first == 1:
                    bt = bt_first
                else:
                    # B[kg*512 : kg*512+512, n0:n0+512] as [128 p, 4 j, 512 n]
                    bt = nl.ndarray((128, 2048), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=bt.reshape((128, 4, 512)),
                                  src=b.ap(pattern=[[N, 128], [128 * N, 4], [1, 512]],
                                           offset=kg * 512 * N + n0))
                for j in range(4):
                    kt = kg * 4 + j
                    for mt in range(4):
                        nisa.nc_matmul(
                            dst=_pick6((4 * q + mt) % 6, acc0, acc1, acc2, acc3, acc4, acc5),
                            stationary=aT[:, kt * 512 + mt * 128: kt * 512 + (mt + 1) * 128],
                            moving=bt[:, j * 512:(j + 1) * 512],
                            accumulate=(kt > 0))
                    if first_nb == 1:
                        if kg + 1 < NG:
                            if j == 0:
                                _tr_half(a, aT, pt, m0, kg + 1, 0, K)
                            if j == 2:
                                _tr_half(a, aT, pt, m0, kg + 1, 1, K)
                    if last_nb == 1:
                        # transposes of the next M-block, written into regions
                        # already consumed by this (final) N-block
                        if j == 1:
                            _tr_half(a, aT, pt, m0 + 512, kg, 0, K)
                        if j == 3:
                            if kg + 1 < NG:
                                _tr_half(a, aT, pt, m0 + 512, kg, 1, K)

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
                _tr_half(a, aT, pt, m0 + 512, NG - 1, 1, K)
    return c


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % (2 * _MB) == 0 and N % _NB == 0 and K % (_KG * _P) == 0
    cfg = {
        "lnc_split": "M",
        "MB": _MB,
        "m_tiles_per_block": _MT,
        "NB": _NB,
        "k_tile": _P,
        "k_tiles_per_group": _KG,
        "psum_acc_banks_rotating": _NACC,
        "psum_transpose_banks": _NPT,
        "a_transpose": "tensor_engine_in_kernel",
        "a_transpose_schedule": "next_mblock_during_last_nblock",
        "first_block_lookahead_groups": 1,
        "first_b_group_split_per_ktile": True,
        "accum": "fp32_psum",
    }
    out = _mm_kernel(a, b)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
