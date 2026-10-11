import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Fixed configuration (identical for every configured shape)
_P = 128      # partition / k-tile size
_MB = 512     # rows of A kept transposed (A^T) resident in SBUF per block (4 m-tiles)
_MT = 4       # m-tiles per block
_NB = 512     # output columns per PSUM accumulation block (one PSUM bank per m-tile)
_KG = 4       # k-tiles (of 128) per DMA group / transpose group
_LA = 2       # transpose look-ahead (in groups) during the first N-block of an M-block

_LAST_CONFIG = {}


def _pick4(i, t0, t1, t2, t3):
    r = t0
    if i == 1:
        r = t1
    if i == 2:
        r = t2
    if i == 3:
        r = t3
    return r


def _transpose_group(a, aT, m0, g, K):
    # A[m0:m0+512, g*512:(g+1)*512] -> SBUF tile laid out [128 p, 4 mt, 512 k]
    at = nl.ndarray((128, 2048), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at.reshape((128, 4, 512)),
                  src=a.ap(pattern=[[K, 128], [128 * K, 4], [1, 512]],
                           offset=m0 * K + g * 512))
    for h in range(2):
        pt = nl.ndarray((128, 1024), dtype=a.dtype, buffer=nl.psum)
        for jj in range(2):
            j = 2 * h + jj
            for mt in range(4):
                nisa.nc_transpose(
                    dst=pt[:, jj * 512 + mt * 128: jj * 512 + (mt + 1) * 128],
                    data=at[:, mt * 512 + j * 128: mt * 512 + (j + 1) * 128],
                    engine=nisa.engine.tensor)
        # aT[k, kt*512 + m] layout; this half covers k-tiles g*4+2h, g*4+2h+1
        base = g * 2048 + h * 1024
        if h == 0:
            nisa.tensor_copy(dst=aT[:, base: base + 1024], src=pt,
                             engine=nisa.engine.vector)
        else:
            nisa.tensor_copy(dst=aT[:, base: base + 1024], src=pt,
                             engine=nisa.engine.scalar)


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

    # resident transposed A block: aT[k_in_tile, kt*512 + m]
    aT = nl.ndarray((128, KT * 512), dtype=a.dtype, buffer=nl.sbuf)

    for mb in range(n_mblk):
        m0 = pid * rows + mb * 512
        # prologue: transpose the first groups of this M-block
        for g in range(min(2, NG)):
            _transpose_group(a, aT, m0, g, K)

        for nbi in range(n_nblk):
            n0 = nbi * 512
            acc0 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            acc1 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            acc2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            acc3 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for kg in range(NG):
                if nbi == 0:
                    if kg + 2 < NG:
                        _transpose_group(a, aT, m0, kg + 2, K)
                # B[kg*512 : kg*512+512, n0:n0+512] as [128 p, 4 j, 512 n]
                bt = nl.ndarray((128, 2048), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=bt.reshape((128, 4, 512)),
                              src=b.ap(pattern=[[N, 128], [128 * N, 4], [1, 512]],
                                       offset=kg * 512 * N + n0))
                for j in range(4):
                    kt = kg * 4 + j
                    for mt in range(4):
                        nisa.nc_matmul(
                            dst=_pick4(mt, acc0, acc1, acc2, acc3),
                            stationary=aT[:, kt * 512 + mt * 128: kt * 512 + (mt + 1) * 128],
                            moving=bt[:, j * 512:(j + 1) * 512],
                            accumulate=(kt > 0))
            # single rounding fp32 -> output dtype, then store
            ot = nl.ndarray((128, 2048), dtype=a.dtype, buffer=nl.sbuf)
            for mt in range(4):
                if mt % 2 == 0:
                    nisa.tensor_copy(dst=ot[:, mt * 512:(mt + 1) * 512],
                                     src=_pick4(mt, acc0, acc1, acc2, acc3),
                                     engine=nisa.engine.vector)
                else:
                    nisa.tensor_copy(dst=ot[:, mt * 512:(mt + 1) * 512],
                                     src=_pick4(mt, acc0, acc1, acc2, acc3),
                                     engine=nisa.engine.scalar)
            nisa.dma_copy(dst=c.ap(pattern=[[N, 128], [128 * N, 4], [1, 512]],
                                   offset=m0 * N + n0),
                          src=ot.reshape((128, 4, 512)))
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
        "transpose_lookahead_groups": _LA,
        "a_transpose": "tensor_engine_in_kernel",
        "accum": "fp32_psum",
    }
    out = _mm_kernel(a, b)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
