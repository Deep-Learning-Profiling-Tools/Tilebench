import os

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    
except ImportError:
    nki = None

NUM_CORES = int(os.environ.get("NKI_MATMUL_NUM_CORES", "2"))

if nki is not None:
    @nki.jit
    def matmul_kernel(lhs, rhs, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,
                      NUM_CORES=1, DOUBLE_ROW=False):

        M, K = lhs.shape
        K_rhs, N = rhs.shape

        BLOCK_M = TILE_M * TILES_IN_BLOCK_M
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N
        BLOCK_K = TILE_K * TILES_IN_BLOCK_K

        assert K == K_rhs, "lhs and rhs must share the contraction dimension"
        assert M % BLOCK_M == 0, "M must be a multiple of BLOCK_M"
        assert N % BLOCK_N == 0, "N must be a multiple of BLOCK_N"
        assert K % BLOCK_K == 0, "K must be a multiple of BLOCK_K"

        NUM_BLOCK_M = M // BLOCK_M
        NUM_BLOCK_N = N // BLOCK_N
        NUM_BLOCK_K = K // BLOCK_K

        assert NUM_BLOCK_M % NUM_CORES == 0, "M blocks must divide across cores"
        BLOCKS_PER_CORE = NUM_BLOCK_M // NUM_CORES

        result = nl.ndarray((M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

        core = nl.program_id(0)
        for mi in nl.affine_range(BLOCKS_PER_CORE):
            m = core * BLOCKS_PER_CORE + mi
        
            result_tiles = nl.zeros((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), N), dtype=nl.float32, buffer=nl.sbuf)

            for k in nl.sequential_range(NUM_BLOCK_K):
                lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), BLOCK_K), dtype=lhs.dtype, buffer=nl.sbuf)
                i_lhs = nl.mgrid[0:TILE_M, 0:BLOCK_K]
                
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    lhsT_tiles[bm] = nl.load(lhs[m * BLOCK_M + bm * TILE_M + i_lhs.p, k * BLOCK_K + i_lhs.x])
                    
                    for bk in nl.affine_range(TILES_IN_BLOCK_K):
                        lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)] = nisa.nc_transpose(
                            lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)]
                        )

                for n in nl.affine_range(NUM_BLOCK_N):
                    if DOUBLE_ROW:
                        rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K // 2, nl.par_dim(TILE_K), 2 * BLOCK_N),
                                               dtype=rhs.dtype, buffer=nl.sbuf)
                        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                        for bk in nl.affine_range(TILES_IN_BLOCK_K // 2):
                            for t in range(2):
                                rhs_tiles[bk, :, nl.ds(t * BLOCK_N, BLOCK_N)] = nl.load(
                                    rhs[(k * TILES_IN_BLOCK_K + 2 * bk + t) * TILE_K + i_rhs.p,
                                        n * BLOCK_N + i_rhs.x]
                                )
                    else:
                        rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                               dtype=rhs.dtype, buffer=nl.sbuf)
                        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            rhs_tiles[bk] = nl.load(
                                rhs[(k * TILES_IN_BLOCK_K + bk) * TILE_K + i_rhs.p,
                                    n * BLOCK_N + i_rhs.x]
                            )

                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        for bn in nl.affine_range(TILES_IN_BLOCK_N):
                            res_psum = nl.zeros((TILE_M, TILE_N),
                                                dtype=nl.float32, buffer=nl.psum)
                            if DOUBLE_ROW:
                                for bk in nl.affine_range(TILES_IN_BLOCK_K // 2):
                                    i_k, i_tm, i_m = nl.mgrid[0:TILE_K, 0:2, 0:TILE_M]
                                    lhsT_double = lhsT_tiles[
                                        bm, i_k, bk * (2 * TILE_M) + i_tm * TILE_M + i_m
                                    ]
                                    i_k2, i_tn, i_n = nl.mgrid[0:TILE_K, 0:2, 0:TILE_N]
                                    rhs_double = rhs_tiles[
                                        bk, i_k2, i_tn * BLOCK_N + bn * TILE_N + i_n
                                    ]
                                    res_psum += nisa.nc_matmul(
                                        lhsT_double, rhs_double,
                                        perf_mode='double_row_gen3',
                                    )
                            else:
                                for bk in nl.affine_range(TILES_IN_BLOCK_K):
                                    res_psum += nisa.nc_matmul(
                                        lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)],
                                        rhs_tiles[bk, :, nl.ds(bn * TILE_N, TILE_N)],
                                    )
                            result_tiles[bm, :, nl.ds(n * BLOCK_N + bn * TILE_N, TILE_N)] += res_psum

            i_res = nl.mgrid[0:TILE_M, 0:N]
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                nl.store(result[m * BLOCK_M + bm * TILE_M + i_res.p, i_res.x],
                         value=nl.copy(result_tiles[bm], dtype=lhs.dtype))

        return result


def _pick_tiles_in_block(dim: int, tile: int, preferred: int) -> int:
    """Largest t <= preferred with (tile * t) dividing dim. Falls back to 1."""
    for t in range(preferred, 0, -1):
        if dim % (tile * t) == 0:
            return t
    return 1


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    """NKI matmul. Output dtype matches input dtype (fp32 / fp16 / fp8)."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions")
    if a.dtype != b.dtype:
        raise ValueError("Incompatible dtypes")
    if a.dtype == torch.float8_e4m3fn:
        raise NotImplementedError(
            "matmul NKI: fp8_e4m3fn is not supported on TRN1/TRN2 "
            "(neuronxcc F8E4M3FN requires TRN3+); use fp8_e5m2 for fp8 coverage"
        )

    M, K = a.shape
    _, N = b.shape

    if M % TILE_M or K % TILE_K or N % TILE_N:
        raise NotImplementedError(
            f"matmul NKI: shape ({M}, {N}, {K}) must be a multiple of "
            f"({TILE_M}, {TILE_N}, {TILE_K})"
        )

    tib_m = _pick_tiles_in_block(M, TILE_M, 4)
    tib_n = _pick_tiles_in_block(N, TILE_N, 2)
    tib_k = _pick_tiles_in_block(K, TILE_K, 8)

    num_cores = NUM_CORES if (M // (TILE_M * tib_m)) % NUM_CORES == 0 else 1

    use_double_row = (a.dtype == torch.float8_e5m2 and tib_k % 2 == 0
                      and os.environ.get("NKI_MATMUL_DOUBLE_ROW", "1") == "1")

    return matmul_kernel[nl.nc(num_cores)](a, b, tib_m, tib_n, tib_k, num_cores,
                                           use_double_row)


def get_last_config() -> dict | None:
    return None
