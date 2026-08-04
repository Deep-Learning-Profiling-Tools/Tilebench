"""NKI batched matmul: BATCH independent (M,K)@(K,N) matmuls.

Self-contained: does not import from matmul_fp32_fp16_fp8/impl_nki.py, but the
tiling/PSUM-accumulation approach mirrors it (same TILE_M/K/N, same
_pick_tiles_in_block, same NUM_CORES M-block splitting).

Structured like Triton's bmm_kernel (impl_triton.py): BATCH is folded into a
single kernel launch (an outer nl.affine_range over BATCH, wrapping the
per-core M-block / PSUM-accumulation tiling) instead of dispatching one
kernel per batch element from Python. affine_range (not sequential_range) is
used for the batch loop because batches are data-independent, so the
compiler is free to pipeline/interleave them the same way Triton's grid lets
NeuronCores interleave (m_block, n_block, batch) program instances.
NEURON_RT_NUM_CORES=1 (set by run_bench.py) only restricts the profiler's
device visibility, not nl.nc()'s compile-time SPMD core count, so nl.nc(2)
gets real multi-core parallelism here.

Note: unlike matmul_fp32_fp16_fp8's kernel, this does not implement the fp8
DOUBLE_ROW perf mode -- fp8 batched matmul is correct but not tuned for it.
"""
import os

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
except ImportError:
    nki = None

NUM_CORES = int(os.environ.get("NKI_MATMUL_NUM_CORES", "2"))


def _pick_tiles_in_block(dim: int, tile: int, preferred: int) -> int:
    """Largest t <= preferred with (tile * t) dividing dim. Falls back to 1."""
    for t in range(preferred, 0, -1):
        if dim % (tile * t) == 0:
            return t
    return 1


if nki is not None:
    @nki.jit
    def batched_matmul_kernel(lhs, rhs, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,
                               NUM_CORES=1):

        BATCH, M, K = lhs.shape
        _, K_rhs, N = rhs.shape

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

        result = nl.ndarray((BATCH, M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

        core = nl.program_id(0)
        for b in nl.affine_range(BATCH):
            for mi in nl.affine_range(BLOCKS_PER_CORE):
                m = core * BLOCKS_PER_CORE + mi

                result_tiles = nl.zeros((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), N), dtype=nl.float32, buffer=nl.sbuf)

                for k in nl.sequential_range(NUM_BLOCK_K):
                    lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), BLOCK_K), dtype=lhs.dtype, buffer=nl.sbuf)
                    i_lhs = nl.mgrid[0:TILE_M, 0:BLOCK_K]

                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        lhsT_tiles[bm] = nl.load(lhs[b, m * BLOCK_M + bm * TILE_M + i_lhs.p, k * BLOCK_K + i_lhs.x])

                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)] = nisa.nc_transpose(
                                lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)]
                            )

                    for n in nl.affine_range(NUM_BLOCK_N):
                        rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                               dtype=rhs.dtype, buffer=nl.sbuf)
                        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            rhs_tiles[bk] = nl.load(
                                rhs[b, (k * TILES_IN_BLOCK_K + bk) * TILE_K + i_rhs.p,
                                    n * BLOCK_N + i_rhs.x]
                            )

                        for bm in nl.affine_range(TILES_IN_BLOCK_M):
                            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                                res_psum = nl.zeros((TILE_M, TILE_N),
                                                    dtype=nl.float32, buffer=nl.psum)
                                for bk in nl.affine_range(TILES_IN_BLOCK_K):
                                    res_psum += nisa.nc_matmul(
                                        lhsT_tiles[bm, :, nl.ds(bk * TILE_M, TILE_M)],
                                        rhs_tiles[bk, :, nl.ds(bn * TILE_N, TILE_N)],
                                    )
                                result_tiles[bm, :, nl.ds(n * BLOCK_N + bn * TILE_N, TILE_N)] += res_psum

                i_res = nl.mgrid[0:TILE_M, 0:N]
                for bm in nl.affine_range(TILES_IN_BLOCK_M):
                    nl.store(result[b, m * BLOCK_M + bm * TILE_M + i_res.p, i_res.x],
                             value=nl.copy(result_tiles[bm], dtype=lhs.dtype))

        return result


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = None, autotune: bool = False, **kwargs) -> torch.Tensor:

    A3 = A.view(BATCH, M, K)
    B3 = B.view(BATCH, K, N)

    Mp = ((M + TILE_M - 1) // TILE_M) * TILE_M
    Kp = ((K + TILE_K - 1) // TILE_K) * TILE_K
    Np = ((N + TILE_N - 1) // TILE_N) * TILE_N

    if Mp > M or Kp > K:
        A3 = torch.nn.functional.pad(A3, (0, Kp - K, 0, Mp - M))
    if Kp > K or Np > N:
        B3 = torch.nn.functional.pad(B3, (0, Np - N, 0, Kp - K))

    tib_m = _pick_tiles_in_block(Mp, TILE_M, 4)
    tib_n = _pick_tiles_in_block(Np, TILE_N, 2)
    tib_k = _pick_tiles_in_block(Kp, TILE_K, 8)

    num_cores = NUM_CORES if (Mp // (TILE_M * tib_m)) % NUM_CORES == 0 else 1

    out = batched_matmul_kernel[nl.nc(num_cores)](A3, B3, tib_m, tib_n, tib_k, num_cores)
    return out[:, :M, :N].reshape(-1)


def get_last_config() -> dict | None:
    return None
