import functools
import os
import re
import subprocess

from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# --- Hardware constants (NeuronCore-v2/v3) ---------------------------------
# nc_matmul: dst[M, N] = stationary[K, M].T @ moving[K, N]
MAX_TILE_M = 128   # stationary free dim (output rows per PE tile)      <= 128
MAX_TILE_K = 128   # contraction dim per matmul (== partition dim)       <= 128
MAX_TILE_N = 512   # moving free dim == one fp32 PSUM bank (128 x 2KB)   <= 512


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel can be launched with.

    The NKI launch degree has to be compatible with the LNC the XLA module is
    compiled for; trn2/trn3 default to LNC=2 unless the compiler/runtime env
    says otherwise.
    """
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").strip().lower()
    if target in ("trn2", "gen3", "trn3", "gen4"):
        return 2
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
        lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
        if lnc:
            return int(lnc.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 1


if nki is not None:
    @nki.jit
    def batched_matmul_kernel(lhs, rhs, TILE_M, TILE_K, TILE_N, NUM_CORES=1):
        """``lhs[B, M, K] @ rhs[B, K, N]`` -> ``[B, M, N]`` in the input dtype.

        The whole batch is handled by a *single* launch: the batch axis is
        split across the ``NUM_CORES`` SPMD programs (``nl.program_id(0)``),
        and each program walks its slice of the batch with a compile-time
        unrolled Python loop.  Per batch element the kernel is the standard
        3-level PE tiling:

            lhsT_tiles[k, m] = transpose(lhs[b, m-tile, k-tile])   (stationary)
            rhs_tiles[k]     = rhs[b, k-tile, :]                   (moving)
            psum[m, n]       = sum_k lhsT_tiles[k, m].T @ rhs_tiles[k][:, n]

        Tile sizes are compile-time ints; the trailing M/K/N tiles are simply
        shorter (nc_matmul accepts any stationary/moving extent within the
        hardware maxima), so no host-side padding is needed.

        Args:
            lhs: (B, M, K) HBM tensor.
            rhs: (B, K, N) HBM tensor, same dtype as ``lhs``.
            TILE_M: output rows per PE tile, <= 128.
            TILE_K: contraction rows per matmul (partition dim), <= 128.
            TILE_N: moving free dim / PSUM free dim, <= 512.
            NUM_CORES: SPMD programs the batch is split across; must equal the
                launch degree the kernel is invoked with.

        Returns:
            (B, M, N) HBM tensor in ``lhs.dtype``.
        """
        B, M, K = lhs.shape
        B_rhs, K_rhs, N = rhs.shape

        assert B == B_rhs, "lhs and rhs must share the batch dimension"
        assert K == K_rhs, "lhs and rhs must share the contraction dimension"
        assert TILE_M <= 128 and TILE_K <= 128 and TILE_N <= 512, \
            "tile sizes exceed the PE/PSUM maxima"
        assert B % NUM_CORES == 0, "batch must divide across cores"

        NUM_M = (M + TILE_M - 1) // TILE_M
        NUM_K = (K + TILE_K - 1) // TILE_K
        NUM_N = (N + TILE_N - 1) // TILE_N
        BATCH_PER_CORE = B // NUM_CORES

        result = nl.ndarray((B, M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

        core = nl.program_id(0)
        for bl in range(BATCH_PER_CORE):
            bi = core * BATCH_PER_CORE + bl

            # nl.par_dim is gone, so the per-tile axes live in the *free*
            # dimensions of one fused SBUF tile (partition axis stays first):
            # lhsT_tiles[0:k_size, m, k, 0:m_size] is the transposed lhs tile
            # for output rows m and contraction rows k -- a valid stationary
            # operand for nc_matmul.
            lhsT_tiles = nl.ndarray((TILE_K, NUM_M, NUM_K, TILE_M),
                                    dtype=lhs.dtype, buffer=nl.sbuf)
            for m in range(NUM_M):
                m_row0 = m * TILE_M
                m_size = min(TILE_M, M - m_row0)
                lhs_tile = nl.ndarray((TILE_M, K), dtype=lhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=lhs_tile[0:m_size, 0:K],
                              src=lhs[bi, m_row0:m_row0 + m_size, 0:K])
                for k in range(NUM_K):
                    k_row0 = k * TILE_K
                    k_size = min(TILE_K, K - k_row0)
                    # nc_transpose writes PSUM; on gen3+ the destination dtype
                    # has to match the input dtype.  Stage back into SBUF so
                    # the whole transposed block stays one flat tile.
                    t_psum = nl.ndarray((TILE_K, TILE_M), dtype=lhs.dtype,
                                        buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=t_psum[0:k_size, 0:m_size],
                        data=lhs_tile[0:m_size, k_row0:k_row0 + k_size],
                    )
                    nisa.tensor_copy(
                        dst=lhsT_tiles[0:k_size, m, k, 0:m_size],
                        src=t_psum[0:k_size, 0:m_size],
                    )

            # The rhs block is shared by every m tile of this batch element,
            # so it is loaded once: rhs_tiles[0:k_size, k, :] is the moving
            # operand for contraction rows k.
            rhs_tiles = nl.ndarray((TILE_K, NUM_K, N), dtype=rhs.dtype,
                                   buffer=nl.sbuf)
            for k in range(NUM_K):
                k_row0 = k * TILE_K
                k_size = min(TILE_K, K - k_row0)
                nisa.dma_copy(dst=rhs_tiles[0:k_size, k, 0:N],
                              src=rhs[bi, k_row0:k_row0 + k_size, 0:N])

            for m in range(NUM_M):
                m_row0 = m * TILE_M
                m_size = min(TILE_M, M - m_row0)
                for n in range(NUM_N):
                    n_col0 = n * TILE_N
                    n_size = min(TILE_N, N - n_col0)

                    # Repeated nc_matmul writes into the same PSUM tile with
                    # accumulate=True sum the contraction tiles in hardware.
                    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32,
                                          buffer=nl.psum)
                    for k in range(NUM_K):
                        k_row0 = k * TILE_K
                        k_size = min(TILE_K, K - k_row0)
                        nisa.nc_matmul(
                            dst=res_psum[0:m_size, 0:n_size],
                            stationary=lhsT_tiles[0:k_size, m, k, 0:m_size],
                            moving=rhs_tiles[0:k_size, k,
                                             n_col0:n_col0 + n_size],
                            accumulate=(k > 0),
                        )

                    # PSUM cannot be DMA'd to HBM directly; stage through SBUF
                    # (which is also where the fp32 accumulator is cast down).
                    out_tile = nl.ndarray((TILE_M, TILE_N), dtype=lhs.dtype,
                                          buffer=nl.sbuf)
                    nisa.tensor_copy(dst=out_tile[0:m_size, 0:n_size],
                                     src=res_psum[0:m_size, 0:n_size])
                    nisa.dma_copy(
                        dst=result[bi, m_row0:m_row0 + m_size,
                                   n_col0:n_col0 + n_size],
                        src=out_tile[0:m_size, 0:n_size],
                    )

        return result


_tuner = NkiAutotuner(batched_matmul_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = None, autotune: bool = False, **kwargs) -> torch.Tensor:
    """Batched matmul on Trainium: one NKI launch covers all BATCH matmuls."""
    if A.dtype != B.dtype:
        raise ValueError("Incompatible dtypes")

    A3 = A.view(BATCH, M, K)
    B3 = B.view(BATCH, K, N)

    tile_m = min(MAX_TILE_M, M)
    tile_k = min(MAX_TILE_K, K)
    tile_n = min(MAX_TILE_N, N)
    _default = SimpleNamespace(block_size_m=tile_m, block_size_k=tile_k, block_size_n=tile_n)
    if autotune:
        _space = [SimpleNamespace(block_size_m=bm, block_size_k=bk, block_size_n=bn)
                  for bm in (64, 128) for bk in (64, 128) for bn in (128, 256, 512)
                  if bm <= M and bk <= K and bn <= N]
        if not any(vars(c) == vars(_default) for c in _space):
            _space.append(_default)
        cfg = _tuner.tune_or_cached(
            shape_key=((BATCH, M, N, K), str(A.dtype)),
            search_space=_space,
            args_fn=lambda cfg: (A3, B3, cfg.block_size_m, cfg.block_size_k, cfg.block_size_n, 1),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _default
    tile_m, tile_k, tile_n = cfg.block_size_m, cfg.block_size_k, cfg.block_size_n

    # The SPMD grid degree and the batch-split factor are the same number: the
    # kernel derives its batch slice from nl.program_id(0).  The launch degree
    # has to match the LNC the module is compiled for, so the batch is split
    # across the LNC cores when it divides evenly, one core otherwise.
    lnc = _lnc_degree()
    num_cores = lnc if BATCH % lnc == 0 else 1

    C = batched_matmul_kernel[num_cores](A3, B3, tile_m, tile_k, tile_n,
                                         num_cores)
    return C.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
