import os

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax

    # Tensor Engine tile limits (the PE array is 128x128, PSUM banks are 128x512 fp32):
    #   stationary tile  (K <= 128, M <= 128)   nl.tile_size.gemm_stationary_fmax
    #   moving tile      (K <= 128, N <= 512)   nl.tile_size.gemm_moving_fmax
    #   PSUM result      (M <= 128, N <= 512)
    # nc_matmul computes stationary.T @ moving, so the CONTRACTION dim (K) must sit
    # on partitions for BOTH operands.
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    
except ImportError:
    nki = None

# Physical NeuronCores per logical core (trn2 `neuron-ls` reports
# logical-neuroncore-config: 2). Override with NKI_MATMUL_NUM_CORES=1 to force
# the single-core kernel.
NUM_CORES = int(os.environ.get("NKI_MATMUL_NUM_CORES", "2"))


if nki is not None:
    @nki.jit
    def matmul_kernel(lhs, rhs, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,
                      NUM_CORES=1, DOUBLE_ROW=False):
        """C[M, N] = A[M, K] @ B[K, N] on the Tensor Engine, blocked along all three dims.

        Args:
            lhs: (M, K) input tile-language tensor in HBM.
            rhs: (K, N) input tensor in HBM.
            TILES_IN_BLOCK_M/N/K: meta-parameters (compile-time ints) giving the
                number of 128 / 512 / 128-wide tiles per block along each dim.
            NUM_CORES: number of physical NeuronCores the launch grid is sharded
                over; each core takes NUM_BLOCK_M / NUM_CORES of the M blocks.

        Returns:
            (M, N) tensor in HBM, dtype == lhs.dtype. Accumulation is fp32
            throughout (the Tensor Engine always accumulates in fp32); the cast
            to the output dtype happens once, in the epilogue -- same as the
            Triton reference.

        Notes:
            Compute-bound by design, and measured to be so: neuron-profile at
            M=N=4096, K=1024 fp16 reports the Tensor Engine 89% active and 78%
            matmul-FLOPs utilisation, against only 36% memory-bandwidth
            utilisation. B is re-read once per M block, so BLOCK_M trades SBUF
            against HBM traffic; DMA is far from the limiter at these shapes.
            (An earlier revision of this note quoted 19% MBU / 41% MFU -- those
            were the single-core kernel, before the nl.nc() sharding below.)

            The (blocks, par_dim, free) SBUF tiles below make the SDK emit
            "DeprecationWarning: Block dimension is deprecated". The warning is
            benign here -- neuronxcc 2.25's own bundled kernels (see
            nki/kernels/double_row_matmul.py) use the same form -- and the
            replacement is slower: folding the block dim into the free dim
            (par_dim first) costs 28% at K=1024 and ~4% at K=8192, because the
            block form gives the compiler independent tiles it can pipeline
            separately. Revisit if a future SDK turns the warning into an error.
        """
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

        # M blocks are independent, so they shard cleanly over the physical
        # NeuronCores of a logical core (launch grid is nl.nc(NUM_CORES); each
        # core owns a contiguous run of M blocks and writes disjoint rows of C).
        core = nl.program_id(0)
        for mi in nl.affine_range(BLOCKS_PER_CORE):
            m = core * BLOCKS_PER_CORE + mi
            # fp32 accumulator for this M block's full row of C, held in SBUF
            # across the K loop (PSUM only accumulates within one K block).
            result_tiles = nl.zeros((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), N), dtype=nl.float32, buffer=nl.sbuf)

            # sequential_range: the K loop carries the accumulation dependency.
            for k in nl.sequential_range(NUM_BLOCK_K):
                # ---- load A block and transpose it so K lands on partitions ----
                # Loaded as (TILE_M rows of A, BLOCK_K) to keep the DMA free dim
                # large and contiguous, then each 128x128 sub-tile is transposed
                # in place (TILE_M == TILE_K == 128 makes this square).
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
                        # ---- load B block staged for Double-FP8 ----
                        # double_row wants a 256-long contraction laid out as
                        # (128 partitions, 2), the 2 on the first free dim. Stage
                        # two consecutive K sub-tiles side by side in the free
                        # dim so rhs_tiles[bk, p, t * BLOCK_N + x] holds K
                        # sub-tile 2*bk + t. The stationary side needs no
                        # restaging: lhsT_tiles already stores K sub-tiles
                        # adjacent along its free dim.
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
                        # ---- load B block: K already on partitions, no transpose ----
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
                            # Repeated nc_matmul into one PSUM tile accumulates
                            # in hardware over this block's K tiles.
                            res_psum = nl.zeros((TILE_M, TILE_N),
                                                dtype=nl.float32, buffer=nl.psum)
                            if DOUBLE_ROW:
                                for bk in nl.affine_range(TILES_IN_BLOCK_K // 2):
                                    # Both operands address K as t * 128 + p over
                                    # the same (t, p), so the 256-long
                                    # contraction pairs up elementwise.
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

    # Measured on trn2 at M=N=4096 (fp16), sweeping TILES_IN_BLOCK_{M,N,K}:
    # (4, 2, 8) was fastest at both ends of the K grid -- 63.5 TFLOP/s at
    # K=1024 and 73.3 at K=8192. Raising M past 4 (BLOCK_M=1024) regresses:
    # the fp32 result_tiles accumulator is TILES_IN_BLOCK_M * 128 * N * 4 bytes,
    # and at 16 MB it starts crowding SBUF.
    tib_m = _pick_tiles_in_block(M, TILE_M, 4)
    tib_n = _pick_tiles_in_block(N, TILE_N, 2)
    tib_k = _pick_tiles_in_block(K, TILE_K, 8)

    # trn2 runs logical-neuroncore-config=2: one logical NeuronCore is two
    # physical cores. A plain kernel occupies only one of them and tops out at
    # half the device's FLOPs (which is exactly the 2x gap against the XLA
    # matmul baseline, since torch shards across both). Fan the M blocks out
    # over both cores when they divide evenly.
    num_cores = NUM_CORES if (M // (TILE_M * tib_m)) % NUM_CORES == 0 else 1

    # Double-FP8: pack two fp8 pairs per PE for 2x Tensor Engine throughput.
    # Without it fp8 costs the same PE cycles as bf16/fp16 (the nc_matmul cost
    # model is max(...) for all three), so the default path tops out at the fp16
    # rate. Consumes K sub-tiles in pairs, hence the even-tib_k requirement.
    use_double_row = (a.dtype == torch.float8_e5m2 and tib_k % 2 == 0
                      and os.environ.get("NKI_MATMUL_DOUBLE_ROW", "1") == "1")

    return matmul_kernel[nl.nc(num_cores)](a, b, tib_m, tib_n, tib_k, num_cores,
                                           use_double_row)


def get_last_config() -> dict | None:
    return None
