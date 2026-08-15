"""NKI histogramming: per-bin count via a fully-unrolled (bin_block x chunk) loop nest.

Known limitation -- compile blowup at large (N, num_bins):

    ``histogram_kernel``'s two Python ``for`` loops (over ``num_bin_blocks =
    ceil(num_bins/128)`` and ``num_chunks = ceil(N/8192)``) are both fully unrolled at
    compile time, giving ``num_bin_blocks * num_chunks`` copies of the loop body in the
    compiled program. The body is 4 instructions per (bin block, chunk) -- a
    broadcasting DMA, a compare, a free-axis reduce and an accumulate -- so the compiled
    instruction count is ~4 * num_bin_blocks * num_chunks.

    Hardware-verified: num_bins=64 at the largest swept N (67,108,864 elements -> 8,192
    unrolled iterations, one bin block), and num_bins=4096 at N=4,194,304 (32 bin blocks
    x 512 chunks = 16,384 unrolled iterations).

    The sweep's largest case, num_bins=4096 at N=67,108,864 (32 bin blocks x 8,192
    chunks = 262,144 unrolled iterations), still does not build. It no longer reaches
    the ``[NCC_EBVF030]`` instruction-count verdict that the pre-Beta-2 version hit
    (35,913,891 instructions vs a 5,000,000 limit): at an identical 1,024-iteration
    configuration this version's NEFF is 0.17 MB against the old version's 2.30 MB
    (~13x fewer instructions, extrapolating to ~2.7M for the largest case), and compile
    wall time dropped correspondingly (the 8,192-iteration case went from ~7 min to
    ~75 s). What blocks it now is the sheer size of the unrolled program on disk:
    ``neuronx-cc`` aborts partway through with ``No space left on device``
    (``[NCC_INLA001]`` / ``LLVM ERROR: IO failure on output stream``) after filling
    ~1 GB of its temporary workdir -- measured on a host whose root filesystem was
    already close to capacity, so the exact threshold is machine-dependent. Both symptoms have the same root cause -- full unrolling -- which is fixed
    elsewhere in this codebase (1d_conv, 3d_conv, block_sparse_attention) via
    ``nl.dynamic_range``, not yet applied here. Left as a known limitation rather than
    rewritten in this pass.
"""
import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

# Number of input elements processed per unrolled inner-loop iteration. Sized so
# that the two [PMAX, CHUNK_SIZE] int32 SBUF tiles (values + compare result) stay
# well inside one SBUF partition (2 x 32 KB of the 192 KB available).
CHUNK_SIZE = 8192


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def histogram_kernel(values, num_bins):
        """Count occurrences of each bin id in ``values``.

        Equivalent to ``torch.bincount(values, minlength=num_bins)`` for inputs whose
        values all lie in ``[0, num_bins)``.

        Bins are laid out along the partition axis: partition ``p`` of bin block ``bi``
        owns bin id ``bi * 128 + p``. Each chunk of the input row is broadcast to all
        partitions, compared against each partition's own bin id, and the equality mask
        is summed along the free axis into a per-partition running count.

        Args:
            values: [1, N] int32 HBM tensor of bin ids.
            num_bins: number of histogram bins (compile-time constant).

        Returns:
            [num_bins, 1] int32 HBM tensor of per-bin counts.

        Notes:
            * The per-partition broadcast of the ``[1, chunk]`` values row is done by
              the load itself: ``values.ap(pattern=[[0, bin_sz], [1, sz]])`` gives the
              partition axis a stride of 0, so every partition DMAs the same HBM row.
              This costs no extra instructions over a plain ``[1, chunk]`` load (and
              avoids an ``nc_matmul``-based broadcast, which would need one matmul per
              512 columns inside an already fully-unrolled loop nest).
            * The compare uses ``nisa.tensor_scalar`` with ``operand0=bin_iota``, a
              ``[P, 1]`` tile broadcast along the free axis by the hardware.
              ``operand0`` must be float32 (the MLIR verifier rejects an int32
              ``operand0``); bin ids are far below 2^24 so the compare is exact.
              The fused ``nisa.tensor_scalar_reduce`` is *not* usable here because it
              additionally requires a floating-point ``data`` input and ``reduce_res``
              (``[NCC_IBVF012]`` / ``[NCC_IBVF013]``), which would cost an extra
              int32 -> fp32 conversion pass over every chunk.
            * Everything downstream of the compare stays int32, so counts above 2^24
              remain exact.
        """
        kernel_assert(len(values.shape) == 2 and values.shape[0] == 1,
                      "values must be a [1, N] row")
        N = values.shape[1]
        num_bin_blocks = div_ceil(num_bins, PMAX)
        num_chunks = div_ceil(N, CHUNK_SIZE)

        hbm_result = nl.ndarray((num_bins, 1), dtype=nl.int32, buffer=nl.shared_hbm)

        for bi in range(num_bin_blocks):
            bin_offset = bi * PMAX
            bin_sz = min(PMAX, num_bins - bin_offset)

            # bin_iota[p, 0] = bin_offset + p, i.e. the bin id owned by partition p.
            bin_iota = nl.ndarray((bin_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=bin_iota, pattern=[[0, 1]], offset=bin_offset,
                      channel_multiplier=1)

            count = nl.ndarray((bin_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dst=count, value=0)

            for ci in range(num_chunks):
                free_offset = ci * CHUNK_SIZE
                # Clamp the tail chunk instead of masking: the tile is sized to the
                # number of real elements, so no out-of-range value is ever compared.
                sz = min(CHUNK_SIZE, N - free_offset)

                # Broadcasting load: partition stride 0 replicates the values row.
                v_bcast = nl.ndarray((bin_sz, sz), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=v_bcast,
                    src=values.ap(pattern=[[0, bin_sz], [1, sz]], offset=free_offset),
                )

                # eq[p, j] = (v_bcast[p, j] == bin_iota[p]); chunk_count = sum_j eq[p, j]
                eq = nl.ndarray((bin_sz, sz), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eq, data=v_bcast, op0=nl.equal,
                                   operand0=bin_iota)

                chunk_count = nl.ndarray((bin_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=chunk_count, op=nl.add, data=eq, axis=(1,))

                nisa.tensor_tensor(dst=count, data1=count, data2=chunk_count,
                                   op=nl.add)

            nisa.dma_copy(dst=hbm_result[bin_offset:bin_offset + bin_sz, 0:1],
                          src=count)

        return hbm_result


def run(input: torch.Tensor, N: int, num_bins: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    values_2d = input.reshape(1, -1).to(torch.int32)
    hist = histogram_kernel(values_2d, num_bins)
    return hist.reshape(-1)


def get_last_config() -> dict | None:
    return None
