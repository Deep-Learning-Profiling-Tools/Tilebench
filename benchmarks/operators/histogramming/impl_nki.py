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
