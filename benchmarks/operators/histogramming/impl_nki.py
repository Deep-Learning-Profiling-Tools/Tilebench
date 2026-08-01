"""NKI histogramming via one-hot compare + reduce.

A true concurrent scatter-add (bin[value[i]] += 1) needs atomic
read-modify-write for colliding indices within a tile, which pigeonhole
guarantees here (num_bins as low as 64 vs. 128 elements/tile). The only
scatter primitive available (`nisa.dma_compute` with `vector_offset`)
requires `unique_indices=True` -- "non-unique indices not yet supported"
per its docstring -- so it cannot express this safely. Instead: put a block
of bin indices on the partition dim, broadcast each chunk of input values
across those partitions, and reduce `sum(value_chunk == bin_iota)` over the
free (element) dim -- the same one-hot-mask-and-reduce idiom used for the
target-class lookup in cross_entropy/impl_nki.py, just reduction instead of
selection.
"""
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def histogram_kernel(values, num_bins):
        N = values.shape[1]
        chunk_size = 8192
        num_bin_blocks = (num_bins + (PMAX - 1)) // PMAX
        num_chunks = (N + chunk_size - 1) // chunk_size

        hbm_result = nl.ndarray((num_bins, 1), dtype=nl.int32, buffer=nl.hbm)

        for bi in range(num_bin_blocks):
            bin_offset = bi * PMAX
            bin_partition_index = nl.arange(PMAX)[:, None]
            mask_bins = bin_partition_index < (num_bins - bin_offset)

            grid = nl.mgrid[0:PMAX, 0:1]
            bin_iota = nl.add(nisa.iota(expr=grid.p, dtype=nl.int32), bin_offset)

            count = nl.zeros((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
            for ci in range(num_chunks):
                free_offset = ci * chunk_size
                free_index = nl.arange(chunk_size)[None, :]
                mask_elems = free_index < (N - free_offset)

                v_tile = nl.load(values[nl.arange(1)[:, None], free_offset + free_index],
                                  mask=mask_elems, dtype=nl.int32)
                v_broadcast = nl.broadcast_to(v_tile, shape=(PMAX, chunk_size))

                eq = nl.equal(bin_iota, v_broadcast, dtype=nl.int32)
                zero_tile = nl.zeros(eq.shape, dtype=nl.int32, buffer=nl.sbuf)
                mask_elems_b = nl.broadcast_to(mask_elems, shape=(PMAX, chunk_size))
                eq_safe = nl.where(mask_elems_b, eq, zero_tile)
                count[...] = nl.add(count, nl.sum(eq_safe, axis=1, keepdims=True))

            nl.store(hbm_result[bin_offset + bin_partition_index, nl.arange(1)[None, :]],
                     value=count, mask=mask_bins)

        return hbm_result


def run(input: torch.Tensor, N: int, num_bins: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    values_2d = input.reshape(1, -1).to(torch.int32)
    hist = histogram_kernel(values_2d, num_bins)
    return hist.reshape(-1)


def get_last_config() -> dict | None:
    return None
