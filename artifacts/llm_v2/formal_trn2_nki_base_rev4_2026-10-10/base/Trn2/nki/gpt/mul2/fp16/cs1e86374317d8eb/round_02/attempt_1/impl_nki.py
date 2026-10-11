import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _mul2_kernel(x, tile_free):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    tile_elements = partitions * tile_free
    tile_count = x.shape[0] // tile_elements
    tiles_per_core = tile_count // nl.num_programs()
    first_tile = nl.program_id(0) * tiles_per_core

    x_tiles = x.reshape((tile_count, partitions, tile_free))
    out_tiles = out.reshape((tile_count, partitions, tile_free))

    previous = nl.ndarray(
        (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=previous, src=x_tiles[first_tile])

    for i in range(1, tiles_per_core):
        current = nl.ndarray(
            (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=current, src=x_tiles[first_tile + i])

        # Fuse doubling into the output DMA. Each input element is loaded
        # from HBM once; both addition operands are the same SBUF value.
        nisa.dma_compute(
            dst=out_tiles[first_tile + i - 1],
            srcs=[previous, previous],
            reduce_op=nl.add,
        )
        previous = current

    nisa.dma_compute(
        dst=out_tiles[first_tile + tiles_per_core - 1],
        srcs=[previous, previous],
        reduce_op=nl.add,
    )
    return out


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    n = x.numel()
    tile_free = min(4096, n // 512)

    result = _mul2_kernel[2](x, tile_free)
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": 128,
        "tile_free": tile_free,
        "tile_elements": 128 * tile_free,
        "tiles_per_core": n // (256 * tile_free),
        "lookahead_tiles": 1,
        "doubling": "dma_compute_add",
        "fused_output_dma": True,
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
