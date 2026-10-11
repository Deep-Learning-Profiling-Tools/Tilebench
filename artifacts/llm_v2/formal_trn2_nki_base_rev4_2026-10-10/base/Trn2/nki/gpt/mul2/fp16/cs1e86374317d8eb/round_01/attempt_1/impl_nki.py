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

    # Keep the next input tile available while processing the previous tile.
    for i in range(1, tiles_per_core):
        current = nl.ndarray(
            (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=current, src=x_tiles[first_tile + i])
        nisa.tensor_scalar(
            dst=previous,
            data=previous,
            op0=nl.multiply,
            operand0=2.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out_tiles[first_tile + i - 1], src=previous
        )
        previous = current

    nisa.tensor_scalar(
        dst=previous,
        data=previous,
        op0=nl.multiply,
        operand0=2.0,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=out_tiles[first_tile + tiles_per_core - 1], src=previous
    )
    return out


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    n = x.numel()
    # The configured lengths make every tile complete. Retain at least
    # two tiles per physical core, with a maximum 1 MiB tile.
    tile_free = min(4096, n // 512)

    result = _mul2_kernel[2](x, tile_free)
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": 128,
        "tile_free": tile_free,
        "tile_elements": 128 * tile_free,
        "tiles_per_core": n // (256 * tile_free),
        "lookahead_tiles": 1,
        "engine": "vector",
        "in_place_sbuf": True,
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
