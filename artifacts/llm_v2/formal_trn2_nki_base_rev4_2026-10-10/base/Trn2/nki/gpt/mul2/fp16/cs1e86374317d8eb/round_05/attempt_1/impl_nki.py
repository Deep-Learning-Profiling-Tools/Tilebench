import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _mul2_kernel(x, tile_free):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    first_tile = 2 * nl.program_id(0)
    x_tiles = x.reshape((4, partitions, tile_free))
    out_tiles = out.reshape((4, partitions, tile_free))

    first = nl.ndarray(
        (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
    )
    second = nl.ndarray(
        (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
    )

    nisa.dma_copy(dst=first, src=x_tiles[first_tile])
    nisa.dma_copy(dst=second, src=x_tiles[first_tile + 1])

    nisa.tensor_scalar(
        dst=first,
        data=first,
        op0=nl.multiply,
        operand0=2.0,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(dst=out_tiles[first_tile], src=first)

    nisa.tensor_scalar(
        dst=second,
        data=second,
        op0=nl.multiply,
        operand0=2.0,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(dst=out_tiles[first_tile + 1], src=second)

    return out


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    n = x.numel()

    # All configured lengths divide exactly into four tiles.
    # Both resident tiles together occupy at most 20 MiB per core.
    tile_free = n // 512
    result = _mul2_kernel[2](x, tile_free)

    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": 128,
        "tile_free": tile_free,
        "tile_elements": 128 * tile_free,
        "tiles_per_core": 2,
        "lookahead_tiles": 1,
        "sbuf_bytes_per_core": n,
        "engine": "vector",
        "arithmetic": "multiply",
        "in_place_sbuf": True,
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
