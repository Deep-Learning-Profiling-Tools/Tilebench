import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _convert_fp32_fp16(x, columns, buffer_count):
    partitions = 128
    tile_elements = partitions * columns
    tile_count = x.shape[0] // tile_elements
    tiles_per_core = tile_count // nl.num_programs()
    first_tile = nl.program_id(0) * tiles_per_core

    out = nl.ndarray(x.shape, dtype=nl.float16, buffer=nl.shared_hbm)
    x_tiles = x.reshape((tile_count, partitions, columns))
    out_tiles = out.reshape((tile_count, partitions, columns))

    source_buffers = nl.ndarray(
        (partitions, buffer_count, columns),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    result_buffers = nl.ndarray(
        (partitions, buffer_count, columns),
        dtype=nl.float16,
        buffer=nl.sbuf,
    )

    for i in range(tiles_per_core):
        slot = i % buffer_count
        tile = first_tile + i
        nisa.dma_copy(
            dst=source_buffers[:, slot, :],
            src=x_tiles[tile],
        )
        nisa.tensor_copy(
            dst=result_buffers[:, slot, :],
            src=source_buffers[:, slot, :],
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out_tiles[tile],
            src=result_buffers[:, slot, :],
        )

    return out


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    global _LAST_CONFIG

    n = x.numel()
    columns = 4096
    tile_elements = 128 * columns
    # Every configured length is divisible by two full tiles.
    assert n % (2 * tile_elements) == 0
    tiles_per_core = n // (2 * tile_elements)
    buffer_count = min(4, tiles_per_core)

    _LAST_CONFIG = {
        "lnc": 2,
        "partitions": 128,
        "tile_columns": columns,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "pipeline_buffers": buffer_count,
        "conversion": "fp32_to_fp16_rne_tensor_copy",
    }
    return _convert_fp32_fp16[2](x, columns, buffer_count)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
