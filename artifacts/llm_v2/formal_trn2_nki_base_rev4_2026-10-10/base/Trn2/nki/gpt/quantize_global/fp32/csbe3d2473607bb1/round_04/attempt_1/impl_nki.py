import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _convert_fp32_fp16(x, columns, buffer_count):
    partitions = 128
    core_elements = x.shape[0] // nl.num_programs()
    core_start = nl.program_id(0) * core_elements
    tile_elements = partitions * columns
    tiles_per_core = (core_elements + tile_elements - 1) // tile_elements

    out = nl.ndarray(x.shape, dtype=nl.float16, buffer=nl.shared_hbm)
    buffers = nl.ndarray(
        (partitions, buffer_count, columns),
        dtype=nl.float16,
        buffer=nl.sbuf,
    )

    # Prefetch into independent buffers. DMA performs the RNE conversion.
    for i in range(buffer_count):
        width = min(columns, (core_elements - i * tile_elements) // partitions)
        start = core_start + i * tile_elements
        nisa.dma_copy(
            dst=buffers[:, i, :width],
            src=x[start:start + partitions * width].reshape((partitions, width)),
        )

    for i in range(tiles_per_core):
        slot = i % buffer_count
        width = min(columns, (core_elements - i * tile_elements) // partitions)
        start = core_start + i * tile_elements
        nisa.dma_copy(
            dst=out[start:start + partitions * width].reshape((partitions, width)),
            src=buffers[:, slot, :width],
        )

        next_tile = i + buffer_count
        if next_tile < tiles_per_core:
            next_width = min(
                columns,
                (core_elements - next_tile * tile_elements) // partitions,
            )
            next_start = core_start + next_tile * tile_elements
            nisa.dma_copy(
                dst=buffers[:, slot, :next_width],
                src=x[
                    next_start:next_start + partitions * next_width
                ].reshape((partitions, next_width)),
            )

    return out


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    global _LAST_CONFIG

    n = x.numel()
    assert n % 256 == 0

    core_elements = n // 2
    columns = min(16384, core_elements // 128)
    tile_elements = 128 * columns
    tiles_per_core = (core_elements + tile_elements - 1) // tile_elements
    buffer_count = min(2, tiles_per_core)

    _LAST_CONFIG = {
        "lnc": 2,
        "partitions": 128,
        "tile_columns": columns,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "pipeline_buffers": buffer_count,
        "prefetch_tiles": buffer_count,
        "tail_elements_per_core": core_elements % tile_elements,
        "conversion": "fp32_to_fp16_rne_dma",
    }

    return _convert_fp32_fp16[2](x, columns, buffer_count)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
