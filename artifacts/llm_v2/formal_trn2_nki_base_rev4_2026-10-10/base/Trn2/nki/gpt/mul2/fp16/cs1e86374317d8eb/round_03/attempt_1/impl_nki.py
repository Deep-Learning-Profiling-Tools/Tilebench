import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _mul2_kernel(x, tile_free, pipeline_depth):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    tile_elements = partitions * tile_free
    core_elements = x.shape[0] // nl.num_programs()
    core_start = nl.program_id(0) * core_elements
    full_tiles = core_elements // tile_elements
    tail_elements = core_elements % tile_elements
    tail_free = tail_elements // partitions

    # Separate input and output slots allow the next input DMA to reuse an
    # input slot as soon as arithmetic finishes, without waiting for a store.
    inputs = nl.ndarray(
        (partitions, pipeline_depth, tile_free),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (partitions, pipeline_depth, tile_free),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )

    for slot in range(pipeline_depth):
        start = core_start + slot * tile_elements
        nisa.dma_copy(
            dst=inputs[:, slot, :],
            src=x[start:start + tile_elements].reshape(
                (partitions, tile_free)
            ),
        )

    if tail_elements > 0:
        tail_input = nl.ndarray(
            (partitions, tail_free), dtype=x.dtype, buffer=nl.sbuf
        )
        tail_output = nl.ndarray(
            (partitions, tail_free), dtype=x.dtype, buffer=nl.sbuf
        )
        tail_start = core_start + full_tiles * tile_elements
        nisa.dma_copy(
            dst=tail_input,
            src=x[tail_start:tail_start + tail_elements].reshape(
                (partitions, tail_free)
            ),
        )

    for tile in range(full_tiles):
        slot = tile % pipeline_depth

        nisa.tensor_scalar(
            dst=outputs[:, slot, :],
            data=inputs[:, slot, :],
            op0=nl.multiply,
            operand0=2.0,
            engine=nisa.engine.vector,
        )

        future_tile = tile + pipeline_depth
        if future_tile < full_tiles:
            future_start = core_start + future_tile * tile_elements
            nisa.dma_copy(
                dst=inputs[:, slot, :],
                src=x[
                    future_start:future_start + tile_elements
                ].reshape((partitions, tile_free)),
            )

        start = core_start + tile * tile_elements
        nisa.dma_copy(
            dst=out[start:start + tile_elements].reshape(
                (partitions, tile_free)
            ),
            src=outputs[:, slot, :],
        )

    if tail_elements > 0:
        nisa.tensor_scalar(
            dst=tail_output,
            data=tail_input,
            op0=nl.multiply,
            operand0=2.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out[tail_start:tail_start + tail_elements].reshape(
                (partitions, tail_free)
            ),
            src=tail_output,
        )

    return out


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    n = x.numel()
    tile_free = min(8192, n // 512)
    tile_elements = 128 * tile_free
    core_elements = n // 2
    full_tiles = core_elements // tile_elements
    pipeline_depth = min(3, full_tiles)

    result = _mul2_kernel[2](x, tile_free, pipeline_depth)
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": 128,
        "tile_free": tile_free,
        "tile_elements": tile_elements,
        "full_tiles_per_core": full_tiles,
        "tail_elements_per_core": core_elements % tile_elements,
        "pipeline_depth": pipeline_depth,
        "engine": "vector",
        "in_place_sbuf": False,
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
