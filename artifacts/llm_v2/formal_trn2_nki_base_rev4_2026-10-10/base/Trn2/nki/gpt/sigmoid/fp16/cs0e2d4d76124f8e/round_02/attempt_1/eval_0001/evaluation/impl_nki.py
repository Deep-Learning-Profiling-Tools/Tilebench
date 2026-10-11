import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _sigmoid_kernel(X, N, free_tile, dma_upcast):
    out = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm)

    partitions = 128
    load_buffers = 4
    compute_buffers = 2
    per_core = N // 2
    base = nl.program_id(0) * per_core
    columns = per_core // partitions
    tiles = (columns + free_tile - 1) // free_tile

    if dma_upcast:
        load_dtype = nl.float32
    else:
        load_dtype = X.dtype

    inputs = nl.ndarray(
        (partitions, load_buffers, free_tile),
        dtype=load_dtype,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (partitions, load_buffers, free_tile),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )

    if not dma_upcast:
        converted = nl.ndarray(
            (partitions, compute_buffers, free_tile),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

    for tile in range(min(tiles, load_buffers)):
        width = min(free_tile, columns - tile * free_tile)
        start = base + tile * partitions * free_tile
        count = partitions * width
        nisa.dma_copy(
            dst=inputs[:, tile, :width],
            src=X[start:start + count].reshape((partitions, width)),
        )

    for tile in range(tiles):
        slot = tile % load_buffers
        width = min(free_tile, columns - tile * free_tile)
        start = base + tile * partitions * free_tile
        count = partitions * width

        if dma_upcast:
            nisa.activation(
                dst=outputs[:, slot, :width],
                op=nl.sigmoid,
                data=inputs[:, slot, :width],
            )
        else:
            compute_slot = tile % compute_buffers
            nisa.tensor_copy(
                dst=converted[:, compute_slot, :width],
                src=inputs[:, slot, :width],
                engine=nisa.engine.vector,
            )
            nisa.activation(
                dst=outputs[:, slot, :width],
                op=nl.sigmoid,
                data=converted[:, compute_slot, :width],
            )

        nisa.dma_copy(
            dst=out[start:start + count].reshape((partitions, width)),
            src=outputs[:, slot, :width],
        )

        next_tile = tile + load_buffers
        if next_tile < tiles:
            next_width = min(free_tile, columns - next_tile * free_tile)
            next_start = base + next_tile * partitions * free_tile
            next_count = partitions * next_width
            nisa.dma_copy(
                dst=inputs[:, slot, :next_width],
                src=X[next_start:next_start + next_count].reshape(
                    (partitions, next_width)
                ),
            )

    remainder = per_core - columns * partitions
    if remainder > 0:
        start = base + columns * partitions
        tail_input = nl.ndarray(
            (1, remainder), dtype=nl.float32, buffer=nl.sbuf
        )
        tail_output = nl.ndarray(
            (1, remainder), dtype=X.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=tail_input,
            src=X[start:start + remainder].reshape((1, remainder)),
        )
        nisa.activation(
            dst=tail_output,
            op=nl.sigmoid,
            data=tail_input,
        )
        nisa.dma_copy(
            dst=out[start:start + remainder].reshape((1, remainder)),
            src=tail_output,
        )

    return out


def run(X: torch.Tensor, N: int, **kwargs):
    global _last_config

    # Use larger DMA blocks once each core has four full blocks of work.
    large_blocks = N >= 2 * 4 * 128 * 8192
    free_tile = 8192 if large_blocks else 4096
    dma_upcast = not large_blocks

    result = _sigmoid_kernel[2](X, N, free_tile, dma_upcast)

    per_core = N // 2
    columns = per_core // 128
    _last_config = {
        "lnc": 2,
        "partitions": 128,
        "free_tile": free_tile,
        "load_buffers": 4,
        "output_buffers": 4,
        "compute_buffers": 0 if dma_upcast else 2,
        "upcast": "dma" if dma_upcast else "vector_copy",
        "tiles_per_core": (columns + free_tile - 1) // free_tile,
        "tail_elements_per_core": per_core % 128,
        "compute_dtype": "float32",
        "activation": "sigmoid",
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
