import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _sigmoid_kernel(X, N, free_tile, buffers):
    out = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm)

    partitions = 128
    pid = nl.program_id(0)
    per_core = N // 2
    base = pid * per_core

    aligned_columns = per_core // partitions
    tiles = (aligned_columns + free_tile - 1) // free_tile

    # Loading into float32 explicitly performs the required upcast.
    inputs = nl.ndarray(
        (partitions, buffers, free_tile),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (partitions, buffers, free_tile),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )

    # Prime the circular input buffers.
    for tile in range(min(tiles, buffers)):
        width = min(free_tile, aligned_columns - tile * free_tile)
        start = base + tile * partitions * free_tile
        count = partitions * width
        nisa.dma_copy(
            dst=inputs[:, tile, :width],
            src=X[start:start + count].reshape((partitions, width)),
        )

    for tile in range(tiles):
        slot = tile % buffers
        width = min(free_tile, aligned_columns - tile * free_tile)
        start = base + tile * partitions * free_tile
        count = partitions * width

        # Evaluate in fp32; the destination performs the single downcast.
        nisa.activation(
            dst=outputs[:, slot, :width],
            op=nl.sigmoid,
            data=inputs[:, slot, :width],
        )
        nisa.dma_copy(
            dst=out[start:start + count].reshape((partitions, width)),
            src=outputs[:, slot, :width],
        )

        next_tile = tile + buffers
        if next_tile < tiles:
            next_width = min(
                free_tile, aligned_columns - next_tile * free_tile
            )
            next_start = base + next_tile * partitions * free_tile
            next_count = partitions * next_width
            nisa.dma_copy(
                dst=inputs[:, slot, :next_width],
                src=X[next_start:next_start + next_count].reshape(
                    (partitions, next_width)
                ),
            )

    remainder = per_core - aligned_columns * partitions
    if remainder > 0:
        start = base + aligned_columns * partitions
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
            dst=tail_output, op=nl.sigmoid, data=tail_input
        )
        nisa.dma_copy(
            dst=out[start:start + remainder].reshape((1, remainder)),
            src=tail_output,
        )

    return out


def run(X: torch.Tensor, N: int, **kwargs):
    global _last_config

    free_tile = 4096
    buffers = 4
    result = _sigmoid_kernel[2](X, N, free_tile, buffers)

    per_core = N // 2
    columns = per_core // 128
    _last_config = {
        "lnc": 2,
        "partitions": 128,
        "free_tile": free_tile,
        "pipeline_buffers": buffers,
        "tiles_per_core": (columns + free_tile - 1) // free_tile,
        "tail_elements_per_core": per_core % 128,
        "compute_dtype": "float32",
        "activation": "sigmoid",
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
