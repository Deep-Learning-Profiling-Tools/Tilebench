import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _sigmoid_kernel(
    X, N, free_tile, compute_tile, io_buffers, dma_upcast,
    prefetch_before_store
):
    out = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm)

    partitions = 128
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
        (partitions, io_buffers, free_tile),
        dtype=load_dtype,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (partitions, io_buffers, free_tile),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )

    if not dma_upcast:
        converted = nl.ndarray(
            (partitions, compute_buffers, compute_tile),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

    for tile in range(min(tiles, io_buffers)):
        width = min(free_tile, columns - tile * free_tile)
        start = base + tile * partitions * free_tile
        count = partitions * width
        nisa.dma_copy(
            dst=inputs[:, tile, :width],
            src=X[start:start + count].reshape((partitions, width)),
        )

    for tile in range(tiles):
        slot = tile % io_buffers
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
            # Keep vector conversion and ScalarE work at the smaller
            # granularity, independently of the DMA transfer size.
            for chunk in range((width + compute_tile - 1) // compute_tile):
                c0 = chunk * compute_tile
                chunk_width = min(compute_tile, width - c0)
                compute_slot = (
                    tile * (free_tile // compute_tile) + chunk
                ) % compute_buffers

                nisa.tensor_copy(
                    dst=converted[:, compute_slot, :chunk_width],
                    src=inputs[:, slot, c0:c0 + chunk_width],
                    engine=nisa.engine.vector,
                )
                nisa.activation(
                    dst=outputs[:, slot, c0:c0 + chunk_width],
                    op=nl.sigmoid,
                    data=converted[:, compute_slot, :chunk_width],
                )

        next_tile = tile + io_buffers

        if prefetch_before_store:
            if next_tile < tiles:
                next_width = min(
                    free_tile, columns - next_tile * free_tile
                )
                next_start = base + next_tile * partitions * free_tile
                next_count = partitions * next_width
                nisa.dma_copy(
                    dst=inputs[:, slot, :next_width],
                    src=X[next_start:next_start + next_count].reshape(
                        (partitions, next_width)
                    ),
                )

        nisa.dma_copy(
            dst=out[start:start + count].reshape((partitions, width)),
            src=outputs[:, slot, :width],
        )

        if not prefetch_before_store:
            if next_tile < tiles:
                next_width = min(
                    free_tile, columns - next_tile * free_tile
                )
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

    if N >= 2 * 4 * 128 * 16384:
        free_tile = 16384
        compute_tile = 8192
        io_buffers = 2
        dma_upcast = False
        prefetch_before_store = True
    elif N >= 2 * 4 * 128 * 8192:
        free_tile = 8192
        compute_tile = 8192
        io_buffers = 4
        dma_upcast = False
        prefetch_before_store = False
    else:
        free_tile = 4096
        compute_tile = 4096
        io_buffers = 4
        dma_upcast = True
        prefetch_before_store = False

    result = _sigmoid_kernel[2](
        X, N, free_tile, compute_tile, io_buffers,
        dma_upcast, prefetch_before_store,
    )

    per_core = N // 2
    columns = per_core // 128
    _last_config = {
        "lnc": 2,
        "partitions": 128,
        "free_tile": free_tile,
        "compute_tile": compute_tile,
        "load_buffers": io_buffers,
        "output_buffers": io_buffers,
        "compute_buffers": 0 if dma_upcast else 2,
        "upcast": "dma" if dma_upcast else "vector_copy",
        "prefetch_before_store": prefetch_before_store,
        "tiles_per_core": (columns + free_tile - 1) // free_tile,
        "tail_elements_per_core": per_core % 128,
        "compute_dtype": "float32",
        "activation": "sigmoid",
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
