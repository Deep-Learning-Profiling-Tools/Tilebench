import torch
import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _load_tile(X, inputs, slot, base, tile, free_tile, per_core):
    start = base + tile * 128 * free_tile
    count = min(128 * free_tile, per_core - tile * 128 * free_tile)
    width = count // 128
    remainder = count % 128

    if width > 0:
        nisa.dma_copy(
            dst=inputs[:, slot, :width],
            src=X[start:start + 128 * width].reshape((128, width)),
        )

    if remainder > 0:
        tail_start = start + 128 * width
        nisa.dma_copy(
            dst=inputs[:remainder, slot, width:width + 1],
            src=X[tail_start:tail_start + remainder].reshape(
                (remainder, 1)
            ),
        )


@nki.jit
def _sigmoid_kernel(
    X, N, free_tile, compute_tile, io_buffers, compute_buffers, dma_upcast
):
    out = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm)

    per_core = N // 2
    base = nl.program_id(0) * per_core
    tile_elements = 128 * free_tile
    tiles = (per_core + tile_elements - 1) // tile_elements

    if dma_upcast:
        load_dtype = nl.float32
    else:
        load_dtype = X.dtype

    inputs = nl.ndarray(
        (128, io_buffers, free_tile),
        dtype=load_dtype,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (128, io_buffers, free_tile),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )

    if not dma_upcast:
        converted = nl.ndarray(
            (128, compute_buffers, compute_tile),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

    for tile in range(min(tiles, io_buffers)):
        _load_tile(
            X, inputs, tile, base, tile, free_tile, per_core
        )

    for tile in range(tiles):
        slot = tile % io_buffers
        start = base + tile * tile_elements
        count = min(tile_elements, per_core - tile * tile_elements)
        width = count // 128
        remainder = count % 128
        compute_width = width
        if remainder > 0:
            compute_width = width + 1

        if dma_upcast:
            nisa.activation(
                dst=outputs[:, slot, :compute_width],
                op=nl.sigmoid,
                data=inputs[:, slot, :compute_width],
            )
        else:
            for chunk in range(
                (compute_width + compute_tile - 1) // compute_tile
            ):
                c0 = chunk * compute_tile
                chunk_width = min(compute_tile, compute_width - c0)
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

        if width > 0:
            nisa.dma_copy(
                dst=out[start:start + 128 * width].reshape((128, width)),
                src=outputs[:, slot, :width],
            )

        if remainder > 0:
            tail_start = start + 128 * width
            nisa.dma_copy(
                dst=out[tail_start:tail_start + remainder].reshape(
                    (remainder, 1)
                ),
                src=outputs[:remainder, slot, width:width + 1],
            )

        next_tile = tile + io_buffers
        if next_tile < tiles:
            _load_tile(
                X, inputs, slot, base, next_tile, free_tile, per_core
            )

    return out


def run(X: torch.Tensor, N: int, **kwargs):
    global _last_config

    large_blocks = N >= 2 * 4 * 128 * 8192
    if large_blocks:
        free_tile = 8192
        compute_tile = 4096
        io_buffers = 3
        compute_buffers = 4
        dma_upcast = False
    else:
        free_tile = 4096
        compute_tile = 4096
        io_buffers = 4
        compute_buffers = 0
        dma_upcast = True

    result = _sigmoid_kernel[2](
        X, N, free_tile, compute_tile, io_buffers,
        compute_buffers, dma_upcast,
    )

    per_core = N // 2
    tile_elements = 128 * free_tile
    _last_config = {
        "lnc": 2,
        "partitions": 128,
        "free_tile": free_tile,
        "compute_tile": compute_tile,
        "load_buffers": io_buffers,
        "output_buffers": io_buffers,
        "compute_buffers": compute_buffers,
        "upcast": "dma" if dma_upcast else "vector_copy",
        "prefetch_before_store": False,
        "tiles_per_core": (
            per_core + tile_elements - 1
        ) // tile_elements,
        "tail_elements_per_core": per_core % 128,
        "tail_mapping": "fused_last_column",
        "compute_dtype": "float32",
        "activation": "sigmoid",
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
