import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _dequantize_rowwise(x, state_x, tile_rows):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=nl.float16, buffer=nl.shared_hbm)

    rows_per_core = rows // nl.num_programs()
    first_row = nl.program_id(0) * rows_per_core
    blocks = rows_per_core // tile_rows
    scales_hbm = state_x.reshape((rows, 1))

    current_values = nl.ndarray(
        (tile_rows, cols), dtype=nl.int8, buffer=nl.sbuf
    )
    current_scale = nl.ndarray(
        (tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=current_scale,
        src=scales_hbm[first_row:first_row + tile_rows, :],
    )
    nisa.dma_copy(
        dst=current_values,
        src=x[first_row:first_row + tile_rows, :],
    )

    next_values = nl.ndarray(
        (tile_rows, cols), dtype=nl.int8, buffer=nl.sbuf
    )
    next_scale = nl.ndarray(
        (tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=next_scale,
        src=scales_hbm[
            first_row + tile_rows:first_row + 2 * tile_rows, :
        ],
    )
    nisa.dma_copy(
        dst=next_values,
        src=x[first_row + tile_rows:first_row + 2 * tile_rows, :],
    )

    for block in range(blocks):
        # Queue future reads before the current store so DMA can overlap
        # arithmetic without waiting for that store's producer.
        if block + 2 < blocks:
            future_row = first_row + (block + 2) * tile_rows
            future_values = nl.ndarray(
                (tile_rows, cols), dtype=nl.int8, buffer=nl.sbuf
            )
            future_scale = nl.ndarray(
                (tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=future_scale,
                src=scales_hbm[future_row:future_row + tile_rows, :],
            )
            nisa.dma_copy(
                dst=future_values,
                src=x[future_row:future_row + tile_rows, :],
            )

        result = nl.ndarray(
            (tile_rows, cols), dtype=nl.float16, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=result,
            data=current_values,
            op0=nl.multiply,
            operand0=current_scale,
            op1=nl.multiply,
            operand1=1.0 / 127.0,
            engine=nisa.engine.vector,
        )

        output_row = first_row + block * tile_rows
        nisa.dma_copy(
            dst=out[output_row:output_row + tile_rows, :],
            src=result,
        )

        if block + 1 < blocks:
            current_values = next_values
            current_scale = next_scale
        if block + 2 < blocks:
            next_values = future_values
            next_scale = future_scale

    return out


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows, cols = x.shape

    _LAST_CONFIG = {
        "kernel": "dequantize_rowwise",
        "lnc": 2,
        "tile_rows": 128,
        "tile_cols": int(cols),
        "prefetch_distance": 2,
        "rows": int(rows),
        "cols": int(cols),
        "arithmetic_dtype": "float32",
        "output_dtype": "float16",
    }

    return _dequantize_rowwise[2](x, state_x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
