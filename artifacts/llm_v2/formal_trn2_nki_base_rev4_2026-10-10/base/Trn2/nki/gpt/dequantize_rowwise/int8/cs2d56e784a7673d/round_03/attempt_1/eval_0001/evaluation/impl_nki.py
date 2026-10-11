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
    scales_hbm = state_x.reshape((rows, 1))

    for block in range(rows_per_core // tile_rows):
        r0 = first_row + block * tile_rows

        scale = nl.ndarray(
            (tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        values = nl.ndarray(
            (tile_rows, cols), dtype=nl.float16, buffer=nl.sbuf
        )
        result = nl.ndarray(
            (tile_rows, cols), dtype=nl.float16, buffer=nl.sbuf
        )

        nisa.dma_copy(
            dst=scale,
            src=scales_hbm[r0:r0 + tile_rows, :],
        )

        # Every int8 value is exactly representable in fp16. Staging in
        # fp16 enables the VectorE's wider narrow-float data path without
        # changing any input value or rounding an intermediate product.
        nisa.dma_copy(
            dst=values,
            src=x[r0:r0 + tile_rows, :],
        )

        # VectorE converts operands to fp32 and performs both multiplies
        # before the sole product rounding into the fp16 destination.
        nisa.tensor_scalar(
            dst=result,
            data=values,
            op0=nl.multiply,
            operand0=scale,
            op1=nl.multiply,
            operand1=1.0 / 127.0,
            engine=nisa.engine.vector,
        )

        nisa.dma_copy(
            dst=out[r0:r0 + tile_rows, :],
            src=result,
        )

    return out


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows, cols = x.shape

    _LAST_CONFIG = {
        "kernel": "dequantize_rowwise",
        "lnc": 2,
        "tile_rows": 128,
        "tile_cols": int(cols),
        "rows": int(rows),
        "cols": int(cols),
        "input_staging_dtype": "float16",
        "arithmetic_dtype": "float32",
        "output_dtype": "float16",
    }

    return _dequantize_rowwise[2](x, state_x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
