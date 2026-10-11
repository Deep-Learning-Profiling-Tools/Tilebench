import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _dequantize_rowwise(x, state_x, tile_rows):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=nl.float16, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    rows_per_core = rows // nl.num_programs()
    scales_hbm = state_x.reshape((rows, 1))

    for block in range(rows_per_core // tile_rows):
        r0 = pid * rows_per_core + block * tile_rows

        scale = nl.ndarray(
            (tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        values = nl.ndarray(
            (tile_rows, cols), dtype=nl.int8, buffer=nl.sbuf
        )
        result = nl.ndarray(
            (tile_rows, cols), dtype=nl.float16, buffer=nl.sbuf
        )

        nisa.dma_copy(
            dst=scale,
            src=scales_hbm[r0:r0 + tile_rows, :],
        )
        nisa.dma_copy(
            dst=values,
            src=x[r0:r0 + tile_rows, :],
        )

        # Vector arithmetic converts int8 inputs to fp32. Both multiplies
        # precede the single conversion to the fp16 destination.
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
        "arithmetic_dtype": "float32",
        "output_dtype": "float16",
    }

    return _dequantize_rowwise[2](x, state_x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
