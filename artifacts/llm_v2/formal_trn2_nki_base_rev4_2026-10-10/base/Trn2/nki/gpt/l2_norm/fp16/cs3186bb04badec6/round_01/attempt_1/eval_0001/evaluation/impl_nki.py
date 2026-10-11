import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _l2_norm_rows(x, eps, row_tile):
    rows, columns = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    tiles = (rows + row_tile - 1) // row_tile

    for tile in range(pid, tiles, programs):
        row_start = tile * row_tile
        active_rows = min(row_tile, rows - row_start)

        # Promote before squaring and retain the input for the apply pass.
        values = nl.ndarray(
            (active_rows, columns), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=values,
            src=x[row_start:row_start + active_rows, :],
        )

        squares = nl.ndarray(
            (active_rows, columns), dtype=nl.float32, buffer=nl.sbuf
        )
        sumsq = nl.ndarray(
            (active_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=squares,
            op=nl.square,
            data=values,
            reduce_op=nl.add,
            reduce_res=sumsq,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        rstd = nl.ndarray(
            (active_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=rstd,
            op=nl.rsqrt,
            data=sumsq,
            bias=eps,
        )

        result = nl.ndarray(
            (active_rows, columns), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=result,
            data=values,
            op0=nl.multiply,
            operand0=rstd,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out[row_start:row_start + active_rows, :],
            src=result,
        )

    return out


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG

    shape = tuple(x.shape)
    rows = shape[0] * shape[1]
    columns = shape[2]
    row_tile = 128

    x_rows = x.reshape(rows, columns)
    result = _l2_norm_rows[2](x_rows, float(eps), row_tile)

    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "row_tile": row_tile,
        "column_tile": columns,
        "row_tiles": (rows + row_tile - 1) // row_tile,
        "statistics_dtype": "float32",
        "fused_square_sum": True,
        "retain_input": True,
        "apply_engine": "vector",
    }
    return result.reshape(shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
