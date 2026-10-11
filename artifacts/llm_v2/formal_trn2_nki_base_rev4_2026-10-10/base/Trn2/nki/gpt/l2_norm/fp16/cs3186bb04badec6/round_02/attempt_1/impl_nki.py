import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _l2_norm_pipelined(x, eps, row_tile):
    rows, columns = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    tiles = (rows + row_tile - 1) // row_tile

    # Alternate input buffers so the next load precedes the current store
    # in the DMA instruction stream. All arithmetic scratch is reused.
    values0 = nl.ndarray(
        (row_tile, columns), dtype=nl.float32, buffer=nl.sbuf
    )
    values1 = nl.ndarray(
        (row_tile, columns), dtype=nl.float32, buffer=nl.sbuf
    )
    squares = nl.ndarray(
        (row_tile, columns), dtype=nl.float32, buffer=nl.sbuf
    )
    sumsq = nl.ndarray((row_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd = nl.ndarray((row_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(
        (row_tile, columns), dtype=x.dtype, buffer=nl.sbuf
    )

    if pid < tiles:
        first_row = pid * row_tile
        first_count = min(row_tile, rows - first_row)
        nisa.dma_copy(
            dst=values0[0:first_count, :],
            src=x[first_row:first_row + first_count, :],
        )

        local_tiles = (tiles - pid + programs - 1) // programs
        for index in range(local_tiles):
            tile = pid + index * programs
            row_start = tile * row_tile
            active_rows = min(row_tile, rows - row_start)

            if index % 2 == 0:
                values = values0
                next_values = values1
            else:
                values = values1
                next_values = values0

            next_tile = tile + programs
            if next_tile < tiles:
                next_row = next_tile * row_tile
                next_count = min(row_tile, rows - next_row)
                nisa.dma_copy(
                    dst=next_values[0:next_count, :],
                    src=x[next_row:next_row + next_count, :],
                )

            # DMA promotion happens before the fp32 square and reduction.
            nisa.activation(
                dst=squares[0:active_rows, :],
                op=nl.square,
                data=values[0:active_rows, :],
                reduce_op=nl.add,
                reduce_res=sumsq[0:active_rows, :],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            nisa.activation(
                dst=rstd[0:active_rows, :],
                op=nl.rsqrt,
                data=sumsq[0:active_rows, :],
                bias=eps,
            )

            # Multiply in fp32 and cast only the final scaled values.
            nisa.tensor_scalar(
                dst=result[0:active_rows, :],
                data=values[0:active_rows, :],
                op0=nl.multiply,
                operand0=rstd[0:active_rows, :],
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=out[row_start:row_start + active_rows, :],
                src=result[0:active_rows, :],
            )

    return out


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG

    shape = tuple(x.shape)
    rows = shape[0] * shape[1]
    columns = shape[2]
    row_tile = 128

    x_rows = x.reshape(rows, columns)
    result = _l2_norm_pipelined[2](x_rows, float(eps), row_tile)

    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "row_tile": row_tile,
        "column_tile": columns,
        "row_tiles": (rows + row_tile - 1) // row_tile,
        "input_buffers": 2,
        "prefetch_tiles": 1,
        "reuse_arithmetic_scratch": True,
        "statistics_dtype": "float32",
        "fused_square_sum": True,
        "retain_input": True,
        "apply_engine": "vector",
    }
    return result.reshape(shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
