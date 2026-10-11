import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _l2_norm_prefetched(x, eps, row_tile):
    rows, columns = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    tiles = (rows + row_tile - 1) // row_tile
    local_tiles = (tiles - pid + programs - 1) // programs

    if local_tiles > 0:
        first_row = pid * row_tile
        first_count = min(row_tile, rows - first_row)
        values = nl.ndarray(
            (row_tile, columns), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=values[0:first_count, :],
            src=x[first_row:first_row + first_count, :],
        )

        if local_tiles > 1:
            second_row = (pid + programs) * row_tile
            second_count = min(row_tile, rows - second_row)
            next_values = nl.ndarray(
                (row_tile, columns), dtype=x.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=next_values[0:second_count, :],
                src=x[second_row:second_row + second_count, :],
            )

        for index in range(local_tiles):
            tile = pid + index * programs
            row_start = tile * row_tile
            active_rows = min(row_tile, rows - row_start)

            if index + 2 < local_tiles:
                future_row = (tile + 2 * programs) * row_tile
                future_count = min(row_tile, rows - future_row)
                future_values = nl.ndarray(
                    (row_tile, columns), dtype=x.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=future_values[0:future_count, :],
                    src=x[future_row:future_row + future_count, :],
                )

            squares = nl.ndarray(
                (active_rows, columns), dtype=nl.float32, buffer=nl.sbuf
            )
            sumsq = nl.ndarray(
                (active_rows, 1), dtype=nl.float32, buffer=nl.sbuf
            )

            # ScalarE promotes its input to fp32 before arithmetic.
            # Both the squared values and the fused sum remain fp32.
            nisa.activation(
                dst=squares,
                op=nl.square,
                data=values[0:active_rows, :],
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
            # VectorE multiplies in fp32 and casts only the final result.
            # Retaining fp16 inputs also enables its wider fp16 data path.
            nisa.tensor_scalar(
                dst=result,
                data=values[0:active_rows, :],
                op0=nl.multiply,
                operand0=rstd,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=out[row_start:row_start + active_rows, :],
                src=result,
            )

            if index + 1 < local_tiles:
                values = next_values
            if index + 2 < local_tiles:
                next_values = future_values

    return out


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG

    shape = tuple(x.shape)
    rows = shape[0] * shape[1]
    columns = shape[2]
    row_tile = 128

    x_rows = x.reshape(rows, columns)
    result = _l2_norm_prefetched[2](x_rows, float(eps), row_tile)

    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "row_tile": row_tile,
        "column_tile": columns,
        "row_tiles": (rows + row_tile - 1) // row_tile,
        "prefetch_tiles": 2,
        "arithmetic_scratch": "independent_per_tile",
        "retained_input_dtype": "float16",
        "statistics_dtype": "float32",
        "arithmetic_dtype": "float32",
        "fused_square_sum": True,
        "retain_input": True,
        "apply_engine": "vector",
    }
    return result.reshape(shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
