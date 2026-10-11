import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _row_sumsq(values, active_rows, columns, use_vector):
    squares = nl.ndarray(
        (active_rows, columns), dtype=nl.float32, buffer=nl.sbuf
    )
    sumsq = nl.ndarray(
        (active_rows, 1), dtype=nl.float32, buffer=nl.sbuf
    )

    if use_vector:
        # VectorE promotes both operands to fp32 before multiplication.
        nisa.tensor_tensor(
            dst=squares,
            data1=values[0:active_rows, :],
            data2=values[0:active_rows, :],
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        nisa.tensor_reduce(
            dst=sumsq,
            data=squares,
            op=nl.add,
            axis=1,
        )
    else:
        # ScalarE squares and accumulates in fp32.
        nisa.activation(
            dst=squares,
            op=nl.square,
            data=values[0:active_rows, :],
            reduce_op=nl.add,
            reduce_res=sumsq,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

    return sumsq


@nki.jit
def _l2_norm_pipelined(x, eps, row_tile, vector_period):
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
        sumsq = _row_sumsq(values, first_count, columns, False)

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
            current_vector = False
            if vector_period > 0:
                current_vector = index % vector_period == 1

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

            if index + 1 < local_tiles:
                next_row = (tile + programs) * row_tile
                next_count = min(row_tile, rows - next_row)
                next_vector = False
                if vector_period > 0:
                    next_vector = (index + 1) % vector_period == 1

                if current_vector:
                    # Queue independent ScalarE statistics before waiting
                    # for the current VectorE reduction.
                    next_sumsq = _row_sumsq(
                        next_values, next_count, columns, next_vector
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
                if not current_vector:
                    # Keep the current apply ahead of the next VectorE
                    # statistics, while allowing ScalarE to run ahead.
                    next_sumsq = _row_sumsq(
                        next_values, next_count, columns, next_vector
                    )
                values = next_values
                sumsq = next_sumsq

            if index + 2 < local_tiles:
                next_values = future_values

    return out


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG

    shape = tuple(x.shape)
    rows = shape[0] * shape[1]
    columns = shape[2]
    row_tile = 128

    # Long rows amortize the extra VectorE reduction instruction.
    vector_period = 3 if columns >= 4096 else 0

    x_rows = x.reshape(rows, columns)
    result = _l2_norm_pipelined[2](
        x_rows, float(eps), row_tile, vector_period
    )

    _LAST_CONFIG = {
        "lnc": 2,
        "launches": 1,
        "row_tile": row_tile,
        "column_tile": columns,
        "row_tiles": (rows + row_tile - 1) // row_tile,
        "prefetch_tiles": 2,
        "vector_statistics_period": vector_period,
        "statistics_pipeline": "independent_engine_lookahead",
        "arithmetic_scratch": "independent_per_tile",
        "retained_input_dtype": "float16",
        "statistics_dtype": "float32",
        "arithmetic_dtype": "float32",
        "retain_input": True,
        "apply_engine": "vector",
    }
    return result.reshape(shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
