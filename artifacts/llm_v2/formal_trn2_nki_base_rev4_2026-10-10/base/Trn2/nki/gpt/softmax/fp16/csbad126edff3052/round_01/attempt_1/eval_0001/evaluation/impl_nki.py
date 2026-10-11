import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _softmax_resident(x, row_tile):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = rows // programs
    tiles_per_program = rows_per_program // row_tile

    # Two staging slots allow adjacent row tiles to overlap without
    # allocating storage for every unrolled iteration.
    input0 = nl.ndarray(
        (row_tile, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    input1 = nl.ndarray(
        (row_tile, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    exp0 = nl.ndarray(
        (row_tile, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    exp1 = nl.ndarray(
        (row_tile, cols), dtype=nl.float32, buffer=nl.sbuf
    )

    for tile in range(tiles_per_program):
        r0 = pid * rows_per_program + tile * row_tile

        if tile % 2 == 0:
            values = input0
            exponentials = exp0
        else:
            values = input1
            exponentials = exp1

        # The entire row is resident; upcast before any arithmetic.
        nisa.dma_copy(
            dst=values,
            src=x[r0:r0 + row_tile, :],
        )

        negative_max = nl.ndarray(
            (row_tile, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        denominator = nl.ndarray(
            (row_tile, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        inverse = nl.ndarray(
            (row_tile, 1), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.tensor_reduce(
            dst=negative_max,
            op=nl.maximum,
            data=values,
            axis=1,
            negate=True,
        )

        # Stable fp32 exponentials and their fp32 row sums.
        nisa.activation(
            dst=exponentials,
            op=nl.exp,
            data=values,
            bias=negative_max,
            reduce_op=nl.add,
            reduce_res=denominator,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        nisa.reciprocal(dst=inverse, data=denominator)
        nisa.tensor_scalar(
            dst=exponentials,
            data=exponentials,
            op0=nl.multiply,
            operand0=inverse,
            engine=nisa.engine.vector,
        )

        # The only downcast is the final output store.
        nisa.dma_copy(
            dst=out[r0:r0 + row_tile, :],
            src=exponentials,
        )

    return out


def run(x):
    global _LAST_CONFIG
    rows, cols = x.shape
    _LAST_CONFIG = {
        "algorithm": "resident_row_softmax",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": int(cols),
        "staging_slots": 2,
        "rows_per_core": int(rows) // 2,
        "statistics_dtype": "float32",
        "fused_exp_sum": True,
    }
    return _softmax_resident[2](x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
