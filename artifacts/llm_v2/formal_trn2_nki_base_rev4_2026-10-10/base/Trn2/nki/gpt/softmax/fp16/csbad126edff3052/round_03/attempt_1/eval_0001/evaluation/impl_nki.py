import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _softmax_resident_pipeline(x, row_tile):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = rows // programs
    row_base = pid * rows_per_program
    tile_count = rows_per_program // row_tile

    # Three resident fp32 tiles allow input preparation, exponentiation,
    # and normalization to operate on different row groups.
    values = nl.ndarray(
        (row_tile, 3, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    output_tiles = nl.ndarray(
        (row_tile, 2, cols), dtype=x.dtype, buffer=nl.sbuf
    )
    negative_max = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    denominator = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    inverse = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )

    # Upcast directly into the resident tile before any computation.
    nisa.dma_copy(
        dst=values[:, 0, :],
        src=x[row_base:row_base + row_tile, :],
    )
    nisa.tensor_reduce(
        dst=negative_max[:, 0, :],
        op=nl.maximum,
        data=values[:, 0, :],
        axis=1,
        negate=True,
    )

    for tile in range(tile_count + 1):
        if tile < tile_count:
            slot = tile % 3
            next_slot = (tile + 1) % 3

            if tile + 1 < tile_count:
                next_r0 = row_base + (tile + 1) * row_tile
                nisa.dma_copy(
                    dst=values[:, next_slot, :],
                    src=x[next_r0:next_r0 + row_tile, :],
                )

            # Each whole row is resident. Its exact maximum is already
            # available, and its fp32 exponentials are retained in place.
            nisa.activation(
                dst=values[:, slot, :],
                op=nl.exp,
                data=values[:, slot, :],
                bias=negative_max[:, slot, :],
                reduce_op=nl.add,
                reduce_res=denominator[:, slot, :],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

            if tile + 1 < tile_count:
                nisa.tensor_reduce(
                    dst=negative_max[:, next_slot, :],
                    op=nl.maximum,
                    data=values[:, next_slot, :],
                    axis=1,
                    negate=True,
                )

        # Delay normalization by one tile so the current exponential
        # instruction does not depend on the previous tile's reciprocal.
        if tile > 0:
            completed = tile - 1
            previous_slot = completed % 3
            output_slot = completed % 2
            r0 = row_base + completed * row_tile

            nisa.reciprocal(
                dst=inverse[:, previous_slot, :],
                data=denominator[:, previous_slot, :],
            )
            nisa.tensor_scalar(
                dst=output_tiles[:, output_slot, :],
                data=values[:, previous_slot, :],
                op0=nl.multiply,
                operand0=inverse[:, previous_slot, :],
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(
                dst=out[r0:r0 + row_tile, :],
                src=output_tiles[:, output_slot, :],
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
        "rows_per_core": int(rows) // 2,
        "resident_slots": 3,
        "output_slots": 2,
        "normalization_delay": 1,
        "direct_fp32_load": True,
        "statistics_dtype": "float32",
        "fused_exp_sum": True,
        "inplace_exponentials": True,
        "prefetch_next_tile": True,
        "normalization_output_dtype": str(x.dtype),
    }
    return _softmax_resident_pipeline[2](x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
