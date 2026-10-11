import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _softmax_resident_pipeline(
    x, row_tile, normalization_width, scalar_columns
):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = rows // programs
    row_base = pid * rows_per_program
    tile_count = rows_per_program // row_tile
    normalization_chunks = (
        cols + normalization_width - 1
    ) // normalization_width

    values = nl.ndarray(
        (row_tile, 4, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    output_tiles = nl.ndarray(
        (row_tile, 2, normalization_width),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    negative_max = nl.ndarray(
        (row_tile, 4, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    denominator = nl.ndarray(
        (row_tile, 4, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    inverse = nl.ndarray(
        (row_tile, 4, 1), dtype=nl.float32, buffer=nl.sbuf
    )

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

    if tile_count > 1:
        nisa.dma_copy(
            dst=values[:, 1, :],
            src=x[row_base + row_tile:row_base + 2 * row_tile, :],
        )

    for tile in range(tile_count + 1):
        if tile < tile_count:
            slot = tile % 4

            if tile + 2 < tile_count:
                prefetch_slot = (tile + 2) % 4
                prefetch_r0 = row_base + (tile + 2) * row_tile
                nisa.dma_copy(
                    dst=values[:, prefetch_slot, :],
                    src=x[prefetch_r0:prefetch_r0 + row_tile, :],
                )

            # Every row is resident in fp32. Retain its exponentials
            # while obtaining the fp32 denominator.
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
                next_slot = (tile + 1) % 4
                nisa.tensor_reduce(
                    dst=negative_max[:, next_slot, :],
                    op=nl.maximum,
                    data=values[:, next_slot, :],
                    axis=1,
                    negate=True,
                )

        if tile > 0:
            completed = tile - 1
            previous_slot = completed % 4
            r0 = row_base + completed * row_tile

            nisa.reciprocal(
                dst=inverse[:, previous_slot, :],
                data=denominator[:, previous_slot, :],
            )

            for chunk in range(normalization_chunks):
                c0 = chunk * normalization_width
                width = min(normalization_width, cols - c0)
                vector_columns = width - scalar_columns
                output_slot = (
                    completed * normalization_chunks + chunk
                ) % 2

                nisa.tensor_scalar(
                    dst=output_tiles[
                        :, output_slot, :vector_columns
                    ],
                    data=values[
                        :, previous_slot, c0:c0 + vector_columns
                    ],
                    op0=nl.multiply,
                    operand0=inverse[:, previous_slot, :],
                    engine=nisa.engine.vector,
                )

                if scalar_columns > 0:
                    # Use a small independent ScalarE stripe to reduce
                    # VectorE normalization pressure. Both engines do
                    # fp32 multiplication and cast only the final result.
                    nisa.tensor_scalar(
                        dst=output_tiles[
                            :, output_slot, vector_columns:width
                        ],
                        data=values[
                            :,
                            previous_slot,
                            c0 + vector_columns:c0 + width,
                        ],
                        op0=nl.multiply,
                        operand0=inverse[:, previous_slot, :],
                        engine=nisa.engine.scalar,
                    )

                nisa.dma_copy(
                    dst=out[r0:r0 + row_tile, c0:c0 + width],
                    src=output_tiles[:, output_slot, :width],
                )

    return out


def run(x):
    global _LAST_CONFIG
    rows, cols = x.shape

    # Prefer contiguous whole-row stores when four fp32 resident slots,
    # two fp16 output slots, and one fp16 DMA workspace fit with margin.
    full_row_bytes_per_partition = 22 * cols + 48
    if full_row_bytes_per_partition <= 208888:
        normalization_width = cols
        if cols >= 4096:
            scalar_columns = ((cols + 2048) // 4096) * 512
        else:
            scalar_columns = 0
    else:
        # Bound the output footprint without reducing input lookahead.
        chunks = (cols + 4095) // 4096
        normalization_width = (
            (cols + chunks * 512 - 1) // (chunks * 512)
        ) * 512
        scalar_columns = 0

    normalization_chunks = (
        cols + normalization_width - 1
    ) // normalization_width

    _LAST_CONFIG = {
        "algorithm": "resident_row_softmax",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": int(cols),
        "rows_per_core": int(rows) // 2,
        "resident_slots": 4,
        "output_slots": 2,
        "input_lookahead": 2,
        "normalization_delay": 1,
        "normalization_width": int(normalization_width),
        "normalization_chunks": int(normalization_chunks),
        "normalization_vector_width": int(
            normalization_width - scalar_columns
        ),
        "normalization_scalar_columns": int(scalar_columns),
        "contiguous_row_stores": normalization_chunks == 1,
        "direct_fp32_load": True,
        "statistics_dtype": "float32",
        "fused_exp_sum": True,
        "inplace_exponentials": True,
        "normalization_output_dtype": str(x.dtype),
    }

    return _softmax_resident_pipeline[2](
        x, 128, normalization_width, scalar_columns
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
