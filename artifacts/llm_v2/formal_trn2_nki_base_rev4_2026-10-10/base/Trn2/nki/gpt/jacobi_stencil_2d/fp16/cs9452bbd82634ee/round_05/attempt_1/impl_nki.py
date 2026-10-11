import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _stencil(halo, result, row_group, width):
    count = row_group * width
    nisa.tensor_tensor(
        dst=result[:, 0:count],
        data1=halo[:, 0:count],
        data2=halo[:, 2 * width:2 * width + count],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=result[:, 0:count],
        data1=result[:, 0:count],
        data2=halo[:, width - 1:width - 1 + count],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=result[:, 0:count],
        data1=result[:, 0:count],
        data2=halo[:, width + 1:width + 1 + count],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_scalar(
        dst=result[:, 0:count],
        data=result[:, 0:count],
        op0=nl.multiply,
        operand0=0.25,
        engine=nisa.engine.vector,
    )


def _load_full_halo(input, halo, r0, row_group, rows, cols):
    count = row_group * cols
    halo_width = (row_group + 2) * cols

    if r0 == 0:
        nisa.dma_copy(
            dst=halo[0:1, cols:halo_width],
            src=input[0:row_group + 1, :].reshape(
                (1, (row_group + 1) * cols)
            ),
        )
        nisa.dma_copy(
            dst=halo[1:128, :],
            src=input.ap(
                pattern=[[count, 127], [1, halo_width]],
                offset=(row_group - 1) * cols,
            ),
        )
        nisa.dma_copy(
            dst=halo[0:1, 0:cols],
            src=halo[0:1, cols:2 * cols],
        )
    elif r0 + 128 * row_group == rows:
        nisa.dma_copy(
            dst=halo[0:127, :],
            src=input.ap(
                pattern=[[count, 127], [1, halo_width]],
                offset=(r0 - 1) * cols,
            ),
        )
        nisa.dma_copy(
            dst=halo[127:128, 0:(row_group + 1) * cols],
            src=input[rows - row_group - 1:rows, :].reshape(
                (1, (row_group + 1) * cols)
            ),
        )
        nisa.dma_copy(
            dst=halo[127:128, (row_group + 1) * cols:halo_width],
            src=halo[127:128, row_group * cols:(row_group + 1) * cols],
        )
    else:
        nisa.dma_copy(
            dst=halo,
            src=input.ap(
                pattern=[[count, 128], [1, halo_width]],
                offset=(r0 - 1) * cols,
            ),
        )


@nki.jit
def _jacobi_full_width(input, rows, cols, row_group):
    output = nl.ndarray(
        (rows, cols), dtype=input.dtype, buffer=nl.shared_hbm
    )
    halo = nl.ndarray(
        (128, (row_group + 2) * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    result = nl.ndarray(
        (128, row_group * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )

    r0 = nl.program_id(0) * (rows // nl.num_programs())
    count = row_group * cols
    _load_full_halo(input, halo, r0, row_group, rows, cols)
    _stencil(halo, result, row_group, cols)

    nisa.tensor_copy(
        dst=result[:, 0:count:cols],
        src=halo[:, cols:cols + count:cols],
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=result[:, cols - 1:count:cols],
        src=halo[:, 2 * cols - 1:cols + count:cols],
        engine=nisa.engine.vector,
    )

    if r0 == 0:
        nisa.dma_copy(
            dst=result[0:1, 0:cols],
            src=halo[0:1, cols:2 * cols],
        )
    if r0 + 128 * row_group == rows:
        nisa.dma_copy(
            dst=result[127:128, count - cols:count],
            src=halo[127:128, count:count + cols],
        )

    nisa.dma_copy(
        dst=output.ap(
            pattern=[[count, 128], [1, count]],
            offset=r0 * cols,
        ),
        src=result,
    )
    return output


def _load_window_halo(
    input, halo, r0, row_group, rows, cols, c0, columns
):
    # Both horizontal halo slots are retained at grid edges as well.
    # Consequently, every packed row has an even, four-byte-aligned stride.
    width = columns + 2
    halo_width = (row_group + 2) * width
    first_col = c0 - 1
    load_start = 0
    loaded_width = width

    if c0 == 0:
        first_col = 0
        load_start = 1
        loaded_width = loaded_width - 1
    if c0 + columns == cols:
        loaded_width = loaded_width - 1

    if c0 == 0 or c0 + columns == cols:
        nisa.memset(
            dst=halo[:, 0:halo_width],
            value=0.0,
            engine=nisa.engine.vector,
        )

    if r0 == 0:
        first = halo[0:1, width:halo_width].reshape(
            (1, row_group + 1, width)
        )
        nisa.dma_copy(
            dst=first[:, :, load_start:load_start + loaded_width],
            src=input.ap(
                pattern=[
                    [0, 1],
                    [cols, row_group + 1],
                    [1, loaded_width],
                ],
                offset=first_col,
            ),
        )
        remaining = halo[1:128, 0:halo_width].reshape(
            (127, row_group + 2, width)
        )
        nisa.dma_copy(
            dst=remaining[:, :, load_start:load_start + loaded_width],
            src=input.ap(
                pattern=[
                    [row_group * cols, 127],
                    [cols, row_group + 2],
                    [1, loaded_width],
                ],
                offset=(row_group - 1) * cols + first_col,
            ),
        )
        nisa.dma_copy(
            dst=halo[0:1, 0:width],
            src=halo[0:1, width:2 * width],
        )
    elif r0 + 128 * row_group == rows:
        preceding = halo[0:127, 0:halo_width].reshape(
            (127, row_group + 2, width)
        )
        nisa.dma_copy(
            dst=preceding[:, :, load_start:load_start + loaded_width],
            src=input.ap(
                pattern=[
                    [row_group * cols, 127],
                    [cols, row_group + 2],
                    [1, loaded_width],
                ],
                offset=(r0 - 1) * cols + first_col,
            ),
        )
        last = halo[
            127:128, 0:(row_group + 1) * width
        ].reshape((1, row_group + 1, width))
        nisa.dma_copy(
            dst=last[:, :, load_start:load_start + loaded_width],
            src=input.ap(
                pattern=[
                    [0, 1],
                    [cols, row_group + 1],
                    [1, loaded_width],
                ],
                offset=(rows - row_group - 1) * cols + first_col,
            ),
        )
        nisa.dma_copy(
            dst=halo[
                127:128, (row_group + 1) * width:halo_width
            ],
            src=halo[
                127:128, row_group * width:(row_group + 1) * width
            ],
        )
    else:
        packed = halo[:, 0:halo_width].reshape(
            (128, row_group + 2, width)
        )
        nisa.dma_copy(
            dst=packed[:, :, load_start:load_start + loaded_width],
            src=input.ap(
                pattern=[
                    [row_group * cols, 128],
                    [cols, row_group + 2],
                    [1, loaded_width],
                ],
                offset=(r0 - 1) * cols + first_col,
            ),
        )


def _compute_window(
    halo, result, output, r0, row_group, rows, cols, c0, columns
):
    width = columns + 2
    count = row_group * width
    _stencil(halo, result, row_group, width)

    if c0 == 0:
        nisa.tensor_copy(
            dst=result[:, 1:count:width],
            src=halo[:, width + 1:width + count:width],
            engine=nisa.engine.vector,
        )
    if c0 + columns == cols:
        nisa.tensor_copy(
            dst=result[:, width - 2:count:width],
            src=halo[:, 2 * width - 2:width + count:width],
            engine=nisa.engine.vector,
        )

    if r0 == 0:
        nisa.dma_copy(
            dst=result[0:1, 0:width],
            src=halo[0:1, width:2 * width],
        )
    if r0 + 128 * row_group == rows:
        nisa.dma_copy(
            dst=result[127:128, count - width:count],
            src=halo[127:128, count:count + width],
        )

    packed = result[:, 0:count].reshape(
        (128, row_group, width)
    )
    nisa.dma_copy(
        dst=output.ap(
            pattern=[
                [row_group * cols, 128],
                [cols, row_group],
                [1, columns],
            ],
            offset=r0 * cols + c0,
        ),
        src=packed[:, :, 1:columns + 1],
    )


@nki.jit
def _jacobi_column_pipeline(
    input, rows, cols, row_group, column_tile, column_tiles
):
    output = nl.ndarray(
        (rows, cols), dtype=input.dtype, buffer=nl.shared_hbm
    )
    halo0 = nl.ndarray(
        (128, (row_group + 2) * (column_tile + 2)),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    halo1 = nl.ndarray(
        (128, (row_group + 2) * (column_tile + 2)),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    result0 = nl.ndarray(
        (128, row_group * (column_tile + 2)),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    result1 = nl.ndarray(
        (128, row_group * (column_tile + 2)),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )

    r0 = nl.program_id(0) * (rows // nl.num_programs())
    column_units = cols // 128
    base_units = column_units // column_tiles
    larger_tiles = column_units % column_tiles

    first_columns = base_units * 128
    if larger_tiles > 0:
        first_columns = first_columns + 128

    _load_window_halo(
        input, halo0, r0, row_group,
        rows, cols, 0, first_columns,
    )

    for tile in range(column_tiles):
        c0 = 128 * (
            tile * base_units + min(tile, larger_tiles)
        )
        columns = base_units * 128
        if tile < larger_tiles:
            columns = columns + 128

        if tile + 1 < column_tiles:
            next_tile = tile + 1
            next_c0 = 128 * (
                next_tile * base_units
                + min(next_tile, larger_tiles)
            )
            next_columns = base_units * 128
            if next_tile < larger_tiles:
                next_columns = next_columns + 128

            if tile % 2 == 0:
                _load_window_halo(
                    input, halo1, r0, row_group,
                    rows, cols, next_c0, next_columns,
                )
            else:
                _load_window_halo(
                    input, halo0, r0, row_group,
                    rows, cols, next_c0, next_columns,
                )

        if tile % 2 == 0:
            _compute_window(
                halo0, result0, output, r0, row_group,
                rows, cols, c0, columns,
            )
        else:
            _compute_window(
                halo1, result1, output, r0, row_group,
                rows, cols, c0, columns,
            )

    return output


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG
    rows = int(rows)
    cols = int(cols)
    row_group = rows // 256
    sbuf_budget = 212984

    full_width_bytes = 4 * (row_group + 1) * cols
    if full_width_bytes <= sbuf_budget:
        _LAST_CONFIG = {
            "kernel": "jacobi_full_width",
            "lnc": 2,
            "partition_tile": 128,
            "rows_per_partition": row_group,
            "column_tile": cols,
            "column_tiles": 1,
            "halo_buffers": 1,
            "result_buffers": 1,
            "sbuf_bytes_per_partition": full_width_bytes,
            "rows": rows,
            "columns": cols,
            "arithmetic": "fp16_sequential_add",
            "scale_engine": "vector",
            "boundary": "on_chip_restore_single_store",
        }
        return _jacobi_full_width[2](
            input, rows, cols, row_group
        )

    # Two halos and two results consume 8*(g+1)*width bytes.
    # Reserve an additional halo-sized allowance for allocation and
    # strided-DMA lowering, rather than filling SBUF to its limit.
    maximum_width = sbuf_budget // (10 * row_group + 12)
    maximum_units = max(1, (maximum_width - 2) // 128)
    column_units = cols // 128
    column_tiles = (
        column_units + maximum_units - 1
    ) // maximum_units
    maximum_balanced_units = (
        column_units + column_tiles - 1
    ) // column_tiles
    column_tile = maximum_balanced_units * 128

    _LAST_CONFIG = {
        "kernel": "jacobi_balanced_column_pipeline",
        "lnc": 2,
        "partition_tile": 128,
        "rows_per_partition": row_group,
        "column_tile": column_tile,
        "column_tile_min": (column_units // column_tiles) * 128,
        "column_tiles": column_tiles,
        "halo_buffers": 2,
        "result_buffers": 2,
        "sbuf_bytes_per_partition": (
            8 * (row_group + 1) * (column_tile + 2)
        ),
        "rows": rows,
        "columns": cols,
        "arithmetic": "fp16_sequential_add",
        "scale_engine": "vector",
        "boundary": "on_chip_restore_single_store",
        "balanced_column_tiles": True,
        "horizontal_halo_slots": 2,
    }

    return _jacobi_column_pipeline[2](
        input, rows, cols, row_group,
        column_tile, column_tiles,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
