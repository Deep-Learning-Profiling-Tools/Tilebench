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
    width = row_group * cols
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
                pattern=[[width, 127], [1, halo_width]],
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
                pattern=[[width, 127], [1, halo_width]],
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
                pattern=[[width, 128], [1, halo_width]],
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
    left = 0
    right = 0
    if c0 > 0:
        left = 1
    if c0 + columns < cols:
        right = 1

    width = columns + left + right
    first_col = c0 - left
    halo_width = (row_group + 2) * width

    if r0 == 0:
        nisa.dma_copy(
            dst=halo[0:1, width:halo_width].reshape(
                (1, row_group + 1, width)
            ),
            src=input.ap(
                pattern=[[0, 1], [cols, row_group + 1], [1, width]],
                offset=first_col,
            ),
        )
        nisa.dma_copy(
            dst=halo[1:128, 0:halo_width].reshape(
                (127, row_group + 2, width)
            ),
            src=input.ap(
                pattern=[
                    [row_group * cols, 127],
                    [cols, row_group + 2],
                    [1, width],
                ],
                offset=(row_group - 1) * cols + first_col,
            ),
        )
        nisa.dma_copy(
            dst=halo[0:1, 0:width],
            src=halo[0:1, width:2 * width],
        )
    elif r0 + 128 * row_group == rows:
        nisa.dma_copy(
            dst=halo[0:127, 0:halo_width].reshape(
                (127, row_group + 2, width)
            ),
            src=input.ap(
                pattern=[
                    [row_group * cols, 127],
                    [cols, row_group + 2],
                    [1, width],
                ],
                offset=(r0 - 1) * cols + first_col,
            ),
        )
        nisa.dma_copy(
            dst=halo[127:128, 0:(row_group + 1) * width].reshape(
                (1, row_group + 1, width)
            ),
            src=input.ap(
                pattern=[[0, 1], [cols, row_group + 1], [1, width]],
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
        nisa.dma_copy(
            dst=halo[:, 0:halo_width].reshape(
                (128, row_group + 2, width)
            ),
            src=input.ap(
                pattern=[
                    [row_group * cols, 128],
                    [cols, row_group + 2],
                    [1, width],
                ],
                offset=(r0 - 1) * cols + first_col,
            ),
        )


def _compute_window(
    halo, result, output, r0, row_group, rows, cols, c0, columns
):
    left = 0
    right = 0
    if c0 > 0:
        left = 1
    if c0 + columns < cols:
        right = 1

    width = columns + left + right
    count = row_group * width
    _stencil(halo, result, row_group, width)

    if c0 == 0:
        nisa.tensor_copy(
            dst=result[:, 0:count:width],
            src=halo[:, width:width + count:width],
            engine=nisa.engine.vector,
        )
    if c0 + columns == cols:
        nisa.tensor_copy(
            dst=result[:, width - 1:count:width],
            src=halo[:, 2 * width - 1:width + count:width],
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

    packed = result[:, 0:count].reshape((128, row_group, width))
    nisa.dma_copy(
        dst=output.ap(
            pattern=[
                [row_group * cols, 128],
                [cols, row_group],
                [1, columns],
            ],
            offset=r0 * cols + c0,
        ),
        src=packed[:, :, left:left + columns],
    )


@nki.jit
def _jacobi_column_pipeline(input, rows, cols, row_group, column_tile):
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
    tiles = (cols + column_tile - 1) // column_tile
    first_columns = min(column_tile, cols)
    _load_window_halo(
        input, halo0, r0, row_group, rows, cols, 0, first_columns
    )

    for tile in range(tiles):
        c0 = tile * column_tile
        columns = min(column_tile, cols - c0)

        if tile + 1 < tiles:
            next_c0 = (tile + 1) * column_tile
            next_columns = min(column_tile, cols - next_c0)
            if tile % 2 == 0:
                _load_window_halo(
                    input, halo1, r0, row_group, rows, cols,
                    next_c0, next_columns,
                )
            else:
                _load_window_halo(
                    input, halo0, r0, row_group, rows, cols,
                    next_c0, next_columns,
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
            "rows": rows,
            "columns": cols,
            "arithmetic": "fp16_sequential_add",
            "scale_engine": "vector",
            "boundary": "on_chip_restore_single_store",
        }
        return _jacobi_full_width[2](input, rows, cols, row_group)

    # Two halo buffers and two result buffers use
    # 8 * (row_group + 1) * (column_tile + 2) bytes per partition.
    # Keep the whole core's row range in each column tile to amortize
    # vertical halos, while allowing both input and output DMA overlap.
    maximum_width = sbuf_budget // (8 * (row_group + 1)) - 2
    column_tile = min(cols, (maximum_width // 128) * 128)

    _LAST_CONFIG = {
        "kernel": "jacobi_column_pipeline",
        "lnc": 2,
        "partition_tile": 128,
        "rows_per_partition": row_group,
        "column_tile": column_tile,
        "column_tiles": (cols + column_tile - 1) // column_tile,
        "halo_buffers": 2,
        "result_buffers": 2,
        "rows": rows,
        "columns": cols,
        "arithmetic": "fp16_sequential_add",
        "scale_engine": "vector",
        "boundary": "on_chip_restore_single_store",
    }
    return _jacobi_column_pipeline[2](
        input, rows, cols, row_group, column_tile
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
