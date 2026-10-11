import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _load_halo(input, halo, r0, row_group, rows, cols):
    p = 128
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
            dst=halo[1:p, 0:halo_width],
            src=input.ap(
                pattern=[[width, p - 1], [1, halo_width]],
                offset=(row_group - 1) * cols,
            ),
        )
        nisa.dma_copy(
            dst=halo[0:1, 0:cols],
            src=halo[0:1, cols:2 * cols],
        )
    elif r0 + p * row_group == rows:
        nisa.dma_copy(
            dst=halo[0:p - 1, 0:halo_width],
            src=input.ap(
                pattern=[[width, p - 1], [1, halo_width]],
                offset=(r0 - 1) * cols,
            ),
        )
        nisa.dma_copy(
            dst=halo[p - 1:p, 0:(row_group + 1) * cols],
            src=input[rows - row_group - 1:rows, :].reshape(
                (1, (row_group + 1) * cols)
            ),
        )
        nisa.dma_copy(
            dst=halo[p - 1:p, (row_group + 1) * cols:halo_width],
            src=halo[p - 1:p, row_group * cols:(row_group + 1) * cols],
        )
    else:
        nisa.dma_copy(
            dst=halo[0:p, 0:halo_width],
            src=input.ap(
                pattern=[[width, p], [1, halo_width]],
                offset=(r0 - 1) * cols,
            ),
        )


def _compute_store(halo, result, output, r0, row_group, rows, cols):
    p = 128
    width = row_group * cols

    nisa.tensor_tensor(
        dst=result[0:p, 0:width],
        data1=halo[0:p, 0:width],
        data2=halo[0:p, 2 * cols:2 * cols + width],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=result[0:p, 0:width],
        data1=result[0:p, 0:width],
        data2=halo[0:p, cols - 1:cols - 1 + width],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=result[0:p, 0:width],
        data1=result[0:p, 0:width],
        data2=halo[0:p, cols + 1:cols + 1 + width],
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.tensor_scalar(
        dst=result[0:p, 0:width],
        data=result[0:p, 0:width],
        op0=nl.multiply,
        operand0=0.25,
        engine=nisa.engine.vector,
    )

    # Restore both column boundaries for every packed row.
    nisa.tensor_copy(
        dst=result[0:p, 0:width:cols],
        src=halo[0:p, cols:cols + width:cols],
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=result[0:p, cols - 1:width:cols],
        src=halo[0:p, 2 * cols - 1:cols + width:cols],
        engine=nisa.engine.vector,
    )

    if r0 == 0:
        nisa.dma_copy(
            dst=result[0:1, 0:cols],
            src=halo[0:1, cols:2 * cols],
        )
    if r0 + p * row_group == rows:
        nisa.dma_copy(
            dst=result[p - 1:p, width - cols:width],
            src=halo[p - 1:p, width:width + cols],
        )

    nisa.dma_copy(
        dst=output.ap(
            pattern=[[width, p], [1, width]],
            offset=r0 * cols,
        ),
        src=result[0:p, 0:width],
    )


@nki.jit
def _jacobi_kernel(input, rows, cols, row_group, halo_buffers):
    output = nl.ndarray(
        (rows, cols), dtype=input.dtype, buffer=nl.shared_hbm
    )
    halo0 = nl.ndarray(
        (128, (row_group + 2) * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    result = nl.ndarray(
        (128, row_group * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )

    if halo_buffers == 2:
        halo1 = nl.ndarray(
            (128, (row_group + 2) * cols),
            dtype=input.dtype,
            buffer=nl.sbuf,
        )

    rows_per_core = rows // nl.num_programs()
    row_units = rows_per_core // 128
    block_count = (row_units + row_group - 1) // row_group
    core_r0 = nl.program_id(0) * rows_per_core

    first_group = min(row_group, row_units)
    _load_halo(input, halo0, core_r0, first_group, rows, cols)

    for block in range(block_count):
        unit0 = block * row_group
        current_group = min(row_group, row_units - unit0)
        r0 = core_r0 + unit0 * 128

        # Queue the next input transfer before the current output transfer.
        # The alternate halo can be filled while VectorE consumes this halo.
        if block + 1 < block_count:
            next_unit = (block + 1) * row_group
            next_group = min(row_group, row_units - next_unit)
            next_r0 = core_r0 + next_unit * 128
            if block % 2 == 0:
                _load_halo(
                    input, halo1, next_r0, next_group, rows, cols
                )
            else:
                _load_halo(
                    input, halo0, next_r0, next_group, rows, cols
                )

        if block % 2 == 0:
            _compute_store(
                halo0, result, output,
                r0, current_group, rows, cols,
            )
        else:
            _compute_store(
                halo1, result, output,
                r0, current_group, rows, cols,
            )

    return output


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG
    rows = int(rows)
    cols = int(cols)

    row_units = rows // 256
    sbuf_bytes_per_partition = 212984

    single_capacity = sbuf_bytes_per_partition // (4 * cols) - 1
    if row_units <= single_capacity:
        row_group = row_units
        halo_buffers = 1
    else:
        # Two input halos and one shared result:
        # 2 * [2 * (g + 2) + g] * cols bytes per partition.
        pipeline_capacity = max(
            1, (sbuf_bytes_per_partition // cols - 8) // 6
        )
        block_count = (
            row_units + pipeline_capacity - 1
        ) // pipeline_capacity
        row_group = (row_units + block_count - 1) // block_count
        halo_buffers = 2

    _LAST_CONFIG = {
        "kernel": "jacobi_pipelined_row_halo",
        "lnc": 2,
        "partition_tile": 128,
        "rows_per_partition": row_group,
        "halo_buffers": halo_buffers,
        "blocks_per_core": (row_units + row_group - 1) // row_group,
        "rows": rows,
        "columns": cols,
        "arithmetic": "fp16_sequential_add",
        "scale_engine": "vector",
        "boundary": "strided_on_chip_copy_before_single_store",
    }

    return _jacobi_kernel[2](
        input, rows, cols, row_group, halo_buffers
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
