import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _sweep_block(input, output, halo, result, r0, row_group, rows, cols):
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
        # The missing neighbour is used only for the discarded boundary row.
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

    # Match the reference's fp16 rounding after each addition.
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

    # Restore boundary values before the single output store.
    for j in range(row_group):
        first = j * cols
        centre_first = (j + 1) * cols
        nisa.tensor_copy(
            dst=result[0:p, first:first + 1],
            src=halo[0:p, centre_first:centre_first + 1],
        )
        nisa.tensor_copy(
            dst=result[0:p, first + cols - 1:first + cols],
            src=halo[0:p, centre_first + cols - 1:centre_first + cols],
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
def _jacobi_kernel(input, rows, cols, row_group):
    output = nl.ndarray(
        (rows, cols), dtype=input.dtype, buffer=nl.shared_hbm
    )

    partition_tile = 128
    halo = nl.ndarray(
        (partition_tile, (row_group + 2) * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )
    result = nl.ndarray(
        (partition_tile, row_group * cols),
        dtype=input.dtype,
        buffer=nl.sbuf,
    )

    pid = nl.program_id(0)
    rows_per_core = rows // nl.num_programs()
    row_units = rows_per_core // partition_tile
    block_count = (row_units + row_group - 1) // row_group

    for block in range(block_count):
        unit0 = block * row_group
        current_group = min(row_group, row_units - unit0)
        r0 = pid * rows_per_core + unit0 * partition_tile

        # Short final blocks still occupy all 128 partitions.
        _sweep_block(
            input, output, halo, result,
            r0, current_group, rows, cols,
        )

    return output


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    global _LAST_CONFIG

    rows = int(rows)
    cols = int(cols)

    # Halo and result together require 4 * (row_group + 1) * cols
    # bytes per partition. Use the documented Trn2 SBUF capacity.
    capacity_group = max(1, 212984 // (4 * cols) - 1)
    row_group = min(8, rows // 256, capacity_group)

    _LAST_CONFIG = {
        "kernel": "jacobi_flat_row_halo",
        "lnc": 2,
        "partition_tile": 128,
        "rows_per_partition": row_group,
        "tail": "full_partition_short_row_group",
        "rows": rows,
        "columns": cols,
        "arithmetic": "fp16_sequential_add",
        "scale_engine": "vector",
        "boundary": "on_chip_select_before_store",
    }

    return _jacobi_kernel[2](input, rows, cols, row_group)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
