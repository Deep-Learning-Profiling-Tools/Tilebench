import math

import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _stage_weights(weight):
    raw = nl.ndarray(
        (128, 128, 3, 3), dtype=weight.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=raw, src=weight)

    transposed = nl.ndarray(
        (128, 3, 3, 128), dtype=weight.dtype, buffer=nl.sbuf
    )
    for kh in range(3):
        for kw in range(3):
            tile = nl.ndarray(
                (128, 128), dtype=weight.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=tile, src=raw[:, :, kh, kw])
            transpose = nl.ndarray(
                (128, 128), dtype=weight.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=transpose,
                data=tile,
                engine=nisa.engine.tensor,
            )
            nisa.tensor_copy(
                dst=transposed[:, kh, kw, :], src=transpose
            )
    return transposed


def _load_core_input(
    input_rows, center, height, core_start, core_rows, guard_rows
):
    source_start = max(0, core_start - 1)
    source_end = min(height, core_start + core_rows + 1)
    destination_start = source_start - core_start + 1 + guard_rows
    valid_rows = source_end - source_start

    nisa.dma_copy(
        dst=center[
            :,
            destination_start:destination_start + valid_rows,
            :,
        ],
        src=input_rows[:, source_start:source_end, :],
    )

    if guard_rows != 0:
        nisa.memset(dst=center[:, 0:guard_rows, :], value=0.0)
        nisa.memset(
            dst=center[
                :,
                guard_rows + core_rows + 2:
                guard_rows + core_rows + 2 + guard_rows,
                :,
            ],
            value=0.0,
        )

    if core_start == 0:
        nisa.memset(
            dst=center[:, guard_rows:guard_rows + 1, :],
            value=0.0,
        )
    if core_start + core_rows == height:
        nisa.memset(
            dst=center[
                :,
                guard_rows + core_rows + 1:
                guard_rows + core_rows + 2,
                :,
            ],
            value=0.0,
        )


def _prepare_horizontal(center, left, right, rows, width):
    nisa.dma_copy(
        dst=left[:, :, 1:width],
        src=center[:, :, 0:width - 1],
    )
    nisa.dma_copy(
        dst=right[:, :, 0:width - 1],
        src=center[:, :, 1:width],
    )


def _compute_resident_band(
    center,
    left,
    right,
    transposed_weight,
    destination,
    start_row,
    rows,
    width,
):
    columns = rows * width
    accumulator = nl.ndarray(
        (128, columns), dtype=nl.float32, buffer=nl.psum
    )
    center_matrix = center.reshape((128, center.shape[1] * width))
    left_matrix = left.reshape((128, left.shape[1] * width))
    right_matrix = right.reshape((128, right.shape[1] * width))

    for kh in range(3):
        for kw in range(3):
            if kw == 0:
                moving_matrix = left_matrix
            elif kw == 1:
                moving_matrix = center_matrix
            else:
                moving_matrix = right_matrix

            for tile in range((columns + 511) // 512):
                output_start = tile * 512
                count = min(512, columns - output_start)
                input_start = (start_row + kh) * width + output_start
                nisa.nc_matmul(
                    dst=accumulator[
                        :, output_start:output_start + count
                    ],
                    stationary=transposed_weight[:, kh, kw, :],
                    moving=moving_matrix[
                        :, input_start:input_start + count
                    ],
                    accumulate=(kh * 3 + kw != 0),
                )

    result = nl.ndarray(
        (128, columns), dtype=center.dtype, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=result, src=accumulator)
    nisa.dma_copy(
        dst=destination,
        src=result.reshape((128, rows, width)),
    )


def _compute_projected_band(
    center,
    first_column,
    last_column,
    zero_column,
    transposed_weight,
    destination,
    start_row,
    rows,
    core_rows,
    width,
):
    columns = rows * width
    accumulator = nl.ndarray(
        (128, columns), dtype=nl.float32, buffer=nl.psum
    )
    center_matrix = center.reshape((128, center.shape[1] * width))
    boundary_end = start_row + rows + 4

    # A dense flattened left shift wraps through the preceding row's
    # last column. Zeroing that column makes every wrapped lane exactly
    # the required padding zero, without constructing a shifted matrix.
    # The analogous operation handles the right shift.
    for kw in range(3):
        if kw == 0:
            nisa.dma_copy(
                dst=center[
                    :, start_row:boundary_end, width - 1:width
                ],
                src=zero_column[:, 0:rows + 4, :],
            )
        elif kw == 1:
            nisa.dma_copy(
                dst=center[
                    :, start_row:boundary_end, width - 1:width
                ],
                src=last_column[:, start_row:boundary_end, :],
            )
        else:
            nisa.dma_copy(
                dst=center[:, start_row:boundary_end, 0:1],
                src=zero_column[:, 0:rows + 4, :],
            )

        for kh in range(3):
            for tile in range((columns + 511) // 512):
                output_start = tile * 512
                count = min(512, columns - output_start)
                input_start = (
                    (start_row + kh + 1) * width
                    + output_start
                    + kw
                    - 1
                )
                nisa.nc_matmul(
                    dst=accumulator[
                        :, output_start:output_start + count
                    ],
                    stationary=transposed_weight[:, kh, kw, :],
                    moving=center_matrix[
                        :, input_start:input_start + count
                    ],
                    accumulate=(kw * 3 + kh != 0),
                )

    # Only the overlap with the next band's input halo needs restoring.
    # All modifications are confined to private on-chip input staging.
    if start_row + rows < core_rows:
        next_row = start_row + rows
        nisa.dma_copy(
            dst=center[:, next_row:next_row + 4, 0:1],
            src=first_column[:, next_row:next_row + 4, :],
        )

    result = nl.ndarray(
        (128, columns), dtype=center.dtype, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=result, src=accumulator)
    nisa.dma_copy(
        dst=destination,
        src=result.reshape((128, rows, width)),
    )


@nki.jit
def _convolution(input_tensor, weight, band_rows, boundary_projection):
    height = input_tensor.shape[2]
    core_rows = height // 2
    core_start = nl.program_id(0) * core_rows

    output = nl.ndarray(
        input_tensor.shape,
        dtype=input_tensor.dtype,
        buffer=nl.shared_hbm,
    )
    input_rows = input_tensor.reshape((128, height, height))
    output_rows = output.reshape((128, height, height))
    transposed_weight = _stage_weights(weight)

    guard_rows = 0
    if boundary_projection:
        guard_rows = 1

    staged_rows = core_rows + 2 + 2 * guard_rows
    center = nl.ndarray(
        (128, staged_rows, height),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    _load_core_input(
        input_rows,
        center,
        height,
        core_start,
        core_rows,
        guard_rows,
    )

    bands = (core_rows + band_rows - 1) // band_rows

    if boundary_projection:
        first_column = nl.ndarray(
            (128, staged_rows, 1),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        last_column = nl.ndarray(
            (128, staged_rows, 1),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        zero_column = nl.ndarray(
            (128, band_rows + 4, 1),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=first_column, src=center[:, :, 0:1])
        nisa.dma_copy(
            dst=last_column, src=center[:, :, height - 1:height]
        )
        nisa.memset(dst=zero_column, value=0.0)

        for band in range(bands):
            start_row = band * band_rows
            rows = min(band_rows, core_rows - start_row)
            _compute_projected_band(
                center,
                first_column,
                last_column,
                zero_column,
                transposed_weight,
                output_rows[
                    :,
                    core_start + start_row:
                    core_start + start_row + rows,
                    :,
                ],
                start_row,
                rows,
                core_rows,
                height,
            )
    else:
        left = nl.ndarray(
            (128, staged_rows, height),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        right = nl.ndarray(
            (128, staged_rows, height),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        nisa.memset(dst=left, value=0.0)
        nisa.memset(dst=right, value=0.0)
        _prepare_horizontal(center, left, right, core_rows, height)

        for band in range(bands):
            start_row = band * band_rows
            rows = min(band_rows, core_rows - start_row)
            _compute_resident_band(
                center,
                left,
                right,
                transposed_weight,
                output_rows[
                    :,
                    core_start + start_row:
                    core_start + start_row + rows,
                    :,
                ],
                start_row,
                rows,
                height,
            )

    return output


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    global _LAST_CONFIG

    batch, channels, height, width = input.shape
    if (
        batch != 1
        or channels != 128
        or height != width
        or height < 16
        or height > 320
        or height % 16 != 0
        or tuple(weight.shape) != (128, 128, 3, 3)
        or stride != 1
        or padding != 1
        or groups != 1
        or input.dtype != torch.float16
        or weight.dtype != input.dtype
    ):
        raise ValueError("Inputs are outside the configured convolution domain.")

    core_rows = height // 2
    row_alignment = 512 // math.gcd(width, 512)
    maximum_rows = min(core_rows, 4096 // width)
    minimum_bands = (core_rows + maximum_rows - 1) // maximum_rows
    required_rows = (core_rows + minimum_bands - 1) // minimum_bands
    aligned_rows = (maximum_rows // row_alignment) * row_alignment

    # Prefer complete TensorE tiles only when alignment does not add
    # another output band.
    band_rows = maximum_rows
    if aligned_rows >= required_rows:
        band_rows = aligned_rows

    boundary_projection = core_rows * width >= 2048
    staged_rows = core_rows + (4 if boundary_projection else 2)

    config = {
        "algorithm": "implicit_gemm",
        "lnc": 2,
        "core_split": "output_rows",
        "channel_tile": 128,
        "reduction_chunk": 128,
        "spatial_tile": 512,
        "psum_column_limit": 4096,
        "band_rows": band_rows,
        "bands_per_core": (core_rows + band_rows - 1) // band_rows,
        "row_alignment": row_alignment,
        "band_policy": "alignment_without_extra_bands",
        "reduction_order": (
            "kw_kh_ic" if boundary_projection else "kh_kw_ic"
        ),
        "accumulation_dtype": "float32",
        "input_staging": "resident_sbuf_core_halo",
        "input_staging_rows": staged_rows,
        "horizontal_staging": (
            "boundary_projection"
            if boundary_projection
            else "resident_core"
        ),
        "boundary_restore": (
            "next_band_halo" if boundary_projection else "not_needed"
        ),
        "weight_staging": "sbuf_transpose",
        "loop_mode": "static_output_bands",
    }

    result = _convolution[2](
        input, weight, band_rows, boundary_projection
    )
    _LAST_CONFIG = config
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
