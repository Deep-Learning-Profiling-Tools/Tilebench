import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _stage_weights(weight):
    raw_weight = nl.ndarray(
        (128, 128, 3, 3), dtype=weight.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=raw_weight, src=weight)

    transposed_weight = nl.ndarray(
        (128, 3, 3, 128), dtype=weight.dtype, buffer=nl.sbuf
    )
    for kh in range(3):
        for kw in range(3):
            weight_tile = nl.ndarray(
                (128, 128), dtype=weight.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=weight_tile, src=raw_weight[:, :, kh, kw]
            )
            weight_transpose = nl.ndarray(
                (128, 128), dtype=weight.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=weight_transpose,
                data=weight_tile,
                engine=nisa.engine.tensor,
            )
            nisa.tensor_copy(
                dst=transposed_weight[:, kh, kw, :],
                src=weight_transpose,
            )
    return transposed_weight


def _load_core_input(input_rows, center, height, core_start, core_rows):
    source_start = max(0, core_start - 1)
    source_end = min(height, core_start + core_rows + 1)
    destination_start = source_start - core_start + 1
    valid_rows = source_end - source_start

    nisa.dma_copy(
        dst=center[
            :,
            destination_start:destination_start + valid_rows,
            :,
        ],
        src=input_rows[:, source_start:source_end, :],
    )

    if core_start == 0:
        nisa.memset(dst=center[:, 0:1, :], value=0.0)
    if core_start + core_rows == height:
        nisa.memset(
            dst=center[:, core_rows + 1:core_rows + 2, :],
            value=0.0,
        )


def _prepare_small_horizontal(center, left, right, width):
    nisa.memset(dst=left, value=0.0)
    nisa.memset(dst=right, value=0.0)
    nisa.dma_copy(
        dst=left[:, :, 1:width],
        src=center[:, :, 0:width - 1],
    )
    nisa.dma_copy(
        dst=right[:, :, 0:width - 1],
        src=center[:, :, 1:width],
    )


def _prepare_flat_horizontal(
    center_matrix, left, right, zero_column, start, columns, width
):
    side_columns = left.shape[1] * width
    left_matrix = left.reshape((128, side_columns))
    right_matrix = right.reshape((128, side_columns))
    extent = columns + 2 * width

    # Dense copies avoid a separate DMA segment for every spatial row.
    # The boundary copies below remove every flattened row-wrap lane.
    if start == 0:
        nisa.dma_copy(
            dst=left_matrix[:, 1:extent],
            src=center_matrix[:, 0:extent - 1],
        )
    else:
        nisa.dma_copy(
            dst=left_matrix[:, 0:extent],
            src=center_matrix[:, start - 1:start + extent - 1],
        )

    if start + extent == center_matrix.shape[1]:
        nisa.dma_copy(
            dst=right_matrix[:, 0:extent - 1],
            src=center_matrix[:, start + 1:start + extent],
        )
    else:
        nisa.dma_copy(
            dst=right_matrix[:, 0:extent],
            src=center_matrix[:, start + 1:start + extent + 1],
        )

    phase = start % width
    left_boundary = (width - phase) % width
    right_boundary = width - 1 - phase

    nisa.dma_copy(
        dst=left[:, :, left_boundary:left_boundary + 1],
        src=zero_column,
    )
    nisa.dma_copy(
        dst=right[:, :, right_boundary:right_boundary + 1],
        src=zero_column,
    )


def _compute_band(
    center_matrix,
    left_matrix,
    right_matrix,
    transposed_weight,
    destination,
    center_start,
    side_start,
    columns,
    width,
):
    accumulator = nl.ndarray(
        (128, columns), dtype=nl.float32, buffer=nl.psum
    )

    # Tap-major reduction: nine native-fp16, K=128 matrix products
    # accumulate into one local fp32 accumulator per output element.
    for kh in range(3):
        for kw in range(3):
            if kw == 0:
                moving_matrix = left_matrix
                moving_start = side_start + kh * width
            elif kw == 1:
                moving_matrix = center_matrix
                moving_start = center_start + kh * width
            else:
                moving_matrix = right_matrix
                moving_start = side_start + kh * width

            for tile in range((columns + 511) // 512):
                output_start = tile * 512
                count = min(512, columns - output_start)
                input_start = moving_start + output_start

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
        (128, columns), dtype=center_matrix.dtype, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=result, src=accumulator)
    nisa.dma_copy(dst=destination, src=result)


@nki.jit
def _convolution(
    input_tensor,
    weight,
    band_columns,
    resident_horizontal,
    small_horizontal,
):
    height = input_tensor.shape[2]
    width = height
    core_rows = height // 2
    core_columns = core_rows * width
    core_start = nl.program_id(0) * core_rows
    output_start = core_start * width

    output = nl.ndarray(
        input_tensor.shape,
        dtype=input_tensor.dtype,
        buffer=nl.shared_hbm,
    )
    input_rows = input_tensor.reshape((128, height, width))
    output_matrix = output.reshape((128, height * width))
    transposed_weight = _stage_weights(weight)

    center = nl.ndarray(
        (128, core_rows + 2, width),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    _load_core_input(
        input_rows, center, height, core_start, core_rows
    )
    center_matrix = center.reshape(
        (128, (core_rows + 2) * width)
    )

    if resident_horizontal:
        side_rows = core_rows
    else:
        side_rows = (band_columns + width - 1) // width

    left = nl.ndarray(
        (128, side_rows + 2, width),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    right = nl.ndarray(
        (128, side_rows + 2, width),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )

    if small_horizontal:
        _prepare_small_horizontal(center, left, right, width)
    else:
        zero_column = nl.ndarray(
            (128, side_rows + 2, 1),
            dtype=input_tensor.dtype,
            buffer=nl.sbuf,
        )
        nisa.memset(dst=zero_column, value=0.0)

        if resident_horizontal:
            _prepare_flat_horizontal(
                center_matrix,
                left,
                right,
                zero_column,
                0,
                core_columns,
                width,
            )

    left_matrix = left.reshape(
        (128, (side_rows + 2) * width)
    )
    right_matrix = right.reshape(
        (128, (side_rows + 2) * width)
    )

    bands = (core_columns + band_columns - 1) // band_columns
    for band in range(bands):
        start = band * band_columns
        columns = min(band_columns, core_columns - start)

        if resident_horizontal:
            side_start = start
        else:
            _prepare_flat_horizontal(
                center_matrix,
                left,
                right,
                zero_column,
                start,
                columns,
                width,
            )
            side_start = 0

        _compute_band(
            center_matrix,
            left_matrix,
            right_matrix,
            transposed_weight,
            output_matrix[
                :,
                output_start + start:
                output_start + start + columns,
            ],
            start,
            side_start,
            columns,
            width,
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
    core_columns = core_rows * width
    band_columns = min(4096, core_columns)

    resident_budget = 192 * 1024
    resident_bytes = (
        3 * (core_rows + 2) * width * 2
        + band_columns * 2
        + 4 * 128 * 9
        + 512
    )
    resident_horizontal = resident_bytes <= resident_budget
    small_horizontal = core_columns < 2048

    side_rows = (
        core_rows
        if resident_horizontal
        else (band_columns + width - 1) // width
    )

    config = {
        "algorithm": "implicit_gemm",
        "lnc": 2,
        "core_split": "output_rows",
        "channel_tile": 128,
        "reduction_chunk": 128,
        "spatial_tile": 512,
        "psum_column_limit": 4096,
        "band_columns": band_columns,
        "bands_per_core": (
            core_columns + band_columns - 1
        ) // band_columns,
        "band_policy": "flat_spatial_full_psum",
        "reduction_order": "kh_kw_ic",
        "accumulation_dtype": "float32",
        "input_staging": "resident_sbuf_core_halo",
        "input_staging_rows": core_rows + 2,
        "horizontal_staging": (
            "resident_core" if resident_horizontal else "per_band"
        ),
        "horizontal_staging_rows": side_rows + 2,
        "horizontal_copy": (
            "row_strided_zero_buffers"
            if small_horizontal
            else "contiguous_shift_boundary_zero"
        ),
        "resident_sbuf_budget_bytes_per_partition": resident_budget,
        "weight_staging": "sbuf_transpose",
        "loop_mode": "static_output_bands",
    }

    result = _convolution[2](
        input,
        weight,
        band_columns,
        resident_horizontal,
        small_horizontal,
    )
    _LAST_CONFIG = config
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
