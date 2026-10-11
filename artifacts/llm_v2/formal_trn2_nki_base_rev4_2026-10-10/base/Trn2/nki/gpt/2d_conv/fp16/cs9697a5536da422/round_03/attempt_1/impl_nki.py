import math

import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


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


def _prepare_horizontal(center, left, right, source_row, rows, width):
    # Each shifted tile is densely packed across spatial rows. Its
    # untouched boundary column remains zero throughout the kernel.
    nisa.dma_copy(
        dst=left[:, 0:rows + 2, 1:width],
        src=center[
            :,
            source_row:source_row + rows + 2,
            0:width - 1,
        ],
    )
    nisa.dma_copy(
        dst=right[:, 0:rows + 2, 0:width - 1],
        src=center[
            :,
            source_row:source_row + rows + 2,
            1:width,
        ],
    )


def _compute_band(
    center,
    left,
    right,
    transposed_weight,
    destination,
    center_row,
    side_row,
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

    # Contract the full channel dimension for each kernel tap. All nine
    # chunks accumulate into the same local fp32 output accumulator.
    for kh in range(3):
        for kw in range(3):
            if kw == 0:
                moving_matrix = left_matrix
                moving_row = side_row + kh
            elif kw == 1:
                moving_matrix = center_matrix
                moving_row = center_row + kh
            else:
                moving_matrix = right_matrix
                moving_row = side_row + kh

            for tile in range((columns + 511) // 512):
                output_start = tile * 512
                count = min(512, columns - output_start)
                input_start = moving_row * width + output_start

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


@nki.jit
def _convolution(input_tensor, weight, band_rows, resident_horizontal):
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
                dst=weight_tile,
                src=raw_weight[:, :, kh, kw],
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

    # The entire core's input region, including its vertical halo, fits
    # in SBUF for every configured shape. No padded global copy is made.
    center = nl.ndarray(
        (128, core_rows + 2, height),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    _load_core_input(
        input_rows, center, height, core_start, core_rows
    )

    if resident_horizontal:
        side_rows = core_rows
    else:
        side_rows = band_rows

    left = nl.ndarray(
        (128, side_rows + 2, height),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    right = nl.ndarray(
        (128, side_rows + 2, height),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=left, value=0.0)
    nisa.memset(dst=right, value=0.0)

    if resident_horizontal:
        _prepare_horizontal(
            center, left, right, 0, core_rows, height
        )

    bands = (core_rows + band_rows - 1) // band_rows
    for band in range(bands):
        start_row = band * band_rows
        rows = min(band_rows, core_rows - start_row)

        if resident_horizontal:
            side_start = start_row
        else:
            _prepare_horizontal(
                center, left, right, start_row, rows, height
            )
            side_start = 0

        _compute_band(
            center,
            left,
            right,
            transposed_weight,
            output_rows[
                :,
                core_start + start_row:core_start + start_row + rows,
                :,
            ],
            start_row,
            side_start,
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
    band_rows = min(core_rows, 4096 // width)
    if band_rows >= row_alignment:
        band_rows = (band_rows // row_alignment) * row_alignment

    # Reserve room for the output cast and weight staging, and keep the
    # estimated live SBUF footprint below a fixed conservative budget.
    resident_budget = 192 * 1024
    resident_bytes = (
        3 * (core_rows + 2) * width * 2
        + 4096 * 2
        + 4 * 128 * 9
        + 512
    )
    resident_horizontal = resident_bytes <= resident_budget
    side_rows = core_rows if resident_horizontal else band_rows

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
        "reduction_order": "kh_kw_ic",
        "accumulation_dtype": "float32",
        "input_staging": "resident_sbuf_core_halo",
        "input_staging_rows": core_rows + 2,
        "horizontal_staging": (
            "resident_core" if resident_horizontal else "per_band"
        ),
        "horizontal_staging_rows": side_rows + 2,
        "resident_sbuf_budget_bytes_per_partition": resident_budget,
        "weight_staging": "sbuf_transpose",
        "loop_mode": "static_output_bands",
    }

    result = _convolution[2](
        input, weight, band_rows, resident_horizontal
    )
    _LAST_CONFIG = config
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
