import math

import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _compute_band(padded_input, transposed_weight, destination, rows, width):
    columns = rows * width
    accumulator = nl.ndarray(
        (128, columns), dtype=nl.float32, buffer=nl.psum
    )

    # Tap-major reduction: each chunk contracts all 128 input channels.
    for kh in range(3):
        for kw in range(3):
            moving = nl.ndarray(
                (128, rows, width),
                dtype=padded_input.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=moving,
                src=padded_input[
                    :, kh:kh + rows, kw:kw + width
                ],
            )
            moving_matrix = moving.reshape((128, columns))

            for tile in range((columns + 511) // 512):
                start = tile * 512
                count = min(512, columns - start)
                nisa.nc_matmul(
                    dst=accumulator[:, start:start + count],
                    stationary=transposed_weight[:, kh, kw, :],
                    moving=moving_matrix[:, start:start + count],
                    accumulate=(kh * 3 + kw != 0),
                )

    result = nl.ndarray(
        (128, columns), dtype=padded_input.dtype, buffer=nl.sbuf
    )
    nisa.tensor_copy(dst=result, src=accumulator)
    nisa.dma_copy(
        dst=destination,
        src=result.reshape((128, rows, width)),
    )


def _static_band(
    input_tensor, output, padded_input, transposed_weight,
    height, start_row, rows
):
    source_start = max(0, start_row - 1)
    source_end = min(height, start_row + rows + 1)
    destination_start = source_start - start_row + 1
    valid_rows = source_end - source_start

    nisa.dma_copy(
        dst=padded_input[
            :,
            destination_start:destination_start + valid_rows,
            1:height + 1,
        ],
        src=input_tensor[0, :, source_start:source_end, :],
    )

    if start_row == 0:
        nisa.memset(dst=padded_input[:, 0:1, :], value=0.0)
    if start_row + rows == height:
        nisa.memset(
            dst=padded_input[:, rows + 1:rows + 2, :],
            value=0.0,
        )

    _compute_band(
        padded_input,
        transposed_weight,
        output[0, :, start_row:start_row + rows, :],
        rows,
        height,
    )


@nki.jit
def _convolution(input_tensor, weight, band_rows):
    height = input_tensor.shape[2]
    output = nl.ndarray(
        input_tensor.shape,
        dtype=input_tensor.dtype,
        buffer=nl.shared_hbm,
    )

    # The weights remain on chip for the entire output-row traversal.
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

    pid = nl.program_id(0)
    rows_per_core = height // 2
    core_start = pid * rows_per_core
    bands = (rows_per_core + band_rows - 1) // band_rows

    padded_input = nl.ndarray(
        (128, band_rows + 2, height + 2),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    # Horizontal halo columns remain zero throughout the traversal.
    nisa.memset(dst=padded_input, value=0.0)

    _static_band(
        input_tensor, output, padded_input, transposed_weight,
        height, core_start, band_rows,
    )

    if bands > 2:
        interior_start = core_start + band_rows

        def interior_body(row_offset):
            source = input_tensor.ap(
                pattern=[
                    [height * height, 128],
                    [height, band_rows + 2],
                    [1, height],
                ],
                offset=(interior_start - 1) * height,
                scalar_offset=row_offset,
                indirect_dim=1,
            )
            nisa.dma_copy(
                dst=padded_input[:, :, 1:height + 1],
                src=source,
            )
            destination = output.ap(
                pattern=[
                    [height * height, 128],
                    [height, band_rows],
                    [1, height],
                ],
                offset=interior_start * height,
                scalar_offset=row_offset,
                indirect_dim=1,
            )
            _compute_band(
                padded_input,
                transposed_weight,
                destination,
                band_rows,
                height,
            )

        nl.fori_loop(
            0,
            (bands - 2) * band_rows,
            interior_body,
            step=band_rows,
        )

    if bands > 1:
        last_start = core_start + (bands - 1) * band_rows
        last_rows = rows_per_core - (bands - 1) * band_rows
        _static_band(
            input_tensor, output, padded_input, transposed_weight,
            height, last_start, last_rows,
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

    # At most eight PSUM banks. Prefer complete 512-column tiles when
    # an integral number of full output rows permits that alignment.
    row_alignment = 512 // math.gcd(width, 512)
    band_rows = min(height // 2, 4096 // width)
    if band_rows >= row_alignment:
        band_rows = (band_rows // row_alignment) * row_alignment

    config = {
        "algorithm": "implicit_gemm",
        "lnc": 2,
        "core_split": "output_rows",
        "channel_tile": 128,
        "reduction_chunk": 128,
        "spatial_tile": 512,
        "psum_column_limit": 4096,
        "band_rows": band_rows,
        "bands_per_core": (height // 2 + band_rows - 1) // band_rows,
        "row_alignment": row_alignment,
        "reduction_order": "kh_kw_ic",
        "accumulation_dtype": "float32",
        "input_staging": "reused_sbuf_halo",
        "weight_staging": "sbuf_transpose",
    }

    result = _convolution[2](input, weight, band_rows)
    _LAST_CONFIG = config
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
