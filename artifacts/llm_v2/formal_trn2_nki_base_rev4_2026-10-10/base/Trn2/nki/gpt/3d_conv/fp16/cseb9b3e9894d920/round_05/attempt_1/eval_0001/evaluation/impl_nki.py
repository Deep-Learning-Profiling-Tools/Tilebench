import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _load_plane(
    x, patch, depth, rows, cols,
    src_row, read_rows, dst_row,
    src_col, read_cols, dst_col, row_index,
):
    slot = ((depth + 1) // 2) % 2
    p0 = ((depth + 1) % 2) * 64
    plane_elements = (rows + 2) * (cols + 2)

    if 0 <= depth and depth < x.shape[2]:
        spatial = patch[
            p0:p0 + 64, slot, 0:plane_elements
        ].reshape((64, rows + 2, cols + 2))

        plane = x[
            0, :, depth, :, src_col:src_col + read_cols
        ]
        if row_index is None:
            source = plane[:, src_row:src_row + read_rows, :]
        else:
            source = plane[:, nl.ds(row_index, read_rows), :]

        nisa.dma_copy(
            dst=spatial[
                :, dst_row:dst_row + read_rows,
                dst_col:dst_col + read_cols
            ],
            src=source,
        )
    else:
        nisa.memset(
            dst=patch[p0:p0 + 64, slot, :],
            value=0.0,
        )


def _multiply_store(
    patch, packed_weight, acc, result, out,
    od, out_row, col_start, cols, rows,
    accumulator_elements, row_index,
):
    pitch = cols + 2
    spatial_elements = rows * pitch

    # Accumulator boundaries need not coincide with row boundaries.
    # Full chunks therefore use all eight PSUM banks without rounding
    # their extent down to a whole number of spatial rows.
    for base in range(0, spatial_elements, accumulator_elements):
        N = min(accumulator_elements, spatial_elements - base)

        for q in range(18):
            depth_pair = q // 9
            position = q % 9
            kh = position // 3
            kw = position % 3

            slot = (od // 2 + depth_pair) % 2
            start = base + kh * pitch + kw

            for n0 in range(0, N, 512):
                n1 = min(n0 + 512, N)
                nisa.nc_matmul(
                    dst=acc[:, n0:n1],
                    stationary=packed_weight[:, q, :],
                    moving=patch[:, slot, start + n0:start + n1],
                    accumulate=(q > 0),
                )

        nisa.tensor_copy(
            dst=result[:, base:base + N],
            src=acc[:, 0:N],
        )

    # Retaining cast results in SBUF permits one strided macro-tile
    # store per output depth, including chunks that split spatial rows.
    for member in range(2):
        if row_index is None:
            destination = out[
                0, :, od + member,
                out_row:out_row + rows,
                col_start:col_start + cols
            ]
        else:
            plane = out[
                0, :, od + member,
                out_row:out.shape[3],
                col_start:col_start + cols
            ]
            destination = plane[:, nl.ds(row_index, rows), :]

        p0 = member * 64
        source = result[
            p0:p0 + 64, :
        ].reshape((64, rows, pitch))

        nisa.dma_copy(
            dst=destination,
            src=source[:, :, 0:cols],
        )


def _spatial_tile(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, rows, accumulator_elements,
    src_row, read_rows, dst_row, out_row,
    src_col, read_cols, dst_col, row_index,
):
    pitch = cols + 2
    spatial_elements = rows * pitch
    accumulator_size = min(accumulator_elements, spatial_elements)
    plane_storage = (rows + 2) * pitch + 2

    patch = nl.ndarray(
        (128, 2, plane_storage),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    acc = nl.ndarray(
        (128, accumulator_size),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    result = nl.ndarray(
        (128, spatial_elements),
        dtype=out.dtype,
        buffer=nl.sbuf,
    )

    nisa.memset(dst=patch[:, 0, :], value=0.0)
    nisa.memset(dst=patch[:, 1, :], value=0.0)

    for depth in range(od_start - 1, od_start + 3):
        _load_plane(
            x, patch, depth, rows, cols,
            src_row, read_rows, dst_row,
            src_col, read_cols, dst_col, row_index,
        )

    # Four input planes reside in a two-slot SBUF ring. Each slot
    # contains two planes in its two 64-partition halves.
    for od in range(od_start, od_end, 2):
        if od > od_start:
            for depth in range(od + 1, od + 3):
                _load_plane(
                    x, patch, depth, rows, cols,
                    src_row, read_rows, dst_row,
                    src_col, read_cols, dst_col, row_index,
                )

        _multiply_store(
            patch, packed_weight, acc, result, out,
            od, out_row, col_start, cols, rows,
            accumulator_elements, row_index,
        )


def _column_strip(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, accumulator_elements, macro_elements,
):
    H = x.shape[3]
    W = x.shape[4]
    rows = min(H, macro_elements // (cols + 2))

    left_pad = 0
    right_pad = 0
    if col_start == 0:
        left_pad = 1
    if col_start + cols == W:
        right_pad = 1

    src_col = max(0, col_start - 1)
    read_cols = cols + 2 - left_pad - right_pad

    _spatial_tile(
        x, packed_weight, out, od_start, od_end,
        col_start, cols, rows, accumulator_elements,
        0, min(H, rows + 1), 1, 0,
        src_col, read_cols, left_pad, None,
    )

    if H > rows:
        last_start = ((H - 1) // rows) * rows
        last_rows = H - last_start

        def interior(i):
            _spatial_tile(
                x, packed_weight, out, od_start, od_end,
                col_start, cols, rows, accumulator_elements,
                0, rows + 2, 0, 1,
                src_col, read_cols, left_pad, i,
            )

        if last_start > rows:
            nl.fori_loop(
                rows - 1, last_start - 1, interior, step=rows
            )

        _spatial_tile(
            x, packed_weight, out, od_start, od_end,
            col_start, cols, last_rows, accumulator_elements,
            last_start - 1, last_rows + 1, 0, last_start,
            src_col, read_cols, left_pad, None,
        )


@nki.jit
def _conv3d_kernel(
    x, weight, tile_cols, accumulator_elements, macro_elements,
):
    out = nl.ndarray(
        x.shape, dtype=x.dtype, buffer=nl.shared_hbm
    )

    packed_weight = nl.ndarray(
        (128, 18, 128),
        dtype=weight.dtype,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=packed_weight, value=0.0)

    # The stationary operand combines two adjacent output depths.
    # Six nonzero blocks hold the original weights; the other two
    # blocks remain zero to suppress the extra input plane.
    for member in range(2):
        for extended_depth in range(4):
            kd = extended_depth - member
            if 0 <= kd and kd < 3:
                p0 = (extended_depth % 2) * 64
                q0 = (extended_depth // 2) * 9
                m0 = member * 64
                nisa.dma_copy(
                    dst=packed_weight[
                        p0:p0 + 64, q0:q0 + 9, m0:m0 + 64
                    ],
                    src=weight.ap(
                        pattern=[
                            [27, 64],
                            [1, 9],
                            [64 * 27, 64],
                        ],
                        offset=kd * 9,
                    ),
                )

    pid = nl.program_id(0)
    depths_per_program = x.shape[2] // nl.num_programs()
    od_start = pid * depths_per_program
    od_end = od_start + depths_per_program

    for col_start in range(0, x.shape[4], tile_cols):
        cols = min(tile_cols, x.shape[4] - col_start)
        _column_strip(
            x, packed_weight, out, od_start, od_end,
            col_start, cols, accumulator_elements, macro_elements,
        )

    return out


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    global _LAST_CONFIG

    batch, channels, D, H, W = input.shape
    assert batch == 1 and channels == 64 and D == 32
    assert W == H and 16 <= H <= 320 and H % 16 == 0
    assert tuple(weight.shape) == (64, 64, 3, 3, 3)
    assert input.dtype == torch.float16 and weight.dtype == input.dtype
    assert stride == 1 and padding == 1 and groups == 1

    tile_cols = W
    accumulator_elements = 4096
    macro_elements = 28672
    pitch = tile_cols + 2
    macro_rows = min(H, macro_elements // pitch)
    macro_tiles = (H + macro_rows - 1) // macro_rows
    tail_rows = H - (macro_tiles - 1) * macro_rows

    _LAST_CONFIG = {
        "lnc": 2,
        "gemm_m": 128,
        "gemm_k": 128,
        "gemm_n_instruction": 512,
        "output_channels_per_depth": 64,
        "output_depths_per_accumulator": 2,
        "max_accumulator_elements": accumulator_elements,
        "max_macro_elements": macro_elements,
        "spatial_tile_columns": tile_cols,
        "spatial_pitch": pitch,
        "spatial_macro_tile_rows": macro_rows,
        "spatial_macro_tiles": macro_tiles,
        "macro_accumulator_chunks": (
            macro_rows * pitch + accumulator_elements - 1
        ) // accumulator_elements,
        "tail_macro_tile_rows": tail_rows,
        "tail_accumulator_chunks": (
            tail_rows * pitch + accumulator_elements - 1
        ) // accumulator_elements,
        "column_tiles": 1,
        "spatial_pitch_padding": 2,
        "accumulator_traversal": "flat_padded_spatial_axis",
        "output_store": "one_macro_dma_per_output_depth",
        "logical_reduction_length": 64 * 27,
        "reduction_chunks": 18,
        "reduction_order": "depth_pair_then_spatial_position_then_channel",
        "reduction_padding": "zero_stationary_lanes",
        "depth_mapping": "contiguous_half_per_core",
        "depth_reuse": "four_plane_sbuf_ring",
        "plane_pair_layout": "two_64_partition_halves",
        "weight_gather_blocks": 6,
        "patch_storage": "sbuf",
        "cast_result_storage": "sbuf",
        "accumulator_dtype": "float32",
    }

    return _conv3d_kernel[2](
        input, weight, tile_cols,
        accumulator_elements, macro_elements,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
