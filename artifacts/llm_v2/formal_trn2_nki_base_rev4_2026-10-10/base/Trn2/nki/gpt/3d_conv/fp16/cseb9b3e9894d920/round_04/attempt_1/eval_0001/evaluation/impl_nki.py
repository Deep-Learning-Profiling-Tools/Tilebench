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
    # Each ring slot holds two consecutive planes, one in each
    # 64-partition half. The pairs start at odd input depths.
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
    od, out_row, col_start, cols,
    row_start, row_count, row_index,
):
    pitch = cols + 2
    N = row_count * pitch

    # Two adjacent output depths share four input planes. Stationary
    # lanes 0:64 produce od, and lanes 64:128 produce od+1. Zero weight
    # lanes suppress the one extra input plane for each output.
    for q in range(18):
        depth_pair = q // 9
        position = q % 9
        kh = position // 3
        kw = position % 3

        slot = (od // 2 + depth_pair) % 2
        start = (row_start + kh) * pitch + kw

        for n0 in range(0, N, 512):
            n1 = min(n0 + 512, N)
            nisa.nc_matmul(
                dst=acc[:, n0:n1],
                stationary=packed_weight[:, q, :],
                moving=patch[:, slot, start + n0:start + n1],
                accumulate=(q > 0),
            )

    nisa.tensor_copy(
        dst=result[:, 0:N],
        src=acc[:, 0:N],
    )

    for member in range(2):
        if row_index is None:
            destination = out[
                0, :, od + member,
                out_row + row_start:out_row + row_start + row_count,
                col_start:col_start + cols
            ]
        else:
            plane = out[
                0, :, od + member,
                out_row + row_start:out.shape[3],
                col_start:col_start + cols
            ]
            destination = plane[:, nl.ds(row_index, row_count), :]

        p0 = member * 64
        source = result[
            p0:p0 + 64, 0:N
        ].reshape((64, row_count, pitch))

        # The extra two columns make the moving operands contiguous;
        # neither extra column is stored.
        nisa.dma_copy(
            dst=destination,
            src=source[:, :, 0:cols],
        )


def _spatial_tile(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, rows, row_chunk,
    src_row, read_rows, dst_row, out_row,
    src_col, read_cols, dst_col, row_index,
):
    pitch = cols + 2
    accumulator_rows = min(rows, row_chunk)
    N = accumulator_rows * pitch
    plane_storage = (rows + 2) * pitch + 2

    patch = nl.ndarray(
        (128, 2, plane_storage),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    acc = nl.ndarray(
        (128, N),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    result = nl.ndarray(
        (128, N),
        dtype=out.dtype,
        buffer=nl.sbuf,
    )

    nisa.memset(dst=patch, value=0.0)

    for depth in range(od_start - 1, od_start + 3):
        _load_plane(
            x, patch, depth, rows, cols,
            src_row, read_rows, dst_row,
            src_col, read_cols, dst_col, row_index,
        )

    # The four-plane ring is entirely in SBUF. A spatial macro-tile
    # contains up to two accumulator-sized row tiles.
    for od in range(od_start, od_end, 2):
        if od > od_start:
            for depth in range(od + 1, od + 3):
                _load_plane(
                    x, patch, depth, rows, cols,
                    src_row, read_rows, dst_row,
                    src_col, read_cols, dst_col, row_index,
                )

        for row_start in range(0, rows, row_chunk):
            row_count = min(row_chunk, rows - row_start)
            _multiply_store(
                patch, packed_weight, acc, result, out,
                od, out_row, col_start, cols,
                row_start, row_count, row_index,
            )


def _column_strip(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, accumulator_elements, macro_row_tiles,
):
    H = x.shape[3]
    W = x.shape[4]
    row_chunk = min(H, accumulator_elements // (cols + 2))
    rows = min(H, macro_row_tiles * row_chunk)

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
        col_start, cols, rows, row_chunk,
        0, min(H, rows + 1), 1, 0,
        src_col, read_cols, left_pad, None,
    )

    if H > rows:
        last_start = ((H - 1) // rows) * rows
        last_rows = H - last_start

        def interior(i):
            _spatial_tile(
                x, packed_weight, out, od_start, od_end,
                col_start, cols, rows, row_chunk,
                0, rows + 2, 0, 1,
                src_col, read_cols, left_pad, i,
            )

        if last_start > rows:
            nl.fori_loop(
                rows - 1, last_start - 1, interior, step=rows
            )

        _spatial_tile(
            x, packed_weight, out, od_start, od_end,
            col_start, cols, last_rows, row_chunk,
            last_start - 1, last_rows + 1, 0, last_start,
            src_col, read_cols, left_pad, None,
        )


@nki.jit
def _conv3d_kernel(
    x, weight, tile_cols, accumulator_elements, macro_row_tiles,
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

    # Gather all nine spatial kernel positions together. There are six
    # nonzero blocks; the remaining two blocks are reduction padding.
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
            col_start, cols, accumulator_elements, macro_row_tiles,
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

    tile_cols = min(W, 126)
    accumulator_elements = 4096
    macro_row_tiles = 2
    column_tiles = (W + tile_cols - 1) // tile_cols
    tail_cols = W - (column_tiles - 1) * tile_cols
    row_chunk = min(H, accumulator_elements // (tile_cols + 2))
    tail_row_chunk = min(H, accumulator_elements // (tail_cols + 2))

    _LAST_CONFIG = {
        "lnc": 2,
        "gemm_m": 128,
        "gemm_k": 128,
        "gemm_n_instruction": 512,
        "output_channels_per_depth": 64,
        "output_depths_per_accumulator": 2,
        "max_accumulator_elements": accumulator_elements,
        "spatial_tile_columns": tile_cols,
        "accumulator_tile_rows": row_chunk,
        "spatial_macro_tile_rows": min(H, macro_row_tiles * row_chunk),
        "macro_row_tiles": macro_row_tiles,
        "tail_tile_columns": tail_cols,
        "tail_accumulator_tile_rows": tail_row_chunk,
        "tail_macro_tile_rows": min(H, macro_row_tiles * tail_row_chunk),
        "column_tiles": column_tiles,
        "spatial_pitch_padding": 2,
        "logical_reduction_length": 64 * 27,
        "reduction_chunks": 18,
        "reduction_order": "depth_pair_then_spatial_position_then_channel",
        "reduction_padding": "zero_stationary_lanes",
        "depth_mapping": "contiguous_half_per_core",
        "depth_reuse": "four_plane_sbuf_ring",
        "plane_pair_layout": "two_64_partition_halves",
        "weight_gather_blocks": 6,
        "patch_storage": "sbuf",
        "accumulator_dtype": "float32",
    }

    return _conv3d_kernel[2](
        input, weight, tile_cols,
        accumulator_elements, macro_row_tiles,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
