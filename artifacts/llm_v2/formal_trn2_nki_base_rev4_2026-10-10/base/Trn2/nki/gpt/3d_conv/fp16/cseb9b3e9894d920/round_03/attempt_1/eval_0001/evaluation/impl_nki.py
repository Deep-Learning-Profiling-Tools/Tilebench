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
    slot = (depth + 3) % 3
    plane_elements = (rows + 2) * (cols + 2)
    storage_elements = plane_elements + 2

    if 0 <= depth and depth < x.shape[2]:
        spatial = patch[
            0:64, slot, 0:plane_elements
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

        # The second partition half supplies the adjacent kernel-column
        # position. Both copies remain on chip.
        nisa.dma_copy(
            dst=patch[64:128, slot, 0:storage_elements - 1],
            src=patch[0:64, slot, 1:storage_elements],
        )
    else:
        nisa.memset(dst=patch[:, slot, :], value=0.0)


def _multiply_store(
    patch, packed_weight, acc, moving, result,
    out, od, out_row, col_start, rows, cols, row_index,
):
    pitch = cols + 2
    N = rows * pitch

    for q in range(14):
        position = 2 * q
        kd = position // 9
        kh = (position // 3) % 3
        kw = position % 3
        slot = (od + kd + 2) % 3
        start = kh * pitch + kw

        contraction = 128
        if q == 13:
            contraction = 64
            moving_source = patch[
                0:64, slot, start:start + N
            ]
        else:
            position1 = position + 1
            kd1 = position1 // 9
            kh1 = (position1 // 3) % 3
            kw1 = position1 % 3

            if kd1 == kd and kh1 == kh and kw1 == kw + 1:
                moving_source = patch[:, slot, start:start + N]
            else:
                slot1 = (od + kd1 + 2) % 3
                start1 = kh1 * pitch + kw1
                nisa.dma_copy(
                    dst=moving[0:64, :],
                    src=patch[0:64, slot, start:start + N],
                )
                nisa.dma_copy(
                    dst=moving[64:128, :],
                    src=patch[0:64, slot1, start1:start1 + N],
                )
                moving_source = moving

        for n0 in range(0, N, 512):
            n1 = min(n0 + 512, N)
            nisa.nc_matmul(
                dst=acc[:, n0:n1],
                stationary=packed_weight[0:contraction, q, :],
                moving=moving_source[:, n0:n1],
                accumulate=(q > 0),
            )

    nisa.tensor_copy(dst=result, src=acc)

    if row_index is None:
        destination = out[
            0, :, od,
            out_row:out_row + rows,
            col_start:col_start + cols
        ]
    else:
        plane = out[
            0, :, od, out_row:out.shape[3],
            col_start:col_start + cols
        ]
        destination = plane[:, nl.ds(row_index, rows), :]

    # Extra columns make the TensorE input accesses contiguous. They are
    # discarded, so only the real output positions are written.
    nisa.dma_copy(
        dst=destination,
        src=result.reshape((64, rows, pitch))[:, :, 0:cols],
    )


def _spatial_tile(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, rows,
    src_row, read_rows, dst_row, out_row,
    src_col, read_cols, dst_col, row_index,
):
    pitch = cols + 2
    N = rows * pitch
    plane_storage = (rows + 2) * pitch + 2

    patch = nl.ndarray(
        (128, 3, plane_storage), dtype=x.dtype, buffer=nl.sbuf
    )
    moving = nl.ndarray((128, N), dtype=x.dtype, buffer=nl.sbuf)
    acc = nl.ndarray((64, N), dtype=nl.float32, buffer=nl.psum)
    result = nl.ndarray((64, N), dtype=out.dtype, buffer=nl.sbuf)

    nisa.memset(dst=patch, value=0.0)

    for depth in range(od_start - 1, od_start + 2):
        _load_plane(
            x, patch, depth, rows, cols,
            src_row, read_rows, dst_row,
            src_col, read_cols, dst_col, row_index,
        )

    # A three-plane SBUF ring retains the overlapping depth windows.
    # Every output still has one full-reduction fp32 accumulator.
    for od in range(od_start, od_end):
        if od > od_start:
            _load_plane(
                x, patch, od + 1, rows, cols,
                src_row, read_rows, dst_row,
                src_col, read_cols, dst_col, row_index,
            )

        _multiply_store(
            patch, packed_weight, acc, moving, result,
            out, od, out_row, col_start, rows, cols, row_index,
        )


def _column_strip(
    x, packed_weight, out, od_start, od_end,
    col_start, cols, accumulator_elements,
):
    H = x.shape[3]
    W = x.shape[4]
    rows = min(H, accumulator_elements // (cols + 2))

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
        col_start, cols, rows,
        0, min(H, rows + 1), 1, 0,
        src_col, read_cols, left_pad, None,
    )

    if H > rows:
        last_start = ((H - 1) // rows) * rows
        last_rows = H - last_start

        def interior(i):
            _spatial_tile(
                x, packed_weight, out, od_start, od_end,
                col_start, cols, rows,
                0, rows + 2, 0, 1,
                src_col, read_cols, left_pad, i,
            )

        if last_start > rows:
            nl.fori_loop(
                rows - 1, last_start - 1, interior, step=rows
            )

        _spatial_tile(
            x, packed_weight, out, od_start, od_end,
            col_start, cols, last_rows,
            last_start - 1, last_rows + 1, 0, last_start,
            src_col, read_cols, left_pad, None,
        )


@nki.jit
def _conv3d_kernel(x, weight, tile_cols, accumulator_elements):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    packed_weight = nl.ndarray(
        (128, 14, 64), dtype=weight.dtype, buffer=nl.sbuf
    )
    for q in range(14):
        nisa.dma_copy(
            dst=packed_weight[0:64, q, :],
            src=weight.ap(
                pattern=[[27, 64], [64 * 27, 64]],
                offset=2 * q,
            ),
        )
        if q < 13:
            nisa.dma_copy(
                dst=packed_weight[64:128, q, :],
                src=weight.ap(
                    pattern=[[27, 64], [64 * 27, 64]],
                    offset=2 * q + 1,
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
            col_start, cols, accumulator_elements,
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
    column_tiles = (W + tile_cols - 1) // tile_cols
    tail_cols = W - (column_tiles - 1) * tile_cols

    _LAST_CONFIG = {
        "lnc": 2,
        "gemm_m": 64,
        "gemm_k": 128,
        "final_gemm_k": 64,
        "gemm_n_instruction": 512,
        "max_accumulator_elements": accumulator_elements,
        "spatial_tile_columns": tile_cols,
        "spatial_tile_rows": min(H, accumulator_elements // (tile_cols + 2)),
        "tail_tile_columns": tail_cols,
        "tail_tile_rows": min(H, accumulator_elements // (tail_cols + 2)),
        "column_tiles": column_tiles,
        "spatial_pitch_padding": 2,
        "reduction_chunks": 14,
        "reduction_order": "kernel_position_then_input_channel",
        "depth_mapping": "contiguous_half_per_core",
        "depth_reuse": "three_plane_sbuf_ring",
        "adjacent_column_pairing": "shifted_partition_half",
        "patch_storage": "sbuf",
        "accumulator_dtype": "float32",
    }

    return _conv3d_kernel[2](
        input, weight, tile_cols, accumulator_elements
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
