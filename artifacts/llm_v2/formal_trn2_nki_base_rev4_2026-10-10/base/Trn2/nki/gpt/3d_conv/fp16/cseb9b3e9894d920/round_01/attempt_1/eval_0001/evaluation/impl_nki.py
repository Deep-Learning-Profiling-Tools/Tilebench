import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _load_rows(
    x, patch, od, src_row, read_rows, dst_row,
    src_col, read_cols, dst_col, row_index
):
    D = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]

    for kd in range(3):
        depth = od + kd - 1
        if depth >= 0 and depth < D:
            offset = depth * H * W + src_row * W + src_col
            if row_index is None:
                source = x.ap(
                    pattern=[[D * H * W, 64], [W, read_rows], [1, read_cols]],
                    offset=offset,
                )
            else:
                source = x.ap(
                    pattern=[[D * H * W, 64], [W, read_rows], [1, read_cols]],
                    offset=offset,
                    scalar_offset=row_index,
                    indirect_dim=1,
                )
            nisa.dma_copy(
                dst=patch[
                    :, kd,
                    dst_row:dst_row + read_rows,
                    dst_col:dst_col + read_cols
                ],
                src=source,
            )


def _multiply_store(
    patch, packed_weight, out, od, out_row,
    col_start, rows, cols, row_index
):
    D = out.shape[2]
    H = out.shape[3]
    W = out.shape[4]
    N = rows * cols

    acc = nl.ndarray((64, N), dtype=nl.float32, buffer=nl.psum)

    # Each contraction chunk contains two kernel positions, each with 64
    # input channels. The final chunk has a zero-filled second half.
    for q in range(14):
        moving = nl.ndarray((128, N), dtype=patch.dtype, buffer=nl.sbuf)

        position = 2 * q
        kd = position // 9
        kh = (position // 3) % 3
        kw = position % 3
        nisa.dma_copy(
            dst=moving[0:64, :],
            src=patch[:, kd, kh:kh + rows, kw:kw + cols],
        )

        position1 = position + 1
        if position1 < 27:
            kd1 = position1 // 9
            kh1 = (position1 // 3) % 3
            kw1 = position1 % 3
            nisa.dma_copy(
                dst=moving[64:128, :],
                src=patch[:, kd1, kh1:kh1 + rows, kw1:kw1 + cols],
            )
        else:
            nisa.memset(dst=moving[64:128, :], value=0.0)

        for n0 in range(0, N, 512):
            n1 = min(n0 + 512, N)
            nisa.nc_matmul(
                dst=acc[:, n0:n1],
                stationary=packed_weight[:, q, :],
                moving=moving[:, n0:n1],
                accumulate=(q > 0),
            )

    result = nl.ndarray((64, N), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=acc)

    offset = od * H * W + out_row * W + col_start
    if row_index is None:
        destination = out.ap(
            pattern=[[D * H * W, 64], [W, rows], [1, cols]],
            offset=offset,
        )
    else:
        destination = out.ap(
            pattern=[[D * H * W, 64], [W, rows], [1, cols]],
            offset=offset,
            scalar_offset=row_index,
            indirect_dim=1,
        )
    nisa.dma_copy(dst=destination, src=result)


def _depth_column_strip(x, packed_weight, out, od, col_start, cols, rows):
    H = x.shape[3]
    W = x.shape[4]

    left_pad = 0
    right_pad = 0
    if col_start == 0:
        left_pad = 1
    if col_start + cols == W:
        right_pad = 1

    src_col = max(0, col_start - 1)
    read_cols = cols + 2 - left_pad - right_pad

    # This halo buffer never leaves SBUF. Its width padding and missing
    # depth planes remain zero while the valid rows are refreshed.
    patch = nl.ndarray(
        (64, 3, rows + 2, cols + 2),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=patch, value=0.0)

    first_read_rows = min(H, rows + 1)
    _load_rows(
        x, patch, od, 0, first_read_rows, 1,
        src_col, read_cols, left_pad, None,
    )
    _multiply_store(
        patch, packed_weight, out, od, 0,
        col_start, rows, cols, None,
    )

    if H > rows:
        last_start = ((H - 1) // rows) * rows
        last_rows = H - last_start

        # i denotes the first input halo row. Output starts at i + 1;
        # the additional row is encoded in the static AP offset.
        def interior(i):
            _load_rows(
                x, patch, od, 0, rows + 2, 0,
                src_col, read_cols, left_pad, i,
            )
            _multiply_store(
                patch, packed_weight, out, od, 1,
                col_start, rows, cols, i,
            )

        if last_start > rows:
            nl.fori_loop(rows - 1, last_start - 1, interior, step=rows)

        # Replace the bottom halo left by the preceding interior tile.
        nisa.memset(
            dst=patch[:, :, last_rows + 1:last_rows + 2, :],
            value=0.0,
        )
        _load_rows(
            x, patch, od, last_start - 1, last_rows + 1, 0,
            src_col, read_cols, left_pad, None,
        )
        _multiply_store(
            patch, packed_weight, out, od, last_start,
            col_start, last_rows, cols, None,
        )


@nki.jit
def _conv3d_kernel(x, weight, tile_rows, tile_cols):
    D = x.shape[2]
    W = x.shape[4]

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    # Pack stationary tiles on chip only. Reduction order is
    # (kernel position, input channel).
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
        if 2 * q + 1 < 27:
            nisa.dma_copy(
                dst=packed_weight[64:128, q, :],
                src=weight.ap(
                    pattern=[[27, 64], [64 * 27, 64]],
                    offset=2 * q + 1,
                ),
            )
        else:
            nisa.memset(dst=packed_weight[64:128, q, :], value=0.0)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    depths_per_program = D // programs

    for od in range(
        pid * depths_per_program,
        (pid + 1) * depths_per_program,
    ):
        for col_start in range(0, W, tile_cols):
            cols = min(tile_cols, W - col_start)
            _depth_column_strip(
                x, packed_weight, out, od,
                col_start, cols, tile_rows,
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

    # Balanced, 16-element-aligned width strips, each no wider than 128.
    width_strips = (W + 127) // 128
    tile_cols = ((W + 16 * width_strips - 1) // (16 * width_strips)) * 16
    tile_rows = min(H, 2048 // tile_cols)

    _LAST_CONFIG = {
        "lnc": 2,
        "gemm_m": 64,
        "gemm_k": 128,
        "gemm_n_instruction": 512,
        "spatial_tile_rows": tile_rows,
        "spatial_tile_columns": tile_cols,
        "spatial_tile_elements": tile_rows * tile_cols,
        "column_tiles": (W + tile_cols - 1) // tile_cols,
        "reduction_chunks": 14,
        "reduction_order": "kernel_position_then_input_channel",
        "depth_mapping": "contiguous_half_per_core",
        "patch_storage": "sbuf",
        "accumulator_dtype": "float32",
    }

    return _conv3d_kernel[2](input, weight, tile_rows, tile_cols)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
