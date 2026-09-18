import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _dequantize_rowwise_kernel(
    x_flat,
    state_x,
    output_flat,
    COLS: ConstInt,
    TILE_N: ConstInt,
    USE_FLAT: ConstBool,
):
    row = ct.bid(0)
    num_tiles = ct.cdiv(COLS, TILE_N)

    scale = ct.load(
        state_x,
        index=(row,),
        shape=(),
        latency=1,
        allow_tma=False,
    )
    scale_h = ct.astype(scale * 0.007874015748031496, np.float16)

    if USE_FLAT:
        base_tile = row * num_tiles
        for j in range(0, num_tiles):
            tile_idx = base_tile + j

            x_tile = ct.load(
                x_flat,
                index=(tile_idx,),
                shape=(TILE_N,),
                latency=1,
                allow_tma=False,
            )

            y = ct.astype(x_tile, np.float16) * scale_h

            ct.store(
                output_flat,
                index=(tile_idx,),
                tile=y,
                latency=1,
                allow_tma=False,
            )
    else:
        row_start = row * COLS
        x_row = x_flat.slice(0, row_start, row_start + COLS)
        out_row = output_flat.slice(0, row_start, row_start + COLS)

        for j in range(0, num_tiles):
            x_tile = ct.load(
                x_row,
                index=(j,),
                shape=(TILE_N,),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )

            y = ct.astype(x_tile, np.float16) * scale_h

            ct.store(
                out_row,
                index=(j,),
                tile=y,
                latency=1,
                allow_tma=False,
            )


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)

    x_flat = x.reshape(rows * cols)
    output_flat = output.reshape(rows * cols)

    stream = torch.cuda.current_stream()

    TILE_N = 4096
    occupancy = 8
    USE_PAD = (cols % TILE_N) != 0
    USE_FLAT = not USE_PAD
    LOOP_TILES_N = (cols + TILE_N - 1) // TILE_N

    grid = (rows, 1, 1)
    kernel = _dequantize_rowwise_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (x_flat, state_x, output_flat, cols, TILE_N, USE_FLAT),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": 1,
            "TILE_N": TILE_N,
            "occupancy": occupancy,
            "USE_PAD": USE_PAD,
            "USE_FLAT": USE_FLAT,
            "GRID_N_FAST": False,
            "LOOP_N": True,
            "LOOP_TILES_N": LOOP_TILES_N,
            "HALF_COMPUTE": True,
            "ROW_SLICE_1D": not USE_FLAT,
            "SCALAR_SCALE": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
