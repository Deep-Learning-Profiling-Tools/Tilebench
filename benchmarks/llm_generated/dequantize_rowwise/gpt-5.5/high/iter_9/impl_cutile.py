import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _dequantize_rowwise_flat_kernel(
    x_flat,
    state_x,
    output_flat,
    NUM_TILES: ConstInt,
    TILE_N: ConstInt,
):
    bid_n = ct.bid(0)
    bid_m = ct.bid(1)

    tile_idx = bid_m * NUM_TILES + bid_n

    scale = ct.load(
        state_x,
        index=(bid_m,),
        shape=(),
        latency=1,
        allow_tma=False,
    )
    scale_h = ct.astype(scale * 0.007874015748031496, np.float16)

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


@ct.kernel(occupancy=8)
def _dequantize_rowwise_padded_kernel(
    x_flat,
    state_x,
    output_flat,
    COLS: ConstInt,
    TILE_N: ConstInt,
):
    bid_n = ct.bid(0)
    bid_m = ct.bid(1)

    row_start = bid_m * COLS

    x_row = x_flat.slice(0, row_start, row_start + COLS)
    out_row = output_flat.slice(0, row_start, row_start + COLS)

    scale = ct.load(
        state_x,
        index=(bid_m,),
        shape=(),
        latency=1,
        allow_tma=False,
    )
    scale_h = ct.astype(scale * 0.007874015748031496, np.float16)

    x_tile = ct.load(
        x_row,
        index=(bid_n,),
        shape=(TILE_N,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    y = ct.astype(x_tile, np.float16) * scale_h

    ct.store(
        out_row,
        index=(bid_n,),
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
    NUM_TILES = (cols + TILE_N - 1) // TILE_N

    grid = (NUM_TILES, rows, 1)

    if USE_FLAT:
        ct.launch(
            stream,
            grid,
            _dequantize_rowwise_flat_kernel,
            (x_flat, state_x, output_flat, NUM_TILES, TILE_N),
        )
    else:
        ct.launch(
            stream,
            grid,
            _dequantize_rowwise_padded_kernel,
            (x_flat, state_x, output_flat, cols, TILE_N),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": 1,
            "TILE_N": TILE_N,
            "occupancy": occupancy,
            "USE_PAD": USE_PAD,
            "USE_FLAT": USE_FLAT,
            "NUM_TILES": NUM_TILES,
            "GRID_N_FAST": True,
            "LOOP_N": False,
            "HALF_COMPUTE": True,
            "ROW_SLICE_1D": not USE_FLAT,
            "SCALAR_SCALE": True,
            "SEPARATE_FLAT_KERNEL": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
