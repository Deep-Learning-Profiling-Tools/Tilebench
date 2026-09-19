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
    USE_PAD: ConstBool,
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

    if USE_PAD:
        x_tile = ct.load(
            x_row,
            index=(bid_n,),
            shape=(TILE_N,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
    else:
        x_tile = ct.load(
            x_row,
            index=(bid_n,),
            shape=(TILE_N,),
            latency=1,
            allow_tma=False,
        )

    x_h = ct.astype(x_tile, np.float16)
    scale_h = ct.astype(scale * 0.007874015748031496, np.float16)
    y = x_h * scale_h

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

    TILE_N = 8192
    occupancy = 8
    USE_PAD = (cols % TILE_N) != 0

    grid = (ct.cdiv(cols, TILE_N), rows, 1)
    kernel = _dequantize_rowwise_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (x_flat, state_x, output_flat, cols, TILE_N, USE_PAD),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": 1,
            "TILE_N": TILE_N,
            "occupancy": occupancy,
            "USE_PAD": USE_PAD,
            "GRID_N_FAST": True,
            "HALF_COMPUTE": True,
            "ROW_SLICE_1D": True,
            "SCALAR_SCALE": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
