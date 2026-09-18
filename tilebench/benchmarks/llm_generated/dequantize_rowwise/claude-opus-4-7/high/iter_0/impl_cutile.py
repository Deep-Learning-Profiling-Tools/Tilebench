import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _dequant_kernel(x, scale, output,
                    ROWS: ConstInt, COLS: ConstInt,
                    TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(TILE_M, TILE_N),
                     padding_mode=ct.PaddingMode.ZERO)
    s_tile = ct.load(scale, index=(bid_m,), shape=(TILE_M,),
                     padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x_tile, np.float32)
    y = x_f * s_tile[:, None] * np.float32(1.0 / 127.0)
    y = ct.astype(y, output.dtype)

    ct.store(output, index=(bid_m, bid_n), tile=y)


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows, cols = x.shape
    output = torch.empty((rows, cols), dtype=torch.float16, device=x.device)
    stream = torch.cuda.current_stream()

    TILE_M = 8
    TILE_N = 512
    occupancy = 4

    grid = (ct.cdiv(rows, TILE_M), ct.cdiv(cols, TILE_N), 1)
    kernel = _dequant_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, state_x, output, rows, cols, TILE_M, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_M": TILE_M, "TILE_N": TILE_N, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
