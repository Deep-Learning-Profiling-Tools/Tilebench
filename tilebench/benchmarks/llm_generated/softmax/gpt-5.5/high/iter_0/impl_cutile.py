import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _softmax_kernel(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(row, 0),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    x_f32 = ct.astype(x_tile, np.float32)

    row_max = ct.max(x_f32, axis=1, keepdims=True)
    shifted = (x_f32 - row_max) * 1.4426950408889634

    numerator = ct.exp2(shifted, flush_to_zero=True)
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    y = numerator / denominator

    ct.store(
        output,
        index=(row, 0),
        tile=ct.astype(y, x.dtype),
        latency=1,
        allow_tma=False,
    )


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    stream = torch.cuda.current_stream()

    TILE = 1 << (n_cols - 1).bit_length()

    occupancy = 8
    if TILE > 1024:
        occupancy = 4
    if TILE > 4096:
        occupancy = 2
    if TILE > 8192:
        occupancy = 1

    grid = (n_rows, 1, 1)
    kernel = _softmax_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
