import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

LOG2_E = 1.4426950408889634


@ct.kernel(occupancy=2)
def _softmax_kernel_large(x, output, n_cols: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    tile_f32 = ct.astype(tile, np.float32)
    max_val = ct.max(tile_f32, axis=1, keepdims=True)
    exp_tile = ct.exp2((tile_f32 - max_val) * LOG2_E, flush_to_zero=True)
    sum_val = ct.sum(exp_tile, axis=1, keepdims=True)
    out_tile = exp_tile / sum_val
    ct.store(output, index=(row, 0), tile=ct.astype(out_tile, x.dtype))


@ct.kernel(occupancy=4)
def _softmax_kernel_small(x, output, n_cols: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    tile_f32 = ct.astype(tile, np.float32)
    max_val = ct.max(tile_f32, axis=1, keepdims=True)
    exp_tile = ct.exp2((tile_f32 - max_val) * LOG2_E, flush_to_zero=True)
    sum_val = ct.sum(exp_tile, axis=1, keepdims=True)
    out_tile = exp_tile / sum_val
    ct.store(output, index=(row, 0), tile=ct.astype(out_tile, x.dtype))


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    stream = torch.cuda.current_stream()

    TILE = 1 << (n_cols - 1).bit_length()
    if TILE < 256:
        TILE = 256

    grid = (n_rows, 1, 1)
    # For large tiles, the tile size dominates shmem so keep occupancy lower.
    # For smaller tiles (≤4096), allow higher occupancy.
    if TILE <= 4096:
        ct.launch(stream, grid, _softmax_kernel_small, (x, output, n_cols, TILE))
        occ = 4
    else:
        ct.launch(stream, grid, _softmax_kernel_large, (x, output, n_cols, TILE))
        occ = 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occ})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
