import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=6)
def _relu_kernel(x, output, n_tiles: ConstInt, TILE: ConstInt, ITERS: ConstInt):
    bid = ct.bid(0)
    num_ctas = ct.num_blocks(0)
    for i in range(ITERS):
        tile_idx = bid + i * num_ctas
        x_tile = ct.load(x, index=(tile_idx,), shape=(TILE,),
                         padding_mode=ct.PaddingMode.ZERO)
        zero = ct.full(x_tile.shape, 0, dtype=x_tile.dtype)
        y_tile = ct.maximum(x_tile, zero)
        ct.store(output, index=(tile_idx,), tile=y_tile)


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 6
    ITERS = 4

    n_tiles = ct.cdiv(n_elements, TILE)
    # Number of CTAs launched: ceil(n_tiles / ITERS)
    n_ctas = ct.cdiv(n_tiles, ITERS)

    grid = (n_ctas, 1, 1)
    ct.launch(stream, grid, _relu_kernel, (x, output, n_tiles, TILE, ITERS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "ITERS": ITERS})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
