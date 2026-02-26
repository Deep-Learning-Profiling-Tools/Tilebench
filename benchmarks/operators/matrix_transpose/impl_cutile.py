import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def transpose_kernel(x, output, TILE: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(TILE, TILE))
    out_tile = ct.transpose(x_tile)
    ct.store(output, index=(bid_n, bid_m), tile=out_tile)


def _tile_dim_from_block_size(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 1
    while tile * 2 <= root:
        tile *= 2
    return tile


def run(x: torch.Tensor, block_size: int = 1024):
    m, n = x.shape
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    tile = _tile_dim_from_block_size(block_size)
    grid = (math.ceil(m / tile), math.ceil(n / tile), 1)
    ct.launch(torch.cuda.current_stream(), grid, transpose_kernel, (x, output, tile))
    return output
