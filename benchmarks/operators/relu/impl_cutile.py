import math

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def relu_kernel(x_ptr, output_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    y_tile = ct.where(x_tile >= 0, x_tile, 0.0)
    ct.store(output_ptr, index=(bid,), tile=y_tile)


def run(x: torch.Tensor, block_size: int = 1024):
    output = torch.empty_like(x)
    n_elements = x.numel()
    tile = block_size
    grid = (math.ceil(n_elements / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, relu_kernel, (x, output, tile))
    return output
