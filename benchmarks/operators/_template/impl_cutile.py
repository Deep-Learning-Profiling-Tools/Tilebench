import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def kernel_template(x_ptr, out_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    y_tile = x_tile  # Replace with real operator math.
    ct.store(out_ptr, index=(bid,), tile=y_tile)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    del kwargs
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (math.ceil(n_elements / block_size), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel_template, (x, out, block_size))
    return out
