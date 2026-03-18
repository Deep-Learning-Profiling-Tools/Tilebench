import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    _block_start = bid * BLOCK_SIZE  # noqa: F841
    x_tile = ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    x_sin = ct.sin(x_tile)
    ct.store(output_ptr, index=(bid,), tile=x_sin)


def run(x: torch.Tensor, block_size: int = 1024):
    n_elements = x.numel()
    output = torch.empty_like(x)
    TILE = block_size
    grid = (math.ceil(n_elements / TILE), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel_function,
        (x, output, n_elements, TILE),
    )
    return output
