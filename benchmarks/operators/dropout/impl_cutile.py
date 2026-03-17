import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _dropout_kernel(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    _block_start = bid * BLOCK_SIZE  # noqa: F841
    x_tile = ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    x_keep_tile = ct.load(x_keep_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    output = x_keep_tile * x_tile / (1.0 - p)
    ct.store(output_ptr, index=(bid,), tile=output)


def run(x, x_keep, p, block_size=1024):
    n_elements = x.numel()
    output = torch.empty_like(x)
    TILE = block_size
    grid = (math.ceil(n_elements / TILE), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _dropout_kernel,
        (x, x_keep, output, n_elements, p, TILE),
    )
    return output
