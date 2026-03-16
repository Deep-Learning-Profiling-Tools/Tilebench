import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]


@ct.kernel
def kernel_function(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    y_tile = ct.load(y_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x_tile))
    out_tile = x_tile * sigmoid_x * y_tile
    ct.store(output_ptr, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    assert x.shape == y.shape
    n_elements = x.numel()
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    TILE = block_size
    grid = (math.ceil(n_elements / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel_function, (x_flat, y_flat, output, n_elements, TILE))
    return output.view(x.shape)
