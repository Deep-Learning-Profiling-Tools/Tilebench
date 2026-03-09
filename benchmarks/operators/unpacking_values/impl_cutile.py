import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def unpack_kernel(x_ptr, out_ptr, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,))
    out_tile = ct.astype(x_tile, ct.float32)
    ct.store(out_ptr, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()
    tile = block_size
    n_padded = math.ceil(n / tile) * tile

    if n_padded != n:
        x_pad = torch.zeros((n_padded,), device=x.device, dtype=x.dtype)
        x_pad[:n] = x_flat
    else:
        x_pad = x_flat

    out_pad = torch.empty((n_padded,), device=x.device, dtype=torch.float32)
    grid = (n_padded // tile, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, unpack_kernel, (x_pad, out_pad, tile))
    return out_pad[:n].view(x.shape)
