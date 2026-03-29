import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def divergence_kernel(x_ptr, y_ptr, out_ptr, eps, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    y_tile = ct.astype(ct.load(y_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    diff = x_tile - y_tile
    out_tile = (diff * diff) / (y_tile * y_tile + eps)
    ct.store(out_ptr, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, y: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")

    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    n = x_flat.numel()
    tile = block_size
    n_padded = math.ceil(n / tile) * tile

    if n_padded != n:
        x_pad = torch.zeros((n_padded,), device=x.device, dtype=x.dtype)
        y_pad = torch.zeros((n_padded,), device=y.device, dtype=y.dtype)
        x_pad[:n] = x_flat
        y_pad[:n] = y_flat
    else:
        x_pad = x_flat
        y_pad = y_flat

    out_pad = torch.empty((n_padded,), device=x.device, dtype=torch.float32)
    grid = (n_padded // tile, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        divergence_kernel,
        (x_pad, y_pad, out_pad, eps, tile),
    )
    return out_pad[:n].view(x.shape)
