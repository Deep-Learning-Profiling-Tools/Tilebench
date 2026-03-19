import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def kl_divergence_kernel(p_ptr, q_ptr, out_ptr, eps, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    p_tile = ct.astype(ct.load(p_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    q_tile = ct.astype(ct.load(q_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    p_tile = ct.maximum(p_tile, eps)
    q_tile = ct.maximum(q_tile, eps)
    out_tile = p_tile * (ct.log(p_tile) - ct.log(q_tile))
    ct.store(out_ptr, index=(bid,), tile=out_tile)


def run(p: torch.Tensor, q: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    if p.shape != q.shape:
        raise ValueError("Input tensors must have the same shape.")

    p_flat = p.contiguous().view(-1)
    q_flat = q.contiguous().view(-1)
    n = p_flat.numel()
    tile = block_size
    n_padded = math.ceil(n / tile) * tile

    if n_padded != n:
        p_pad = torch.zeros((n_padded,), device=p.device, dtype=p.dtype)
        q_pad = torch.zeros((n_padded,), device=q.device, dtype=q.dtype)
        p_pad[:n] = p_flat
        q_pad[:n] = q_flat
    else:
        p_pad = p_flat
        q_pad = q_flat

    out_pad = torch.empty((n_padded,), device=p.device, dtype=torch.float32)
    grid = (n_padded // tile, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kl_divergence_kernel, (p_pad, q_pad, out_pad, eps, tile))
    return out_pad[:n].view(p.shape)
