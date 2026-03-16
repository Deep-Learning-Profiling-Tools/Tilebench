import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def fused_kernel(x_ptr, gate_ptr, bias_ptr, out_ptr, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(ct.load(x_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    gate_tile = ct.astype(ct.load(gate_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    bias_tile = ct.astype(ct.load(bias_ptr, index=(bid,), shape=(BLOCK_SIZE,)), ct.float32)
    z = x_tile * gate_tile + bias_tile
    out_tile = ct.maximum(z, 0.0)
    ct.store(out_ptr, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, block_size: int = 1024, **kwargs):
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")

    x_flat = x.contiguous().view(-1)
    gate_flat = gate.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)
    n = x_flat.numel()
    tile = block_size
    n_padded = math.ceil(n / tile) * tile

    if n_padded != n:
        x_pad = torch.zeros((n_padded,), device=x.device, dtype=x.dtype)
        gate_pad = torch.zeros((n_padded,), device=gate.device, dtype=gate.dtype)
        bias_pad = torch.zeros((n_padded,), device=bias.device, dtype=bias.dtype)
        x_pad[:n] = x_flat
        gate_pad[:n] = gate_flat
        bias_pad[:n] = bias_flat
    else:
        x_pad = x_flat
        gate_pad = gate_flat
        bias_pad = bias_flat

    out_pad = torch.empty((n_padded,), device=x.device, dtype=torch.float32)
    grid = (n_padded // tile, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, fused_kernel, (x_pad, gate_pad, bias_pad, out_pad, tile))
    return out_pad[:n].view(x.shape)
