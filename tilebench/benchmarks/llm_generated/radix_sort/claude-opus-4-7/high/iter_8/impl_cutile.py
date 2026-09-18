import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _count_ones_kernel(x, block_ones, N, bit, BLOCK: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bit_val = ct.astype((x_tile >> bit) & 1, np.int32)
    bit_val = ct.where(valid, bit_val, 0)
    s = ct.sum(bit_val, axis=0, keepdims=True)
    ct.store(block_ones, index=(pid,), tile=s)


@ct.kernel(occupancy=2)
def _sum_and_scan_super_kernel(block_ones, super_ones, num_blocks, SUPER: ConstInt):
    pid = ct.bid(0)
    v = ct.load(block_ones, index=(pid,), shape=(SUPER,),
                padding_mode=ct.PaddingMode.ZERO)
    inc = ct.cumsum(v, axis=0)
    exc = inc - v
    ct.store(block_ones, index=(pid,), tile=exc)
    s = ct.sum(v, axis=0, keepdims=True)
    ct.store(super_ones, index=(pid,), tile=s)


@ct.kernel(occupancy=1)
def _scan_super_kernel(super_ones, total_ones, BLOCK: ConstInt):
    v = ct.load(super_ones, index=(0,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(v, axis=0, keepdims=True)
    ct.store(total_ones, index=(0,), tile=total)
    inc = ct.cumsum(v, axis=0)
    exc = inc - v
    ct.store(super_ones, index=(0,), tile=exc)


@ct.kernel(occupancy=2)
def _scatter_kernel(x, out, block_prefix, super_prefix, total_ones,
                    N, bit, SUPER_SIZE: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    super_idx = pid // SUPER_SIZE

    bp = ct.load(block_prefix, index=(pid,), shape=(1,))
    sp = ct.load(super_prefix, index=(super_idx,), shape=(1,))
    to = ct.load(total_ones, index=(0,), shape=(1,))

    ones_offset = bp + sp
    total_zeros = N - to

    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bit_val = ct.astype((x_tile >> bit) & 1, np.int32)
    bit_val_m = ct.where(valid, bit_val, 0)

    inc = ct.cumsum(bit_val_m, axis=0)
    local_ones = inc - bit_val_m
    local_zeros = ct.arange(BLOCK, dtype=np.int32) - local_ones

    dest_if_one = total_zeros + ones_offset + local_ones
    dest_if_zero = (pid * BLOCK - ones_offset) + local_zeros

    dest = ct.where(bit_val_m == 1, dest_if_one, dest_if_zero)
    dest = ct.where(valid, dest, N)
    ct.scatter(out, (dest,), x_tile)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024

    n = N
    x = input.contiguous().clone()
    out = torch.empty_like(x)

    num_blocks = (n + BLOCK - 1) // BLOCK
    num_super = (num_blocks + SUPER_SIZE - 1) // SUPER_SIZE
    SUPER_TILE = 1
    while SUPER_TILE < num_super:
        SUPER_TILE *= 2
    SUPER_TILE = max(SUPER_TILE, 16)

    block_ones = torch.empty(num_blocks, dtype=torch.int32, device=x.device)
    super_ones = torch.empty(max(num_super, 1), dtype=torch.int32, device=x.device)
    total_ones = torch.empty(1, dtype=torch.int32, device=x.device)

    stream = torch.cuda.current_stream()

    for bit in range(32):
        ct.launch(stream, (num_blocks, 1, 1), _count_ones_kernel,
                  (x, block_ones, n, bit, BLOCK))
        ct.launch(stream, (num_super, 1, 1), _sum_and_scan_super_kernel,
                  (block_ones, super_ones, num_blocks, SUPER_SIZE))
        ct.launch(stream, (1, 1, 1), _scan_super_kernel,
                  (super_ones, total_ones, SUPER_TILE))
        ct.launch(stream, (num_blocks, 1, 1), _scatter_kernel,
                  (x, out, block_ones, super_ones, total_ones,
                   n, bit, SUPER_SIZE, BLOCK))
        x, out = out, x

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK, "SUPER_SIZE": SUPER_SIZE, "SUPER_TILE": SUPER_TILE,
    })
    return x


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
