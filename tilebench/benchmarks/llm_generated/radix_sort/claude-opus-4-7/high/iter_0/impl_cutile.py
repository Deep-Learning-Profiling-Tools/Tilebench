import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel
def _count_ones_kernel(x, out, N, bit, BLOCK: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bit_val = (x_tile >> bit) & 1
    bit_val_m = ct.where(valid, bit_val, 0)
    s = ct.sum(bit_val_m)
    ct.store(out, index=(pid,), tile=s)


@ct.kernel
def _sum_super_kernel(block_ones, super_ones, num_blocks, BLOCK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(block_ones, index=(pid,), shape=(BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < num_blocks
    tile_m = ct.where(valid, tile, 0)
    s = ct.sum(tile_m)
    ct.store(super_ones, index=(pid,), tile=s)


@ct.kernel
def _scan_within_super_kernel(block_ones, num_blocks, BLOCK: ConstInt):
    pid = ct.bid(0)
    tile = ct.load(block_ones, index=(pid,), shape=(BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < num_blocks
    tile_m = ct.where(valid, tile, 0)
    inc = ct.cumsum(tile_m, axis=0)
    exc = inc - tile_m
    ct.store(block_ones, index=(pid,), tile=exc)


@ct.kernel
def _scan_super_kernel(super_ones, total_ones, num_super, BLOCK: ConstInt):
    tile = ct.load(super_ones, index=(0,), shape=(BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    offs = ct.arange(BLOCK, dtype=np.int32)
    valid = offs < num_super
    tile_m = ct.where(valid, tile, 0)
    total = ct.sum(tile_m)
    ct.store(total_ones, index=(0,), tile=total)
    inc = ct.cumsum(tile_m, axis=0)
    exc = inc - tile_m
    ct.store(super_ones, index=(0,), tile=exc)


@ct.kernel
def _scatter_kernel(x, out, block_prefix, super_prefix, total_ones,
                    N, bit, SUPER_SIZE: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    super_idx = pid // SUPER_SIZE

    block_pref = ct.load(block_prefix, index=(pid,), shape=())
    super_pref = ct.load(super_prefix, index=(super_idx,), shape=())
    ones_offset = block_pref + super_pref

    total_ones_val = ct.load(total_ones, index=(0,), shape=())
    total_zeros = N - total_ones_val

    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N

    bit_val = (x_tile >> bit) & 1
    bit_val_m = ct.where(valid, bit_val, 0)

    inc = ct.cumsum(bit_val_m, axis=0)
    local_ones_prefix = inc - bit_val_m
    arange_b = ct.arange(BLOCK, dtype=np.int32)
    local_zeros_prefix = arange_b - local_ones_prefix

    dest_if_one = total_zeros + ones_offset + local_ones_prefix
    zeros_before = pid * BLOCK - ones_offset
    dest_if_zero = zeros_before + local_zeros_prefix

    dest = ct.where(bit_val_m == 1, dest_if_one, dest_if_zero)
    dest = ct.where(valid, dest, N)  # OOB indices get dropped by scatter

    ct.scatter(out, (dest,), x_tile)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024
    SUPER_TILE = 128
    occupancy = 2

    n = N
    x = input.contiguous().clone()
    out = torch.empty_like(x)

    num_blocks = (n + BLOCK - 1) // BLOCK
    num_super = (num_blocks + SUPER_SIZE - 1) // SUPER_SIZE

    block_ones = torch.empty(num_blocks, dtype=torch.int32, device=x.device)
    super_ones = torch.empty(max(num_super, 1), dtype=torch.int32, device=x.device)
    total_ones = torch.empty(1, dtype=torch.int32, device=x.device)

    stream = torch.cuda.current_stream()

    k_count = _count_ones_kernel.with_hints(occupancy=occupancy)
    k_sum = _sum_super_kernel.with_hints(occupancy=occupancy)
    k_scan_in = _scan_within_super_kernel.with_hints(occupancy=occupancy)
    k_scan_sup = _scan_super_kernel.with_hints(occupancy=1)
    k_scatter = _scatter_kernel.with_hints(occupancy=occupancy)

    for bit in range(32):
        ct.launch(stream, (num_blocks, 1, 1), k_count,
                  (x, block_ones, n, bit, BLOCK))
        ct.launch(stream, (num_super, 1, 1), k_sum,
                  (block_ones, super_ones, num_blocks, SUPER_SIZE))
        ct.launch(stream, (num_super, 1, 1), k_scan_in,
                  (block_ones, num_blocks, SUPER_SIZE))
        ct.launch(stream, (1, 1, 1), k_scan_sup,
                  (super_ones, total_ones, num_super, SUPER_TILE))
        ct.launch(stream, (num_blocks, 1, 1), k_scatter,
                  (x, out, block_ones, super_ones, total_ones,
                   n, bit, SUPER_SIZE, BLOCK))
        x, out = out, x

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK, "SUPER_SIZE": SUPER_SIZE, "SUPER_TILE": SUPER_TILE,
        "occupancy": occupancy,
    })
    return x


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
