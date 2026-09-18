import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _count_ones_kernel(src, counts, N, bit, flip, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    r = ct.arange(BLOCK_SIZE, dtype=np.int32)
    offs = bid * BLOCK_SIZE + r
    valid = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(BLOCK_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    bit_vals = ((vals >> bit) & 1) ^ flip
    bit_vals = ct.where(valid, bit_vals, 0)

    ones = ct.sum(bit_vals, axis=0)
    ct.store(counts, index=(bid,), tile=ones, allow_tma=False)


@ct.kernel
def _prefix_counts_kernel(counts, prefix, group_counts, BLOCK_COUNTS: ConstInt):
    gid = ct.bid(0)

    c = ct.load(
        counts,
        index=(gid,),
        shape=(BLOCK_COUNTS,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    cs = ct.cumsum(c, axis=0)
    excl = cs - c

    ct.store(prefix, index=(gid,), tile=excl, allow_tma=False)
    total = ct.sum(c, axis=0)
    ct.store(group_counts, index=(gid,), tile=total, allow_tma=False)


@ct.kernel
def _prefix_groups_kernel(group_counts, group_prefix, total_ones, GROUP_TILE: ConstInt):
    c = ct.load(
        group_counts,
        index=(0,),
        shape=(GROUP_TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    cs = ct.cumsum(c, axis=0)
    excl = cs - c

    ct.store(group_prefix, index=(0,), tile=excl, allow_tma=False)
    total = ct.sum(c, axis=0)
    ct.store(total_ones, index=(0,), tile=total, allow_tma=False)


@ct.kernel
def _scatter_kernel(src, dst, prefix, group_prefix, total_ones,
                    N, bit, flip,
                    BLOCK_SIZE: ConstInt, BLOCK_COUNTS: ConstInt):
    bid = ct.bid(0)
    r = ct.arange(BLOCK_SIZE, dtype=np.int32)
    offs = bid * BLOCK_SIZE + r
    valid = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(BLOCK_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    bit_vals = ((vals >> bit) & 1) ^ flip
    bit_vals = ct.where(valid, bit_vals, 0)

    ones_incl = ct.cumsum(bit_vals, axis=0)
    ones_excl = ones_incl - bit_vals
    zeros_excl = r - ones_excl

    block_prefix = ct.load(prefix, index=(bid,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    group_prefix_val = ct.load(
        group_prefix,
        index=(bid // BLOCK_COUNTS,),
        shape=(),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    ones_before = block_prefix + group_prefix_val

    total = ct.load(total_ones, index=(0,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    total_zeros = N - total
    zeros_before = bid * BLOCK_SIZE - ones_before

    dest_zero = zeros_before + zeros_excl
    dest_one = total_zeros + ones_before + ones_excl
    dest = ct.where(bit_vals == 0, dest_zero, dest_one)
    dest = ct.where(valid, dest, N)

    ct.scatter(dst, dest, vals, check_bounds=True)


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty_like(input)
    tmp = torch.empty_like(input)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 1024
    GROUP_TILE = 256
    occupancy = 4

    num_blocks = ct.cdiv(n, BLOCK_SIZE)
    num_groups = ct.cdiv(num_blocks, BLOCK_COUNTS)

    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int32)
    prefix = torch.empty((num_blocks,), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((num_groups,), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((num_groups,), device=input.device, dtype=torch.int32)
    total_ones = torch.empty((1,), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()

    count_kernel = _count_ones_kernel.with_hints(occupancy=occupancy)
    prefix_counts_kernel = _prefix_counts_kernel.with_hints(occupancy=occupancy)
    prefix_groups_kernel = _prefix_groups_kernel.with_hints(occupancy=occupancy)
    scatter_kernel = _scatter_kernel.with_hints(occupancy=occupancy)

    grid_blocks = (num_blocks, 1, 1)
    grid_groups = (num_groups, 1, 1)

    for bit in range(32):
        flip = 1 if bit == 31 else 0

        if bit == 0:
            src = input
            dst = tmp
        elif (bit & 1) == 1:
            src = tmp
            dst = output
        else:
            src = output
            dst = tmp

        ct.launch(stream, grid_blocks, count_kernel, (src, counts, n, bit, flip, BLOCK_SIZE))
        ct.launch(stream, grid_groups, prefix_counts_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
        ct.launch(stream, (1, 1, 1), prefix_groups_kernel, (group_counts, group_prefix, total_ones, GROUP_TILE))
        ct.launch(
            stream,
            grid_blocks,
            scatter_kernel,
            (src, dst, prefix, group_prefix, total_ones, n, bit, flip, BLOCK_SIZE, BLOCK_COUNTS),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "BLOCK_COUNTS": BLOCK_COUNTS,
        "GROUP_TILE": GROUP_TILE,
        "RADIX_BITS": 1,
        "PASSES": 32,
        "SIGNED_INT_ORDER": True,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
