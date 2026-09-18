import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count_ones_kernel(src_ptr, counts_ptr, N, bit, flip, BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    bit_vals = ((vals >> bit) & 1) ^ flip
    bit_vals = tl.where(mask, bit_vals, 0)

    ones = tl.sum(bit_vals, axis=0)
    tl.store(counts_ptr + bid, ones)


@triton.jit
def _prefix_counts_kernel(counts_ptr, prefix_ptr, group_counts_ptr,
                          num_blocks, BLOCK_COUNTS: tl.constexpr):
    gid = tl.program_id(0)
    r = tl.arange(0, BLOCK_COUNTS)
    offs = gid * BLOCK_COUNTS + r
    mask = offs < num_blocks

    c = tl.load(counts_ptr + offs, mask=mask, other=0)
    cs = tl.cumsum(c, axis=0)
    excl = cs - c

    tl.store(prefix_ptr + offs, excl, mask=mask)
    total = tl.sum(c, axis=0)
    tl.store(group_counts_ptr + gid, total)


@triton.jit
def _prefix_groups_kernel(group_counts_ptr, group_prefix_ptr, total_ones_ptr,
                          num_groups, GROUP_TILE: tl.constexpr):
    r = tl.arange(0, GROUP_TILE)
    mask = r < num_groups

    c = tl.load(group_counts_ptr + r, mask=mask, other=0)
    cs = tl.cumsum(c, axis=0)
    excl = cs - c

    tl.store(group_prefix_ptr + r, excl, mask=mask)
    total = tl.sum(c, axis=0)
    tl.store(total_ones_ptr, total)


@triton.jit
def _scatter_kernel(src_ptr, dst_ptr, prefix_ptr, group_prefix_ptr, total_ones_ptr,
                    N, bit, flip,
                    BLOCK_SIZE: tl.constexpr, BLOCK_COUNTS: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    bit_vals = ((vals >> bit) & 1) ^ flip
    bit_vals = tl.where(mask, bit_vals, 0)

    ones_incl = tl.cumsum(bit_vals, axis=0)
    ones_excl = ones_incl - bit_vals
    zeros_excl = r - ones_excl

    block_prefix = tl.load(prefix_ptr + bid)
    group_prefix = tl.load(group_prefix_ptr + (bid // BLOCK_COUNTS))
    ones_before = block_prefix + group_prefix

    total_ones = tl.load(total_ones_ptr)
    total_zeros = N - total_ones
    zeros_before = bid * BLOCK_SIZE - ones_before

    dest_zero = zeros_before + zeros_excl
    dest_one = total_zeros + ones_before + ones_excl
    dest = tl.where(bit_vals == 0, dest_zero, dest_one)

    tl.store(dst_ptr + dest, vals, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty_like(input)
    tmp = torch.empty_like(input)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 1024
    GROUP_TILE = 256
    num_warps = 8
    num_stages = 3

    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    num_groups = triton.cdiv(num_blocks, BLOCK_COUNTS)

    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int32)
    prefix = torch.empty((num_blocks,), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((num_groups,), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((num_groups,), device=input.device, dtype=torch.int32)
    total_ones = torch.empty((1,), device=input.device, dtype=torch.int32)

    grid_blocks = (num_blocks,)
    grid_groups = (num_groups,)

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

        _count_ones_kernel[grid_blocks](
            src, counts, n, bit, flip,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _prefix_counts_kernel[grid_groups](
            counts, prefix, group_counts, num_blocks,
            BLOCK_COUNTS=BLOCK_COUNTS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _prefix_groups_kernel[(1,)](
            group_counts, group_prefix, total_ones, num_groups,
            GROUP_TILE=GROUP_TILE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _scatter_kernel[grid_blocks](
            src, dst, prefix, group_prefix, total_ones,
            n, bit, flip,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_COUNTS=BLOCK_COUNTS,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "BLOCK_COUNTS": BLOCK_COUNTS,
        "GROUP_TILE": GROUP_TILE,
        "RADIX_BITS": 1,
        "PASSES": 32,
        "SIGNED_INT_ORDER": True,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
