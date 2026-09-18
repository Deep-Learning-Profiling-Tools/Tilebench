import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count_ones_kernel(x_ptr, out_ptr, N, bit, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bit_val = ((x >> bit) & 1).to(tl.int32)
    bit_val = tl.where(mask, bit_val, 0)
    s = tl.sum(bit_val, axis=0)
    tl.store(out_ptr + pid, s)


@triton.jit
def _sum_super_kernel(block_ones_ptr, super_ones_ptr, num_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_blocks
    v = tl.load(block_ones_ptr + offs, mask=mask, other=0)
    s = tl.sum(v, axis=0)
    tl.store(super_ones_ptr + pid, s)


@triton.jit
def _scan_within_super_kernel(block_ones_ptr, num_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_blocks
    v = tl.load(block_ones_ptr + offs, mask=mask, other=0)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(block_ones_ptr + offs, exc, mask=mask)


@triton.jit
def _scan_super_kernel(super_ones_ptr, total_ones_ptr, num_super, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < num_super
    v = tl.load(super_ones_ptr + offs, mask=mask, other=0)
    total = tl.sum(v, axis=0)
    tl.store(total_ones_ptr, total)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(super_ones_ptr + offs, exc, mask=mask)


@triton.jit
def _scatter_kernel(x_ptr, out_ptr, block_prefix_ptr, super_prefix_ptr, total_ones_ptr,
                    N, bit, SUPER_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    super_idx = pid // SUPER_SIZE

    block_pref = tl.load(block_prefix_ptr + pid)
    super_pref = tl.load(super_prefix_ptr + super_idx)
    ones_offset = block_pref + super_pref

    total_ones = tl.load(total_ones_ptr)
    total_zeros = N - total_ones

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bit_val = ((x >> bit) & 1).to(tl.int32)
    bit_val_m = tl.where(mask, bit_val, 0)

    inc = tl.cumsum(bit_val_m, axis=0)
    local_ones_prefix = inc - bit_val_m
    arange_b = tl.arange(0, BLOCK)
    local_zeros_prefix = arange_b - local_ones_prefix

    dest_if_one = total_zeros + ones_offset + local_ones_prefix
    zeros_before = pid * BLOCK - ones_offset
    dest_if_zero = zeros_before + local_zeros_prefix

    dest = tl.where(bit_val_m == 1, dest_if_one, dest_if_zero)
    tl.store(out_ptr + dest, x, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024
    SUPER_TILE = 128
    num_warps = 4
    num_stages = 2

    n = N
    x = input.contiguous().clone()
    out = torch.empty_like(x)

    num_blocks = (n + BLOCK - 1) // BLOCK
    num_super = (num_blocks + SUPER_SIZE - 1) // SUPER_SIZE

    block_ones = torch.empty(num_blocks, dtype=torch.int32, device=x.device)
    super_ones = torch.empty(max(num_super, 1), dtype=torch.int32, device=x.device)
    total_ones = torch.empty(1, dtype=torch.int32, device=x.device)

    for bit in range(32):
        _count_ones_kernel[(num_blocks,)](
            x, block_ones, n, bit, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages)
        _sum_super_kernel[(num_super,)](
            block_ones, super_ones, num_blocks, BLOCK=SUPER_SIZE,
            num_warps=num_warps, num_stages=num_stages)
        _scan_within_super_kernel[(num_super,)](
            block_ones, num_blocks, BLOCK=SUPER_SIZE,
            num_warps=num_warps, num_stages=num_stages)
        _scan_super_kernel[(1,)](
            super_ones, total_ones, num_super, BLOCK=SUPER_TILE,
            num_warps=1, num_stages=1)
        _scatter_kernel[(num_blocks,)](
            x, out, block_ones, super_ones, total_ones,
            n, bit, SUPER_SIZE=SUPER_SIZE, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages)
        x, out = out, x

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK, "SUPER_SIZE": SUPER_SIZE, "SUPER_TILE": SUPER_TILE,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return x


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
