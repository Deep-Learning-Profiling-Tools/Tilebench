import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count_kernel(x_ptr, counts_ptr, N, bit, num_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bin_val = ((x >> bit) & 3).to(tl.int32)
    bin_val_m = tl.where(mask, bin_val, -1)
    c0 = tl.sum((bin_val_m == 0).to(tl.int32), axis=0)
    c1 = tl.sum((bin_val_m == 1).to(tl.int32), axis=0)
    c2 = tl.sum((bin_val_m == 2).to(tl.int32), axis=0)
    c3 = tl.sum((bin_val_m == 3).to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, c0)
    tl.store(counts_ptr + num_blocks + pid, c1)
    tl.store(counts_ptr + 2 * num_blocks + pid, c2)
    tl.store(counts_ptr + 3 * num_blocks + pid, c3)


@triton.jit
def _scan_super_per_bin(counts_ptr, super_sums_ptr, num_blocks, num_super,
                        SUPER: tl.constexpr):
    pid = tl.program_id(0)
    bin_idx = tl.program_id(1)
    offs = pid * SUPER + tl.arange(0, SUPER)
    mask = offs < num_blocks
    ptrs = counts_ptr + bin_idx * num_blocks + offs
    v = tl.load(ptrs, mask=mask, other=0)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(ptrs, exc, mask=mask)
    s = tl.sum(v, axis=0)
    tl.store(super_sums_ptr + bin_idx * num_super + pid, s)


@triton.jit
def _scan_top(super_sums_ptr, bin_starts_ptr, num_super,
              NBINS: tl.constexpr, BLOCK: tl.constexpr):
    bin_idx = tl.arange(0, NBINS)
    offs = tl.arange(0, BLOCK)
    mask = offs < num_super
    ptrs = super_sums_ptr + bin_idx[:, None] * num_super + offs[None, :]
    v = tl.load(ptrs, mask=mask[None, :], other=0)
    inc = tl.cumsum(v, axis=1)
    exc = inc - v
    tl.store(ptrs, exc, mask=mask[None, :])
    total = tl.sum(v, axis=1)
    total_inc = tl.cumsum(total, axis=0)
    bin_starts = total_inc - total
    tl.store(bin_starts_ptr + bin_idx, bin_starts)


@triton.jit
def _scatter_kernel(x_ptr, out_ptr, counts_ptr, super_sums_ptr, bin_starts_ptr,
                    N, bit, num_blocks, num_super,
                    SUPER_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    super_idx = pid // SUPER_SIZE

    p0 = tl.load(counts_ptr + pid)
    p1 = tl.load(counts_ptr + num_blocks + pid)
    p2 = tl.load(counts_ptr + 2 * num_blocks + pid)
    p3 = tl.load(counts_ptr + 3 * num_blocks + pid)
    s0 = tl.load(super_sums_ptr + super_idx)
    s1 = tl.load(super_sums_ptr + num_super + super_idx)
    s2 = tl.load(super_sums_ptr + 2 * num_super + super_idx)
    s3 = tl.load(super_sums_ptr + 3 * num_super + super_idx)
    bs0 = tl.load(bin_starts_ptr + 0)
    bs1 = tl.load(bin_starts_ptr + 1)
    bs2 = tl.load(bin_starts_ptr + 2)
    bs3 = tl.load(bin_starts_ptr + 3)

    pref0 = bs0 + s0 + p0
    pref1 = bs1 + s1 + p1
    pref2 = bs2 + s2 + p2
    pref3 = bs3 + s3 + p3

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bin_val = ((x >> bit) & 3).to(tl.int32)
    bin_val_m = tl.where(mask, bin_val, -1)

    is0 = (bin_val_m == 0).to(tl.int32)
    is1 = (bin_val_m == 1).to(tl.int32)
    is2 = (bin_val_m == 2).to(tl.int32)
    is3 = (bin_val_m == 3).to(tl.int32)

    r0 = tl.cumsum(is0, axis=0) - is0
    r1 = tl.cumsum(is1, axis=0) - is1
    r2 = tl.cumsum(is2, axis=0) - is2
    r3 = tl.cumsum(is3, axis=0) - is3

    dest = tl.where(bin_val_m == 0, pref0 + r0,
             tl.where(bin_val_m == 1, pref1 + r1,
              tl.where(bin_val_m == 2, pref2 + r2,
                                       pref3 + r3)))

    tl.store(out_ptr + dest, x, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024
    NBINS = 4
    NBITS = 2
    num_warps = 4
    num_stages = 2

    n = N
    x = input.contiguous().clone()
    out = torch.empty_like(x)

    num_blocks = (n + BLOCK - 1) // BLOCK
    num_super = (num_blocks + SUPER_SIZE - 1) // SUPER_SIZE
    SUPER_TILE = 1
    while SUPER_TILE < num_super:
        SUPER_TILE *= 2
    SUPER_TILE = max(SUPER_TILE, 16)

    counts = torch.empty(NBINS * num_blocks, dtype=torch.int32, device=x.device)
    super_sums = torch.empty(NBINS * max(num_super, 1), dtype=torch.int32,
                              device=x.device)
    bin_starts = torch.empty(NBINS, dtype=torch.int32, device=x.device)

    for bit in range(0, 32, NBITS):
        _count_kernel[(num_blocks,)](
            x, counts, n, bit, num_blocks, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages)
        _scan_super_per_bin[(num_super, NBINS)](
            counts, super_sums, num_blocks, num_super, SUPER=SUPER_SIZE,
            num_warps=num_warps, num_stages=num_stages)
        _scan_top[(1,)](
            super_sums, bin_starts, num_super,
            NBINS=NBINS, BLOCK=SUPER_TILE,
            num_warps=1, num_stages=1)
        _scatter_kernel[(num_blocks,)](
            x, out, counts, super_sums, bin_starts,
            n, bit, num_blocks, num_super,
            SUPER_SIZE=SUPER_SIZE, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages)
        x, out = out, x

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK, "SUPER_SIZE": SUPER_SIZE, "SUPER_TILE": SUPER_TILE,
        "NBINS": NBINS, "NBITS": NBITS,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return x


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
