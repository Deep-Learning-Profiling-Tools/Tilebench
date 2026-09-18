import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count4_kernel(src_ptr, counts_ptr, N, num_blocks, shift,
                   BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 3

    b0 = tl.where((digit == 0) & mask, 1, 0)
    b1 = tl.where((digit == 1) & mask, 1, 0)
    b2 = tl.where((digit == 2) & mask, 1, 0)

    c0 = tl.sum(b0, axis=0)
    c1 = tl.sum(b1, axis=0)
    c2 = tl.sum(b2, axis=0)
    valid_count = tl.minimum(N - bid * BLOCK_SIZE, BLOCK_SIZE)
    c3 = valid_count - c0 - c1 - c2

    tl.store(counts_ptr + 0 * num_blocks + bid, c0)
    tl.store(counts_ptr + 1 * num_blocks + bid, c1)
    tl.store(counts_ptr + 2 * num_blocks + bid, c2)
    tl.store(counts_ptr + 3 * num_blocks + bid, c3)


@triton.jit
def _count2_kernel(src_ptr, counts_ptr, N, num_blocks, shift,
                   BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 1

    b0 = tl.where((digit == 0) & mask, 1, 0)
    c0 = tl.sum(b0, axis=0)
    valid_count = tl.minimum(N - bid * BLOCK_SIZE, BLOCK_SIZE)
    c1 = valid_count - c0

    tl.store(counts_ptr + 0 * num_blocks + bid, c0)
    tl.store(counts_ptr + 1 * num_blocks + bid, c1)


@triton.jit
def _prefix_counts_kernel(counts_ptr, prefix_ptr, group_counts_ptr,
                          num_blocks,
                          BLOCK_COUNTS: tl.constexpr,
                          GROUP_TILE: tl.constexpr):
    gid = tl.program_id(0)
    bin_id = tl.program_id(1)

    r = tl.arange(0, BLOCK_COUNTS)
    offs = gid * BLOCK_COUNTS + r
    mask = offs < num_blocks

    c = tl.load(counts_ptr + bin_id * num_blocks + offs, mask=mask, other=0)
    cs = tl.cumsum(c, axis=0)
    excl = cs - c

    tl.store(prefix_ptr + bin_id * num_blocks + offs, excl, mask=mask)
    total = tl.sum(c, axis=0)
    tl.store(group_counts_ptr + bin_id * GROUP_TILE + gid, total)


@triton.jit
def _scatter4_kernel(src_ptr, dst_ptr, prefix_ptr, group_counts_ptr,
                     N, num_blocks, shift,
                     BLOCK_SIZE: tl.constexpr,
                     BLOCK_COUNTS: tl.constexpr,
                     GROUP_TILE: tl.constexpr):
    bid = tl.program_id(0)

    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 3

    b0 = tl.where((digit == 0) & mask, 1, 0)
    b1 = tl.where((digit == 1) & mask, 1, 0)
    b2 = tl.where((digit == 2) & mask, 1, 0)

    c0 = tl.cumsum(b0, axis=0)
    c1 = tl.cumsum(b1, axis=0)
    c2 = tl.cumsum(b2, axis=0)

    rank0 = c0 - b0
    rank1 = c1 - b1
    rank2 = c2 - b2
    rank3 = r - rank0 - rank1 - rank2

    local_rank = tl.where(
        digit == 0,
        rank0,
        tl.where(digit == 1, rank1, tl.where(digit == 2, rank2, rank3)),
    )

    p0 = tl.load(prefix_ptr + 0 * num_blocks + bid)
    p1 = tl.load(prefix_ptr + 1 * num_blocks + bid)
    p2 = tl.load(prefix_ptr + 2 * num_blocks + bid)
    p3 = tl.load(prefix_ptr + 3 * num_blocks + bid)

    gid = bid // BLOCK_COUNTS
    gr = tl.arange(0, GROUP_TILE)

    gc0 = tl.load(group_counts_ptr + 0 * GROUP_TILE + gr)
    gc1 = tl.load(group_counts_ptr + 1 * GROUP_TILE + gr)
    gc2 = tl.load(group_counts_ptr + 2 * GROUP_TILE + gr)
    gc3 = tl.load(group_counts_ptr + 3 * GROUP_TILE + gr)

    gp0 = tl.sum(tl.where(gr < gid, gc0, 0), axis=0)
    gp1 = tl.sum(tl.where(gr < gid, gc1, 0), axis=0)
    gp2 = tl.sum(tl.where(gr < gid, gc2, 0), axis=0)
    gp3 = tl.sum(tl.where(gr < gid, gc3, 0), axis=0)

    total0 = tl.sum(gc0, axis=0)
    total1 = tl.sum(gc1, axis=0)
    total2 = tl.sum(gc2, axis=0)

    before = tl.where(
        digit == 0,
        p0 + gp0,
        tl.where(digit == 1, p1 + gp1, tl.where(digit == 2, p2 + gp2, p3 + gp3)),
    )

    bucket_base = tl.where(
        digit == 0,
        0,
        tl.where(digit == 1, total0, tl.where(digit == 2, total0 + total1, total0 + total1 + total2)),
    )

    dest = bucket_base + before + local_rank
    dest = tl.where(mask, dest, N)

    tl.store(dst_ptr + dest, vals)


@triton.jit
def _scatter2_kernel(src_ptr, dst_ptr, prefix_ptr, group_counts_ptr,
                     N, num_blocks, shift,
                     BLOCK_SIZE: tl.constexpr,
                     BLOCK_COUNTS: tl.constexpr,
                     GROUP_TILE: tl.constexpr):
    bid = tl.program_id(0)

    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 1

    b0 = tl.where((digit == 0) & mask, 1, 0)
    c0 = tl.cumsum(b0, axis=0)

    rank0 = c0 - b0
    rank1 = r - rank0
    local_rank = tl.where(digit == 0, rank0, rank1)

    p0 = tl.load(prefix_ptr + 0 * num_blocks + bid)
    p1 = tl.load(prefix_ptr + 1 * num_blocks + bid)

    gid = bid // BLOCK_COUNTS
    gr = tl.arange(0, GROUP_TILE)

    gc0 = tl.load(group_counts_ptr + 0 * GROUP_TILE + gr)
    gc1 = tl.load(group_counts_ptr + 1 * GROUP_TILE + gr)

    gp0 = tl.sum(tl.where(gr < gid, gc0, 0), axis=0)
    gp1 = tl.sum(tl.where(gr < gid, gc1, 0), axis=0)
    total0 = tl.sum(gc0, axis=0)

    before = tl.where(digit == 0, p0 + gp0, p1 + gp1)
    bucket_base = tl.where(digit == 0, 0, total0)

    dest = bucket_base + before + local_rank
    dest = tl.where(mask, dest, N)

    tl.store(dst_ptr + dest, vals)


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty((n + 1,), device=input.device, dtype=input.dtype)
    tmp = torch.empty((n + 1,), device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 1024
    GROUP_TILE = 32
    RADIX_BITS = 2
    LOWER_PASSES = 15
    FINAL_RADIX_BITS = 1
    PASSES = 16
    num_warps = 8
    num_stages = 3

    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)

    grid_blocks = (num_blocks,)
    grid_prefix4 = (GROUP_TILE, 4)
    grid_prefix2 = (GROUP_TILE, 2)

    for pass_id in range(LOWER_PASSES):
        shift = pass_id * RADIX_BITS

        if pass_id == 0:
            src = input
            dst = tmp
        elif (pass_id & 1) == 1:
            src = tmp
            dst = output
        else:
            src = output
            dst = tmp

        _count4_kernel[grid_blocks](
            src, counts, n, num_blocks, shift,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _prefix_counts_kernel[grid_prefix4](
            counts, prefix, group_counts, num_blocks,
            BLOCK_COUNTS=BLOCK_COUNTS,
            GROUP_TILE=GROUP_TILE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _scatter4_kernel[grid_blocks](
            src, dst, prefix, group_counts,
            n, num_blocks, shift,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_COUNTS=BLOCK_COUNTS,
            GROUP_TILE=GROUP_TILE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    shift = 30
    src = tmp
    dst = output

    _count2_kernel[grid_blocks](
        src, counts, n, num_blocks, shift,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _prefix_counts_kernel[grid_prefix2](
        counts, prefix, group_counts, num_blocks,
        BLOCK_COUNTS=BLOCK_COUNTS,
        GROUP_TILE=GROUP_TILE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _scatter2_kernel[grid_blocks](
        src, dst, prefix, group_counts,
        n, num_blocks, shift,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_COUNTS=BLOCK_COUNTS,
        GROUP_TILE=GROUP_TILE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "BLOCK_COUNTS": BLOCK_COUNTS,
        "GROUP_TILE": GROUP_TILE,
        "RADIX_BITS": RADIX_BITS,
        "LOWER_PASSES": LOWER_PASSES,
        "FINAL_RADIX_BITS": FINAL_RADIX_BITS,
        "PASSES": PASSES,
        "ASSUME_NONNEGATIVE_INT32": True,
        "FUSED_GROUP_PREFIX_IN_SCATTER": True,
        "LOCAL_CUMSUMS_LOWER": 3,
        "LOCAL_CUMSUMS_FINAL": 1,
        "COUNT_C3_FROM_VALID": True,
        "GUARD_ELEMENTS": 1,
        "UNCHECKED_SCATTER": True,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output[:n]


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
