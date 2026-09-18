import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count8_kernel(src_ptr, counts_ptr, N, num_blocks, shift,
                   BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 7

    b0 = tl.where((digit == 0) & mask, 1, 0)
    b1 = tl.where((digit == 1) & mask, 1, 0)
    b2 = tl.where((digit == 2) & mask, 1, 0)
    b3 = tl.where((digit == 3) & mask, 1, 0)
    b4 = tl.where((digit == 4) & mask, 1, 0)
    b5 = tl.where((digit == 5) & mask, 1, 0)
    b6 = tl.where((digit == 6) & mask, 1, 0)

    c0 = tl.sum(b0, axis=0)
    c1 = tl.sum(b1, axis=0)
    c2 = tl.sum(b2, axis=0)
    c3 = tl.sum(b3, axis=0)
    c4 = tl.sum(b4, axis=0)
    c5 = tl.sum(b5, axis=0)
    c6 = tl.sum(b6, axis=0)
    valid_count = tl.minimum(N - bid * BLOCK_SIZE, BLOCK_SIZE)
    c7 = valid_count - c0 - c1 - c2 - c3 - c4 - c5 - c6

    tl.store(counts_ptr + 0 * num_blocks + bid, c0)
    tl.store(counts_ptr + 1 * num_blocks + bid, c1)
    tl.store(counts_ptr + 2 * num_blocks + bid, c2)
    tl.store(counts_ptr + 3 * num_blocks + bid, c3)
    tl.store(counts_ptr + 4 * num_blocks + bid, c4)
    tl.store(counts_ptr + 5 * num_blocks + bid, c5)
    tl.store(counts_ptr + 6 * num_blocks + bid, c6)
    tl.store(counts_ptr + 7 * num_blocks + bid, c7)


@triton.jit
def _count4_kernel(src_ptr, counts_ptr, N, num_blocks, shift, flip,
                   BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (((vals >> shift) & 3) ^ flip)

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
def _group_offsets8_kernel(group_counts_ptr, group_prefix_ptr, bin_offsets_ptr,
                           GROUP_TILE: tl.constexpr):
    gr = tl.arange(0, GROUP_TILE)

    gc0 = tl.load(group_counts_ptr + 0 * GROUP_TILE + gr)
    gc1 = tl.load(group_counts_ptr + 1 * GROUP_TILE + gr)
    gc2 = tl.load(group_counts_ptr + 2 * GROUP_TILE + gr)
    gc3 = tl.load(group_counts_ptr + 3 * GROUP_TILE + gr)
    gc4 = tl.load(group_counts_ptr + 4 * GROUP_TILE + gr)
    gc5 = tl.load(group_counts_ptr + 5 * GROUP_TILE + gr)
    gc6 = tl.load(group_counts_ptr + 6 * GROUP_TILE + gr)
    gc7 = tl.load(group_counts_ptr + 7 * GROUP_TILE + gr)

    cs0 = tl.cumsum(gc0, axis=0)
    cs1 = tl.cumsum(gc1, axis=0)
    cs2 = tl.cumsum(gc2, axis=0)
    cs3 = tl.cumsum(gc3, axis=0)
    cs4 = tl.cumsum(gc4, axis=0)
    cs5 = tl.cumsum(gc5, axis=0)
    cs6 = tl.cumsum(gc6, axis=0)
    cs7 = tl.cumsum(gc7, axis=0)

    tl.store(group_prefix_ptr + 0 * GROUP_TILE + gr, cs0 - gc0)
    tl.store(group_prefix_ptr + 1 * GROUP_TILE + gr, cs1 - gc1)
    tl.store(group_prefix_ptr + 2 * GROUP_TILE + gr, cs2 - gc2)
    tl.store(group_prefix_ptr + 3 * GROUP_TILE + gr, cs3 - gc3)
    tl.store(group_prefix_ptr + 4 * GROUP_TILE + gr, cs4 - gc4)
    tl.store(group_prefix_ptr + 5 * GROUP_TILE + gr, cs5 - gc5)
    tl.store(group_prefix_ptr + 6 * GROUP_TILE + gr, cs6 - gc6)
    tl.store(group_prefix_ptr + 7 * GROUP_TILE + gr, cs7 - gc7)

    t0 = tl.sum(gc0, axis=0)
    t1 = tl.sum(gc1, axis=0)
    t2 = tl.sum(gc2, axis=0)
    t3 = tl.sum(gc3, axis=0)
    t4 = tl.sum(gc4, axis=0)
    t5 = tl.sum(gc5, axis=0)
    t6 = tl.sum(gc6, axis=0)

    tl.store(bin_offsets_ptr + 0, 0)
    tl.store(bin_offsets_ptr + 1, t0)
    tl.store(bin_offsets_ptr + 2, t0 + t1)
    tl.store(bin_offsets_ptr + 3, t0 + t1 + t2)
    tl.store(bin_offsets_ptr + 4, t0 + t1 + t2 + t3)
    tl.store(bin_offsets_ptr + 5, t0 + t1 + t2 + t3 + t4)
    tl.store(bin_offsets_ptr + 6, t0 + t1 + t2 + t3 + t4 + t5)
    tl.store(bin_offsets_ptr + 7, t0 + t1 + t2 + t3 + t4 + t5 + t6)


@triton.jit
def _group_offsets4_kernel(group_counts_ptr, group_prefix_ptr, bin_offsets_ptr,
                           GROUP_TILE: tl.constexpr):
    gr = tl.arange(0, GROUP_TILE)

    gc0 = tl.load(group_counts_ptr + 0 * GROUP_TILE + gr)
    gc1 = tl.load(group_counts_ptr + 1 * GROUP_TILE + gr)
    gc2 = tl.load(group_counts_ptr + 2 * GROUP_TILE + gr)
    gc3 = tl.load(group_counts_ptr + 3 * GROUP_TILE + gr)

    cs0 = tl.cumsum(gc0, axis=0)
    cs1 = tl.cumsum(gc1, axis=0)
    cs2 = tl.cumsum(gc2, axis=0)
    cs3 = tl.cumsum(gc3, axis=0)

    tl.store(group_prefix_ptr + 0 * GROUP_TILE + gr, cs0 - gc0)
    tl.store(group_prefix_ptr + 1 * GROUP_TILE + gr, cs1 - gc1)
    tl.store(group_prefix_ptr + 2 * GROUP_TILE + gr, cs2 - gc2)
    tl.store(group_prefix_ptr + 3 * GROUP_TILE + gr, cs3 - gc3)

    t0 = tl.sum(gc0, axis=0)
    t1 = tl.sum(gc1, axis=0)
    t2 = tl.sum(gc2, axis=0)

    tl.store(bin_offsets_ptr + 0, 0)
    tl.store(bin_offsets_ptr + 1, t0)
    tl.store(bin_offsets_ptr + 2, t0 + t1)
    tl.store(bin_offsets_ptr + 3, t0 + t1 + t2)


@triton.jit
def _scatter8_kernel(src_ptr, dst_ptr, prefix_ptr, group_prefix_ptr, bin_offsets_ptr,
                     N, num_blocks, shift,
                     BLOCK_SIZE: tl.constexpr,
                     BLOCK_COUNTS: tl.constexpr,
                     GROUP_TILE: tl.constexpr):
    bid = tl.program_id(0)

    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (vals >> shift) & 7

    b0 = tl.where((digit == 0) & mask, 1, 0)
    b1 = tl.where((digit == 1) & mask, 1, 0)
    b2 = tl.where((digit == 2) & mask, 1, 0)
    b3 = tl.where((digit == 3) & mask, 1, 0)
    b4 = tl.where((digit == 4) & mask, 1, 0)
    b5 = tl.where((digit == 5) & mask, 1, 0)
    b6 = tl.where((digit == 6) & mask, 1, 0)

    c0 = tl.cumsum(b0, axis=0)
    c1 = tl.cumsum(b1, axis=0)
    c2 = tl.cumsum(b2, axis=0)
    c3 = tl.cumsum(b3, axis=0)
    c4 = tl.cumsum(b4, axis=0)
    c5 = tl.cumsum(b5, axis=0)
    c6 = tl.cumsum(b6, axis=0)

    rank0 = c0 - b0
    rank1 = c1 - b1
    rank2 = c2 - b2
    rank3 = c3 - b3
    rank4 = c4 - b4
    rank5 = c5 - b5
    rank6 = c6 - b6
    rank7 = r - rank0 - rank1 - rank2 - rank3 - rank4 - rank5 - rank6

    local_rank = tl.where(
        digit == 0,
        rank0,
        tl.where(
            digit == 1,
            rank1,
            tl.where(
                digit == 2,
                rank2,
                tl.where(
                    digit == 3,
                    rank3,
                    tl.where(
                        digit == 4,
                        rank4,
                        tl.where(digit == 5, rank5, tl.where(digit == 6, rank6, rank7)),
                    ),
                ),
            ),
        ),
    )

    gid = bid // BLOCK_COUNTS

    p0 = tl.load(prefix_ptr + 0 * num_blocks + bid)
    p1 = tl.load(prefix_ptr + 1 * num_blocks + bid)
    p2 = tl.load(prefix_ptr + 2 * num_blocks + bid)
    p3 = tl.load(prefix_ptr + 3 * num_blocks + bid)
    p4 = tl.load(prefix_ptr + 4 * num_blocks + bid)
    p5 = tl.load(prefix_ptr + 5 * num_blocks + bid)
    p6 = tl.load(prefix_ptr + 6 * num_blocks + bid)
    p7 = tl.load(prefix_ptr + 7 * num_blocks + bid)

    gp0 = tl.load(group_prefix_ptr + 0 * GROUP_TILE + gid)
    gp1 = tl.load(group_prefix_ptr + 1 * GROUP_TILE + gid)
    gp2 = tl.load(group_prefix_ptr + 2 * GROUP_TILE + gid)
    gp3 = tl.load(group_prefix_ptr + 3 * GROUP_TILE + gid)
    gp4 = tl.load(group_prefix_ptr + 4 * GROUP_TILE + gid)
    gp5 = tl.load(group_prefix_ptr + 5 * GROUP_TILE + gid)
    gp6 = tl.load(group_prefix_ptr + 6 * GROUP_TILE + gid)
    gp7 = tl.load(group_prefix_ptr + 7 * GROUP_TILE + gid)

    o0 = tl.load(bin_offsets_ptr + 0)
    o1 = tl.load(bin_offsets_ptr + 1)
    o2 = tl.load(bin_offsets_ptr + 2)
    o3 = tl.load(bin_offsets_ptr + 3)
    o4 = tl.load(bin_offsets_ptr + 4)
    o5 = tl.load(bin_offsets_ptr + 5)
    o6 = tl.load(bin_offsets_ptr + 6)
    o7 = tl.load(bin_offsets_ptr + 7)

    before = tl.where(
        digit == 0,
        p0 + gp0,
        tl.where(
            digit == 1,
            p1 + gp1,
            tl.where(
                digit == 2,
                p2 + gp2,
                tl.where(
                    digit == 3,
                    p3 + gp3,
                    tl.where(
                        digit == 4,
                        p4 + gp4,
                        tl.where(digit == 5, p5 + gp5, tl.where(digit == 6, p6 + gp6, p7 + gp7)),
                    ),
                ),
            ),
        ),
    )

    bucket_base = tl.where(
        digit == 0,
        o0,
        tl.where(
            digit == 1,
            o1,
            tl.where(
                digit == 2,
                o2,
                tl.where(
                    digit == 3,
                    o3,
                    tl.where(digit == 4, o4, tl.where(digit == 5, o5, tl.where(digit == 6, o6, o7))),
                ),
            ),
        ),
    )

    dest = bucket_base + before + local_rank
    tl.store(dst_ptr + dest, vals, mask=mask)


@triton.jit
def _scatter4_kernel(src_ptr, dst_ptr, prefix_ptr, group_prefix_ptr, bin_offsets_ptr,
                     N, num_blocks, shift, flip,
                     BLOCK_SIZE: tl.constexpr,
                     BLOCK_COUNTS: tl.constexpr,
                     GROUP_TILE: tl.constexpr):
    bid = tl.program_id(0)

    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    digit = (((vals >> shift) & 3) ^ flip)

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

    gid = bid // BLOCK_COUNTS

    p0 = tl.load(prefix_ptr + 0 * num_blocks + bid)
    p1 = tl.load(prefix_ptr + 1 * num_blocks + bid)
    p2 = tl.load(prefix_ptr + 2 * num_blocks + bid)
    p3 = tl.load(prefix_ptr + 3 * num_blocks + bid)

    gp0 = tl.load(group_prefix_ptr + 0 * GROUP_TILE + gid)
    gp1 = tl.load(group_prefix_ptr + 1 * GROUP_TILE + gid)
    gp2 = tl.load(group_prefix_ptr + 2 * GROUP_TILE + gid)
    gp3 = tl.load(group_prefix_ptr + 3 * GROUP_TILE + gid)

    o0 = tl.load(bin_offsets_ptr + 0)
    o1 = tl.load(bin_offsets_ptr + 1)
    o2 = tl.load(bin_offsets_ptr + 2)
    o3 = tl.load(bin_offsets_ptr + 3)

    before = tl.where(
        digit == 0,
        p0 + gp0,
        tl.where(digit == 1, p1 + gp1, tl.where(digit == 2, p2 + gp2, p3 + gp3)),
    )
    bucket_base = tl.where(digit == 0, o0, tl.where(digit == 1, o1, tl.where(digit == 2, o2, o3)))

    dest = bucket_base + before + local_rank
    tl.store(dst_ptr + dest, vals, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty_like(input)
    tmp = torch.empty_like(input)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 1024
    GROUP_TILE = 32
    RADIX_BITS = 3
    FINAL_RADIX_BITS = 2
    LOWER_PASSES = 10
    PASSES = 11
    num_warps = 8
    num_stages = 3
    group_num_warps = 1

    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((8, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((8, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((8, GROUP_TILE), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((8, GROUP_TILE), device=input.device, dtype=torch.int32)
    bin_offsets = torch.empty((8,), device=input.device, dtype=torch.int32)

    grid_blocks = (num_blocks,)
    grid_prefix8 = (GROUP_TILE, 8)
    grid_prefix4 = (GROUP_TILE, 4)
    grid_group = (1,)

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

        _count8_kernel[grid_blocks](
            src, counts, n, num_blocks, shift,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _prefix_counts_kernel[grid_prefix8](
            counts, prefix, group_counts, num_blocks,
            BLOCK_COUNTS=BLOCK_COUNTS,
            GROUP_TILE=GROUP_TILE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _group_offsets8_kernel[grid_group](
            group_counts, group_prefix, bin_offsets,
            GROUP_TILE=GROUP_TILE,
            num_warps=group_num_warps,
            num_stages=num_stages,
        )
        _scatter8_kernel[grid_blocks](
            src, dst, prefix, group_prefix, bin_offsets,
            n, num_blocks, shift,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_COUNTS=BLOCK_COUNTS,
            GROUP_TILE=GROUP_TILE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _count4_kernel[grid_blocks](
        output, counts, n, num_blocks, 30, 2,
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
    _group_offsets4_kernel[grid_group](
        group_counts, group_prefix, bin_offsets,
        GROUP_TILE=GROUP_TILE,
        num_warps=group_num_warps,
        num_stages=num_stages,
    )
    _scatter4_kernel[grid_blocks](
        output, tmp, prefix, group_prefix, bin_offsets,
        n, num_blocks, 30, 2,
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
        "FINAL_RADIX_BITS": FINAL_RADIX_BITS,
        "LOWER_PASSES": LOWER_PASSES,
        "PASSES": PASSES,
        "SIGNED_INT_ORDER": True,
        "PRECOMPUTED_GROUP_PREFIX": True,
        "LOCAL_CUMSUMS_LOWER": 7,
        "LOCAL_CUMSUMS_FINAL": 3,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_num_warps": group_num_warps,
    })
    return tmp


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
