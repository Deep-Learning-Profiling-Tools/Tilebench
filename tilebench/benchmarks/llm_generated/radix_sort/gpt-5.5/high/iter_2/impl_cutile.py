import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _count8_kernel(src, counts, N, shift, BLOCK_SIZE: ConstInt):
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
    digit = (vals >> shift) & 7

    b0 = ct.where((digit == 0) & valid, 1, 0)
    b1 = ct.where((digit == 1) & valid, 1, 0)
    b2 = ct.where((digit == 2) & valid, 1, 0)
    b3 = ct.where((digit == 3) & valid, 1, 0)
    b4 = ct.where((digit == 4) & valid, 1, 0)
    b5 = ct.where((digit == 5) & valid, 1, 0)
    b6 = ct.where((digit == 6) & valid, 1, 0)
    b7 = ct.where((digit == 7) & valid, 1, 0)

    ct.store(counts, index=(0, bid), tile=ct.sum(b0), allow_tma=False)
    ct.store(counts, index=(1, bid), tile=ct.sum(b1), allow_tma=False)
    ct.store(counts, index=(2, bid), tile=ct.sum(b2), allow_tma=False)
    ct.store(counts, index=(3, bid), tile=ct.sum(b3), allow_tma=False)
    ct.store(counts, index=(4, bid), tile=ct.sum(b4), allow_tma=False)
    ct.store(counts, index=(5, bid), tile=ct.sum(b5), allow_tma=False)
    ct.store(counts, index=(6, bid), tile=ct.sum(b6), allow_tma=False)
    ct.store(counts, index=(7, bid), tile=ct.sum(b7), allow_tma=False)


@ct.kernel
def _count4_kernel(src, counts, N, shift, flip, BLOCK_SIZE: ConstInt):
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
    digit = ((vals >> shift) & 3) ^ flip

    b0 = ct.where((digit == 0) & valid, 1, 0)
    b1 = ct.where((digit == 1) & valid, 1, 0)
    b2 = ct.where((digit == 2) & valid, 1, 0)
    b3 = ct.where((digit == 3) & valid, 1, 0)

    ct.store(counts, index=(0, bid), tile=ct.sum(b0), allow_tma=False)
    ct.store(counts, index=(1, bid), tile=ct.sum(b1), allow_tma=False)
    ct.store(counts, index=(2, bid), tile=ct.sum(b2), allow_tma=False)
    ct.store(counts, index=(3, bid), tile=ct.sum(b3), allow_tma=False)


@ct.kernel
def _prefix_counts_kernel(counts, prefix, group_counts, BLOCK_COUNTS: ConstInt):
    gid = ct.bid(0)
    bin_id = ct.bid(1)

    c = ct.load(
        counts,
        index=(bin_id, gid),
        shape=(1, BLOCK_COUNTS),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    cs = ct.cumsum(c, axis=1)
    excl = cs - c

    ct.store(prefix, index=(bin_id, gid), tile=excl, allow_tma=False)
    total = ct.sum(c)
    ct.store(group_counts, index=(bin_id, gid), tile=total, allow_tma=False)


@ct.kernel
def _group_offsets8_kernel(group_counts, group_prefix, bin_offsets, GROUP_TILE: ConstInt):
    gc0 = ct.load(group_counts, index=(0, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc1 = ct.load(group_counts, index=(1, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc2 = ct.load(group_counts, index=(2, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc3 = ct.load(group_counts, index=(3, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc4 = ct.load(group_counts, index=(4, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc5 = ct.load(group_counts, index=(5, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc6 = ct.load(group_counts, index=(6, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc7 = ct.load(group_counts, index=(7, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    ct.store(group_prefix, index=(0, 0), tile=ct.cumsum(gc0, axis=1) - gc0, allow_tma=False)
    ct.store(group_prefix, index=(1, 0), tile=ct.cumsum(gc1, axis=1) - gc1, allow_tma=False)
    ct.store(group_prefix, index=(2, 0), tile=ct.cumsum(gc2, axis=1) - gc2, allow_tma=False)
    ct.store(group_prefix, index=(3, 0), tile=ct.cumsum(gc3, axis=1) - gc3, allow_tma=False)
    ct.store(group_prefix, index=(4, 0), tile=ct.cumsum(gc4, axis=1) - gc4, allow_tma=False)
    ct.store(group_prefix, index=(5, 0), tile=ct.cumsum(gc5, axis=1) - gc5, allow_tma=False)
    ct.store(group_prefix, index=(6, 0), tile=ct.cumsum(gc6, axis=1) - gc6, allow_tma=False)
    ct.store(group_prefix, index=(7, 0), tile=ct.cumsum(gc7, axis=1) - gc7, allow_tma=False)

    t0 = ct.sum(gc0)
    t1 = ct.sum(gc1)
    t2 = ct.sum(gc2)
    t3 = ct.sum(gc3)
    t4 = ct.sum(gc4)
    t5 = ct.sum(gc5)
    t6 = ct.sum(gc6)

    ct.store(bin_offsets, index=(0,), tile=0, allow_tma=False)
    ct.store(bin_offsets, index=(1,), tile=t0, allow_tma=False)
    ct.store(bin_offsets, index=(2,), tile=t0 + t1, allow_tma=False)
    ct.store(bin_offsets, index=(3,), tile=t0 + t1 + t2, allow_tma=False)
    ct.store(bin_offsets, index=(4,), tile=t0 + t1 + t2 + t3, allow_tma=False)
    ct.store(bin_offsets, index=(5,), tile=t0 + t1 + t2 + t3 + t4, allow_tma=False)
    ct.store(bin_offsets, index=(6,), tile=t0 + t1 + t2 + t3 + t4 + t5, allow_tma=False)
    ct.store(bin_offsets, index=(7,), tile=t0 + t1 + t2 + t3 + t4 + t5 + t6, allow_tma=False)


@ct.kernel
def _group_offsets4_kernel(group_counts, group_prefix, bin_offsets, GROUP_TILE: ConstInt):
    gc0 = ct.load(group_counts, index=(0, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc1 = ct.load(group_counts, index=(1, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc2 = ct.load(group_counts, index=(2, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gc3 = ct.load(group_counts, index=(3, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    ct.store(group_prefix, index=(0, 0), tile=ct.cumsum(gc0, axis=1) - gc0, allow_tma=False)
    ct.store(group_prefix, index=(1, 0), tile=ct.cumsum(gc1, axis=1) - gc1, allow_tma=False)
    ct.store(group_prefix, index=(2, 0), tile=ct.cumsum(gc2, axis=1) - gc2, allow_tma=False)
    ct.store(group_prefix, index=(3, 0), tile=ct.cumsum(gc3, axis=1) - gc3, allow_tma=False)

    t0 = ct.sum(gc0)
    t1 = ct.sum(gc1)
    t2 = ct.sum(gc2)

    ct.store(bin_offsets, index=(0,), tile=0, allow_tma=False)
    ct.store(bin_offsets, index=(1,), tile=t0, allow_tma=False)
    ct.store(bin_offsets, index=(2,), tile=t0 + t1, allow_tma=False)
    ct.store(bin_offsets, index=(3,), tile=t0 + t1 + t2, allow_tma=False)


@ct.kernel
def _scatter8_kernel(src, dst, prefix, group_prefix, bin_offsets,
                     N, shift,
                     BLOCK_SIZE: ConstInt,
                     BLOCK_COUNTS: ConstInt,
                     GROUP_TILE: ConstInt):
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
    digit = (vals >> shift) & 7

    b0 = ct.where((digit == 0) & valid, 1, 0)
    b1 = ct.where((digit == 1) & valid, 1, 0)
    b2 = ct.where((digit == 2) & valid, 1, 0)
    b3 = ct.where((digit == 3) & valid, 1, 0)
    b4 = ct.where((digit == 4) & valid, 1, 0)
    b5 = ct.where((digit == 5) & valid, 1, 0)
    b6 = ct.where((digit == 6) & valid, 1, 0)

    rank0 = ct.cumsum(b0, axis=0) - b0
    rank1 = ct.cumsum(b1, axis=0) - b1
    rank2 = ct.cumsum(b2, axis=0) - b2
    rank3 = ct.cumsum(b3, axis=0) - b3
    rank4 = ct.cumsum(b4, axis=0) - b4
    rank5 = ct.cumsum(b5, axis=0) - b5
    rank6 = ct.cumsum(b6, axis=0) - b6
    rank7 = r - rank0 - rank1 - rank2 - rank3 - rank4 - rank5 - rank6

    local_rank = ct.where(
        digit == 0,
        rank0,
        ct.where(
            digit == 1,
            rank1,
            ct.where(
                digit == 2,
                rank2,
                ct.where(
                    digit == 3,
                    rank3,
                    ct.where(
                        digit == 4,
                        rank4,
                        ct.where(digit == 5, rank5, ct.where(digit == 6, rank6, rank7)),
                    ),
                ),
            ),
        ),
    )

    gid = bid // BLOCK_COUNTS

    p0 = ct.load(prefix, index=(0, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p1 = ct.load(prefix, index=(1, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p2 = ct.load(prefix, index=(2, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p3 = ct.load(prefix, index=(3, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p4 = ct.load(prefix, index=(4, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p5 = ct.load(prefix, index=(5, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p6 = ct.load(prefix, index=(6, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p7 = ct.load(prefix, index=(7, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    gp0 = ct.load(group_prefix, index=(0, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp1 = ct.load(group_prefix, index=(1, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp2 = ct.load(group_prefix, index=(2, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp3 = ct.load(group_prefix, index=(3, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp4 = ct.load(group_prefix, index=(4, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp5 = ct.load(group_prefix, index=(5, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp6 = ct.load(group_prefix, index=(6, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp7 = ct.load(group_prefix, index=(7, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    o0 = ct.load(bin_offsets, index=(0,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o1 = ct.load(bin_offsets, index=(1,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o2 = ct.load(bin_offsets, index=(2,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o3 = ct.load(bin_offsets, index=(3,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o4 = ct.load(bin_offsets, index=(4,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o5 = ct.load(bin_offsets, index=(5,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o6 = ct.load(bin_offsets, index=(6,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o7 = ct.load(bin_offsets, index=(7,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    before = ct.where(
        digit == 0,
        p0 + gp0,
        ct.where(
            digit == 1,
            p1 + gp1,
            ct.where(
                digit == 2,
                p2 + gp2,
                ct.where(
                    digit == 3,
                    p3 + gp3,
                    ct.where(
                        digit == 4,
                        p4 + gp4,
                        ct.where(digit == 5, p5 + gp5, ct.where(digit == 6, p6 + gp6, p7 + gp7)),
                    ),
                ),
            ),
        ),
    )

    bucket_base = ct.where(
        digit == 0,
        o0,
        ct.where(
            digit == 1,
            o1,
            ct.where(
                digit == 2,
                o2,
                ct.where(
                    digit == 3,
                    o3,
                    ct.where(digit == 4, o4, ct.where(digit == 5, o5, ct.where(digit == 6, o6, o7))),
                ),
            ),
        ),
    )

    dest = bucket_base + before + local_rank
    dest = ct.where(valid, dest, N)
    ct.scatter(dst, dest, vals, check_bounds=True)


@ct.kernel
def _scatter4_kernel(src, dst, prefix, group_prefix, bin_offsets,
                     N, shift, flip,
                     BLOCK_SIZE: ConstInt,
                     BLOCK_COUNTS: ConstInt,
                     GROUP_TILE: ConstInt):
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
    digit = ((vals >> shift) & 3) ^ flip

    b0 = ct.where((digit == 0) & valid, 1, 0)
    b1 = ct.where((digit == 1) & valid, 1, 0)
    b2 = ct.where((digit == 2) & valid, 1, 0)

    rank0 = ct.cumsum(b0, axis=0) - b0
    rank1 = ct.cumsum(b1, axis=0) - b1
    rank2 = ct.cumsum(b2, axis=0) - b2
    rank3 = r - rank0 - rank1 - rank2

    local_rank = ct.where(
        digit == 0,
        rank0,
        ct.where(digit == 1, rank1, ct.where(digit == 2, rank2, rank3)),
    )

    gid = bid // BLOCK_COUNTS

    p0 = ct.load(prefix, index=(0, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p1 = ct.load(prefix, index=(1, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p2 = ct.load(prefix, index=(2, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    p3 = ct.load(prefix, index=(3, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    gp0 = ct.load(group_prefix, index=(0, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp1 = ct.load(group_prefix, index=(1, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp2 = ct.load(group_prefix, index=(2, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    gp3 = ct.load(group_prefix, index=(3, gid), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    o0 = ct.load(bin_offsets, index=(0,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o1 = ct.load(bin_offsets, index=(1,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o2 = ct.load(bin_offsets, index=(2,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    o3 = ct.load(bin_offsets, index=(3,), shape=(), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)

    before = ct.where(
        digit == 0,
        p0 + gp0,
        ct.where(digit == 1, p1 + gp1, ct.where(digit == 2, p2 + gp2, p3 + gp3)),
    )
    bucket_base = ct.where(digit == 0, o0, ct.where(digit == 1, o1, ct.where(digit == 2, o2, o3)))

    dest = bucket_base + before + local_rank
    dest = ct.where(valid, dest, N)
    ct.scatter(dst, dest, vals, check_bounds=True)


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
    occupancy = 4

    num_blocks = ct.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((8, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((8, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((8, GROUP_TILE), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((8, GROUP_TILE), device=input.device, dtype=torch.int32)
    bin_offsets = torch.empty((8,), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()
    grid_blocks = (num_blocks, 1, 1)
    grid_prefix8 = (GROUP_TILE, 8, 1)
    grid_prefix4 = (GROUP_TILE, 4, 1)
    grid_group = (1, 1, 1)

    count8_kernel = _count8_kernel.with_hints(occupancy=occupancy)
    count4_kernel = _count4_kernel.with_hints(occupancy=occupancy)
    prefix_kernel = _prefix_counts_kernel.with_hints(occupancy=occupancy)
    group8_kernel = _group_offsets8_kernel.with_hints(occupancy=occupancy)
    group4_kernel = _group_offsets4_kernel.with_hints(occupancy=occupancy)
    scatter8_kernel = _scatter8_kernel.with_hints(occupancy=occupancy)
    scatter4_kernel = _scatter4_kernel.with_hints(occupancy=occupancy)

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

        ct.launch(stream, grid_blocks, count8_kernel, (src, counts, n, shift, BLOCK_SIZE))
        ct.launch(stream, grid_prefix8, prefix_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
        ct.launch(stream, grid_group, group8_kernel, (group_counts, group_prefix, bin_offsets, GROUP_TILE))
        ct.launch(
            stream,
            grid_blocks,
            scatter8_kernel,
            (src, dst, prefix, group_prefix, bin_offsets, n, shift, BLOCK_SIZE, BLOCK_COUNTS, GROUP_TILE),
        )

    ct.launch(stream, grid_blocks, count4_kernel, (output, counts, n, 30, 2, BLOCK_SIZE))
    ct.launch(stream, grid_prefix4, prefix_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
    ct.launch(stream, grid_group, group4_kernel, (group_counts, group_prefix, bin_offsets, GROUP_TILE))
    ct.launch(
        stream,
        grid_blocks,
        scatter4_kernel,
        (output, tmp, prefix, group_prefix, bin_offsets, n, 30, 2, BLOCK_SIZE, BLOCK_COUNTS, GROUP_TILE),
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
        "occupancy": occupancy,
    })
    return tmp


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
