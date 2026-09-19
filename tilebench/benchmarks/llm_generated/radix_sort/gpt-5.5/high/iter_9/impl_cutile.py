import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _count4_kernel(src, counts, N, shift, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    r = ct.arange(BLOCK_SIZE, dtype=np.int32)
    offs = bid * BLOCK_SIZE + r
    valid = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(BLOCK_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    digit = (vals >> shift) & 3

    d0 = digit == 0
    d1 = digit == 1
    d2 = digit == 2
    d3 = digit == 3

    inc01 = ct.where(d0 & valid, 1, ct.where(d1 & valid, 65536, 0))
    inc23 = ct.where(d2 & valid, 1, ct.where(d3 & valid, 65536, 0))

    pack01 = ct.sum(inc01)
    pack23 = ct.sum(inc23)

    c0 = pack01 & 65535
    c1 = pack01 >> 16
    c2 = pack23 & 65535
    c3 = pack23 >> 16

    ct.store(counts, index=(0, bid), tile=c0, allow_tma=False)
    ct.store(counts, index=(1, bid), tile=c1, allow_tma=False)
    ct.store(counts, index=(2, bid), tile=c2, allow_tma=False)
    ct.store(counts, index=(3, bid), tile=c3, allow_tma=False)


@ct.kernel(occupancy=4)
def _count2_kernel(src, counts, N, shift, BLOCK_SIZE: ConstInt):
    bid = ct.bid(0)
    r = ct.arange(BLOCK_SIZE, dtype=np.int32)
    offs = bid * BLOCK_SIZE + r
    valid = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(BLOCK_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    digit = (vals >> shift) & 1

    b0 = ct.where((digit == 0) & valid, 1, 0)
    c0 = ct.sum(b0)
    valid_count = ct.minimum(N - bid * BLOCK_SIZE, BLOCK_SIZE)
    c1 = valid_count - c0

    ct.store(counts, index=(0, bid), tile=c0, allow_tma=False)
    ct.store(counts, index=(1, bid), tile=c1, allow_tma=False)


@ct.kernel(occupancy=4)
def _prefix_counts_kernel(counts, prefix, group_counts, BLOCK_COUNTS: ConstInt):
    gid = ct.bid(0)
    bin_id = ct.bid(1)

    c = ct.load(
        counts,
        index=(bin_id, gid),
        shape=(1, BLOCK_COUNTS),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    cs = ct.cumsum(c, axis=1)
    excl = cs - c

    ct.store(prefix, index=(bin_id, gid), tile=excl, allow_tma=False)
    total = ct.sum(c)
    ct.store(group_counts, index=(bin_id, gid), tile=total, allow_tma=False)


@ct.kernel(occupancy=4)
def _scatter4_kernel(src, dst, prefix, group_counts,
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
        latency=1,
        allow_tma=False,
    )
    digit = (vals >> shift) & 3

    d0 = digit == 0
    d1 = digit == 1
    d2 = digit == 2
    d3 = digit == 3

    inc01 = ct.where(d0 & valid, 1, ct.where(d1 & valid, 65536, 0))
    inc23 = ct.where(d2 & valid, 1, ct.where(d3 & valid, 65536, 0))

    cs01 = ct.cumsum(inc01, axis=0)
    cs23 = ct.cumsum(inc23, axis=0)

    rank0 = (cs01 & 65535) - 1
    rank1 = (cs01 >> 16) - 1
    rank2 = (cs23 & 65535) - 1
    rank3 = (cs23 >> 16) - 1

    local_rank = ct.where(
        d0,
        rank0,
        ct.where(d1, rank1, ct.where(d2, rank2, rank3)),
    )

    p0 = ct.load(prefix, index=(0, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    p1 = ct.load(prefix, index=(1, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    p2 = ct.load(prefix, index=(2, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    p3 = ct.load(prefix, index=(3, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)

    gid = bid // BLOCK_COUNTS
    gr = ct.arange(GROUP_TILE, dtype=np.int32)[None, :]

    gc0 = ct.load(group_counts, index=(0, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    gc1 = ct.load(group_counts, index=(1, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    gc2 = ct.load(group_counts, index=(2, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    gc3 = ct.load(group_counts, index=(3, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)

    gp0 = ct.sum(ct.where(gr < gid, gc0, 0))
    gp1 = ct.sum(ct.where(gr < gid, gc1, 0))
    gp2 = ct.sum(ct.where(gr < gid, gc2, 0))
    gp3 = ct.sum(ct.where(gr < gid, gc3, 0))

    total0 = ct.sum(gc0)
    total1 = ct.sum(gc1)
    total2 = ct.sum(gc2)

    before = ct.where(
        d0,
        p0 + gp0,
        ct.where(d1, p1 + gp1, ct.where(d2, p2 + gp2, p3 + gp3)),
    )

    bucket_base = ct.where(
        d0,
        0,
        ct.where(d1, total0, ct.where(d2, total0 + total1, total0 + total1 + total2)),
    )

    dest = bucket_base + before + local_rank
    dest = ct.where(valid, dest, N)

    ct.scatter(dst, dest, vals, check_bounds=False, latency=1)


@ct.kernel(occupancy=4)
def _scatter2_kernel(src, dst, prefix, group_counts,
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
        latency=1,
        allow_tma=False,
    )
    digit = (vals >> shift) & 1

    b0 = ct.where((digit == 0) & valid, 1, 0)
    c0 = ct.cumsum(b0, axis=0)

    rank0 = c0 - b0
    rank1 = r - rank0
    local_rank = ct.where(digit == 0, rank0, rank1)

    p0 = ct.load(prefix, index=(0, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    p1 = ct.load(prefix, index=(1, bid), shape=(), padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)

    gid = bid // BLOCK_COUNTS
    gr = ct.arange(GROUP_TILE, dtype=np.int32)[None, :]

    gc0 = ct.load(group_counts, index=(0, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)
    gc1 = ct.load(group_counts, index=(1, 0), shape=(1, GROUP_TILE),
                  padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False)

    gp0 = ct.sum(ct.where(gr < gid, gc0, 0))
    gp1 = ct.sum(ct.where(gr < gid, gc1, 0))
    total0 = ct.sum(gc0)

    before = ct.where(digit == 0, p0 + gp0, p1 + gp1)
    bucket_base = ct.where(digit == 0, 0, total0)

    dest = bucket_base + before + local_rank
    dest = ct.where(valid, dest, N)

    ct.scatter(dst, dest, vals, check_bounds=False, latency=1)


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty((n + 1,), device=input.device, dtype=input.dtype)
    tmp = torch.empty((n + 1,), device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 2048
    GROUP_TILE = 16
    RADIX_BITS = 2
    LOWER_PASSES = 15
    PASSES = 16
    FINAL_RADIX_BITS = 1
    occupancy = 4
    LATENCY_HINT = 1

    num_blocks = ct.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()
    grid_blocks = (num_blocks, 1, 1)
    grid_prefix4 = (GROUP_TILE, 4, 1)
    grid_prefix2 = (GROUP_TILE, 2, 1)

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

        ct.launch(stream, grid_blocks, _count4_kernel, (src, counts, n, shift, BLOCK_SIZE))
        ct.launch(stream, grid_prefix4, _prefix_counts_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
        ct.launch(
            stream,
            grid_blocks,
            _scatter4_kernel,
            (src, dst, prefix, group_counts, n, shift, BLOCK_SIZE, BLOCK_COUNTS, GROUP_TILE),
        )

    shift = 30
    src = tmp
    dst = output

    ct.launch(stream, grid_blocks, _count2_kernel, (src, counts, n, shift, BLOCK_SIZE))
    ct.launch(stream, grid_prefix2, _prefix_counts_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
    ct.launch(
        stream,
        grid_blocks,
        _scatter2_kernel,
        (src, dst, prefix, group_counts, n, shift, BLOCK_SIZE, BLOCK_COUNTS, GROUP_TILE),
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
        "PACKED_PAIR_CUMSUMS": True,
        "LOCAL_CUMSUMS_LOWER": 2,
        "LOCAL_CUMSUMS_FINAL": 1,
        "COUNT_PACKED_PAIRS": True,
        "GUARD_ELEMENTS": 1,
        "UNCHECKED_SCATTER": True,
        "LATENCY_HINT": LATENCY_HINT,
        "occupancy": occupancy,
    })
    return output[:n]


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
