```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _count4_kernel(src_ptr, counts_ptr, N, num_blocks, shift, flip,
                   BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
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
def _prefix_counts4_kernel(counts_ptr, prefix_ptr, group_counts_ptr,
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
def _scatter4_precomputed_kernel(src_ptr, dst_ptr, prefix_ptr, group_prefix_ptr, bin_offsets_ptr,
                                 N, num_blocks, shift, flip,
                                 BLOCK_SIZE: tl.constexpr,
                                 BLOCK_COUNTS: tl.constexpr,
                                 GROUP_TILE: tl.constexpr):
    bid = tl.program_id(0)

    r = tl.arange(0, BLOCK_SIZE)
    offs = bid * BLOCK_SIZE + r
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
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
    tl.store(dst_ptr + dest, vals, mask=mask, cache_modifier=".cg", eviction_policy="evict_first")


def run(input: torch.Tensor, N: int, **kwargs):
    n = int(N)

    output = torch.empty_like(input)
    tmp = torch.empty_like(input)

    BLOCK_SIZE = 1024
    BLOCK_COUNTS = 1024
    GROUP_TILE = 32
    RADIX_BITS = 2
    PASSES = 16
    num_warps = 8
    num_stages = 3
    group_num_warps = 1

    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)
    bin_offsets = torch.empty((4,), device=input.device, dtype=torch.int32)

    grid_blocks = (num_blocks,)
    grid_prefix = (GROUP_TILE, 4)
    grid_group = (1,)

    for pass_id in range(PASSES):
        shift = pass_id * RADIX_BITS
        flip = 2 if shift == 30 else 0

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
            src, counts, n, num_blocks, shift, flip,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _prefix_counts4_kernel[grid_prefix](
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
        _scatter4_precomputed_kernel[grid_blocks](
            src, dst, prefix, group_prefix, bin_offsets,
            n, num_blocks, shift, flip,
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
        "PASSES": PASSES,
        "SIGNED_INT_ORDER": True,
        "PRECOMPUTED_GROUP_PREFIX": True,
        "LOCAL_CUMSUMS": 3,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_num_warps": group_num_warps,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
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

    c0 = ct.sum(b0)
    c1 = ct.sum(b1)
    c2 = ct.sum(b2)
    valid_count = ct.minimum(N - bid * BLOCK_SIZE, BLOCK_SIZE)
    c3 = valid_count - c0 - c1 - c2

    ct.store(counts, index=(0, bid), tile=c0, allow_tma=False)
    ct.store(counts, index=(1, bid), tile=c1, allow_tma=False)
    ct.store(counts, index=(2, bid), tile=c2, allow_tma=False)
    ct.store(counts, index=(3, bid), tile=c3, allow_tma=False)


@ct.kernel(occupancy=4)
def _prefix_counts4_kernel(counts, prefix, group_counts, BLOCK_COUNTS: ConstInt):
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


@ct.kernel(occupancy=4)
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


@ct.kernel(occupancy=4)
def _scatter4_precomputed_kernel(src, dst, prefix, group_prefix, bin_offsets,
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

    c0 = ct.cumsum(b0, axis=0)
    c1 = ct.cumsum(b1, axis=0)
    c2 = ct.cumsum(b2, axis=0)

    rank0 = c0 - b0
    rank1 = c1 - b1
    rank2 = c2 - b2
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
    RADIX_BITS = 2
    PASSES = 16
    occupancy = 4

    num_blocks = ct.cdiv(n, BLOCK_SIZE)

    counts = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    prefix = torch.empty((4, num_blocks), device=input.device, dtype=torch.int32)
    group_counts = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)
    group_prefix = torch.empty((4, GROUP_TILE), device=input.device, dtype=torch.int32)
    bin_offsets = torch.empty((4,), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()
    grid_blocks = (num_blocks, 1, 1)
    grid_prefix = (GROUP_TILE, 4, 1)
    grid_group = (1, 1, 1)

    for pass_id in range(PASSES):
        shift = pass_id * RADIX_BITS
        flip = 2 if shift == 30 else 0

        if pass_id == 0:
            src = input
            dst = tmp
        elif (pass_id & 1) == 1:
            src = tmp
            dst = output
        else:
            src = output
            dst = tmp

        ct.launch(stream, grid_blocks, _count4_kernel, (src, counts, n, shift, flip, BLOCK_SIZE))
        ct.launch(stream, grid_prefix, _prefix_counts4_kernel, (counts, prefix, group_counts, BLOCK_COUNTS))
        ct.launch(stream, grid_group, _group_offsets4_kernel, (group_counts, group_prefix, bin_offsets, GROUP_TILE))
        ct.launch(
            stream,
            grid_blocks,
            _scatter4_precomputed_kernel,
            (src, dst, prefix, group_prefix, bin_offsets, n, shift, flip, BLOCK_SIZE, BLOCK_COUNTS, GROUP_TILE),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "BLOCK_COUNTS": BLOCK_COUNTS,
        "GROUP_TILE": GROUP_TILE,
        "RADIX_BITS": RADIX_BITS,
        "PASSES": PASSES,
        "SIGNED_INT_ORDER": True,
        "PRECOMPUTED_GROUP_PREFIX": True,
        "LOCAL_CUMSUMS": 3,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
