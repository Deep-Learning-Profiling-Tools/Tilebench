```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _copy_kernel(data_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _radix_hist_kernel(src_ptr, hist_ptr, n_elements, shift,
                       BLOCK: tl.constexpr, RADIX: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    vals = tl.load(src_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
    vals_f32 = vals.to(tl.float32)
    bits = vals_f32.to(tl.uint32, bitcast=True)

    zero_u = tl.full((BLOCK,), 0, tl.uint32)
    one_u = tl.full((BLOCK,), 1, tl.uint32)
    sign_mask = one_u << 31
    all_ones = ~zero_u

    sign = bits & sign_mask
    key = tl.where(sign != zero_u, bits ^ all_ones, bits ^ sign_mask)
    digit = ((key >> shift) & 15).to(tl.int32)

    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.store(hist_ptr + pid * RADIX + d, cnt)


@triton.jit
def _radix_prefix_kernel(hist_ptr, prefix_ptr, totals_ptr, num_blocks,
                         HIST_BLOCK: tl.constexpr, RADIX: tl.constexpr):
    digit = tl.program_id(0)
    offs = tl.arange(0, HIST_BLOCK)
    mask = offs < num_blocks

    counts = tl.load(
        hist_ptr + offs * RADIX + digit,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.int32)

    scan = tl.cumsum(counts, axis=0)
    pref = scan - counts
    tl.store(prefix_ptr + offs * RADIX + digit, pref, mask=mask)

    total = tl.sum(counts, axis=0)
    tl.store(totals_ptr + digit, total)


@triton.jit
def _radix_base_kernel(totals_ptr, bases_ptr, RADIX: tl.constexpr):
    offs = tl.arange(0, RADIX)
    counts = tl.load(totals_ptr + offs, eviction_policy="evict_first").to(tl.int32)
    scan = tl.cumsum(counts, axis=0)
    bases = scan - counts
    tl.store(bases_ptr + offs, bases)


@triton.jit
def _radix_scatter_kernel(src_ptr, dst_ptr, prefix_ptr, bases_ptr, n_elements, shift,
                          BLOCK: tl.constexpr, RADIX: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    offs = pid * BLOCK + local
    mask = offs < n_elements

    vals = tl.load(src_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
    vals_f32 = vals.to(tl.float32)
    bits = vals_f32.to(tl.uint32, bitcast=True)

    zero_u = tl.full((BLOCK,), 0, tl.uint32)
    one_u = tl.full((BLOCK,), 1, tl.uint32)
    sign_mask = one_u << 31
    all_ones = ~zero_u

    sign = bits & sign_mask
    key = tl.where(sign != zero_u, bits ^ all_ones, bits ^ sign_mask)
    digit = ((key >> shift) & 15).to(tl.int32)

    for d in tl.static_range(0, RADIX):
        eq = (digit == d) & mask
        rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
        block_prefix = tl.load(prefix_ptr + pid * RADIX + d).to(tl.int32)
        bucket_base = tl.load(bases_ptr + d).to(tl.int32)
        pos = bucket_base + block_prefix + rank
        tl.store(dst_ptr + pos, vals, mask=eq)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    DATA_BLOCK = 2048
    RADIX_BITS = 4
    RADIX = 16
    PASSES = 8
    data_num_warps = 8
    prefix_num_warps = 8
    num_stages = 2

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 2,
            "DATA_BLOCK": DATA_BLOCK,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "passes": 0,
            "data_num_warps": data_num_warps,
            "prefix_num_warps": prefix_num_warps,
            "num_stages": num_stages,
            "num_blocks": 0,
            "HIST_BLOCK": 0,
            "copy_kernel": 1,
        })
        return out

    if n_int == 1:
        out = torch.empty((n_int,), device=data.device, dtype=data.dtype)
        _copy_kernel[(1,)](
            data, out, n_int,
            BLOCK=DATA_BLOCK,
            num_warps=data_num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 2,
            "DATA_BLOCK": DATA_BLOCK,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "passes": 0,
            "data_num_warps": data_num_warps,
            "prefix_num_warps": prefix_num_warps,
            "num_stages": num_stages,
            "num_blocks": 1,
            "HIST_BLOCK": 1,
            "copy_kernel": 1,
        })
        return out

    num_blocks = triton.cdiv(n_int, DATA_BLOCK)
    hist_block = 1 << ((num_blocks - 1).bit_length())

    buf1 = torch.empty((n_int,), device=data.device, dtype=data.dtype)
    buf2 = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    hist = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    prefix = torch.empty((num_blocks * RADIX,), device=data.device, dtype=torch.int32)
    totals = torch.empty((RADIX,), device=data.device, dtype=torch.int32)
    bases = torch.empty((RADIX,), device=data.device, dtype=torch.int32)

    data_grid = (num_blocks,)
    prefix_grid = (RADIX,)
    base_grid = (1,)

    src = data
    dst = buf1

    for pass_idx in range(PASSES):
        shift = pass_idx * RADIX_BITS

        _radix_hist_kernel[data_grid](
            src, hist, n_int, shift,
            BLOCK=DATA_BLOCK,
            RADIX=RADIX,
            num_warps=data_num_warps,
            num_stages=num_stages,
        )
        _radix_prefix_kernel[prefix_grid](
            hist, prefix, totals, num_blocks,
            HIST_BLOCK=hist_block,
            RADIX=RADIX,
            num_warps=prefix_num_warps,
            num_stages=num_stages,
        )
        _radix_base_kernel[base_grid](
            totals, bases,
            RADIX=RADIX,
            num_warps=1,
            num_stages=num_stages,
        )
        _radix_scatter_kernel[data_grid](
            src, dst, prefix, bases, n_int, shift,
            BLOCK=DATA_BLOCK,
            RADIX=RADIX,
            num_warps=data_num_warps,
            num_stages=num_stages,
        )

        src = dst
        dst = buf2 if (pass_idx % 2 == 0) else buf1

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 2,
        "DATA_BLOCK": DATA_BLOCK,
        "RADIX_BITS": RADIX_BITS,
        "RADIX": RADIX,
        "passes": PASSES,
        "data_num_warps": data_num_warps,
        "prefix_num_warps": prefix_num_warps,
        "num_stages": num_stages,
        "num_blocks": num_blocks,
        "HIST_BLOCK": hist_block,
        "copy_kernel": 0,
    })
    return src


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _copy_kernel(data, out, N, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    active = offs < N
    out_idx = ct.where(active, offs, N)
    ct.scatter(out, out_idx, vals, check_bounds=True)


@ct.kernel
def _radix_hist_kernel(src, hist, N, shift, TILE: ConstInt, RADIX: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    active = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f32 = ct.astype(vals, np.float32)
    bits = ct.bitcast(vals_f32, ct.uint32)

    zero_u = ct.full((TILE,), 0, dtype=ct.uint32)
    one_u = ct.full((TILE,), 1, dtype=ct.uint32)
    sign_mask = ct.bitwise_lshift(one_u, 31)
    all_ones = ct.bitwise_not(zero_u)

    sign = bits & sign_mask
    key_neg = bits ^ all_ones
    key_pos = bits ^ sign_mask
    key = ct.where(sign != zero_u, key_neg, key_pos)
    digit = ct.bitwise_rshift(key, shift) & 15

    for d in range(0, RADIX):
        eq = (digit == d) & active
        cnt = ct.sum(ct.astype(eq, np.int32), axis=0)
        ct.store(hist, index=(bid, d), tile=cnt)


@ct.kernel
def _radix_prefix_kernel(hist, prefix, totals, num_blocks, HIST_BLOCK: ConstInt):
    digit = ct.bid(0)
    offs = ct.arange(HIST_BLOCK, dtype=np.int32)

    counts = ct.gather(
        hist,
        (offs, digit),
        padding_value=0,
        check_bounds=True,
    )
    scan = ct.cumsum(counts, axis=0)
    pref = scan - counts

    ct.scatter(prefix, (offs, digit), pref, check_bounds=True)

    total = ct.sum(counts, axis=0)
    ct.store(totals, index=(digit,), tile=total)


@ct.kernel
def _radix_base_kernel(totals, bases, RADIX: ConstInt):
    counts = ct.load(
        totals,
        index=(0,),
        shape=(RADIX,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scan = ct.cumsum(counts, axis=0)
    base = scan - counts
    ct.store(bases, index=(0,), tile=base)


@ct.kernel
def _radix_scatter_kernel(src, dst, prefix, bases, N, shift,
                          TILE: ConstInt, RADIX: ConstInt):
    bid = ct.bid(0)
    local = ct.arange(TILE, dtype=np.int32)
    offs = bid * TILE + local
    active = offs < N

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f32 = ct.astype(vals, np.float32)
    bits = ct.bitcast(vals_f32, ct.uint32)

    zero_u = ct.full((TILE,), 0, dtype=ct.uint32)
    one_u = ct.full((TILE,), 1, dtype=ct.uint32)
    sign_mask = ct.bitwise_lshift(one_u, 31)
    all_ones = ct.bitwise_not(zero_u)

    sign = bits & sign_mask
    key_neg = bits ^ all_ones
    key_pos = bits ^ sign_mask
    key = ct.where(sign != zero_u, key_neg, key_pos)
    digit = ct.bitwise_rshift(key, shift) & 15

    for d in range(0, RADIX):
        eq = (digit == d) & active
        rank = ct.cumsum(ct.astype(eq, np.int32), axis=0) - 1

        block_prefix = ct.load(prefix, index=(bid, d), shape=())
        bucket_base = ct.load(bases, index=(d,), shape=())

        pos = bucket_base + block_prefix + rank
        out_idx = ct.where(eq, pos, N)
        ct.scatter(dst, out_idx, vals, check_bounds=True)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)

    TILE = 2048
    RADIX_BITS = 4
    RADIX = 16
    PASSES = 8
    occupancy = 4

    stream = torch.cuda.current_stream()

    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 2,
            "TILE": TILE,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "passes": 0,
            "occupancy": occupancy,
            "num_blocks": 0,
            "HIST_BLOCK": 0,
            "copy_kernel": 1,
        })
        return out

    if n_int == 1:
        out = torch.empty((n_int,), device=data.device, dtype=data.dtype)
        kernel = _copy_kernel.with_hints(occupancy=occupancy)
        ct.launch(stream, (1, 1, 1), kernel, (data, out, n_int, TILE))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "algorithm": 2,
            "TILE": TILE,
            "RADIX_BITS": RADIX_BITS,
            "RADIX": RADIX,
            "passes": 0,
            "occupancy": occupancy,
            "num_blocks": 1,
            "HIST_BLOCK": 1,
            "copy_kernel": 1,
        })
        return out

    num_blocks = ct.cdiv(n_int, TILE)
    hist_block = 1 << ((int(num_blocks) - 1).bit_length())

    buf1 = torch.empty((n_int,), device=data.device, dtype=data.dtype)
    buf2 = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    hist = torch.empty((int(num_blocks), RADIX), device=data.device, dtype=torch.int32)
    prefix = torch.empty((int(num_blocks), RADIX), device=data.device, dtype=torch.int32)
    totals = torch.empty((RADIX,), device=data.device, dtype=torch.int32)
    bases = torch.empty((RADIX,), device=data.device, dtype=torch.int32)

    hist_kernel = _radix_hist_kernel.with_hints(occupancy=occupancy)
    prefix_kernel = _radix_prefix_kernel.with_hints(occupancy=occupancy)
    base_kernel = _radix_base_kernel.with_hints(occupancy=occupancy)
    scatter_kernel = _radix_scatter_kernel.with_hints(occupancy=occupancy)

    data_grid = (int(num_blocks), 1, 1)
    prefix_grid = (RADIX, 1, 1)
    base_grid = (1, 1, 1)

    src = data
    dst = buf1

    for pass_idx in range(PASSES):
        shift = pass_idx * RADIX_BITS

        ct.launch(stream, data_grid, hist_kernel, (src, hist, n_int, shift, TILE, RADIX))
        ct.launch(stream, prefix_grid, prefix_kernel, (hist, prefix, totals, int(num_blocks), hist_block))
        ct.launch(stream, base_grid, base_kernel, (totals, bases, RADIX))
        ct.launch(stream, data_grid, scatter_kernel, (src, dst, prefix, bases, n_int, shift, TILE, RADIX))

        src = dst
        dst = buf2 if (pass_idx % 2 == 0) else buf1

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "algorithm": 2,
        "TILE": TILE,
        "RADIX_BITS": RADIX_BITS,
        "RADIX": RADIX,
        "passes": PASSES,
        "occupancy": occupancy,
        "num_blocks": int(num_blocks),
        "HIST_BLOCK": hist_block,
        "copy_kernel": 0,
    })
    return src


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
