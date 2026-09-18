Looking at the trajectory:
- **Triton**: Iter 0 (BLOCK=1024) was best at 64.2%. Iter 3 regressed to 57.7% with BLOCK=2048. The merged kernel from iter 3 saves launches but BLOCK=2048 hurt. I'll revert to BLOCK=1024 but **keep the merged sum_and_scan kernel** (saves 32 launches across all bits).
- **cuTile**: Iter 2 (NBINS=4, 8 passes) at 26.9% was best. Iter 3 broke verify. Revert to iter 2 exactly.

```python title="impl_triton.py"
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
def _sum_and_scan_super_kernel(block_ones_ptr, super_ones_ptr, num_blocks,
                                SUPER: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * SUPER + tl.arange(0, SUPER)
    mask = offs < num_blocks
    v = tl.load(block_ones_ptr + offs, mask=mask, other=0)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(block_ones_ptr + offs, exc, mask=mask)
    s = tl.sum(v, axis=0)
    tl.store(super_ones_ptr + pid, s)


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

    block_ones = torch.empty(num_blocks, dtype=torch.int32, device=x.device)
    super_ones = torch.empty(max(num_super, 1), dtype=torch.int32, device=x.device)
    total_ones = torch.empty(1, dtype=torch.int32, device=x.device)

    for bit in range(32):
        _count_ones_kernel[(num_blocks,)](
            x, block_ones, n, bit, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages)
        _sum_and_scan_super_kernel[(num_super,)](
            block_ones, super_ones, num_blocks, SUPER=SUPER_SIZE,
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _hist_kernel(x, hist, N, bit, BLOCK: ConstInt, NBINS: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bucket = (x_tile >> bit) & (NBINS - 1)
    bucket = ct.where(valid, bucket, NBINS)
    bins = ct.arange(NBINS, dtype=np.int32)
    is_in = (bucket[:, None] == bins[None, :])
    is_in_i = ct.astype(is_in, np.int32)
    counts = ct.sum(is_in_i, axis=0, keepdims=True)
    ct.store(hist, index=(pid, 0), tile=counts)


@ct.kernel(occupancy=2)
def _sum_and_scan_super_kernel(hist, super_hist, num_blocks,
                                NBINS: ConstInt, SUPER: ConstInt):
    pid = ct.bid(0)
    v = ct.load(hist, index=(pid, 0), shape=(SUPER, NBINS),
                padding_mode=ct.PaddingMode.ZERO)
    inc = ct.cumsum(v, axis=0)
    exc = inc - v
    ct.store(hist, index=(pid, 0), tile=exc)
    s = ct.sum(v, axis=0, keepdims=True)
    ct.store(super_hist, index=(pid, 0), tile=s)


@ct.kernel(occupancy=1)
def _scan_super_kernel(super_hist, bucket_start, num_super,
                       NBINS: ConstInt, BLOCK: ConstInt):
    v = ct.load(super_hist, index=(0, 0), shape=(BLOCK, NBINS),
                padding_mode=ct.PaddingMode.ZERO)
    inc = ct.cumsum(v, axis=0)
    exc = inc - v
    ct.store(super_hist, index=(0, 0), tile=exc)
    total = ct.sum(v, axis=0, keepdims=False)
    inc_total = ct.cumsum(total, axis=0)
    exc_total = inc_total - total
    ct.store(bucket_start, index=(0,), tile=exc_total)


@ct.kernel(occupancy=2)
def _scatter_kernel(x, out, hist, super_hist, bucket_start,
                    N, bit, SUPER_SIZE: ConstInt,
                    NBINS: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    super_idx = pid // SUPER_SIZE

    block_pref = ct.load(hist, index=(pid, 0), shape=(1, NBINS))
    super_pref = ct.load(super_hist, index=(super_idx, 0), shape=(1, NBINS))
    bucket_starts = ct.load(bucket_start, index=(0,), shape=(NBINS,))
    global_pref = block_pref + super_pref + bucket_starts[None, :]

    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bucket = (x_tile >> bit) & (NBINS - 1)
    bucket = ct.where(valid, bucket, NBINS)
    bins = ct.arange(NBINS, dtype=np.int32)
    is_in = (bucket[:, None] == bins[None, :])
    is_in_i = ct.astype(is_in, np.int32)
    inc = ct.cumsum(is_in_i, axis=0)
    local_pref = inc - is_in_i

    local_pref_elem = ct.sum(is_in_i * local_pref, axis=1, keepdims=False)
    global_pref_elem = ct.sum(is_in_i * global_pref, axis=1, keepdims=False)
    dest = global_pref_elem + local_pref_elem
    dest = ct.where(valid, dest, N)
    ct.scatter(out, (dest,), x_tile)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024
    NBINS = 4
    NBITS = 2

    n = N
    x = input.contiguous().clone()
    out = torch.empty_like(x)

    num_blocks = (n + BLOCK - 1) // BLOCK
    num_super = (num_blocks + SUPER_SIZE - 1) // SUPER_SIZE
    SUPER_TILE = 1
    while SUPER_TILE < num_super:
        SUPER_TILE *= 2
    SUPER_TILE = max(SUPER_TILE, 16)

    hist = torch.empty((num_blocks, NBINS), dtype=torch.int32, device=x.device)
    super_hist = torch.empty((max(num_super, 1), NBINS), dtype=torch.int32, device=x.device)
    bucket_start = torch.empty(NBINS, dtype=torch.int32, device=x.device)

    stream = torch.cuda.current_stream()
    npasses = 32 // NBITS

    for pass_idx in range(npasses):
        bit = pass_idx * NBITS
        ct.launch(stream, (num_blocks, 1, 1), _hist_kernel,
                  (x, hist, n, bit, BLOCK, NBINS))
        ct.launch(stream, (num_super, 1, 1), _sum_and_scan_super_kernel,
                  (hist, super_hist, num_blocks, NBINS, SUPER_SIZE))
        ct.launch(stream, (1, 1, 1), _scan_super_kernel,
                  (super_hist, bucket_start, num_super, NBINS, SUPER_TILE))
        ct.launch(stream, (num_blocks, 1, 1), _scatter_kernel,
                  (x, out, hist, super_hist, bucket_start,
                   n, bit, SUPER_SIZE, NBINS, BLOCK))
        x, out = out, x

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK, "SUPER_SIZE": SUPER_SIZE, "SUPER_TILE": SUPER_TILE,
        "NBINS": NBINS, "NBITS": NBITS,
    })
    return x


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverting to the best-known configs: Triton uses BLOCK=1024 (iter 0's win) plus iter 3's merged sum_and_scan kernel (saves 32 launches without changing the bandwidth profile). cuTile reverts to iter 2's exact code (NBINS=4, 8 passes instead of 32).
