Looking at the trajectory, the triton 1-bit approach has plateaued at 68.1%. The roofline assumes 32 passes (1 bit each). Switching to a 2-bit radix (4 bins, 16 passes) halves the memory traffic, which should push us past 100% of the roofline assuming bandwidth-bound. The cuTile 1-bit version was stuck at 30.4%, so multi-bit should also help there.

The key challenge in 2-bit is computing per-bin local ranks in the scatter kernel — I use 4 separate cumsums on one-hot indicators, then select the right one per element via nested `where`. The global prefix structure (count → per-super scan → top scan with bin-starts) is straightforward.

```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _count_kernel(x, counts, N, bit, num_blocks, BLOCK: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bin_val = ct.astype((x_tile >> bit) & 3, np.int32)
    bin_val_m = ct.where(valid, bin_val, -1)
    c0 = ct.sum(ct.astype(bin_val_m == 0, np.int32), axis=0, keepdims=True)
    c1 = ct.sum(ct.astype(bin_val_m == 1, np.int32), axis=0, keepdims=True)
    c2 = ct.sum(ct.astype(bin_val_m == 2, np.int32), axis=0, keepdims=True)
    c3 = ct.sum(ct.astype(bin_val_m == 3, np.int32), axis=0, keepdims=True)
    ct.store(counts, index=(pid,), tile=c0)
    ct.store(counts, index=(num_blocks + pid,), tile=c1)
    ct.store(counts, index=(2 * num_blocks + pid,), tile=c2)
    ct.store(counts, index=(3 * num_blocks + pid,), tile=c3)


@ct.kernel(occupancy=2)
def _scan_super_per_bin(counts, super_sums, num_blocks, num_super,
                        SUPER: ConstInt):
    pid = ct.bid(0)
    bin_idx = ct.bid(1)
    bin_offset = bin_idx * num_blocks
    bin_slice = counts.slice(0, bin_offset, bin_offset + num_blocks)
    v = ct.load(bin_slice, index=(pid,), shape=(SUPER,),
                padding_mode=ct.PaddingMode.ZERO)
    inc = ct.cumsum(v, axis=0)
    exc = inc - v
    ct.store(bin_slice, index=(pid,), tile=exc)
    s = ct.sum(v, axis=0, keepdims=True)
    ct.store(super_sums, index=(bin_idx * num_super + pid,), tile=s)


@ct.kernel(occupancy=1)
def _scan_top(super_sums, bin_starts, num_super, TOP_TILE: ConstInt):
    s0_slice = super_sums.slice(0, 0, num_super)
    s1_slice = super_sums.slice(0, num_super, 2 * num_super)
    s2_slice = super_sums.slice(0, 2 * num_super, 3 * num_super)
    s3_slice = super_sums.slice(0, 3 * num_super, 4 * num_super)

    v0 = ct.load(s0_slice, index=(0,), shape=(TOP_TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    v1 = ct.load(s1_slice, index=(0,), shape=(TOP_TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    v2 = ct.load(s2_slice, index=(0,), shape=(TOP_TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    v3 = ct.load(s3_slice, index=(0,), shape=(TOP_TILE,),
                 padding_mode=ct.PaddingMode.ZERO)

    total0 = ct.sum(v0, axis=0, keepdims=True)
    total1 = ct.sum(v1, axis=0, keepdims=True)
    total2 = ct.sum(v2, axis=0, keepdims=True)

    inc0 = ct.cumsum(v0, axis=0)
    inc1 = ct.cumsum(v1, axis=0)
    inc2 = ct.cumsum(v2, axis=0)
    inc3 = ct.cumsum(v3, axis=0)

    ct.store(s0_slice, index=(0,), tile=inc0 - v0)
    ct.store(s1_slice, index=(0,), tile=inc1 - v1)
    ct.store(s2_slice, index=(0,), tile=inc2 - v2)
    ct.store(s3_slice, index=(0,), tile=inc3 - v3)

    bs0 = ct.zeros((1,), dtype=np.int32)
    bs1 = total0
    bs2 = total0 + total1
    bs3 = total0 + total1 + total2

    ct.store(bin_starts, index=(0,), tile=bs0)
    ct.store(bin_starts, index=(1,), tile=bs1)
    ct.store(bin_starts, index=(2,), tile=bs2)
    ct.store(bin_starts, index=(3,), tile=bs3)


@ct.kernel(occupancy=2)
def _scatter_kernel(x, out, counts, super_sums, bin_starts,
                    N, bit, num_blocks, num_super,
                    SUPER_SIZE: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    super_idx = pid // SUPER_SIZE

    p0 = ct.load(counts, index=(pid,), shape=(1,))
    p1 = ct.load(counts, index=(num_blocks + pid,), shape=(1,))
    p2 = ct.load(counts, index=(2 * num_blocks + pid,), shape=(1,))
    p3 = ct.load(counts, index=(3 * num_blocks + pid,), shape=(1,))
    s0 = ct.load(super_sums, index=(super_idx,), shape=(1,))
    s1 = ct.load(super_sums, index=(num_super + super_idx,), shape=(1,))
    s2 = ct.load(super_sums, index=(2 * num_super + super_idx,), shape=(1,))
    s3 = ct.load(super_sums, index=(3 * num_super + super_idx,), shape=(1,))
    bs0 = ct.load(bin_starts, index=(0,), shape=(1,))
    bs1 = ct.load(bin_starts, index=(1,), shape=(1,))
    bs2 = ct.load(bin_starts, index=(2,), shape=(1,))
    bs3 = ct.load(bin_starts, index=(3,), shape=(1,))

    pref0 = bs0 + s0 + p0
    pref1 = bs1 + s1 + p1
    pref2 = bs2 + s2 + p2
    pref3 = bs3 + s3 + p3

    x_tile = ct.load(x, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < N
    bin_val = ct.astype((x_tile >> bit) & 3, np.int32)
    bin_val_m = ct.where(valid, bin_val, -1)

    is0 = ct.astype(bin_val_m == 0, np.int32)
    is1 = ct.astype(bin_val_m == 1, np.int32)
    is2 = ct.astype(bin_val_m == 2, np.int32)
    is3 = ct.astype(bin_val_m == 3, np.int32)

    r0 = ct.cumsum(is0, axis=0) - is0
    r1 = ct.cumsum(is1, axis=0) - is1
    r2 = ct.cumsum(is2, axis=0) - is2
    r3 = ct.cumsum(is3, axis=0) - is3

    dest = ct.where(bin_val_m == 0, pref0 + r0,
             ct.where(bin_val_m == 1, pref1 + r1,
              ct.where(bin_val_m == 2, pref2 + r2,
                                       pref3 + r3)))
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

    counts = torch.empty(NBINS * num_blocks, dtype=torch.int32, device=x.device)
    super_sums = torch.empty(NBINS * max(num_super, 1), dtype=torch.int32,
                              device=x.device)
    bin_starts = torch.empty(NBINS, dtype=torch.int32, device=x.device)

    stream = torch.cuda.current_stream()

    for bit in range(0, 32, NBITS):
        ct.launch(stream, (num_blocks, 1, 1), _count_kernel,
                  (x, counts, n, bit, num_blocks, BLOCK))
        ct.launch(stream, (num_super, NBINS, 1), _scan_super_per_bin,
                  (counts, super_sums, num_blocks, num_super, SUPER_SIZE))
        ct.launch(stream, (1, 1, 1), _scan_top,
                  (super_sums, bin_starts, num_super, SUPER_TILE))
        ct.launch(stream, (num_blocks, 1, 1), _scatter_kernel,
                  (x, out, counts, super_sums, bin_starts,
                   n, bit, num_blocks, num_super, SUPER_SIZE, BLOCK))
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
