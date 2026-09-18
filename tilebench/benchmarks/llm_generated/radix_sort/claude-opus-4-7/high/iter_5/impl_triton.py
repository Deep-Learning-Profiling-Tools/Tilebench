import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_kernel(x_ptr, hist_ptr, N, bit, BLOCK: tl.constexpr, NBINS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bucket = ((x >> bit) & (NBINS - 1)).to(tl.int32)
    bucket = tl.where(mask, bucket, NBINS)
    bins = tl.arange(0, NBINS)
    is_in = (bucket[:, None] == bins[None, :]).to(tl.int32)
    counts = tl.sum(is_in, axis=0)
    tl.store(hist_ptr + pid * NBINS + bins, counts)


@triton.jit
def _scan_super_kernel(hist_ptr, super_hist_ptr, num_blocks,
                       NBINS: tl.constexpr, SUPER: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * SUPER + tl.arange(0, SUPER)
    bins = tl.arange(0, NBINS)
    offs = rows[:, None] * NBINS + bins[None, :]
    mask = rows[:, None] < num_blocks
    v = tl.load(hist_ptr + offs, mask=mask, other=0)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(hist_ptr + offs, exc, mask=mask)
    s = tl.sum(v, axis=0)
    tl.store(super_hist_ptr + pid * NBINS + bins, s)


@triton.jit
def _final_scan_kernel(super_hist_ptr, bucket_start_ptr, num_super,
                       NBINS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    bins = tl.arange(0, NBINS)
    offs = rows[:, None] * NBINS + bins[None, :]
    mask = rows[:, None] < num_super
    v = tl.load(super_hist_ptr + offs, mask=mask, other=0)
    inc = tl.cumsum(v, axis=0)
    exc = inc - v
    tl.store(super_hist_ptr + offs, exc, mask=mask)
    totals = tl.sum(v, axis=0)
    inc_t = tl.cumsum(totals, axis=0)
    exc_t = inc_t - totals
    tl.store(bucket_start_ptr + bins, exc_t)


@triton.jit
def _scatter_kernel(x_ptr, out_ptr, hist_ptr, super_hist_ptr, bucket_start_ptr,
                    N, bit, SUPER_SIZE: tl.constexpr, NBINS: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    super_idx = pid // SUPER_SIZE
    bins = tl.arange(0, NBINS)
    block_pref = tl.load(hist_ptr + pid * NBINS + bins)
    super_pref = tl.load(super_hist_ptr + super_idx * NBINS + bins)
    bucket_starts = tl.load(bucket_start_ptr + bins)
    global_pref = block_pref + super_pref + bucket_starts

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bucket = ((x >> bit) & (NBINS - 1)).to(tl.int32)
    bucket = tl.where(mask, bucket, NBINS)
    is_in = (bucket[:, None] == bins[None, :]).to(tl.int32)
    inc = tl.cumsum(is_in, axis=0)
    local_pref = inc - is_in

    local_pref_elem = tl.sum(is_in * local_pref, axis=1)
    global_pref_elem = tl.sum(is_in * global_pref[None, :], axis=1)
    dest = global_pref_elem + local_pref_elem
    tl.store(out_ptr + dest, x, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    BLOCK = 1024
    SUPER_SIZE = 1024
    NBINS = 16
    NBITS = 4
    num_warps = 8
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

    hist = torch.empty((num_blocks, NBINS), dtype=torch.int32, device=x.device)
    super_hist = torch.empty((max(num_super, 1), NBINS), dtype=torch.int32, device=x.device)
    bucket_start = torch.empty(NBINS, dtype=torch.int32, device=x.device)

    npasses = 32 // NBITS
    for pass_idx in range(npasses):
        bit = pass_idx * NBITS
        _hist_kernel[(num_blocks,)](
            x, hist, n, bit, BLOCK=BLOCK, NBINS=NBINS,
            num_warps=num_warps, num_stages=num_stages)
        _scan_super_kernel[(num_super,)](
            hist, super_hist, num_blocks, NBINS=NBINS, SUPER=SUPER_SIZE,
            num_warps=num_warps, num_stages=num_stages)
        _final_scan_kernel[(1,)](
            super_hist, bucket_start, num_super, NBINS=NBINS, BLOCK=SUPER_TILE,
            num_warps=1, num_stages=1)
        _scatter_kernel[(num_blocks,)](
            x, out, hist, super_hist, bucket_start, n, bit,
            SUPER_SIZE=SUPER_SIZE, NBINS=NBINS, BLOCK=BLOCK,
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
