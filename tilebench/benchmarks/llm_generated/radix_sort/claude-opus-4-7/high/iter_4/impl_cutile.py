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
