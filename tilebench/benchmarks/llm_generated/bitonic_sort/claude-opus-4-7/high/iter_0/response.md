Looking at this operator, I need to implement a multi-launch bitonic sort. My strategy:

1. **Pad** input to next-pow2 with `+inf` so padding ends up at the tail
2. **Cross-segment kernel**: for `j ≥ BLOCK_ELEM`, each CTA processes a pair of distant contiguous tiles (compare-exchange between `data[base..base+B)` and `data[base+j..base+j+B)`)
3. **Fused local kernel**: for `j ≤ BLOCK_ELEM/2`, one CTA loads a full segment and performs ALL compare-exchange stages for that `k` in-register using `tl.flip`/`ct.cat` swaps — this collapses `log₂(BLOCK_ELEM)` global passes into one

This reduces global passes from O(log²M) to roughly O(log²M / log BLOCK).

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_cross(data_ptr, k,
                   BLOCK_PAIRS: tl.constexpr,
                   LOG_BLOCK_PAIRS: tl.constexpr,
                   LOG_J: tl.constexpr):
    pid = tl.program_id(0)
    pid_outer = pid >> (LOG_J - LOG_BLOCK_PAIRS)
    pid_inner = pid & ((1 << (LOG_J - LOG_BLOCK_PAIRS)) - 1)
    left_base = (pid_outer << (LOG_J + 1)) + (pid_inner << LOG_BLOCK_PAIRS)
    right_base = left_base + (1 << LOG_J)

    local = tl.arange(0, BLOCK_PAIRS)
    left_offs = left_base + local
    right_offs = right_base + local

    a = tl.load(data_ptr + left_offs)
    b = tl.load(data_ptr + right_offs)

    asc = (left_offs & k) == 0
    mn = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    new_a = tl.where(asc, mn, mx)
    new_b = tl.where(asc, mx, mn)
    tl.store(data_ptr + left_offs, new_a)
    tl.store(data_ptr + right_offs, new_b)


@triton.jit
def _bitonic_local(data_ptr, k,
                   BLOCK_ELEM: tl.constexpr,
                   LOG_J_START: tl.constexpr):
    pid = tl.program_id(0)
    seg_start = pid * BLOCK_ELEM
    local = tl.arange(0, BLOCK_ELEM)
    offs = seg_start + local
    x = tl.load(data_ptr + offs)
    asc = (offs & k) == 0

    for log_j in tl.static_range(LOG_J_START, -1, -1):
        j = 1 << log_j
        G = BLOCK_ELEM // (2 * j)
        x3 = tl.reshape(x, (G, 2, j))
        y3 = tl.flip(x3, 1)
        y = tl.reshape(y3, (BLOCK_ELEM,))
        is_left = (local & j) == 0
        cond = asc != is_left
        x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK_ELEM = 2048
    BLOCK_PAIRS = BLOCK_ELEM // 2
    LOG_BLOCK_ELEM = BLOCK_ELEM.bit_length() - 1
    LOG_BLOCK_PAIRS = BLOCK_PAIRS.bit_length() - 1
    num_warps = 4
    num_stages = 2

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, BLOCK_ELEM)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_blocks = M // BLOCK_ELEM
    grid = (n_blocks,)

    k = 2
    while k <= M:
        if k <= BLOCK_ELEM:
            log_j_start = (k.bit_length() - 1) - 1
            _bitonic_local[grid](
                work, k,
                BLOCK_ELEM=BLOCK_ELEM,
                LOG_J_START=log_j_start,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            j = k // 2
            while j >= BLOCK_ELEM:
                log_j = j.bit_length() - 1
                _bitonic_cross[grid](
                    work, k,
                    BLOCK_PAIRS=BLOCK_PAIRS,
                    LOG_BLOCK_PAIRS=LOG_BLOCK_PAIRS,
                    LOG_J=log_j,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                j //= 2
            log_j_start = LOG_BLOCK_ELEM - 1
            _bitonic_local[grid](
                work, k,
                BLOCK_ELEM=BLOCK_ELEM,
                LOG_J_START=log_j_start,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_ELEM": BLOCK_ELEM,
        "BLOCK_PAIRS": BLOCK_PAIRS,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return work[:N].contiguous()


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
def _bitonic_cross(data, k: ConstInt,
                   BLOCK_PAIRS: ConstInt,
                   LOG_BLOCK_PAIRS: ConstInt,
                   LOG_J: ConstInt):
    pid = ct.bid(0)
    log_stride = LOG_J - LOG_BLOCK_PAIRS
    stride = 1 << log_stride
    pid_outer = pid >> log_stride
    pid_inner = pid & (stride - 1)
    left_tile = (pid_outer << (log_stride + 1)) | pid_inner
    right_tile = left_tile + stride

    a = ct.load(data, index=(left_tile,), shape=(BLOCK_PAIRS,))
    b = ct.load(data, index=(right_tile,), shape=(BLOCK_PAIRS,))

    p_local = ct.arange(BLOCK_PAIRS, dtype=np.int32)
    left_offs = left_tile * BLOCK_PAIRS + p_local
    asc = (left_offs & k) == 0

    mn = ct.minimum(a, b)
    mx = ct.maximum(a, b)
    new_a = ct.where(asc, mn, mx)
    new_b = ct.where(asc, mx, mn)

    ct.store(data, index=(left_tile,), tile=new_a)
    ct.store(data, index=(right_tile,), tile=new_b)


@ct.kernel
def _bitonic_local(data, k: ConstInt,
                   BLOCK_ELEM: ConstInt,
                   LOG_J_START: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK_ELEM, dtype=np.int32)
    offs = pid * BLOCK_ELEM + local
    x = ct.load(data, index=(pid,), shape=(BLOCK_ELEM,))
    asc = (offs & k) == 0

    for log_j in range(LOG_J_START, -1, -1):
        j = 1 << log_j
        G = BLOCK_ELEM // (2 * j)
        x3 = ct.reshape(x, (G, 2, j))
        left_half = ct.extract(x3, (0, 0, 0), (G, 1, j))
        right_half = ct.extract(x3, (0, 1, 0), (G, 1, j))
        y3 = ct.cat((right_half, left_half), axis=1)
        y = ct.reshape(y3, (BLOCK_ELEM,))
        is_left = (local & j) == 0
        cond = asc != is_left
        x = ct.where(cond, ct.maximum(x, y), ct.minimum(x, y))

    ct.store(data, index=(pid,), tile=x)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK_ELEM = 2048
    BLOCK_PAIRS = BLOCK_ELEM // 2
    LOG_BLOCK_ELEM = BLOCK_ELEM.bit_length() - 1
    LOG_BLOCK_PAIRS = BLOCK_PAIRS.bit_length() - 1
    occupancy = 2

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, BLOCK_ELEM)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_blocks = M // BLOCK_ELEM
    grid = (n_blocks, 1, 1)

    cross_kernel = _bitonic_cross.with_hints(occupancy=occupancy)
    local_kernel = _bitonic_local.with_hints(occupancy=occupancy)

    k = 2
    while k <= M:
        if k <= BLOCK_ELEM:
            log_j_start = (k.bit_length() - 1) - 1
            ct.launch(stream, grid, local_kernel,
                      (work, k, BLOCK_ELEM, log_j_start))
        else:
            j = k // 2
            while j >= BLOCK_ELEM:
                log_j = j.bit_length() - 1
                ct.launch(stream, grid, cross_kernel,
                          (work, k, BLOCK_PAIRS, LOG_BLOCK_PAIRS, log_j))
                j //= 2
            log_j_start = LOG_BLOCK_ELEM - 1
            ct.launch(stream, grid, local_kernel,
                      (work, k, BLOCK_ELEM, log_j_start))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_ELEM": BLOCK_ELEM,
        "BLOCK_PAIRS": BLOCK_PAIRS,
        "occupancy": occupancy,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
