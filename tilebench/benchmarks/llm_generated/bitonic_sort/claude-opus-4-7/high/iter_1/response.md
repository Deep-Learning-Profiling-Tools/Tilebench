Looking at the errors:
1. **Triton**: `tl.static_range(LOG_J_START, -1, -1)` (negative step) was producing tensor-typed loop variables instead of constexpr ints, breaking the `reshape` shape tuple. Fix: use positive iteration over `tl.static_range(N)` so the loop variable is a clean constexpr int.
2. **cuTile**: `with_hints` is not a method on `@ct.kernel`. Fix: pass `occupancy=` directly into the `@ct.kernel(...)` decorator.

I'll restructure into three kernels (init does all phases for k≤BLOCK_ELEM; cross does one inter-block step; finalize does the in-block tail) so iteration counts are all compile-time constants.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_init(data_ptr,
                  BLOCK_ELEM: tl.constexpr,
                  LOG_BLOCK_ELEM: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK_ELEM)
    offs = pid * BLOCK_ELEM + local
    x = tl.load(data_ptr + offs)

    for log_k_m1 in tl.static_range(LOG_BLOCK_ELEM):
        log_k = log_k_m1 + 1
        k_val = 1 << log_k
        asc = (offs & k_val) == 0
        for stage in tl.static_range(log_k):
            log_j = log_k - 1 - stage
            j_val = 1 << log_j
            g_val = BLOCK_ELEM // (2 * j_val)
            x3 = tl.reshape(x, (g_val, 2, j_val))
            y = tl.reshape(tl.flip(x3, 1), (BLOCK_ELEM,))
            is_left = (local & j_val) == 0
            cond = asc != is_left
            x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_finalize(data_ptr, k,
                      BLOCK_ELEM: tl.constexpr,
                      LOG_BLOCK_ELEM: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK_ELEM)
    offs = pid * BLOCK_ELEM + local
    x = tl.load(data_ptr + offs)
    asc = (offs & k) == 0

    for stage in tl.static_range(LOG_BLOCK_ELEM):
        log_j = LOG_BLOCK_ELEM - 1 - stage
        j_val = 1 << log_j
        g_val = BLOCK_ELEM // (2 * j_val)
        x3 = tl.reshape(x, (g_val, 2, j_val))
        y = tl.reshape(tl.flip(x3, 1), (BLOCK_ELEM,))
        is_left = (local & j_val) == 0
        cond = asc != is_left
        x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_cross(data_ptr, k,
                   BLOCK_ELEM: tl.constexpr,
                   LOG_BLOCK_ELEM: tl.constexpr,
                   LOG_J: tl.constexpr):
    pid = tl.program_id(0)
    log_stride = LOG_J - LOG_BLOCK_ELEM
    stride = 1 << log_stride
    pid_outer = pid >> log_stride
    pid_inner = pid & (stride - 1)
    block_left = (pid_outer << (log_stride + 1)) | pid_inner
    block_right = block_left + stride

    local = tl.arange(0, BLOCK_ELEM)
    left_offs = block_left * BLOCK_ELEM + local
    right_offs = block_right * BLOCK_ELEM + local

    a = tl.load(data_ptr + left_offs)
    b = tl.load(data_ptr + right_offs)

    asc = (left_offs & k) == 0
    mn = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    new_a = tl.where(asc, mn, mx)
    new_b = tl.where(asc, mx, mn)

    tl.store(data_ptr + left_offs, new_a)
    tl.store(data_ptr + right_offs, new_b)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK_ELEM = 2048
    LOG_BLOCK_ELEM = 11
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
    cross_grid = (max(n_blocks // 2, 1),)

    _bitonic_init[grid](
        work,
        BLOCK_ELEM=BLOCK_ELEM,
        LOG_BLOCK_ELEM=LOG_BLOCK_ELEM,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    k = 2 * BLOCK_ELEM
    while k <= M:
        j = k // 2
        while j >= BLOCK_ELEM:
            log_j = j.bit_length() - 1
            _bitonic_cross[cross_grid](
                work, k,
                BLOCK_ELEM=BLOCK_ELEM,
                LOG_BLOCK_ELEM=LOG_BLOCK_ELEM,
                LOG_J=log_j,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            j //= 2
        _bitonic_finalize[grid](
            work, k,
            BLOCK_ELEM=BLOCK_ELEM,
            LOG_BLOCK_ELEM=LOG_BLOCK_ELEM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_ELEM": BLOCK_ELEM,
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


@ct.kernel(occupancy=2)
def _bitonic_init(data,
                  BLOCK_ELEM: ConstInt,
                  LOG_BLOCK_ELEM: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK_ELEM, dtype=np.int32)
    offs = pid * BLOCK_ELEM + local
    x = ct.load(data, index=(pid,), shape=(BLOCK_ELEM,))

    for log_k_m1 in range(LOG_BLOCK_ELEM):
        log_k = log_k_m1 + 1
        k_val = 1 << log_k
        asc = (offs & k_val) == 0
        for stage in range(log_k):
            log_j = log_k - 1 - stage
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


@ct.kernel(occupancy=2)
def _bitonic_finalize(data, k,
                      BLOCK_ELEM: ConstInt,
                      LOG_BLOCK_ELEM: ConstInt):
    pid = ct.bid(0)
    local = ct.arange(BLOCK_ELEM, dtype=np.int32)
    offs = pid * BLOCK_ELEM + local
    x = ct.load(data, index=(pid,), shape=(BLOCK_ELEM,))
    asc = (offs & k) == 0

    for stage in range(LOG_BLOCK_ELEM):
        log_j = LOG_BLOCK_ELEM - 1 - stage
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


@ct.kernel(occupancy=2)
def _bitonic_cross(data, k,
                   BLOCK_ELEM: ConstInt,
                   LOG_BLOCK_ELEM: ConstInt,
                   LOG_J: ConstInt):
    pid = ct.bid(0)
    log_stride = LOG_J - LOG_BLOCK_ELEM
    stride = 1 << log_stride
    pid_outer = pid >> log_stride
    pid_inner = pid & (stride - 1)
    block_left = (pid_outer << (log_stride + 1)) | pid_inner
    block_right = block_left + stride

    a = ct.load(data, index=(block_left,), shape=(BLOCK_ELEM,))
    b = ct.load(data, index=(block_right,), shape=(BLOCK_ELEM,))

    local = ct.arange(BLOCK_ELEM, dtype=np.int32)
    left_offs = block_left * BLOCK_ELEM + local
    asc = (left_offs & k) == 0

    mn = ct.minimum(a, b)
    mx = ct.maximum(a, b)
    new_a = ct.where(asc, mn, mx)
    new_b = ct.where(asc, mx, mn)

    ct.store(data, index=(block_left,), tile=new_a)
    ct.store(data, index=(block_right,), tile=new_b)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK_ELEM = 2048
    LOG_BLOCK_ELEM = 11

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, BLOCK_ELEM)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_blocks = M // BLOCK_ELEM
    grid = (n_blocks, 1, 1)
    cross_grid = (max(n_blocks // 2, 1), 1, 1)

    ct.launch(stream, grid, _bitonic_init,
              (work, BLOCK_ELEM, LOG_BLOCK_ELEM))

    k = 2 * BLOCK_ELEM
    while k <= M:
        j = k // 2
        while j >= BLOCK_ELEM:
            log_j = j.bit_length() - 1
            ct.launch(stream, cross_grid, _bitonic_cross,
                      (work, k, BLOCK_ELEM, LOG_BLOCK_ELEM, log_j))
            j //= 2
        ct.launch(stream, grid, _bitonic_finalize,
                  (work, k, BLOCK_ELEM, LOG_BLOCK_ELEM))
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_ELEM": BLOCK_ELEM,
        "occupancy": 2,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
