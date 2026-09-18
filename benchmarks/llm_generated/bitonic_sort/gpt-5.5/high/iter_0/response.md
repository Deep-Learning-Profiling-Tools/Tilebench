```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _init_kernel(data_ptr, work_ptr, n_elements, m_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_mask = offs < n_elements
    out_mask = offs < m_elements
    vals = tl.load(data_ptr + offs, mask=in_mask, other=float("inf"))
    tl.store(work_ptr + offs, vals, mask=out_mask)


@triton.jit
def _bitonic_pass_kernel(work_ptr, m_elements, j, j_log, k,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pair = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    half = m_elements >> 1
    mask = pair < half

    low = ((pair >> j_log) << (j_log + 1)) + (pair & (j - 1))
    high = low + j

    a = tl.load(work_ptr + low, mask=mask, other=0.0)
    b = tl.load(work_ptr + high, mask=mask, other=0.0)

    ascending = (low & k) == 0
    need_swap = tl.where(ascending, a > b, a < b)

    low_val = tl.where(need_swap, b, a)
    high_val = tl.where(need_swap, a, b)

    tl.store(work_ptr + low, low_val, mask=mask)
    tl.store(work_ptr + high, high_val, mask=mask)


@triton.jit
def _copy_kernel(work_ptr, out_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(work_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)
    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_SIZE": 1024,
            "num_warps": 4,
            "num_stages": 2,
            "log2_M": 0,
            "passes": 0,
        })
        return out

    m_int = 1 << ((n_int - 1).bit_length())
    log_m = m_int.bit_length() - 1

    work = torch.empty((m_int,), device=data.device, dtype=data.dtype)
    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    BLOCK_SIZE = 1024
    num_warps = 4
    num_stages = 2

    init_grid = (triton.cdiv(m_int, BLOCK_SIZE),)
    _init_kernel[init_grid](
        data, work, n_int, m_int,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if m_int > 1:
        pass_grid = (triton.cdiv(m_int >> 1, BLOCK_SIZE),)
        k = 2
        while k <= m_int:
            j = k >> 1
            while j > 0:
                j_log = j.bit_length() - 1
                _bitonic_pass_kernel[pass_grid](
                    work, m_int, j, j_log, k,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                j >>= 1
            k <<= 1

    copy_grid = (triton.cdiv(n_int, BLOCK_SIZE),)
    _copy_kernel[copy_grid](
        work, out, n_int,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "log2_M": log_m,
        "passes": (log_m * (log_m + 1)) // 2,
    })
    return out


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
def _init_kernel(data, work, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.POS_INF,
    )
    ct.store(work, index=(bid,), tile=vals)


@ct.kernel
def _bitonic_pass_kernel(work, M, j, j_log, k, TILE: ConstInt):
    bid = ct.bid(0)
    pair = bid * TILE + ct.arange(TILE, dtype=np.int32)
    half = M // 2
    active = pair < half

    low = ((pair >> j_log) << (j_log + 1)) + (pair & (j - 1))
    high = low + j

    a = ct.gather(work, low, padding_value=0.0, check_bounds=True)
    b = ct.gather(work, high, padding_value=0.0, check_bounds=True)

    ascending = (low & k) == 0
    need_swap = ct.where(ascending, a > b, a < b)

    low_val = ct.where(need_swap, b, a)
    high_val = ct.where(need_swap, a, b)

    oob = M
    low_idx = ct.where(active, low, oob)
    high_idx = ct.where(active, high, oob)

    ct.scatter(work, low_idx, low_val, check_bounds=True)
    ct.scatter(work, high_idx, high_val, check_bounds=True)


@ct.kernel
def _copy_kernel(work, out, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        work,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(out, index=(bid,), tile=vals)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)
    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": 512,
            "occupancy": 8,
            "log2_M": 0,
            "passes": 0,
        })
        return out

    m_int = 1 << ((n_int - 1).bit_length())
    log_m = m_int.bit_length() - 1

    work = torch.empty((m_int,), device=data.device, dtype=data.dtype)
    out = torch.empty((n_int,), device=data.device, dtype=data.dtype)

    TILE = 512
    occupancy = 8
    stream = torch.cuda.current_stream()

    init_kernel = _init_kernel.with_hints(occupancy=occupancy)
    pass_kernel = _bitonic_pass_kernel.with_hints(occupancy=occupancy)
    copy_kernel = _copy_kernel.with_hints(occupancy=occupancy)

    init_grid = (ct.cdiv(m_int, TILE), 1, 1)
    ct.launch(stream, init_grid, init_kernel, (data, work, TILE))

    if m_int > 1:
        pass_grid = (ct.cdiv(m_int // 2, TILE), 1, 1)
        k = 2
        while k <= m_int:
            j = k // 2
            while j > 0:
                j_log = j.bit_length() - 1
                ct.launch(stream, pass_grid, pass_kernel, (work, m_int, j, j_log, k, TILE))
                j //= 2
            k *= 2

    copy_grid = (ct.cdiv(n_int, TILE), 1, 1)
    ct.launch(stream, copy_grid, copy_kernel, (work, out, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "log2_M": log_m,
        "passes": (log_m * (log_m + 1)) // 2,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
