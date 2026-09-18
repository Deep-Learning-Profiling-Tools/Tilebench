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
