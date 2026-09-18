import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _local_sort_kernel(data_ptr, work_ptr, n_elements, m_elements,
                       BLOCK: tl.constexpr, LOG_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    global_offs = pid * BLOCK + local

    vals = tl.load(
        data_ptr + global_offs,
        mask=global_offs < n_elements,
        other=float("inf"),
        eviction_policy="evict_first",
    )

    # Execute the bitonic network stages whose compare partners are wholly
    # inside this tile.  The global offset is used for the direction bits so
    # the produced blocks match the full global bitonic network exactly
    # (alternating ascending/descending blocks).
    for p in tl.static_range(1, LOG_BLOCK + 1):
        k_stage = 1 << p
        for q in tl.static_range(0, p):
            j_stage = 1 << (p - 1 - q)
            partner = local ^ j_stage
            other = tl.gather(vals, partner, 0)

            lower = local < partner
            ascending = (global_offs & k_stage) == 0

            take_other_asc = tl.where(lower, vals > other, vals < other)
            take_other_desc = tl.where(lower, vals < other, vals > other)
            take_other = tl.where(ascending, take_other_asc, take_other_desc)
            vals = tl.where(take_other, other, vals)

    tl.store(work_ptr + global_offs, vals, mask=global_offs < m_elements)


@triton.jit
def _bitonic_cross_pass_kernel(work_ptr, m_elements, j, j_log, k,
                               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pair = pid * BLOCK + tl.arange(0, BLOCK)
    half = m_elements >> 1
    mask = pair < half

    low = ((pair >> j_log) << (j_log + 1)) + (pair & (j - 1))
    high = low + j

    a = tl.load(work_ptr + low, mask=mask, other=0.0, eviction_policy="evict_first")
    b = tl.load(work_ptr + high, mask=mask, other=0.0, eviction_policy="evict_first")

    ascending = (low & k) == 0
    need_swap = tl.where(ascending, a > b, a < b)

    low_val = tl.where(need_swap, b, a)
    high_val = tl.where(need_swap, a, b)

    tl.store(work_ptr + low, low_val, mask=mask)
    tl.store(work_ptr + high, high_val, mask=mask)


@triton.jit
def _inblock_merge_kernel(work_ptr, m_elements, k,
                          BLOCK: tl.constexpr, LOG_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    global_offs = pid * BLOCK + local

    vals = tl.load(
        work_ptr + global_offs,
        mask=global_offs < m_elements,
        other=float("inf"),
        eviction_policy="evict_first",
    )

    # Fuse the j < BLOCK tail of each global merge stage in registers.
    for q in tl.static_range(0, LOG_BLOCK):
        j_stage = 1 << (LOG_BLOCK - 1 - q)
        partner = local ^ j_stage
        other = tl.gather(vals, partner, 0)

        lower = local < partner
        ascending = (global_offs & k) == 0

        take_other_asc = tl.where(lower, vals > other, vals < other)
        take_other_desc = tl.where(lower, vals < other, vals > other)
        take_other = tl.where(ascending, take_other_asc, take_other_desc)
        vals = tl.where(take_other, other, vals)

    tl.store(work_ptr + global_offs, vals, mask=global_offs < m_elements)


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)
    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "SORT_BLOCK": 4096,
            "PASS_BLOCK": 1024,
            "sort_num_warps": 8,
            "pass_num_warps": 4,
            "num_stages": 2,
            "log2_M": 0,
            "network_passes": 0,
            "global_cross_passes": 0,
            "inblock_merge_launches": 0,
            "copy_kernel": 0,
        })
        return out

    m_int = 1 << ((n_int - 1).bit_length())
    log_m = m_int.bit_length() - 1

    work = torch.empty((m_int,), device=data.device, dtype=data.dtype)

    SORT_BLOCK = 4096
    LOG_SORT_BLOCK = 12
    PASS_BLOCK = 1024
    num_stages = 2

    sort_num_warps = 8
    pass_num_warps = 4

    sort_grid = (triton.cdiv(m_int, SORT_BLOCK),)
    _local_sort_kernel[sort_grid](
        data, work, n_int, m_int,
        BLOCK=SORT_BLOCK,
        LOG_BLOCK=LOG_SORT_BLOCK,
        num_warps=sort_num_warps,
        num_stages=num_stages,
    )

    global_cross_passes = 0
    inblock_merge_launches = 0

    if m_int > SORT_BLOCK:
        pass_grid = (triton.cdiv(m_int >> 1, PASS_BLOCK),)
        merge_grid = (triton.cdiv(m_int, SORT_BLOCK),)

        k = SORT_BLOCK << 1
        while k <= m_int:
            j = k >> 1
            while j >= SORT_BLOCK:
                j_log = j.bit_length() - 1
                _bitonic_cross_pass_kernel[pass_grid](
                    work, m_int, j, j_log, k,
                    BLOCK=PASS_BLOCK,
                    num_warps=pass_num_warps,
                    num_stages=num_stages,
                )
                global_cross_passes += 1
                j >>= 1

            _inblock_merge_kernel[merge_grid](
                work, m_int, k,
                BLOCK=SORT_BLOCK,
                LOG_BLOCK=LOG_SORT_BLOCK,
                num_warps=sort_num_warps,
                num_stages=num_stages,
            )
            inblock_merge_launches += 1
            k <<= 1

    out = work[:n_int]

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "SORT_BLOCK": SORT_BLOCK,
        "PASS_BLOCK": PASS_BLOCK,
        "sort_num_warps": sort_num_warps,
        "pass_num_warps": pass_num_warps,
        "num_stages": num_stages,
        "log2_M": log_m,
        "network_passes": (log_m * (log_m + 1)) // 2,
        "global_cross_passes": global_cross_passes,
        "inblock_merge_launches": inblock_merge_launches,
        "copy_kernel": 0,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
