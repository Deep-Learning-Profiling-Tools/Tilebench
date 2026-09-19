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
