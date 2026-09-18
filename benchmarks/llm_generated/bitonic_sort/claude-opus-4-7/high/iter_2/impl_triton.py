import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_local_init(data_ptr,
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
            partner_idx = local ^ j_val
            y = tl.gather(x, partner_idx, axis=0)
            is_left = (local & j_val) == 0
            cond = asc != is_left
            x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_local_finalize(data_ptr, k,
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
        partner_idx = local ^ j_val
        y = tl.gather(x, partner_idx, axis=0)
        is_left = (local & j_val) == 0
        cond = asc != is_left
        x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_cross(data_ptr, k, j,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Pair index `tid` -> element index i (lower of the pair (i, i+j))
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = tl.load(data_ptr + i)
    aj = tl.load(data_ptr + ixj)
    asc = (i & k) == 0
    mn = tl.minimum(ai, aj)
    mx = tl.maximum(ai, aj)
    tl.store(data_ptr + i, tl.where(asc, mn, mx))
    tl.store(data_ptr + ixj, tl.where(asc, mx, mn))


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK_ELEM = 1024
    LOG_BLOCK_ELEM = 10
    STEP_BLOCK = 256
    num_warps = 4
    num_stages = 2

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, BLOCK_ELEM)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_blocks = M // BLOCK_ELEM
    n_pairs = M // 2

    # Phase 1: local bitonic sort within each BLOCK_ELEM chunk
    _bitonic_local_init[(n_blocks,)](
        work,
        BLOCK_ELEM=BLOCK_ELEM, LOG_BLOCK_ELEM=LOG_BLOCK_ELEM,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Phase 2: cross-block merges
    k = 2 * BLOCK_ELEM
    while k <= M:
        j = k // 2
        while j >= BLOCK_ELEM:
            cross_grid = (max(n_pairs // STEP_BLOCK, 1),)
            _bitonic_cross[cross_grid](
                work, k, j,
                BLOCK_SIZE=STEP_BLOCK,
                num_warps=num_warps, num_stages=num_stages,
            )
            j //= 2
        # Finish all j < BLOCK_ELEM steps with one in-block sweep
        _bitonic_local_finalize[(n_blocks,)](
            work, k,
            BLOCK_ELEM=BLOCK_ELEM, LOG_BLOCK_ELEM=LOG_BLOCK_ELEM,
            num_warps=num_warps, num_stages=num_stages,
        )
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_ELEM": BLOCK_ELEM,
        "STEP_BLOCK": STEP_BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
