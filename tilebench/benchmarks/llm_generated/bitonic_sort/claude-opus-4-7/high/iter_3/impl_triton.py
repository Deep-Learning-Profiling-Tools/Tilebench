import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_local_init(data_ptr,
                        B: tl.constexpr,
                        LOG_B: tl.constexpr):
    """In-block bitonic sort: handles k = 2, 4, ..., B (all stages internal)."""
    pid = tl.program_id(0)
    base = pid * B
    local = tl.arange(0, B)
    offs = base + local
    x = tl.load(data_ptr + offs)

    for log_k_m1 in tl.static_range(LOG_B):
        log_k = log_k_m1 + 1
        k_val = 1 << log_k
        asc = (offs & k_val) == 0
        for stage in tl.static_range(log_k):
            log_j = log_k - 1 - stage
            j_val = 1 << log_j
            partner = local ^ j_val
            y = tl.gather(x, partner, 0)
            is_left = (local & j_val) == 0
            cond = asc != is_left
            x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_local_finalize(data_ptr, k,
                            B: tl.constexpr,
                            LOG_B: tl.constexpr):
    """Finish stages j = B/2, B/4, ..., 1 for a given k (all fit in-block)."""
    pid = tl.program_id(0)
    base = pid * B
    local = tl.arange(0, B)
    offs = base + local
    x = tl.load(data_ptr + offs)
    asc = (offs & k) == 0

    for stage in tl.static_range(LOG_B):
        log_j = LOG_B - 1 - stage
        j_val = 1 << log_j
        partner = local ^ j_val
        y = tl.gather(x, partner, 0)
        is_left = (local & j_val) == 0
        cond = asc != is_left
        x = tl.where(cond, tl.maximum(x, y), tl.minimum(x, y))

    tl.store(data_ptr + offs, x)


@triton.jit
def _bitonic_cross(data_ptr, k, j, n_pairs,
                   BLOCK: tl.constexpr):
    """One cross-block compare-exchange step at distance j."""
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_pairs
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = tl.load(data_ptr + i, mask=mask)
    aj = tl.load(data_ptr + ixj, mask=mask)
    asc = (i & k) == 0
    mn = tl.minimum(ai, aj)
    mx = tl.maximum(ai, aj)
    tl.store(data_ptr + i, tl.where(asc, mn, mx), mask=mask)
    tl.store(data_ptr + ixj, tl.where(asc, mx, mn), mask=mask)


@triton.jit
def _bitonic_step_simple(data_ptr, k, j, n_pairs,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_pairs
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = tl.load(data_ptr + i, mask=mask)
    aj = tl.load(data_ptr + ixj, mask=mask)
    asc = (i & k) == 0
    mn = tl.minimum(ai, aj)
    mx = tl.maximum(ai, aj)
    tl.store(data_ptr + i, tl.where(asc, mn, mx), mask=mask)
    tl.store(data_ptr + ixj, tl.where(asc, mx, mn), mask=mask)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    B = 256
    LOG_B = 8
    BLOCK_CROSS = 1024
    num_warps = 4
    num_stages = 2

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * B)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_blocks_local = M // B
    n_pairs = M // 2

    use_local = (M >= 2 * B)

    if use_local:
        # Phase 1: full in-block bitonic sort (handles k = 2 .. B)
        _bitonic_local_init[(n_blocks_local,)](
            work, B=B, LOG_B=LOG_B,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Phase 2: for each k > B, do cross-block steps then one local finalize
        k = 2 * B
        while k <= M:
            j = k // 2
            while j >= B:
                _bitonic_cross[(triton.cdiv(n_pairs, BLOCK_CROSS),)](
                    work, k, j, n_pairs,
                    BLOCK=BLOCK_CROSS,
                    num_warps=num_warps, num_stages=num_stages,
                )
                j //= 2
            _bitonic_local_finalize[(n_blocks_local,)](
                work, k, B=B, LOG_B=LOG_B,
                num_warps=num_warps, num_stages=num_stages,
            )
            k *= 2
    else:
        # Fallback: simple per-step
        k = 2
        while k <= M:
            j = k // 2
            while j > 0:
                _bitonic_step_simple[(triton.cdiv(n_pairs, BLOCK_CROSS),)](
                    work, k, j, n_pairs,
                    BLOCK=BLOCK_CROSS,
                    num_warps=num_warps, num_stages=num_stages,
                )
                j //= 2
            k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "B": B,
        "BLOCK_CROSS": BLOCK_CROSS,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
