import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_step(data_ptr, k, j, n_pairs,
                  BLOCK: tl.constexpr):
    """One bitonic compare-exchange step at distance j within length-k subsequences."""
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

    BLOCK = 1024
    num_warps = 4
    num_stages = 2

    M = 1 << ((N - 1).bit_length())

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    n_pairs = M // 2
    grid = (triton.cdiv(n_pairs, BLOCK),)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            _bitonic_step[grid](
                work, k, j, n_pairs,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            j //= 2
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
