import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _bitonic_step(data, k, j, BLOCK: ConstInt):
    pid = ct.bid(0)
    tid = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = (tid // j) * (2 * j) + (tid % j)
    ixj = i + j

    ai = ct.gather(data, i, check_bounds=False)
    aj = ct.gather(data, ixj, check_bounds=False)
    asc = (i & k) == 0
    mn = ct.minimum(ai, aj)
    mx = ct.maximum(ai, aj)
    ct.scatter(data, i, ct.where(asc, mn, mx), check_bounds=False)
    ct.scatter(data, ixj, ct.where(asc, mx, mn), check_bounds=False)


def run(data: torch.Tensor, N: int, **kwargs):
    if N <= 1:
        return data.clone()

    BLOCK = 2048

    M_natural = 1 << ((N - 1).bit_length())
    M = max(M_natural, 2 * BLOCK)

    work = torch.empty(M, dtype=data.dtype, device=data.device)
    work[:N] = data
    if M > N:
        work[N:] = float('inf')

    stream = torch.cuda.current_stream()
    n_pairs = M // 2
    grid = (max(n_pairs // BLOCK, 1), 1, 1)

    kernel = _bitonic_step.with_hints(occupancy=4)

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            ct.launch(stream, grid, kernel, (work, k, j, BLOCK))
            j //= 2
        k *= 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK": BLOCK, "occupancy": 4})
    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
