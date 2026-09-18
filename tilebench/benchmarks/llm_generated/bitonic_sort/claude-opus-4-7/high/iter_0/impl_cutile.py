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
