import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _init_kernel(data, work, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        data,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.POS_INF,
    )
    ct.store(work, index=(bid,), tile=vals)


@ct.kernel(occupancy=8)
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


def run(data: torch.Tensor, N: int, **kwargs):
    n_int = int(N)
    if n_int <= 0:
        out = torch.empty((0,), device=data.device, dtype=data.dtype)
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": 1024,
            "occupancy": 8,
            "log2_M": 0,
            "passes": 0,
            "copy_kernel": 0,
        })
        return out

    m_int = 1 << ((n_int - 1).bit_length())
    log_m = m_int.bit_length() - 1

    work = torch.empty((m_int,), device=data.device, dtype=data.dtype)

    TILE = 1024
    occupancy = 8
    stream = torch.cuda.current_stream()

    init_grid = (ct.cdiv(m_int, TILE), 1, 1)
    ct.launch(stream, init_grid, _init_kernel, (data, work, TILE))

    if m_int > 1:
        pass_grid = (ct.cdiv(m_int // 2, TILE), 1, 1)
        k = 2
        while k <= m_int:
            j = k // 2
            while j > 0:
                j_log = j.bit_length() - 1
                ct.launch(stream, pass_grid, _bitonic_pass_kernel, (work, m_int, j, j_log, k, TILE))
                j //= 2
            k *= 2

    out = work[:n_int]

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "log2_M": log_m,
        "passes": (log_m * (log_m + 1)) // 2,
        "copy_kernel": 0,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
