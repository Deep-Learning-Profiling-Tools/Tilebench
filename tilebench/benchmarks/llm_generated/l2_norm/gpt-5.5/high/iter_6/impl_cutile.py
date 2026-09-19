import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _l2_norm_kernel(x, output, eps, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

    x_tile = ct.load(
        x,
        index=(b, m, 0),
        shape=(1, 1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    x_f = ct.astype(x_tile, np.float32)

    sum_sq = ct.sum(x_f * x_f)
    eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
    rstd = ct.rsqrt(ct.maximum(sum_sq, eps_f * eps_f))

    y = ct.astype(x_f * rstd, x.dtype)
    ct.store(
        output,
        index=(b, m, 0),
        tile=y,
        latency=1,
        allow_tma=False,
    )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    TILE = 16384
    occupancy = 1

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel, (x, output, eps, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "MATH_RSQRT": 1})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
