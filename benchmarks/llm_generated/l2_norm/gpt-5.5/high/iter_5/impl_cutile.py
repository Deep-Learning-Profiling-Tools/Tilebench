import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _l2_norm_kernel(x, output, eps, TILE_MAIN: ConstInt, TILE_TAIL: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]
    k_size = x.shape[2]

    b = row // m_size
    m = row - b * m_size

    x0 = ct.load(
        x,
        index=(b, m, 0),
        shape=(1, 1, TILE_MAIN),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    x0_f = ct.astype(x0, np.float32)
    sum_sq = ct.sum(x0_f * x0_f)

    if k_size > TILE_MAIN:
        x1 = ct.load(
            x,
            index=(b, m, TILE_MAIN),
            shape=(1, 1, TILE_TAIL),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x1_f = ct.astype(x1, np.float32)
        sum_sq = sum_sq + ct.sum(x1_f * x1_f)

    eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    y0 = ct.astype(x0_f * rstd, x.dtype)
    ct.store(
        output,
        index=(b, m, 0),
        tile=y0,
        latency=1,
        allow_tma=False,
    )

    if k_size > TILE_MAIN:
        y1 = ct.astype(x1_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, TILE_MAIN),
            tile=y1,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    TILE_MAIN = 8192
    TILE_TAIL = 2048
    occupancy = 1

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel, (x, output, eps, TILE_MAIN, TILE_TAIL))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_MAIN": TILE_MAIN,
            "TILE_TAIL": TILE_TAIL,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
