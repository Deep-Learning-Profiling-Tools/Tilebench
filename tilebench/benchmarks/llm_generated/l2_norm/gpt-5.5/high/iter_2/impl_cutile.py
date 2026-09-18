import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _l2_norm_kernel(x, output, eps, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

    x0 = ct.load(
        x,
        index=(b, m, 0),
        shape=(1, 1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    x0f = ct.astype(x0, np.float32)
    sum_sq = ct.sum(x0f * x0f)

    if x.shape[2] > TILE:
        x1 = ct.load(
            x,
            index=(b, m, 1),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x1f = ct.astype(x1, np.float32)
        sum_sq = sum_sq + ct.sum(x1f * x1f)

    if x.shape[2] > 2 * TILE:
        x2 = ct.load(
            x,
            index=(b, m, 2),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x2f = ct.astype(x2, np.float32)
        sum_sq = sum_sq + ct.sum(x2f * x2f)

    if x.shape[2] > 3 * TILE:
        x3 = ct.load(
            x,
            index=(b, m, 3),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x3f = ct.astype(x3, np.float32)
        sum_sq = sum_sq + ct.sum(x3f * x3f)

    if x.shape[2] > 4 * TILE:
        x4 = ct.load(
            x,
            index=(b, m, 4),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x4f = ct.astype(x4, np.float32)
        sum_sq = sum_sq + ct.sum(x4f * x4f)

    eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    y0 = ct.astype(ct.astype(x0, np.float32) * rstd, x.dtype)
    ct.store(
        output,
        index=(b, m, 0),
        tile=y0,
        latency=1,
        allow_tma=False,
    )

    if x.shape[2] > TILE:
        y1 = ct.astype(ct.astype(x1, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 1),
            tile=y1,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 2 * TILE:
        y2 = ct.astype(ct.astype(x2, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 2),
            tile=y2,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 3 * TILE:
        y3 = ct.astype(ct.astype(x3, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 3),
            tile=y3,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 4 * TILE:
        y4 = ct.astype(ct.astype(x4, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 4),
            tile=y4,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    TILE = 2048
    MAX_TILES = 5
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, eps, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "MAX_TILES": MAX_TILES,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
