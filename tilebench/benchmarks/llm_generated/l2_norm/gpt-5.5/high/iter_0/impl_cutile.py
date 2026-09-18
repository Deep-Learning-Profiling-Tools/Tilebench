import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _l2_norm_kernel(x, output, eps, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

    num_tiles = ct.cdiv(K, TILE)

    acc = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(b, m, j),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.astype(xj, np.float32)
        acc = acc + xj_f * xj_f

    sum_sq = ct.sum(acc)
    eps_f = sum_sq * 0.0 + eps
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(b, m, j),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.astype(xj, np.float32)
        y = ct.astype(xj_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, j),
            tile=y,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]

    TILE = 1024
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, eps, K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
