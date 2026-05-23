import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rmsnorm_kernel(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # Pass 1: sum of squares in fp32
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        acc = acc + xj * xj

    mean_sq = ct.sum(acc, axis=1, keepdims=True) / N
    rstd = ct.rsqrt(mean_sq + eps)

    # Pass 2: normalize and scale
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        yj = xj * rstd * wj[None, :]
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)

    # TILE=2048 was best (iter1: 75.6%). Increase occupancy from 2 -> 4 so more
    # CTAs are resident per SM, giving the scheduler more outstanding memory
    # transactions to hide DRAM latency. Add latency=1 hints (Pattern C) to
    # encourage aggressive async-copy issue.
    TILE = 2048
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)
    ct.launch(stream, grid, _rmsnorm_kernel, (x2, rms_w, o2, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
