import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _ln_kernel(x, w, b, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    inv_N = 1.0 / N

    # Pass 1: accumulate scalar partial sums (small register footprint)
    sum_x = 0.0
    sum_xx = 0.0
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        sum_x = sum_x + ct.sum(xj)
        sum_xx = sum_xx + ct.sum(xj * xj)

    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = ct.rsqrt(var + eps)

    # Pass 2: normalize and write
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        bj = ct.astype(
            ct.load(b, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        yj = (xj - mean) * rstd * wj.reshape((1, TILE)) + bj.reshape((1, TILE))
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    out = torch.empty_like(x)
    K = x.shape[-1]
    rows = x.numel() // K
    x2d = x.contiguous().view(rows, K)
    out2d = out.view(rows, K)
    stream = torch.cuda.current_stream()

    TILE = 2048

    ct.launch(stream, (rows, 1, 1), _ln_kernel,
              (x2d, weight, bias, out2d, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 4, "scalar_accum": True})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
