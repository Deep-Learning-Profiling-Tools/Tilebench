import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
def _ln_kernel(x, w, b, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)

    # Single-pass: load entire row in one tile, kept resident in registers/shmem
    x_tile = ct.astype(ct.load(x, index=(row, 0), shape=(1, TILE),
                               padding_mode=ct.PaddingMode.ZERO),
                       np.float32)

    inv_N = 1.0 / N
    # OOB pads to zero so sum & sum_xx are correct over the valid K elements
    mean = ct.sum(x_tile) * inv_N
    mean_xx = ct.sum(x_tile * x_tile) * inv_N
    var = mean_xx - mean * mean
    rstd = ct.rsqrt(var + eps)

    w_tile = ct.astype(ct.load(w, index=(0,), shape=(TILE,),
                               padding_mode=ct.PaddingMode.ZERO),
                       np.float32)
    b_tile = ct.astype(ct.load(b, index=(0,), shape=(TILE,),
                               padding_mode=ct.PaddingMode.ZERO),
                       np.float32)

    y = (x_tile - mean) * rstd * w_tile.reshape((1, TILE)) + b_tile.reshape((1, TILE))
    ct.store(out, index=(row, 0), tile=ct.astype(y, x.dtype))


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    out = torch.empty_like(x)
    K = x.shape[-1]
    rows = x.numel() // K
    x2d = x.contiguous().view(rows, K)
    out2d = out.view(rows, K)
    stream = torch.cuda.current_stream()

    # TILE chosen so that the entire row (K ≤ 10240) fits in a single tile,
    # eliminating the second read of x present in the two-pass layout.
    TILE = 16384

    ct.launch(stream, (rows, 1, 1), _ln_kernel,
              (x2d, weight, bias, out2d, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 1, "single_pass": True})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
