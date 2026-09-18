import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _ln_kernel(x, w, b, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)

    # Single-pass: load the entire row in one tile.
    # OOB positions are zero, so sum(x) and sum(x*x) are unaffected.
    x_tile = ct.astype(
        ct.load(x, index=(row, 0), shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32)

    inv_N = 1.0 / N
    sum_x = ct.sum(x_tile)
    sum_xx = ct.sum(x_tile * x_tile)
    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = ct.rsqrt(var + eps)

    w_tile = ct.astype(
        ct.load(w, index=(0,), shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32)
    b_tile = ct.astype(
        ct.load(b, index=(0,), shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32)

    # OOB positions: x_tile=0 -> contributes -mean*rstd*0 + 0 = 0,
    # and the store at OOB column indices is silently ignored.
    y = (x_tile - mean) * rstd * w_tile.reshape((1, TILE)) + b_tile.reshape((1, TILE))
    ct.store(out, index=(row, 0), tile=ct.astype(y, x.dtype))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    out = torch.empty_like(x)
    K = x.shape[-1]
    rows = x.numel() // K
    x2d = x.contiguous().view(rows, K)
    out2d = out.view(rows, K)
    stream = torch.cuda.current_stream()

    TILE = max(512, _next_pow2(K))

    # Larger tiles need lower occupancy to fit registers / shmem.
    if TILE >= 16384:
        occupancy = 1
    elif TILE >= 8192:
        occupancy = 2
    elif TILE >= 4096:
        occupancy = 3
    else:
        occupancy = 4

    ct.launch(stream, (rows, 1, 1),
              _ln_kernel.with_hints(occupancy=occupancy),
              (x2d, weight, bias, out2d, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "single_pass": True})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
