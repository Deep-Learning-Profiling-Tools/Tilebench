import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _l2_norm_kernel(x, out, eps: float, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)

    # Single-pass: load whole row once, keep in registers.
    xj = ct.astype(
        ct.load(x, index=(row, 0), shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO),
        np.float32,
    )
    total = ct.sum(xj * xj)
    inv = 1.0 / ct.maximum(ct.sqrt(total), eps)
    yj = xj * inv
    ct.store(out, index=(row, 0), tile=ct.astype(yj, x.dtype))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    TILE = max(_next_pow2(K), 256)
    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel,
              (x_flat, out_flat, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 2, "single_pass": True})
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
