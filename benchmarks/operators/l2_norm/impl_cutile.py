from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [256, 512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _l2_norm_kernel(x, out, eps, N: ConstInt, TILE_SIZE: ConstInt):
    """One CTA normalises one row with tiled two-pass L2 norm."""
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    # Pass 1: accumulate sum(x²).
    _sum_sq = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        _sum_sq = _sum_sq + xj * xj

    rstd = ct.rsqrt(ct.sum(_sum_sq, axis=1, keepdims=False) + eps)

    # Pass 2: normalise.
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = ct.astype(xj * rstd, x.dtype)
        ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)


def run(x: torch.Tensor, eps: float = 1e-6, autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config

    orig_shape = x.shape
    K = orig_shape[-1]
    batch_M = x.numel() // K

    x_2d  = x.contiguous().reshape(batch_M, K)
    out   = torch.empty_like(x)
    out_2d = out.reshape(batch_M, K)

    stream = torch.cuda.current_stream()
    grid   = (batch_M, 1, 1)

    if autotune:
        result = ct.tune.exhaustive_search(
            _SEARCH_SPACE,
            stream,
            grid_fn=lambda cfg: grid,
            kernel=_l2_norm_kernel,
            args_fn=lambda cfg: (x_2d, out_2d, eps, K, cfg.tile_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        cfg = result.best.config
        _last_autotune_config = {
            "tile_size": cfg.tile_size,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    ct.launch(stream, grid, _l2_norm_kernel,
              (x_2d, out_2d, eps, K, cfg.tile_size))

    return out


def get_last_config() -> dict | None:
    return _last_autotune_config
