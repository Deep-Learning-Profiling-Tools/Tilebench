"""cuTile implementation of L2 normalisation (per-row).

Algorithm (one CTA per row):

  Pass 1 – tiled accumulation of sum(x²):
      for j in range(num_tiles):
          xj = ct.load(x, index=(row, j), shape=(1, TILE_SIZE), padding_mode=ZERO)
          _sum_sq += xj * xj
      rstd = ct.rsqrt(sum(_sum_sq) + eps)

  Pass 2 – tiled normalise:
      for j in range(num_tiles):
          yj = xj * rstd
          ct.store(out, index=(row, j), tile=yj)

padding_mode=ZERO on the last tile: OOB elements load as 0 → contribute 0
to sum(x²), so rstd is computed correctly without any masking.
"""
from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [256, 512, 1024, 2048]
    for occ in [1, 2, 4, 8]
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

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=_l2_norm_kernel,
            args_fn=lambda cfg: (x_2d, out_2d, eps, K, cfg.tile_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile_size": result.tuned_config.tile_size,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(stream, grid, _l2_norm_kernel,
                  (x_2d, out_2d, eps, K, cfg.tile_size))

    return out


def get_last_config() -> dict | None:
    return _last_autotune_config
