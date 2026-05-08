"""cuTile implementation of RMSNorm using tiled two-pass loops.

Algorithm (one CTA per row):

  Pass 1 – tiled accumulation of sum(x²):
      for j in range(num_tiles):
          xj = ct.load(x, index=(row, j), shape=(1, TILE_SIZE), padding_mode=ZERO)
          _rms += xj * xj
      rstd = ct.rsqrt(sum(_rms) / N + eps)

  Pass 2 – tiled normalize and scale:
      for j in range(num_tiles):
          yj = xj * rstd * wj
          ct.store(out, index=(row, j), tile=yj)

padding_mode=ZERO on the last tile handles non-power-of-2 N correctly:
out-of-bounds elements load as zero, contributing nothing to sum(x²).

Autotune parameters: TILE_SIZE (tile width, must be power of 2), occupancy.
"""
from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _rmsnorm_kernel(x, rms_w, out, eps, N: ConstInt, TILE_SIZE: ConstInt):
    """One CTA normalises one row using tiled two-pass RMSNorm."""
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    # Pass 1: accumulate sum(x²); padding_mode=ZERO handles the last partial tile.
    _rms = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        _rms = _rms + xj * xj

    rstd = ct.rsqrt(ct.sum(_rms, axis=1, keepdims=False) / N + eps)

    # Pass 2: normalize and scale.
    for j in range(0, num_tiles):
        wj = ct.astype(
            ct.load(rms_w, index=(j,), shape=(TILE_SIZE,),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = ct.astype(xj * rstd * wj, x.dtype)
        ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_rmsnorm_kernel)


def run(
    x: torch.Tensor,
    rms_w: torch.Tensor,
    eps: float = 1e-6,
    autotune: bool = False,
) -> torch.Tensor:
    global _last_autotune_config

    orig_shape = x.shape
    K       = orig_shape[-1]
    batch_M = x.numel() // K

    x_2d    = x.contiguous().reshape(batch_M, K)
    out     = torch.empty_like(x)
    out_2d  = out.reshape(batch_M, K)
    rms_w_c = rms_w.contiguous()

    stream = torch.cuda.current_stream()
    grid   = (batch_M, 1, 1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(batch_M, K),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (x_2d, rms_w_c, out_2d, eps, K, cfg.tile_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {
            "tile_size": cfg.tile_size,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (x_2d, rms_w_c, out_2d, eps, K, cfg.tile_size))

    return out


def get_last_config() -> dict | None:
    return _last_autotune_config
