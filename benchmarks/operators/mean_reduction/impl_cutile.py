"""cuTile implementation of row-wise mean reduction.

Algorithm (one CTA per row):

  Tiled accumulation of sum(x):
      for j in range(num_tiles):
          xj = ct.load(x, index=(row, j), shape=(1, TILE_SIZE), padding_mode=ZERO)
          _acc += xj
      mean = ct.sum(_acc, axis=1, keepdims=False) / N

  Write one value per row to a compact (M, 1) output buffer:
      ct.store(out, index=(row, 0), tile=ct.full((1,1), 0.0) + mean)

ZERO padding on the last partial tile contributes 0 to the sum, so
the mean is computed correctly: sum(real) / N (not sum / TILE_SIZE).
"""
from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _mean_rowwise_kernel(x, out, N: ConstInt, TILE_SIZE: ConstInt):
    """One CTA per row. Accumulates sum in tiles, writes mean to out[row, 0]."""
    row       = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    _acc = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        _acc = _acc + xj

    mean = ct.sum(_acc, axis=1, keepdims=False) / N

    # Write a single value per row into the compact (M, 1) output buffer.
    out_tile = ct.full((1, 1), 0.0, dtype=np.float32) + mean
    ct.store(out, index=(row, 0), tile=out_tile, allow_tma=False, latency=1)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_mean_rowwise_kernel)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    """
    cuTile row-wise mean reduction.
    Input:  (M, N)  — any floating dtype
    Output: (M,) float32
    """
    global _last_autotune_config

    # Permute so that `dim` is last, treat all other dims as rows.
    if x.ndim == 2 and dim == 1:
        x2d = x.contiguous()
    else:
        dims = list(range(x.ndim))
        dims.remove(dim % x.ndim)
        dims.append(dim % x.ndim)
        x2d = x.permute(dims).contiguous().reshape(-1, x.shape[dim])

    M, N = x2d.shape

    # Compact output: one float32 per row.
    out = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    stream = torch.cuda.current_stream()
    grid   = (M, 1, 1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(M, N),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (x2d, out, N, cfg.tile_size),
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
              (x2d, out, N, cfg.tile_size))

    return out.squeeze(1)  # (M,) float32


def get_last_config() -> dict | None:
    return _last_autotune_config
