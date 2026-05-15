from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _layernorm_kernel(x, weight, bias, out, eps, N: ConstInt, TILE_SIZE: ConstInt):
    """One CTA normalises one row using tiled two-pass LayerNorm."""
    row       = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    # Pass 1: accumulate sum(x) and sum(x²) in a single scan.
    _sum_x  = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    _sum_x2 = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        _sum_x  = _sum_x  + xj
        _sum_x2 = _sum_x2 + xj * xj

    mean = ct.sum(_sum_x,  axis=1, keepdims=False) / N
    var  = ct.sum(_sum_x2, axis=1, keepdims=False) / N - mean * mean
    rstd = ct.rsqrt(var + eps)

    # Pass 2: normalize, scale, and shift.
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        wj = ct.astype(
            ct.load(weight, index=(j,), shape=(TILE_SIZE,),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        bj = ct.astype(
            ct.load(bias, index=(j,), shape=(TILE_SIZE,),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = ct.astype((xj - mean) * rstd * wj + bj, x.dtype)
        ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_layernorm_kernel)


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    autotune: bool = False,
) -> torch.Tensor:

    orig_shape = x.shape
    K       = orig_shape[-1]
    batch_M = x.numel() // K

    x_2d     = x.contiguous().reshape(batch_M, K)
    out      = torch.empty_like(x)
    out_2d   = out.reshape(batch_M, K)
    weight_c = weight.contiguous()
    bias_c   = bias.contiguous()

    stream = torch.cuda.current_stream()
    grid   = (batch_M, 1, 1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(batch_M, K),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (x_2d, weight_c, bias_c, out_2d, eps, K, cfg.tile_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile_size": cfg.tile_size,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (x_2d, weight_c, bias_c, out_2d, eps, K, cfg.tile_size))

    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
