from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile_size=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile_size=ts, occupancy=occ)
    for ts in [512, 1024, 2048]
    for occ in [4, 8, 16]
]


@ct.kernel
def rmsnorm_kernel(x, rms_w, out, eps, N: ConstInt, TILE_SIZE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE_SIZE)


    _rms = ct.full((1, TILE_SIZE), 0.0, dtype=ct.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        _rms = _rms + xj * xj

    rstd = ct.rsqrt(ct.sum(_rms, axis=1, keepdims=False) / N + eps)


    for j in range(0, num_tiles):
        wj = ct.astype(
            ct.load(rms_w, index=(j,), shape=(TILE_SIZE,),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE_SIZE),
                    allow_tma=False, latency=1,
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        yj = ct.astype(xj * rstd * wj, x.dtype)
        ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)


_tuner = CutileAutotuner(rmsnorm_kernel)


def run(
    x: torch.Tensor,
    rms_w: torch.Tensor,
    eps: float = 1e-6,
    autotune: bool = False,
) -> torch.Tensor:

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
            shape_key=(batch_M, K, str(x.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (x_2d, rms_w_c, out_2d, eps, K, cfg.tile_size),
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
              (x_2d, rms_w_c, out_2d, eps, K, cfg.tile_size))

    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
