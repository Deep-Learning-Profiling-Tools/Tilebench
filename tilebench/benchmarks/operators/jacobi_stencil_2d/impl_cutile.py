from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile_r=1, tile_c=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile_r=tr, tile_c=tc, occupancy=occ)
    for tr in [1, 2, 4]
    for tc in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def jacobi_stencil_kernel(input_2d, output_2d, rows, cols,
                           TILE_R: ConstInt, TILE_C: ConstInt):
    bid_r = ct.bid(0)
    bid_c = ct.bid(1)

    offs_r = bid_r * TILE_R + ct.arange(TILE_R, dtype=ct.int32)
    offs_c = bid_c * TILE_C + ct.arange(TILE_C, dtype=ct.int32)
    offs_r_2d = offs_r[:, None]
    offs_c_2d = offs_c[None, :]


    mask_edge = (
        (offs_r_2d - 1 >= 0) & (offs_r_2d < rows - 1)
        & (offs_c_2d - 1 >= 0) & (offs_c_2d < cols - 1)
    )
    mask_center = (offs_r_2d < rows) & (offs_c_2d < cols)


    top    = ct.gather(input_2d, (offs_r_2d - 1, offs_c_2d),     padding_value=0.0)
    bottom = ct.gather(input_2d, (offs_r_2d + 1, offs_c_2d),     padding_value=0.0)
    left   = ct.gather(input_2d, (offs_r_2d,     offs_c_2d - 1), padding_value=0.0)
    right  = ct.gather(input_2d, (offs_r_2d,     offs_c_2d + 1), padding_value=0.0)
    center = ct.gather(input_2d, (offs_r_2d,     offs_c_2d),     padding_value=0.0)

    avg = 0.25 * (top + bottom + left + right)
    tile_out = ct.where(mask_edge, avg, center)


    ct.store(output_2d, index=(bid_r, bid_c), tile=tile_out)


_tuner = CutileAutotuner(jacobi_stencil_kernel)


def run(input: torch.Tensor, rows: int, cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(rows, cols, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                ct.cdiv(rows, cfg.tile_r),
                ct.cdiv(cols, cfg.tile_c),
                1,
            ),
            args_fn=lambda cfg: (
                input, output, rows, cols, cfg.tile_r, cfg.tile_c,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile_r":    cfg.tile_r,
            "tile_c":    cfg.tile_c,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (
        ct.cdiv(rows, cfg.tile_r),
        ct.cdiv(cols, cfg.tile_c),
        1,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (input, output, rows, cols, cfg.tile_r, cfg.tile_c))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
