from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile_r=1, tile_c=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile_r=tr, tile_c=tc, occupancy=occ)
    for tr in [1, 2, 4]
    for tc in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config = None


@ct.kernel
def _jacobi_stencil_kernel(input_2d, output_2d, rows, cols,
                           TILE_R: ConstInt, TILE_C: ConstInt):
    """
    2D 5-point Jacobi stencil — direct mirror of Triton's per-pixel-offset method:
      - Each block handles a (TILE_R, TILE_C) output tile.
      - For each output cell, ct.gather its four cardinal neighbors by running-time
        2D indices (the cuTile equivalent of Triton's pointer arithmetic with strides).
      - Interior cells get the average; boundary cells get the center value (via ct.where).
    """
    bid_r = ct.bid(0)
    bid_c = ct.bid(1)

    offs_r = bid_r * TILE_R + ct.arange(TILE_R, dtype=np.int32)
    offs_c = bid_c * TILE_C + ct.arange(TILE_C, dtype=np.int32)
    offs_r_2d = offs_r[:, None]   # (TILE_R, 1)
    offs_c_2d = offs_c[None, :]   # (1, TILE_C)

    # Interior mask (all 4 cardinal neighbors in-bounds); center mask (within grid).
    mask_edge = (
        (offs_r_2d - 1 >= 0) & (offs_r_2d < rows - 1)
        & (offs_c_2d - 1 >= 0) & (offs_c_2d < cols - 1)
    )
    mask_center = (offs_r_2d < rows) & (offs_c_2d < cols)

    # ct.gather broadcasts (TILE_R, 1) x (1, TILE_C) → (TILE_R, TILE_C) per-element indices.
    # Negative or out-of-range indices return padding_value=0 — harmless because the
    # result at boundary positions is overwritten by `center` via ct.where below.
    top    = ct.gather(input_2d, (offs_r_2d - 1, offs_c_2d),     padding_value=0.0)
    bottom = ct.gather(input_2d, (offs_r_2d + 1, offs_c_2d),     padding_value=0.0)
    left   = ct.gather(input_2d, (offs_r_2d,     offs_c_2d - 1), padding_value=0.0)
    right  = ct.gather(input_2d, (offs_r_2d,     offs_c_2d + 1), padding_value=0.0)
    center = ct.gather(input_2d, (offs_r_2d,     offs_c_2d),     padding_value=0.0)

    avg = 0.25 * (top + bottom + left + right)
    tile_out = ct.where(mask_edge, avg, center)

    # Tile-aligned store; positions beyond (rows, cols) are silently dropped (mask_center
    # is implicit in ct.store's OOB semantics).
    ct.store(output_2d, index=(bid_r, bid_c), tile=tile_out)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_jacobi_stencil_kernel)


def run(input: torch.Tensor, rows: int, cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 2D 5-point Jacobi stencil — direct mirror of Triton's method using
    ct.gather for per-pixel runtime-indexed reads (equivalent to Triton's
    tl.load with per-pixel pointer offsets).
    """
    global _last_autotune_config

    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(rows, cols),
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
        _last_autotune_config = {
            "tile_r":    cfg.tile_r,
            "tile_c":    cfg.tile_c,
            "occupancy": cfg.occupancy,
        }
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
    return _last_autotune_config
