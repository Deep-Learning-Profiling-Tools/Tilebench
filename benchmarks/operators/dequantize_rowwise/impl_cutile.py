"""cuTile dequantize_rowwise (mirrors impl_triton.py).

Each CTA dequantises one row. Since cuTile tile dims must be powers of
two and case_grid restricts cols to powers of two, the entire row
loads in one ct.load((1, COLS)) — no inner tiled loop, no padding.

Autotune knob: occupancy (cuTile's analogue of Triton num_warps).
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(occupancy=occ)
    for occ in [2, 4, 8, 16, 32]
]

_INV_127 = 1.0 / 127.0


@ct.kernel
def _dequantize_rowwise_kernel(x, state_x, output, COLS: ConstInt):
    bid = ct.bid(0)
    # x: (rows, cols) int8 -- load row `bid` as (1, COLS).
    x_tile = ct.load(x, index=(bid, 0), shape=(1, COLS))
    # state_x: (rows,) fp32 -- load element `bid` as (1,), reshape for broadcast.
    scale = ct.load(state_x, index=(bid,), shape=(1,))
    scale_2d = ct.reshape(scale, (1, 1))

    x_f32 = ct.astype(x_tile, ct.float32)
    out_f32 = x_f32 * scale_2d * _INV_127
    ct.store(output, index=(bid, 0), tile=ct.astype(out_f32, ct.float16))


_tuner = CutileAutotuner(_dequantize_rowwise_kernel)


def run(x: torch.Tensor, state_x: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config

    rows, cols = x.shape
    output = torch.empty(rows, cols, device=x.device, dtype=torch.float16)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(rows, cols),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (rows, 1, 1),
            args_fn=lambda cfg: (x, state_x, output, cols),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {"occupancy": cfg.occupancy}
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, (rows, 1, 1), kernel, (x, state_x, output, cols))
    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
