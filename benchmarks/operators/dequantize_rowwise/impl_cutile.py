"""cuTile dequantize_rowwise (mirrors impl_triton.py).

Each CTA dequantises one row chunk.  The previous version loaded the entire
(1, COLS) row tile in one CTA, which keeps thousands of elements live for
COLS=8192.  This version splits each row into CHUNK-sized tiles to reduce
register/local-memory pressure.

Autotune knobs: chunk size and occupancy.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(chunk=512, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(chunk=ch, occupancy=occ)
    for ch in [256, 512, 1024]
    for occ in [4, 8, 16]
]

_INV_127 = 1.0 / 127.0


@ct.kernel
def dequantize_rowwise_kernel(x, state_x, output, COLS: ConstInt, CHUNK: ConstInt):
    row = ct.bid(0)
    col_tile = ct.bid(1)

    x_tile = ct.load(
        x, index=(row, col_tile), shape=(1, CHUNK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f32 = ct.astype(x_tile, ct.float32)
    scale = ct.load(state_x, index=(row,), shape=(1,))
    scale_f32 = ct.astype(scale, ct.float32)
    out = x_f32 * ct.reshape(scale_f32, (1, 1)) * _INV_127
    ct.store(output, index=(row, col_tile), tile=ct.astype(out, ct.float16))


_tuner = CutileAutotuner(dequantize_rowwise_kernel)


def run(x: torch.Tensor, state_x: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:

    rows, cols = x.shape
    output = torch.empty(rows, cols, device=x.device, dtype=torch.float16)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(rows, cols),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (rows, (cols + cfg.chunk - 1) // cfg.chunk, 1),
            args_fn=lambda cfg: (x, state_x, output, cols, cfg.chunk),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"chunk": cfg.chunk, "occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = (rows, (cols + cfg.chunk - 1) // cfg.chunk, 1)
    ct.launch(stream, grid, kernel, (x, state_x, output, cols, cfg.chunk))
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
