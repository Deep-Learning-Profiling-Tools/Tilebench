"""cuTile fp32 -> fp16 quantization (elementwise cast).

1D grid; each CTA loads a TILE of fp32 and stores it as fp16. Pure
bandwidth-bound, so the autotune sweep is over (TILE, occupancy) only.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

# 1:1 mirrors impl_triton.py: tile <-> BLOCK_SIZE, occupancy <-> num_warps
# (nw * occ ~= 64 on B200, so Triton's nw in [4, 8, 16] pairs with
# cuTile's occ in [16, 8, 4]).
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [2048, 4096, 8192, 16384]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def quantize_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(output, index=(bid,), tile=ct.astype(x_tile, ct.float16))


_tuner = CutileAutotuner(quantize_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:

    x = x.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n_elements,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1)
    ct.launch(stream, grid, kernel, (x, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
