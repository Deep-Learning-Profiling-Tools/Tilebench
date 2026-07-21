from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]


@ct.kernel
def interleave_kernel(a_ptr, b_ptr, out_ptr_2d, TILE: ConstInt):
    """Direct interleave without cat/transpose materialization.

    Output is viewed as (N, 2).  Each CTA writes one A tile to column 0 and
    one B tile to column 1.  This preserves the final flat layout
    [a0, b0, a1, b1, ...] while avoiding the old (2,TILE)->transpose->reshape
    intermediate that spilled for fp32 at large TILE.
    """
    bid = ct.bid(0)
    a_tile = ct.load(a_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out_ptr_2d, index=(bid, 0), tile=ct.reshape(a_tile, (TILE, 1)))
    ct.store(out_ptr_2d, index=(bid, 1), tile=ct.reshape(b_tile, (TILE, 1)))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(interleave_kernel)


def run(A: torch.Tensor, B: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    output_2d = output.view(N, 2)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N, str(A.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((N + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (A, B, output_2d, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile": cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = ((N + cfg.tile - 1) // cfg.tile, 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (A, B, output_2d, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
