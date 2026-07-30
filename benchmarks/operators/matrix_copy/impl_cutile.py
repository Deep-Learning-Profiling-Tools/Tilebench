from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)


@ct.kernel
def matrix_copy_kernel(a_ptr, b_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    tile = ct.load(a_ptr, index=(bid,), shape=(TILE,))
    ct.store(b_ptr, index=(bid,), tile=tile)


_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096]
    for occ in [4, 8, 16]
]


_tuner = CutileAutotuner(matrix_copy_kernel)


def run(A: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    B = torch.empty_like(A)
    total = N * N
    stream = torch.cuda.current_stream()

    A_flat = A.contiguous().view(-1)
    B_flat = B.view(-1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((total + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (A_flat, B_flat, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile": cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream,
        ((total + cfg.tile - 1) // cfg.tile, 1, 1),
        kernel,
        (A_flat, B_flat, cfg.tile),
    )

    return B


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
