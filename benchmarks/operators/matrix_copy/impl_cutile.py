from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)


@ct.kernel
def matrix_copy_kernel(a_ptr, b_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    tile = ct.load(a_ptr, index=(bid,), shape=(TILE,))
    ct.store(b_ptr, index=(bid,), tile=tile)


_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]


def run(A: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_autotune_config
    B = torch.empty_like(A)
    total = N * N
    stream = torch.cuda.current_stream()

    A_flat = A.contiguous().view(-1)
    B_flat = B.view(-1)

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((total + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=matrix_copy_kernel,
            args_fn=lambda cfg: (A_flat, B_flat, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(
            stream,
            ((total + cfg.tile - 1) // cfg.tile, 1, 1),
            matrix_copy_kernel,
            (A_flat, B_flat, cfg.tile),
        )

    return B


def get_last_config() -> dict | None:
    return _last_autotune_config
