from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]


@ct.kernel
def _reverse_kernel(x_ptr, out_ptr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offsets = ct.arange(TILE, dtype=ct.int32) + bid * TILE
    rev_indices = N - 1 - offsets
    vals = ct.gather(x_ptr, rev_indices, padding_value=0)
    ct.store(out_ptr, index=(bid,), tile=vals)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_autotune_config
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((N + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_reverse_kernel,
            args_fn=lambda cfg: (input, output, N, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = ((N + cfg.tile - 1) // cfg.tile, 1, 1)
        ct.launch(stream, grid, _reverse_kernel, (input, output, N, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
