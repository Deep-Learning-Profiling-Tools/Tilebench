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

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096]
    for occ in [1, 2, 4]
]


@ct.kernel
def _interleave_kernel(a_ptr, b_ptr, out_ptr, N, TILE: ConstInt):
    bid = ct.bid(0)
    a_tile = ct.load(a_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    base = bid * TILE
    offsets = ct.arange(TILE, dtype=ct.int32) + base
    ct.scatter(out_ptr, offsets * 2, a_tile)
    ct.scatter(out_ptr, offsets * 2 + 1, b_tile)


def run(A: torch.Tensor, B: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_autotune_config
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((N + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_interleave_kernel,
            args_fn=lambda cfg: (A, B, output, N, cfg.tile),
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
        ct.launch(stream, grid, _interleave_kernel, (A, B, output, N, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
