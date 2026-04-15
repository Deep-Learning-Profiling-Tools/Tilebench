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
def _interleave_kernel(a_ptr, b_ptr, out_ptr, TILE: ConstInt):
    """
    In-tile interleave matching Triton's tl.interleave method:
      1. Coalesced load of TILE consecutive elements from A and B.
      2. Stack into (2, TILE), transpose to (TILE, 2), flatten. Row-major
         layout of (TILE, 2) is [a0, b0, a1, b1, ..., a_{TILE-1}, b_{TILE-1}].
      3. Single coalesced contiguous store of 2*TILE elements at bid*2*TILE.

    Out-of-bounds tail elements are zeroed on load (padding_mode=ZERO) and
    silently dropped on store.
    """
    bid = ct.bid(0)
    a_tile = ct.load(a_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    a_2d = ct.reshape(a_tile, (1, TILE))
    b_2d = ct.reshape(b_tile, (1, TILE))
    stacked = ct.cat((a_2d, b_2d), axis=0)          # (2, TILE) — rows are A, B
    transposed = ct.transpose(stacked)               # (TILE, 2) — row i = (a_i, b_i)
    interleaved = ct.reshape(transposed, (2 * TILE,))

    ct.store(out_ptr, index=(bid,), tile=interleaved)


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
            args_fn=lambda cfg: (A, B, output, cfg.tile),
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
        ct.launch(stream, grid, _interleave_kernel, (A, B, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
