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
def interleave_kernel(a_ptr, b_ptr, out_ptr, TILE: ConstInt):
    """Merge in-tile, then one contiguous store — the ct analogue of Triton's
    tl.interleave.

    cat((TILE,1),(TILE,1), axis=1) -> (TILE,2), row-major reshape to
    (2*TILE,) yields [a0, b0, a1, b1, ...]; the single contiguous store
    vectorises (STG.E.128) where the previous per-column strided stores
    degraded to per-element STG.E.U8 at int8. Unlike the old
    (2,TILE)->transpose->reshape form, the axis=1 cat has no transpose and
    does not spill at fp32 (probe: 52 regs, no STL/LDL).
    """
    bid = ct.bid(0)
    a_tile = ct.load(a_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    merged = ct.cat((ct.reshape(a_tile, (TILE, 1)), ct.reshape(b_tile, (TILE, 1))), axis=1)
    ct.store(out_ptr, index=(bid,), tile=ct.reshape(merged, (2 * TILE,)))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(interleave_kernel)


def run(A: torch.Tensor, B: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N, str(A.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((N + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (A, B, output, cfg.tile),
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
    ct.launch(stream, grid, kernel, (A, B, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
