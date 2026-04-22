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
    for t in [256, 512, 1024, 2048, 4096]
    for occ in [4, 8, 16]
]


@ct.kernel
def _dequant_kernel(x_ptr, s_ptr, out_ptr, N, TILE_SIZE, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE
    offsets = ct.arange(TILE, dtype=ct.int32) + base

    # Load X tile (flattened 1D)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    # Compute scale indices into 2D S array
    row = offsets // N
    col = offsets % N
    s_row = row // TILE_SIZE
    s_col = col // TILE_SIZE

    # Gather scales from 2D scale array
    scale_tile = ct.gather(s_ptr, (s_row, s_col), padding_value=0)

    # Compute in float32, cast back to input dtype
    x_f32 = ct.astype(x_tile, ct.float32)
    scale_f32 = ct.astype(scale_tile, ct.float32)
    result = x_f32 * scale_f32
    result_cast = ct.astype(result, x_ptr.dtype)

    ct.store(out_ptr, index=(bid,), tile=result_cast)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_autotune_config
    output = torch.empty(M, N, dtype=X.dtype, device=X.device)
    X_flat = X.contiguous().view(-1)
    out_flat = output.view(-1)
    total = M * N
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((total + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_dequant_kernel,
            args_fn=lambda cfg: (X_flat, S, out_flat, N, TILE_SIZE, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = ((total + cfg.tile - 1) // cfg.tile, 1, 1)
        ct.launch(stream, grid, _dequant_kernel,
                  (X_flat, S, out_flat, N, TILE_SIZE, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
