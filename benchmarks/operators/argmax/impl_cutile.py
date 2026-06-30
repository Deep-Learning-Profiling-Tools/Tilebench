from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(block_n=256, occupancy=8)
_SEARCH_SPACE_BASE = [
    SimpleNamespace(block_n=bn, occupancy=occ)
    for bn in [256, 512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]
_last_autotune_config: dict = {}


@ct.kernel
def _argmax_rowwise_kernel(
    input_flat,
    output_flat,
    N,
    N_TILES: ConstInt,
    BLOCK_N: ConstInt,
):
    """
    Chunked row-wise argmax matching Triton's BLOCK_N-tiled scan.
    Each CTA processes one row in N_TILES chunks of BLOCK_N.
    Per chunk: ct.max + ct.argmax → scalar, then sequential comparison
    (mirrors Triton's tl.max / tl.argmax + scalar best_val / best_idx).
    """
    row = ct.bid(0)
    base = row * N

    best_val = ct.full((), -float("inf"), dtype=ct.float32)
    best_idx = ct.full((), 0, dtype=ct.int64)

    for i in range(N_TILES):
        start = i * BLOCK_N
        offsets = start + ct.arange(BLOCK_N, dtype=ct.int32)
        valid = offsets < N

        idx = base + offsets
        idx_safe = ct.where(valid, idx, -1)
        chunk = ct.gather(input_flat, idx_safe, padding_value=-float("inf"))
        chunk = ct.astype(chunk, ct.float32)

        tile_max = ct.max(chunk)
        tile_arg = ct.astype(ct.argmax(chunk), ct.int64)

        better = tile_max > best_val
        best_val = ct.where(better, tile_max, best_val)
        best_idx = ct.where(better, start + tile_arg, best_idx)

    ct.store(output_flat, index=(row,), tile=ct.reshape(best_idx, (1,)))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_argmax_rowwise_kernel)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    assert x.is_cuda, "x must be on CUDA"

    if dim == 1:
        x2d = x.contiguous()
    else:
        x2d = x.transpose(0, 1).contiguous()

    M, N = x2d.shape
    input_flat = x2d.view(-1)

    output = torch.empty(M, dtype=torch.int64, device=x.device)
    stream = torch.cuda.current_stream()

    if autotune:
        # n_tiles depends on N, so derive search_space per shape.
        search_space = [
            SimpleNamespace(block_n=cfg.block_n,
                            n_tiles=(N + cfg.block_n - 1) // cfg.block_n,
                            occupancy=cfg.occupancy)
            for cfg in _SEARCH_SPACE_BASE
        ]
        cfg = _tuner.tune_or_cached(
            shape_key=(M, N),
            search_space=search_space,
            stream=stream,
            grid_fn=lambda cfg: (M, 1, 1),
            args_fn=lambda cfg: (input_flat, output, N, cfg.n_tiles, cfg.block_n),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "block_n": cfg.block_n,
            "occupancy": cfg.occupancy,
        })
        n_tiles = cfg.n_tiles
    else:
        cfg = _DEFAULT_CONFIG
        n_tiles = (N + cfg.block_n - 1) // cfg.block_n

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, (M, 1, 1), kernel,
              (input_flat, output, N, n_tiles, cfg.block_n))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
