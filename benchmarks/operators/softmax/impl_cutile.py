from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(block_size=1024, occupancy=8)
_SEARCH_SPACE_BASE = [
    SimpleNamespace(block_size=bs, occupancy=occ)
    for bs in [512, 1024, 2048]
    for occ in [4, 8, 16]
]


@ct.kernel
def softmax_online_kernel(
    input_tensor,
    output_tensor,
    N_COLS,
    N_TILES: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Online softmax matching Triton's two-pass BLOCK_SIZE-tiled algorithm.
    Each CTA processes one row in N_TILES chunks of BLOCK_SIZE.
    Pass 1: running max/sum; Pass 2: re-read, normalize, store.
    """
    row_idx = ct.bid(0)

    # Pass 1: online max + sum
    m = ct.full((), -float('inf'), dtype=np.float32)
    l = ct.full((), 0.0, dtype=np.float32)

    for i in range(N_TILES):
        tile = ct.load(
            input_tensor, index=(row_idx, i), shape=(1, BLOCK_SIZE),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile, np.float32)

        block_max = ct.max(tile)
        m_new = ct.maximum(m, block_max)
        # exp(-inf - m_new) = 0 for padded lanes, so they don't affect the sum
        l = l * ct.exp(m - m_new) + ct.sum(ct.exp(tile - m_new))
        m = m_new

    # Pass 2: normalize and write back. OOB writes are silently dropped.
    for i in range(N_TILES):
        tile = ct.load(
            input_tensor, index=(row_idx, i), shape=(1, BLOCK_SIZE),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile = ct.astype(tile, np.float32)

        y = ct.exp(tile - m) / l
        y = ct.astype(y, output_tensor.dtype)
        ct.store(output_tensor, index=(row_idx, i), tile=y)


def run(x: torch.Tensor, block_size: int = None, autotune: bool = False):
    global _last_autotune_config

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = (n_rows, 1, 1)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        search_space = [
            SimpleNamespace(
                block_size=cfg.block_size,
                n_tiles=(n_cols + cfg.block_size - 1) // cfg.block_size,
                occupancy=cfg.occupancy,
            )
            for cfg in _SEARCH_SPACE_BASE
        ]
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=softmax_online_kernel,
            args_fn=lambda cfg: (x, output, n_cols, cfg.n_tiles, cfg.block_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=search_space,
        )
        _last_autotune_config = {
            "block_size": result.tuned_config.block_size,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        n_tiles = (n_cols + cfg.block_size - 1) // cfg.block_size
        ct.launch(
            stream, grid, softmax_online_kernel,
            (x, output, n_cols, n_tiles, cfg.block_size),
        )

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
