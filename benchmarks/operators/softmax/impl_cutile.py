from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(block_size=1024, occupancy=8)
# Excluded cfgs that hang the cuTile codegen indefinitely on B200 / cuTile 1.3.0
# (observed via per-cfg 60s probe; see tilebench_run/autotune_timing_rerun/
# softmax_probe.out). Pattern: occupancy=16 with non-1024 block_size.
_HANG_CFGS = {(512, 16), (1024, 16), (2048, 16)}
_SEARCH_SPACE_BASE = [
    SimpleNamespace(block_size=bs, occupancy=occ)
    for bs in [512, 1024, 2048]
    for occ in [4, 8, 16]
    if (bs, occ) not in _HANG_CFGS
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


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
# See core/cutile_autotune.py for why both layers matter.
_tuner = CutileAutotuner(softmax_online_kernel)


def run(x: torch.Tensor, block_size: int = None, autotune: bool = False):

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = (n_rows, 1, 1)
    stream = torch.cuda.current_stream()

    if autotune:
        # n_tiles depends on n_cols, so derive search_space per shape.
        search_space = [
            SimpleNamespace(
                block_size=cfg.block_size,
                n_tiles=(n_cols + cfg.block_size - 1) // cfg.block_size,
                occupancy=cfg.occupancy,
            )
            for cfg in _SEARCH_SPACE_BASE
        ]
        cfg = _tuner.tune_or_cached(
            shape_key=(n_rows, n_cols),
            search_space=search_space,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (x, output, n_cols, cfg.n_tiles, cfg.block_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "block_size": cfg.block_size,
            "occupancy": cfg.occupancy,
        })
        n_tiles = cfg.n_tiles
    else:
        cfg = _DEFAULT_CONFIG
        n_tiles = (n_cols + cfg.block_size - 1) // cfg.block_size

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (x, output, n_cols, n_tiles, cfg.block_size),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
