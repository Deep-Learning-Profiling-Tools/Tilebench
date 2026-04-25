from types import SimpleNamespace

import torch
import cuda.tile as ct
import math

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]

@ct.kernel
def softmax_kernel(
    input_tensor,
    output_tensor,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt
):
    """
    input_tensor: (Rows, Cols)
    output_tensor: (Rows, Cols)
    """
    row_idx = ct.bid(0)

    tile = ct.load(input_tensor, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.NEG_INF)

    max_val = ct.max(tile)

    exp_tile = ct.exp(tile - max_val)

    sum_val = ct.sum(exp_tile)

    output_tile = exp_tile / sum_val
    ct.store(output_tensor, index=(row_idx, 0), tile=output_tile)


def run(x: torch.Tensor, block_size: int, autotune: bool = False):
    global _last_autotune_config
    n_rows, n_cols = x.shape


    if block_size < n_cols:
        block_size = n_cols

    block_size = 1 << (block_size - 1).bit_length()

    output = torch.empty_like(x)


    grid = (n_rows, 1, 1)

    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=softmax_kernel,
            args_fn=lambda cfg: (x, output, n_cols, block_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {"occupancy": result.tuned_config.occupancy}
    else:
        ct.launch(
            stream,
            grid,
            softmax_kernel,
            (x, output, n_cols, block_size)
        )

    return output

def get_last_config() -> dict | None:
    return _last_autotune_config
