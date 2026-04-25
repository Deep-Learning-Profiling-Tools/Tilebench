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
def rope_kernel(
    Q,                  # Rank 4: [TotalTokens, Heads, 2, HalfDim]
    Cos,                # Rank 2: [SeqLen, HalfDim]
    Sin,                # Rank 2: [SeqLen, HalfDim]
    SeqLen: ConstInt,
    TILE_DIM: ConstInt  # HalfDim
):

    row_id = ct.bid(0)   # Batch*Seq
    head_id = ct.bid(1)  # Heads

    seq_idx = row_id % SeqLen

    cos_tile = ct.load(Cos, index=(seq_idx, 0), shape=(1, TILE_DIM))
    sin_tile = ct.load(Sin, index=(seq_idx, 0), shape=(1, TILE_DIM))

    q1 = ct.load(Q, index=(row_id, head_id, 0, 0), shape=(1, 1, 1, TILE_DIM))

    q2 = ct.load(Q, index=(row_id, head_id, 1, 0), shape=(1, 1, 1, TILE_DIM))

    out1 = q1 * cos_tile - q2 * sin_tile
    out2 = q2 * cos_tile + q1 * sin_tile

    ct.store(Q, index=(row_id, head_id, 0, 0), tile=out1)
    ct.store(Q, index=(row_id, head_id, 1, 0), tile=out2)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = None, autotune: bool = False):
    global _last_autotune_config
    output = q.clone().contiguous()

    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2

    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)

    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    grid = (batch * seq_len, n_heads, 1)

    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        # rope is in-place; tune on a tmp buffer so trials don't accumulate rotations.
        tmp = output.clone()
        tmp_view = tmp.view(batch * seq_len, n_heads, 2, half_dim)
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=rope_kernel,
            args_fn=lambda cfg: (tmp_view, cos_view, sin_view, seq_len, half_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {"occupancy": result.tuned_config.occupancy}
        # Apply the chosen config to the real output via a single-config autotune call.
        ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=rope_kernel,
            args_fn=lambda cfg: (output_view, cos_view, sin_view, seq_len, half_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=[result.tuned_config],
        )
    else:
        ct.launch(
            stream,
            grid,
            rope_kernel,
            (output_view, cos_view, sin_view, seq_len, half_dim)
        )

    return output

def get_last_config() -> dict | None:
    return _last_autotune_config
