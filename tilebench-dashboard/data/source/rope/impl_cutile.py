from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(group_size=16, occupancy=4)

_SEARCH_SPACE = [
    SimpleNamespace(group_size=gs, occupancy=occ)
    for gs in [1, 2, 4, 8, 16]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def rope_embedding(
    Q,
    Cos,
    Sin,
    SeqLen:     ConstInt,
    TILE_DIM:   ConstInt,
    GROUP_SIZE: ConstInt,
):
    row_id = ct.bid(0)
    group_id = ct.bid(1)

    seq_idx = row_id % SeqLen


    cos_tile = ct.reshape(ct.load(Cos, index=(seq_idx, 0), shape=(1, TILE_DIM)),
                          (1, 1, 1, TILE_DIM))
    sin_tile = ct.reshape(ct.load(Sin, index=(seq_idx, 0), shape=(1, TILE_DIM)),
                          (1, 1, 1, TILE_DIM))


    q1 = ct.load(Q, index=(row_id, group_id, 0, 0),
                 shape=(1, GROUP_SIZE, 1, TILE_DIM),
                 padding_mode=ct.PaddingMode.ZERO)
    q2 = ct.load(Q, index=(row_id, group_id, 1, 0),
                 shape=(1, GROUP_SIZE, 1, TILE_DIM),
                 padding_mode=ct.PaddingMode.ZERO)

    out1 = q1 * cos_tile - q2 * sin_tile
    out2 = q2 * cos_tile + q1 * sin_tile

    ct.store(Q, index=(row_id, group_id, 0, 0), tile=out1)
    ct.store(Q, index=(row_id, group_id, 1, 0), tile=out2)


_tuner = CutileAutotuner(rope_embedding)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        block_size: int = None, autotune: bool = False):


    output = q.clone().contiguous()

    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2

    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)
    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    stream = torch.cuda.current_stream()

    if autotune:


        tmp = output.clone()
        tmp_view = tmp.view(batch * seq_len, n_heads, 2, half_dim)
        cfg = _tuner.tune_or_cached(
            shape_key=(batch, seq_len, n_heads, head_dim, str(q.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                batch * seq_len,
                (n_heads + cfg.group_size - 1) // cfg.group_size,
                1,
            ),
            args_fn=lambda cfg: (
                tmp_view, cos_view, sin_view, seq_len, half_dim, cfg.group_size,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"group_size": cfg.group_size, "occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    grid = (
        batch * seq_len,
        (n_heads + cfg.group_size - 1) // cfg.group_size,
        1,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (output_view, cos_view, sin_view, seq_len, half_dim, cfg.group_size),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
