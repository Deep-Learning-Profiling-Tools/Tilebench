from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(group_size=4, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(group_size=gs, occupancy=occ)
    for gs in [1, 2, 4, 8, 16]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def rope_kernel(
    Q,                    # Rank 4: [TotalTokens, Heads, 2, HalfDim]
    Cos,                  # Rank 2: [SeqLen, HalfDim]
    Sin,                  # Rank 2: [SeqLen, HalfDim]
    SeqLen:     ConstInt,
    TILE_DIM:   ConstInt,  # HalfDim
    GROUP_SIZE: ConstInt,
):
    """RoPE in-place. One CTA processes GROUP_SIZE consecutive heads of one
    (batch, seq) row, sharing a single cos/sin tile across them — mirrors
    Triton's ROPE_GROUP_SIZE loop.
    """
    row_id = ct.bid(0)    # Batch*Seq
    group_id = ct.bid(1)  # Head group

    seq_idx = row_id % SeqLen

    # cos/sin shared across all heads in this group.
    cos_tile = ct.load(Cos, index=(seq_idx, 0), shape=(1, TILE_DIM))
    sin_tile = ct.load(Sin, index=(seq_idx, 0), shape=(1, TILE_DIM))

    head_start = group_id * GROUP_SIZE
    for k in range(GROUP_SIZE):
        head_id = head_start + k
        # padding_mode=ZERO handles the partial last group when
        # n_heads % GROUP_SIZE != 0; OOB stores are silently dropped.
        q1 = ct.load(Q, index=(row_id, head_id, 0, 0),
                     shape=(1, 1, 1, TILE_DIM),
                     padding_mode=ct.PaddingMode.ZERO)
        q2 = ct.load(Q, index=(row_id, head_id, 1, 0),
                     shape=(1, 1, 1, TILE_DIM),
                     padding_mode=ct.PaddingMode.ZERO)

        out1 = q1 * cos_tile - q2 * sin_tile
        out2 = q2 * cos_tile + q1 * sin_tile

        ct.store(Q, index=(row_id, head_id, 0, 0), tile=out1)
        ct.store(Q, index=(row_id, head_id, 1, 0), tile=out2)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(rope_kernel)


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        block_size: int = None, autotune: bool = False):

    # RoPE is in-place; clone so the caller's q stays pristine across backends.
    output = q.clone().contiguous()

    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2

    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)
    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    stream = torch.cuda.current_stream()

    if autotune:
        # rope is in-place; tune on a tmp buffer so trial rotations don't
        # accumulate into the real output.
        tmp = output.clone()
        tmp_view = tmp.view(batch * seq_len, n_heads, 2, half_dim)
        cfg = _tuner.tune_or_cached(
            shape_key=(batch, seq_len, n_heads, head_dim),
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
