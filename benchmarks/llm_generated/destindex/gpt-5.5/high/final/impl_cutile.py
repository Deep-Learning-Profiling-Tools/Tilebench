import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _destindex_flat2048_kernel(
    kv_nope_flat,
    kv_rope_2d,
    dest_loc,
    out_nope_flat,
    out_rope_flat,
    ROW_NOPE: ConstInt,
    ROW_ROPE: ConstInt,
    BLOCK_NOPE: ConstInt,
    SUB_COL: ConstInt,
    PHASES: ConstInt,
    ROWS_PER_GROUP: ConstInt,
    ROPE_BLOCKS: ConstInt,
    TILE_ROPE_T: ConstInt,
    TILE_ROPE: ConstInt,
):
    bid = ct.bid(0)

    # NOPE fast path:
    # ROW_NOPE = 1536 and BLOCK_NOPE = 2048, so every 3 CTAs cover 4 rows.
    # This avoids per-element div/mod while still using contiguous source loads.
    offs = ct.arange(BLOCK_NOPE, dtype=np.int32)
    vals = ct.load(
        kv_nope_flat,
        index=(bid,),
        shape=(BLOCK_NOPE,),
        latency=1,
        allow_tma=False,
    )

    group = bid // PHASES
    phase = bid - group * PHASES

    row0 = group * ROWS_PER_GROUP + phase
    start_col = phase * SUB_COL
    cut = ROW_NOPE - start_col

    dest0 = ct.load(dest_loc, index=(row0,), shape=(), latency=1)
    dest1 = ct.load(dest_loc, index=(row0 + 1,), shape=(), latency=1)

    first_row = offs < cut
    idx0 = dest0 * ROW_NOPE + start_col + offs
    idx1 = dest1 * ROW_NOPE + (offs - cut)
    out_idx = ct.where(first_row, idx0, idx1)

    ct.scatter(out_nope_flat, out_idx, vals, check_bounds=False, latency=1)

    # ROPE: one 8x64 tile per CTA for the first ROPE_BLOCKS CTAs.
    if bid < ROPE_BLOCKS:
        rcols = ct.arange(TILE_ROPE, dtype=np.int32)
        dest = ct.load(dest_loc, index=(bid,), shape=(TILE_ROPE_T,), latency=1)
        rope_tile = ct.load(
            kv_rope_2d,
            index=(bid, 0),
            shape=(TILE_ROPE_T, TILE_ROPE),
            latency=1,
            allow_tma=False,
        )
        rope_idx = dest[:, None] * ROW_ROPE + rcols[None, :]
        ct.scatter(out_rope_flat, rope_idx, rope_tile, check_bounds=False, latency=1)


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    T = kv_nope.shape[0]
    row_nope = kv_nope.numel() // T
    row_rope = kv_rope.numel() // T

    kv_nope_flat = kv_nope.reshape(-1)
    kv_rope_2d = kv_rope.reshape(T, row_rope)
    out_nope_flat = out_nope.reshape(-1)
    out_rope_flat = out_rope.reshape(-1)

    BLOCK_NOPE = 2048
    SUB_COL = 512
    PHASES = 3
    ROWS_PER_GROUP = 4
    TILE_ROPE_T = 8
    TILE_ROPE = 64
    occupancy = 4

    nope_blocks = ct.cdiv(T * row_nope, BLOCK_NOPE)
    rope_blocks = ct.cdiv(T, TILE_ROPE_T)

    stream = torch.cuda.current_stream()
    grid = (nope_blocks, 1, 1)

    ct.launch(
        stream,
        grid,
        _destindex_flat2048_kernel,
        (
            kv_nope_flat,
            kv_rope_2d,
            dest_loc,
            out_nope_flat,
            out_rope_flat,
            row_nope,
            row_rope,
            BLOCK_NOPE,
            SUB_COL,
            PHASES,
            ROWS_PER_GROUP,
            rope_blocks,
            TILE_ROPE_T,
            TILE_ROPE,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_NOPE": BLOCK_NOPE,
            "SUB_COL": SUB_COL,
            "PHASES": PHASES,
            "ROWS_PER_GROUP": ROWS_PER_GROUP,
            "TILE_ROPE_T": TILE_ROPE_T,
            "TILE_ROPE": TILE_ROPE,
            "ROW_NOPE": row_nope,
            "ROW_ROPE": row_rope,
            "NOPE_BLOCKS": nope_blocks,
            "ROPE_BLOCKS": rope_blocks,
            "occupancy": occupancy,
            "flat2048_phase_scatter": True,
            "allow_tma": False,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
