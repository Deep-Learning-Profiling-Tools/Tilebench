import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@ct.kernel(occupancy=4)
def _destindex_kernel(
    kv_nope,
    kv_rope,
    dest_loc,
    out_nope_flat,
    out_rope_flat,
    TILE_T: ConstInt,
    TILE_COL: ConstInt,
    TILE_ROPE: ConstInt,
    ROW_NOPE: ConstInt,
    ROW_ROPE: ConstInt,
    EVEN_T: ConstBool,
    EVEN_NOPE: ConstBool,
    EVEN_ROPE: ConstBool,
):
    tb = ct.bid(0)
    c = ct.bid(1)

    row_ids = tb * TILE_T + ct.arange(TILE_T, dtype=np.int32)

    if EVEN_T:
        dest = ct.load(dest_loc, index=(tb,), shape=(TILE_T,))
    else:
        dest = ct.load(
            dest_loc,
            index=(tb,),
            shape=(TILE_T,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        dest = ct.where(row_ids < kv_nope.shape[0], dest, -1)

    cols = c * TILE_COL + ct.arange(TILE_COL, dtype=np.int32)

    if EVEN_NOPE:
        nope_tile = ct.load(kv_nope, index=(tb, c), shape=(TILE_T, TILE_COL))
        nope_idx = dest[:, None] * ROW_NOPE + cols[None, :]
        ct.scatter(out_nope_flat, nope_idx, nope_tile, check_bounds=False)
    else:
        nope_tile = ct.load(
            kv_nope,
            index=(tb, c),
            shape=(TILE_T, TILE_COL),
            padding_mode=ct.PaddingMode.ZERO,
        )
        nope_idx = dest[:, None] * ROW_NOPE + cols[None, :]
        valid_nope = (row_ids[:, None] < kv_nope.shape[0]) & (cols[None, :] < ROW_NOPE)
        nope_idx = ct.where(valid_nope, nope_idx, -1)
        ct.scatter(out_nope_flat, nope_idx, nope_tile)

    if c == 0:
        rcols = ct.arange(TILE_ROPE, dtype=np.int32)
        if EVEN_ROPE:
            rope_tile = ct.load(kv_rope, index=(tb, 0), shape=(TILE_T, TILE_ROPE))
            rope_idx = dest[:, None] * ROW_ROPE + rcols[None, :]
            ct.scatter(out_rope_flat, rope_idx, rope_tile, check_bounds=False)
        else:
            rope_tile = ct.load(
                kv_rope,
                index=(tb, 0),
                shape=(TILE_T, TILE_ROPE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            rope_idx = dest[:, None] * ROW_ROPE + rcols[None, :]
            valid_rope = (row_ids[:, None] < kv_rope.shape[0]) & (rcols[None, :] < ROW_ROPE)
            rope_idx = ct.where(valid_rope, rope_idx, -1)
            ct.scatter(out_rope_flat, rope_idx, rope_tile)


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

    kv_nope_2d = kv_nope.reshape(T, row_nope)
    kv_rope_2d = kv_rope.reshape(T, row_rope)
    out_nope_flat = out_nope.reshape(-1)
    out_rope_flat = out_rope.reshape(-1)

    TILE_T = 8
    TILE_COL = 512
    TILE_ROPE = _next_power_of_2(row_rope)
    occupancy = 4

    EVEN_T = (T % TILE_T) == 0
    EVEN_NOPE = EVEN_T and ((row_nope % TILE_COL) == 0)
    EVEN_ROPE = EVEN_T and (row_rope == TILE_ROPE)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(T, TILE_T), ct.cdiv(row_nope, TILE_COL), 1)
    ct.launch(
        stream,
        grid,
        _destindex_kernel,
        (
            kv_nope_2d,
            kv_rope_2d,
            dest_loc,
            out_nope_flat,
            out_rope_flat,
            TILE_T,
            TILE_COL,
            TILE_ROPE,
            row_nope,
            row_rope,
            EVEN_T,
            EVEN_NOPE,
            EVEN_ROPE,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_T": TILE_T,
            "TILE_COL": TILE_COL,
            "TILE_ROPE": TILE_ROPE,
            "ROW_NOPE": row_nope,
            "ROW_ROPE": row_rope,
            "EVEN_T": EVEN_T,
            "EVEN_NOPE": EVEN_NOPE,
            "EVEN_ROPE": EVEN_ROPE,
            "occupancy": occupancy,
            "flat_scatter": True,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
