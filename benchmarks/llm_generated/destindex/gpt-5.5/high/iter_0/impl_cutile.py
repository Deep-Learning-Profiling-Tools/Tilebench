import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _destindex_kernel(
    kv_nope,
    kv_rope,
    dest_loc,
    out_nope,
    out_rope,
    TILE_H_NOPE: ConstInt,
    TILE_D_NOPE: ConstInt,
    TILE_H_ROPE: ConstInt,
    TILE_D_ROPE: ConstInt,
):
    t = ct.bid(0)
    h_block = ct.bid(1)
    dest = ct.load(dest_loc, index=(t,), shape=())

    h_base = h_block * TILE_H_NOPE
    nope_tile = ct.load(
        kv_nope,
        index=(t, h_block, 0),
        shape=(1, TILE_H_NOPE, TILE_D_NOPE),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((TILE_H_NOPE, TILE_D_NOPE))

    h_nope = h_base + ct.arange(TILE_H_NOPE, dtype=np.int32)[:, None]
    d_nope = ct.arange(TILE_D_NOPE, dtype=np.int32)[None, :]
    ct.scatter(out_nope, (dest, h_nope, d_nope), nope_tile)

    if h_block == 0:
        rope_tile = ct.load(
            kv_rope,
            index=(t, 0, 0),
            shape=(1, TILE_H_ROPE, TILE_D_ROPE),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_H_ROPE, TILE_D_ROPE))

        h_rope = ct.arange(TILE_H_ROPE, dtype=np.int32)[:, None]
        d_rope = ct.arange(TILE_D_ROPE, dtype=np.int32)[None, :]
        ct.scatter(out_rope, (dest, h_rope, d_rope), rope_tile)


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    TILE_H_NOPE = 8
    TILE_D_NOPE = max(_next_power_of_2(kv_nope.shape[2]), 512 // TILE_H_NOPE)
    TILE_H_ROPE = _next_power_of_2(kv_rope.shape[1])
    TILE_D_ROPE = max(_next_power_of_2(kv_rope.shape[2]), max(1, 512 // TILE_H_ROPE))
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (kv_nope.shape[0], ct.cdiv(kv_nope.shape[1], TILE_H_NOPE), 1)
    kernel = _destindex_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (
            kv_nope,
            kv_rope,
            dest_loc,
            out_nope,
            out_rope,
            TILE_H_NOPE,
            TILE_D_NOPE,
            TILE_H_ROPE,
            TILE_D_ROPE,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_H_NOPE": TILE_H_NOPE,
            "TILE_D_NOPE": TILE_D_NOPE,
            "TILE_H_ROPE": TILE_H_ROPE,
            "TILE_D_ROPE": TILE_D_ROPE,
            "occupancy": occupancy,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
