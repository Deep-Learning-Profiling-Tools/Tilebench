```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _destindex_rowstore_kernel(
    kv_nope,
    kv_rope,
    dest_loc,
    out_nope,
    out_rope,
    TILE_T: ConstInt,
    TILE_H_NOPE: ConstInt,
    TILE_H_ROPE: ConstInt,
    TILE_D_NOPE: ConstInt,
    TILE_D_ROPE: ConstInt,
    NUM_NOPE_H_BLOCKS: ConstInt,
    NUM_ROPE_H_BLOCKS: ConstInt,
    EVEN_T: ConstBool,
    EVEN_NOPE: ConstBool,
    EVEN_ROPE: ConstBool,
):
    tb = ct.bid(0)

    for rr in range(0, TILE_T):
        t = tb * TILE_T + rr

        if EVEN_T:
            dest = ct.load(dest_loc, index=(t,), shape=(), latency=1)

            if EVEN_NOPE:
                for hb in range(0, NUM_NOPE_H_BLOCKS):
                    nope_tile = ct.load(
                        kv_nope,
                        index=(t, hb, 0),
                        shape=(1, TILE_H_NOPE, TILE_D_NOPE),
                        latency=1,
                        allow_tma=False,
                    )
                    ct.store(
                        out_nope,
                        index=(dest, hb, 0),
                        tile=nope_tile,
                        latency=1,
                        allow_tma=False,
                    )
            else:
                for hb in range(0, NUM_NOPE_H_BLOCKS):
                    nope_tile = ct.load(
                        kv_nope,
                        index=(t, hb, 0),
                        shape=(1, TILE_H_NOPE, TILE_D_NOPE),
                        padding_mode=ct.PaddingMode.ZERO,
                        latency=1,
                        allow_tma=False,
                    )
                    ct.store(
                        out_nope,
                        index=(dest, hb, 0),
                        tile=nope_tile,
                        latency=1,
                        allow_tma=False,
                    )

            if EVEN_ROPE:
                for hb in range(0, NUM_ROPE_H_BLOCKS):
                    rope_tile = ct.load(
                        kv_rope,
                        index=(t, hb, 0),
                        shape=(1, TILE_H_ROPE, TILE_D_ROPE),
                        latency=1,
                        allow_tma=False,
                    )
                    ct.store(
                        out_rope,
                        index=(dest, hb, 0),
                        tile=rope_tile,
                        latency=1,
                        allow_tma=False,
                    )
            else:
                for hb in range(0, NUM_ROPE_H_BLOCKS):
                    rope_tile = ct.load(
                        kv_rope,
                        index=(t, hb, 0),
                        shape=(1, TILE_H_ROPE, TILE_D_ROPE),
                        padding_mode=ct.PaddingMode.ZERO,
                        latency=1,
                        allow_tma=False,
                    )
                    ct.store(
                        out_rope,
                        index=(dest, hb, 0),
                        tile=rope_tile,
                        latency=1,
                        allow_tma=False,
                    )
        else:
            if t < kv_nope.shape[0]:
                dest = ct.load(dest_loc, index=(t,), shape=(), latency=1)

                if EVEN_NOPE:
                    for hb in range(0, NUM_NOPE_H_BLOCKS):
                        nope_tile = ct.load(
                            kv_nope,
                            index=(t, hb, 0),
                            shape=(1, TILE_H_NOPE, TILE_D_NOPE),
                            latency=1,
                            allow_tma=False,
                        )
                        ct.store(
                            out_nope,
                            index=(dest, hb, 0),
                            tile=nope_tile,
                            latency=1,
                            allow_tma=False,
                        )
                else:
                    for hb in range(0, NUM_NOPE_H_BLOCKS):
                        nope_tile = ct.load(
                            kv_nope,
                            index=(t, hb, 0),
                            shape=(1, TILE_H_NOPE, TILE_D_NOPE),
                            padding_mode=ct.PaddingMode.ZERO,
                            latency=1,
                            allow_tma=False,
                        )
                        ct.store(
                            out_nope,
                            index=(dest, hb, 0),
                            tile=nope_tile,
                            latency=1,
                            allow_tma=False,
                        )

                if EVEN_ROPE:
                    for hb in range(0, NUM_ROPE_H_BLOCKS):
                        rope_tile = ct.load(
                            kv_rope,
                            index=(t, hb, 0),
                            shape=(1, TILE_H_ROPE, TILE_D_ROPE),
                            latency=1,
                            allow_tma=False,
                        )
                        ct.store(
                            out_rope,
                            index=(dest, hb, 0),
                            tile=rope_tile,
                            latency=1,
                            allow_tma=False,
                        )
                else:
                    for hb in range(0, NUM_ROPE_H_BLOCKS):
                        rope_tile = ct.load(
                            kv_rope,
                            index=(t, hb, 0),
                            shape=(1, TILE_H_ROPE, TILE_D_ROPE),
                            padding_mode=ct.PaddingMode.ZERO,
                            latency=1,
                            allow_tma=False,
                        )
                        ct.store(
                            out_rope,
                            index=(dest, hb, 0),
                            tile=rope_tile,
                            latency=1,
                            allow_tma=False,
                        )


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
    h_nope = kv_nope.shape[1]
    d_nope = kv_nope.shape[2]
    h_rope = kv_rope.shape[1]
    d_rope = kv_rope.shape[2]

    TILE_T = 8
    TILE_H_NOPE = 4
    TILE_H_ROPE = 1
    TILE_D_NOPE = _next_power_of_2(d_nope)
    TILE_D_ROPE = _next_power_of_2(d_rope)

    NUM_NOPE_H_BLOCKS = ct.cdiv(h_nope, TILE_H_NOPE)
    NUM_ROPE_H_BLOCKS = ct.cdiv(h_rope, TILE_H_ROPE)

    EVEN_T = (T % TILE_T) == 0
    EVEN_NOPE = (h_nope % TILE_H_NOPE == 0) and (d_nope == TILE_D_NOPE)
    EVEN_ROPE = (h_rope % TILE_H_ROPE == 0) and (d_rope == TILE_D_ROPE)

    occupancy = 8
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(T, TILE_T), 1, 1)
    kernel = _destindex_rowstore_kernel.with_hints(occupancy=occupancy)

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
            TILE_T,
            TILE_H_NOPE,
            TILE_H_ROPE,
            TILE_D_NOPE,
            TILE_D_ROPE,
            NUM_NOPE_H_BLOCKS,
            NUM_ROPE_H_BLOCKS,
            EVEN_T,
            EVEN_NOPE,
            EVEN_ROPE,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_T": TILE_T,
            "TILE_H_NOPE": TILE_H_NOPE,
            "TILE_H_ROPE": TILE_H_ROPE,
            "TILE_D_NOPE": TILE_D_NOPE,
            "TILE_D_ROPE": TILE_D_ROPE,
            "NUM_NOPE_H_BLOCKS": NUM_NOPE_H_BLOCKS,
            "NUM_ROPE_H_BLOCKS": NUM_ROPE_H_BLOCKS,
            "EVEN_T": EVEN_T,
            "EVEN_NOPE": EVEN_NOPE,
            "EVEN_ROPE": EVEN_ROPE,
            "occupancy": occupancy,
            "row_dynamic_store": True,
            "allow_tma": False,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
