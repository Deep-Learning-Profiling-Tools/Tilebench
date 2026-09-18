```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _destindex_kernel(
    kv_nope,
    kv_rope,
    dest_loc,
    out_nope,
    out_rope,
    ROW_NOPE: tl.constexpr,
    ROW_ROPE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    t = tl.program_id(0)
    dest = tl.load(dest_loc + t)

    offs_nope = tl.arange(0, BLOCK_NOPE)
    mask_nope = offs_nope < ROW_NOPE
    vals_nope = tl.load(kv_nope + t * ROW_NOPE + offs_nope, mask=mask_nope, other=0)
    tl.store(out_nope + dest * ROW_NOPE + offs_nope, vals_nope, mask=mask_nope)

    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_rope = offs_rope < ROW_ROPE
    vals_rope = tl.load(kv_rope + t * ROW_ROPE + offs_rope, mask=mask_rope, other=0)
    tl.store(out_rope + dest * ROW_ROPE + offs_rope, vals_rope, mask=mask_rope)


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

    BLOCK_NOPE = max(512, _next_power_of_2(row_nope))
    BLOCK_ROPE = max(512, _next_power_of_2(row_rope))
    num_warps = 8
    num_stages = 2

    grid = (T,)
    _destindex_kernel[grid](
        kv_nope,
        kv_rope,
        dest_loc,
        out_nope,
        out_rope,
        ROW_NOPE=row_nope,
        ROW_ROPE=row_rope,
        BLOCK_NOPE=BLOCK_NOPE,
        BLOCK_ROPE=BLOCK_ROPE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_NOPE": BLOCK_NOPE,
            "BLOCK_ROPE": BLOCK_ROPE,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
```
