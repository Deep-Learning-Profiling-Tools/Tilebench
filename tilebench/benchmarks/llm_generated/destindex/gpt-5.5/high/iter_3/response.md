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
    num_warps = 4
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
```
