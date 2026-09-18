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
    T,
    ROW_NOPE: tl.constexpr,
    ROW_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    EVEN_T: tl.constexpr,
    EVEN_NOPE: tl.constexpr,
    EVEN_ROPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)

    if EVEN_T:
        dests = tl.load(dest_loc + rows)
    else:
        row_mask = rows < T
        dests = tl.load(dest_loc + rows, mask=row_mask, other=0)

    if EVEN_NOPE:
        vals_nope = tl.load(
            kv_nope + rows[:, None] * ROW_NOPE + cols[None, :],
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_nope + dests[:, None] * ROW_NOPE + cols[None, :],
            vals_nope,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
    else:
        row_mask = rows < T
        mask_nope = row_mask[:, None] & (cols[None, :] < ROW_NOPE)
        vals_nope = tl.load(
            kv_nope + rows[:, None] * ROW_NOPE + cols[None, :],
            mask=mask_nope,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_nope + dests[:, None] * ROW_NOPE + cols[None, :],
            vals_nope,
            mask=mask_nope,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

    if pid_c == 0:
        rcols = tl.arange(0, BLOCK_ROPE)
        if EVEN_ROPE:
            vals_rope = tl.load(
                kv_rope + rows[:, None] * ROW_ROPE + rcols[None, :],
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            tl.store(
                out_rope + dests[:, None] * ROW_ROPE + rcols[None, :],
                vals_rope,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
        else:
            row_mask = rows < T
            mask_rope = row_mask[:, None] & (rcols[None, :] < ROW_ROPE)
            vals_rope = tl.load(
                kv_rope + rows[:, None] * ROW_ROPE + rcols[None, :],
                mask=mask_rope,
                other=0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            tl.store(
                out_rope + dests[:, None] * ROW_ROPE + rcols[None, :],
                vals_rope,
                mask=mask_rope,
                cache_modifier=".cg",
                eviction_policy="evict_first",
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
    row_nope = kv_nope.numel() // T
    row_rope = kv_rope.numel() // T

    BLOCK_T = 8
    BLOCK_COL = 512
    BLOCK_ROPE = _next_power_of_2(row_rope)

    EVEN_T = (T % BLOCK_T) == 0
    EVEN_NOPE = EVEN_T and ((row_nope % BLOCK_COL) == 0)
    EVEN_ROPE = EVEN_T and (row_rope == BLOCK_ROPE)

    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(row_nope, BLOCK_COL))
    _destindex_kernel[grid](
        kv_nope,
        kv_rope,
        dest_loc,
        out_nope,
        out_rope,
        T,
        ROW_NOPE=row_nope,
        ROW_ROPE=row_rope,
        BLOCK_T=BLOCK_T,
        BLOCK_COL=BLOCK_COL,
        BLOCK_ROPE=BLOCK_ROPE,
        EVEN_T=EVEN_T,
        EVEN_NOPE=EVEN_NOPE,
        EVEN_ROPE=EVEN_ROPE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_T": BLOCK_T,
            "BLOCK_COL": BLOCK_COL,
            "BLOCK_ROPE": BLOCK_ROPE,
            "EVEN_T": EVEN_T,
            "EVEN_NOPE": EVEN_NOPE,
            "EVEN_ROPE": EVEN_ROPE,
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
    out_nope,
    out_rope,
    TILE_T: ConstInt,
    TILE_COL: ConstInt,
    TILE_ROPE: ConstInt,
    EVEN_T: ConstBool,
    EVEN_NOPE: ConstBool,
    EVEN_ROPE: ConstBool,
):
    tb = ct.bid(0)
    c = ct.bid(1)

    if EVEN_T:
        dest = ct.load(dest_loc, index=(tb,), shape=(TILE_T,))
    else:
        dest = ct.load(
            dest_loc,
            index=(tb,),
            shape=(TILE_T,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        row_ids = tb * TILE_T + ct.arange(TILE_T, dtype=np.int32)
        dest = ct.where(row_ids < kv_nope.shape[0], dest, -1)

    cols = c * TILE_COL + ct.arange(TILE_COL, dtype=np.int32)

    if EVEN_NOPE:
        nope_tile = ct.load(kv_nope, index=(tb, c), shape=(TILE_T, TILE_COL))
        ct.scatter(
            out_nope,
            (dest[:, None], cols[None, :]),
            nope_tile,
            check_bounds=False,
        )
    else:
        nope_tile = ct.load(
            kv_nope,
            index=(tb, c),
            shape=(TILE_T, TILE_COL),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ct.scatter(out_nope, (dest[:, None], cols[None, :]), nope_tile)

    if c == 0:
        rcols = ct.arange(TILE_ROPE, dtype=np.int32)
        if EVEN_ROPE:
            rope_tile = ct.load(kv_rope, index=(tb, 0), shape=(TILE_T, TILE_ROPE))
            ct.scatter(
                out_rope,
                (dest[:, None], rcols[None, :]),
                rope_tile,
                check_bounds=False,
            )
        else:
            rope_tile = ct.load(
                kv_rope,
                index=(tb, 0),
                shape=(TILE_T, TILE_ROPE),
                padding_mode=ct.PaddingMode.ZERO,
            )
            ct.scatter(out_rope, (dest[:, None], rcols[None, :]), rope_tile)


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
    out_nope_2d = out_nope.reshape(T, row_nope)
    out_rope_2d = out_rope.reshape(T, row_rope)

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
            out_nope_2d,
            out_rope_2d,
            TILE_T,
            TILE_COL,
            TILE_ROPE,
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
            "EVEN_T": EVEN_T,
            "EVEN_NOPE": EVEN_NOPE,
            "EVEN_ROPE": EVEN_ROPE,
            "occupancy": occupancy,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
