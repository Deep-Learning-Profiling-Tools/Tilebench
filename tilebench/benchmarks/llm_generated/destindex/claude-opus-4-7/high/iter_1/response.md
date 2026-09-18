Looking at iter 0:
- Triton got 58% mean (worst is int8 at 28.8%) — tile too small for int8 throughput.
- cuTile crashed on `.with_hints` (not in the API at our version).

**Strategy:** (1) Increase BLOCK_T per program in Triton to amortize per-row destination lookup and pump up bytes/CTA (critical for int8). (2) For cuTile, switch to decorator-baked `@ct.kernel(occupancy=...)` and drop the `.with_hints` call.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _scatter_kernel(kv_ptr, dest_loc_ptr, out_ptr,
                    HD: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_HD: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    hd_offs = pid_b * BLOCK_HD + tl.arange(0, BLOCK_HD)

    dest = tl.load(dest_loc_ptr + t_offs).to(tl.int64)

    src_ptrs = kv_ptr + t_offs[:, None].to(tl.int64) * HD + hd_offs[None, :]
    src = tl.load(src_ptrs)

    dst_ptrs = out_ptr + dest[:, None] * HD + hd_offs[None, :]
    tl.store(dst_ptrs, src)


def run(kv_nope: torch.Tensor,
        kv_rope: torch.Tensor,
        dest_loc: torch.Tensor,
        o_nope: torch.Tensor,
        o_rope: torch.Tensor):
    T = kv_nope.shape[0]
    HD_nope = kv_nope.shape[1] * kv_nope.shape[2]   # 1536
    HD_rope = kv_rope.shape[1] * kv_rope.shape[2]   #   64

    # dest_loc is a permutation → every output row is written.
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    # Larger BLOCK_T → more bytes/CTA, critical for int8 (1B element).
    BLOCK_T_NOPE = 8
    BLOCK_HD_NOPE = 512   # 1536 / 512 = 3 column blocks
    NW_NOPE = 4

    BLOCK_T_ROPE = 32
    BLOCK_HD_ROPE = 64    # 64 / 64 = 1 column block
    NW_ROPE = 4

    NS = 2

    grid_nope = (T // BLOCK_T_NOPE, triton.cdiv(HD_nope, BLOCK_HD_NOPE))
    _scatter_kernel[grid_nope](
        kv_nope, dest_loc, out_nope,
        HD=HD_nope, BLOCK_T=BLOCK_T_NOPE, BLOCK_HD=BLOCK_HD_NOPE,
        num_warps=NW_NOPE, num_stages=NS,
    )

    grid_rope = (T // BLOCK_T_ROPE, triton.cdiv(HD_rope, BLOCK_HD_ROPE))
    _scatter_kernel[grid_rope](
        kv_rope, dest_loc, out_rope,
        HD=HD_rope, BLOCK_T=BLOCK_T_ROPE, BLOCK_HD=BLOCK_HD_ROPE,
        num_warps=NW_ROPE, num_stages=NS,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_T_NOPE":  BLOCK_T_NOPE,  "BLOCK_HD_NOPE": BLOCK_HD_NOPE, "NW_NOPE": NW_NOPE,
        "BLOCK_T_ROPE":  BLOCK_T_ROPE,  "BLOCK_HD_ROPE": BLOCK_HD_ROPE, "NW_ROPE": NW_ROPE,
        "num_stages": NS,
    })
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


@ct.kernel(occupancy=4)
def _scatter_kernel(kv, dest_loc, out,
                    BLOCK_T: ConstInt, TILE: ConstInt):
    bt = ct.bid(0)
    j  = ct.bid(1)

    src = ct.load(kv, index=(bt, j), shape=(BLOCK_T, TILE),
                  padding_mode=ct.PaddingMode.ZERO)
    dest = ct.load(dest_loc, index=(bt,), shape=(BLOCK_T,))

    col_offs_1d = ct.arange(TILE, dtype=np.int32) + j * TILE
    row_idx = ct.broadcast_to(dest[:, None],        (BLOCK_T, TILE))
    col_idx = ct.broadcast_to(col_offs_1d[None, :], (BLOCK_T, TILE))

    ct.scatter(out, (row_idx, col_idx), src)


def run(kv_nope: torch.Tensor,
        kv_rope: torch.Tensor,
        dest_loc: torch.Tensor,
        o_nope: torch.Tensor,
        o_rope: torch.Tensor):
    T = kv_nope.shape[0]
    HD_nope = kv_nope.shape[1] * kv_nope.shape[2]   # 1536
    HD_rope = kv_rope.shape[1] * kv_rope.shape[2]   #   64

    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    # Collapse (T, H, D) -> (T, H*D); both are contiguous.
    kv_nope_2d  = kv_nope.view(T, HD_nope)
    kv_rope_2d  = kv_rope.view(T, HD_rope)
    out_nope_2d = out_nope.view(T, HD_nope)
    out_rope_2d = out_rope.view(T, HD_rope)

    stream = torch.cuda.current_stream()

    BLOCK_T_NOPE = 8
    TILE_NOPE    = 512   # 1536 / 512 = 3
    BLOCK_T_ROPE = 32
    TILE_ROPE    = 64

    grid_nope = (T // BLOCK_T_NOPE, ct.cdiv(HD_nope, TILE_NOPE), 1)
    ct.launch(stream, grid_nope, _scatter_kernel,
              (kv_nope_2d, dest_loc, out_nope_2d, BLOCK_T_NOPE, TILE_NOPE))

    grid_rope = (T // BLOCK_T_ROPE, ct.cdiv(HD_rope, TILE_ROPE), 1)
    ct.launch(stream, grid_rope, _scatter_kernel,
              (kv_rope_2d, dest_loc, out_rope_2d, BLOCK_T_ROPE, TILE_ROPE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_T_NOPE": BLOCK_T_NOPE, "TILE_NOPE": TILE_NOPE,
        "BLOCK_T_ROPE": BLOCK_T_ROPE, "TILE_ROPE": TILE_ROPE,
        "occupancy": 4,
    })
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Increased `BLOCK_T` (1→8 for nope, 8→32 for rope) to grow per-CTA bytes and amortize the `dest_loc` gather, which should especially lift the int8 case. Fixed cuTile by hoisting `occupancy` into the `@ct.kernel(...)` decorator and removing the unsupported `.with_hints` call.
