Looking at this operator, it's a memory-bound scatter copy. Since `dest_loc` is a permutation, every row of the output is written, so I can use `empty_like` without copying `o_*` first.

For both impls: process source rows in blocks (BLOCK_T rows × BLOCK_HD columns per program). The hard dims are `HD_nope=12*128=1536` (divisible by 512) and `HD_rope=1*64=64`. Since `T = 2048*i`, any pow-2 `BLOCK_T ≤ 2048` divides T — no row-side masking needed. I pick `BLOCK_T=1, TILE=512` for nope and `BLOCK_T=8, TILE=64` for rope so each block does ≥ 512 elements.

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

    # T is always a multiple of BLOCK_T (T = 2048*i, BLOCK_T pow2 <= 2048)
    # BLOCK_HD divides HD (1536 % 512 == 0, 64 % 64 == 0) — no masks needed.
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
    HD_nope = kv_nope.shape[1] * kv_nope.shape[2]   # 12*128 = 1536
    HD_rope = kv_rope.shape[1] * kv_rope.shape[2]   #  1* 64 =   64

    # dest_loc is a permutation → every output row is written.
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    BLOCK_T_NOPE = 1
    BLOCK_HD_NOPE = 512   # 1536 / 512 = 3
    NW_NOPE = 4

    BLOCK_T_ROPE = 8
    BLOCK_HD_ROPE = 64    # 64 / 64 = 1
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


@ct.kernel
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

    # OOB column writes are silently dropped by ct.scatter (check_bounds=True).
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

    BLOCK_T_NOPE = 1
    TILE_NOPE    = 512   # 1536 / 512 = 3
    BLOCK_T_ROPE = 8
    TILE_ROPE    = 64    # 8 * 64 = 512 elems per block
    OCC          = 4

    grid_nope = (T // BLOCK_T_NOPE, ct.cdiv(HD_nope, TILE_NOPE), 1)
    ct.launch(stream, grid_nope,
              _scatter_kernel.with_hints(occupancy=OCC),
              (kv_nope_2d, dest_loc, out_nope_2d, BLOCK_T_NOPE, TILE_NOPE))

    grid_rope = (T // BLOCK_T_ROPE, ct.cdiv(HD_rope, TILE_ROPE), 1)
    ct.launch(stream, grid_rope,
              _scatter_kernel.with_hints(occupancy=OCC),
              (kv_rope_2d, dest_loc, out_rope_2d, BLOCK_T_ROPE, TILE_ROPE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_T_NOPE": BLOCK_T_NOPE, "TILE_NOPE": TILE_NOPE,
        "BLOCK_T_ROPE": BLOCK_T_ROPE, "TILE_ROPE": TILE_ROPE,
        "occupancy": OCC,
    })
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
