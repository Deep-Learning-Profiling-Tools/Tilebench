Looking at iter 4 (best so far at 58.6%): fp32 is already at 82.7%, but the smaller dtypes (fp16/bf16 at ~56%, int8 at 39%) lag significantly. This indicates the kernel needs more in-flight memory transactions per SM. Since this is a pure scatter (write to random permuted destinations), higher block-level parallelism / occupancy should help saturate the memory pipeline for the small-element-size cases.

I'll keep the iter-4 tile shape (which was the best so far) but **double the occupancy from 8 to 16** to increase memory-level parallelism. I'll also ensure the kv_rope path uses a wide enough tile (64 covers the full HD_rope row).

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=16)
def _scatter_kernel(kv, dest_loc, out,
                    BLOCK_T: ConstInt, TILE: ConstInt):
    bt = ct.bid(0)
    j  = ct.bid(1)

    # Coalesced contiguous read: BLOCK_T rows x TILE cols.
    src = ct.load(kv, index=(bt, j), shape=(BLOCK_T, TILE),
                  padding_mode=ct.PaddingMode.ZERO,
                  latency=10)

    # Load BLOCK_T destination indices.
    dest = ct.load(dest_loc, index=(bt,), shape=(BLOCK_T,))

    col_offs_1d = ct.arange(TILE, dtype=np.int32) + j * TILE
    row_idx = ct.broadcast_to(dest[:, None],        (BLOCK_T, TILE))
    col_idx = ct.broadcast_to(col_offs_1d[None, :], (BLOCK_T, TILE))

    ct.scatter(out, (row_idx, col_idx), src,
               check_bounds=False,
               latency=10)


def _pick_block_t(default, T):
    if T % default == 0:
        return default
    for cand in [default // 2, default // 4, default // 8, 8, 4, 2, 1]:
        if cand > 0 and T % cand == 0:
            return cand
    return 1


def run(kv_nope: torch.Tensor,
        kv_rope: torch.Tensor,
        dest_loc: torch.Tensor,
        o_nope: torch.Tensor,
        o_rope: torch.Tensor):
    T = kv_nope.shape[0]
    HD_nope = kv_nope.shape[1] * kv_nope.shape[2]   # 1536
    HD_rope = kv_rope.shape[1] * kv_rope.shape[2]   #   64

    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    kv_nope_2d  = kv_nope.view(T, HD_nope)
    kv_rope_2d  = kv_rope.view(T, HD_rope)
    out_nope_2d = out_nope.view(T, HD_nope)
    out_rope_2d = out_rope.view(T, HD_rope)

    stream = torch.cuda.current_stream()

    BLOCK_T_NOPE = 4
    TILE_NOPE    = 512    # divides 1536 evenly (3 column tiles)
    BLOCK_T_ROPE = 16
    TILE_ROPE    = 64     # full HD_rope row in one tile

    BLOCK_T_NOPE = _pick_block_t(BLOCK_T_NOPE, T)
    BLOCK_T_ROPE = _pick_block_t(BLOCK_T_ROPE, T)

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
        "occupancy": 16,
        "check_bounds": False,
        "latency": 10,
    })
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: keep iter-4's winning tile shape (BLOCK_T_NOPE=4, TILE=512; BLOCK_T_ROPE=16, TILE=64) and bump occupancy 8→16. Pure scatter is bandwidth-bound; the smaller-dtype cases (fp16/bf16/int8) need more concurrent in-flight memory transactions per SM, which higher occupancy provides without changing tile geometry.
