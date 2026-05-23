Looking at the trajectory, iter 6's scatter approach (BLOCK_T=2, TILE=512) plateaued at 66.2%. The fundamental bottleneck is the **scatter write**: contiguous reads but random writes that cannot use TMA. The int8 case is particularly bad (41.6%) because its launch overhead dominates the smaller per-byte work.

**New approach: invert the permutation then gather + contiguous store.** Since `dest_loc` is a permutation, `out[i] = kv[inv_dest_loc[i]]` is equivalent to the original scatter. This converts the slow scatter write to a contiguous TMA-friendly store, with the random access moved to the read side (gather), which has better spatial locality at row granularity.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _invert_kernel(dest_loc, inv_dest, INV_TILE: ConstInt):
    # Build inverse permutation: inv_dest[dest_loc[i]] = i.
    # Small kernel: dest_loc has only T int32 = a few KB, fits easily.
    bid = ct.bid(0)
    t = ct.load(dest_loc, index=(bid,), shape=(INV_TILE,))
    offs = ct.arange(INV_TILE, dtype=np.int32) + bid * INV_TILE
    ct.scatter(inv_dest, (t,), offs, check_bounds=False)


@ct.kernel(occupancy=16)
def _gather_kernel(kv, inv_dest_loc, out,
                   BLOCK_T: ConstInt, TILE: ConstInt):
    # For destination tile (bt, j) of size (BLOCK_T, TILE):
    #   - read which source rows produce these destination rows
    #   - gather their (TILE,) columns
    #   - store contiguously to (bt, j) — uses TMA (static index!)
    bt = ct.bid(0)
    j  = ct.bid(1)

    src_indices = ct.load(inv_dest_loc, index=(bt,), shape=(BLOCK_T,))

    col_offs = ct.arange(TILE, dtype=np.int32) + j * TILE
    row_idx = ct.broadcast_to(src_indices[:, None], (BLOCK_T, TILE))
    col_idx = ct.broadcast_to(col_offs[None, :],    (BLOCK_T, TILE))

    src = ct.gather(kv, (row_idx, col_idx),
                    check_bounds=False, latency=10)

    # Contiguous (TMA-friendly) store — no scatter on the hot path.
    ct.store(out, index=(bt, j), tile=src, latency=10)


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

    # Since dest_loc is a permutation, every output row is overwritten,
    # so we don't need to clone o_nope / o_rope.
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    kv_nope_2d  = kv_nope.view(T, HD_nope)
    kv_rope_2d  = kv_rope.view(T, HD_rope)
    out_nope_2d = out_nope.view(T, HD_nope)
    out_rope_2d = out_rope.view(T, HD_rope)

    # Workspace for inverse permutation (T int32 = tiny, ~160 KB at T=40K).
    inv_dest = torch.empty_like(dest_loc)

    stream = torch.cuda.current_stream()

    # ---- 1) build inverse permutation ----
    INV_TILE = 1024
    for cand in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if T % cand == 0:
            INV_TILE = cand
            break
    ct.launch(stream, (T // INV_TILE, 1, 1), _invert_kernel,
              (dest_loc, inv_dest, INV_TILE))

    # ---- 2) gather + contiguous store ----
    # Match the empirically winning block geometry (iter 6: BLOCK_T_NOPE=2,
    # TILE_NOPE=512 / BLOCK_T_ROPE=8, TILE_ROPE=64) but with the random
    # access now on the READ side instead of the WRITE side.
    BLOCK_T_NOPE = 2
    TILE_NOPE    = 512    # divides 1536 evenly (3 column tiles)
    BLOCK_T_ROPE = 8
    TILE_ROPE    = 64     # full HD_rope row in one tile

    BLOCK_T_NOPE = _pick_block_t(BLOCK_T_NOPE, T)
    BLOCK_T_ROPE = _pick_block_t(BLOCK_T_ROPE, T)

    grid_nope = (T // BLOCK_T_NOPE, ct.cdiv(HD_nope, TILE_NOPE), 1)
    ct.launch(stream, grid_nope, _gather_kernel,
              (kv_nope_2d, inv_dest, out_nope_2d, BLOCK_T_NOPE, TILE_NOPE))

    grid_rope = (T // BLOCK_T_ROPE, ct.cdiv(HD_rope, TILE_ROPE), 1)
    ct.launch(stream, grid_rope, _gather_kernel,
              (kv_rope_2d, inv_dest, out_rope_2d, BLOCK_T_ROPE, TILE_ROPE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "approach": "invert+gather",
        "INV_TILE": INV_TILE,
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

The key insight: scatter writes can't use TMA (random destination), but a contiguous `ct.store` with static `(bt, j)` index can. By inverting the permutation in a tiny preprocessing kernel (O(T) int32), we replace the scatter on the hot path with a gather, keeping the store contiguous and TMA-eligible.
