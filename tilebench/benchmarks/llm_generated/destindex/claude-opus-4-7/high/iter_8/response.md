Looking at the trajectory: iter 6 (BLOCK_T_NOPE=2, BLOCK_T_ROPE=8) was the best at 66.2%. The remaining gap is striking: fp32 hits 69.5% but int8 only 22.6% — strongly suggesting we're losing to **redundant work**, not tile sizing.

**Key insight**: The PyTorch reference clones `o_nope`/`o_rope` then does `index_copy_`. But `dest_loc` is guaranteed to be a **permutation of {0..T-1}** — every output row gets overwritten. The clone is **completely wasted work** (extra 2n bytes per tensor). The roofline of `n * dtype_size * 2` assumes 1 read + 1 write per element; with the clone we were doing 2 reads + 2 writes (= 4n bytes), capping us near ~50% effective regardless of tile tuning.

Using `torch.empty_like` instead of `.clone()` should roughly **double** the effective bandwidth and push us toward 100% on the int8 case (currently most penalized because it has the highest relative clone overhead vs already-coalesced scatter writes). Keeping iter 6's winning tile config since the kernel itself was healthy.

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

    # KEY OPTIMIZATION (iter 8):
    # The reference does `o_nope.clone()` + `index_copy_`. But the spec
    # guarantees `dest_loc` is a permutation of {0..T-1}, so EVERY output
    # row is overwritten. The clone is pure wasted bandwidth (an extra
    # 2n bytes for nope and 2n bytes for rope).
    # Roofline is `n * dtype_size * 2` (1 read + 1 write); with the clone
    # we were doing 2r+2w = 4n bytes, capping effective bandwidth at ~50%.
    # Switch to `torch.empty_like` -> ~2x speedup, especially impactful
    # for int8 where the kernel itself is most launch/latency limited.
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    kv_nope_2d  = kv_nope.view(T, HD_nope)
    kv_rope_2d  = kv_rope.view(T, HD_rope)
    out_nope_2d = out_nope.view(T, HD_nope)
    out_rope_2d = out_rope.view(T, HD_rope)

    stream = torch.cuda.current_stream()

    # Iter 6's winning tile config (66.2% with redundant clone -> aiming
    # for ~100% without it).
    BLOCK_T_NOPE = 2
    TILE_NOPE    = 512    # divides 1536 evenly (3 column tiles)
    BLOCK_T_ROPE = 8
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
        "skip_clone": True,
    })
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

The single high-leverage change: drop the wasted `o_nope.clone()`/`o_rope.clone()` since `dest_loc` is a permutation — every output position is overwritten by the scatter. This halves total memory traffic and should roughly double effective bandwidth on every dtype while keeping iter 6's proven `(BLOCK_T=2, TILE=512)` / `(BLOCK_T=8, TILE=64)` kernel config.
