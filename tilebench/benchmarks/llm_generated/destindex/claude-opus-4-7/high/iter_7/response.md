Looking at the trajectory, decreasing `BLOCK_T_NOPE` consistently improved performance (4→2 jumped 58%→66%). The bottleneck signal is that fp32 hits 85% while int8 only hits 42% — at the same bandwidth, int8 means smaller per-CTA byte volume, suggesting per-CTA bookkeeping overhead. Counterintuitively though, smaller BLOCK_T helps. Following the trend, I'll push BLOCK_T to 1 (each CTA handles one source row × one column tile). This eliminates broadcasting of dest, simplifies scatter (each CTA writes one contiguous row range), and maximizes CTA parallelism.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=16)
def _scatter_kernel_bt1(kv, dest_loc, out, TILE: ConstInt):
    """One CTA = one source row × one column tile.

    With BLOCK_T=1 the dest index is a scalar broadcast, the scatter
    writes a single contiguous strip per CTA → fully coalesced.
    """
    t = ct.bid(0)
    j = ct.bid(1)

    # Read TILE contiguous elements of source row t.
    src = ct.load(kv, index=(t, j), shape=(1, TILE),
                  padding_mode=ct.PaddingMode.ZERO,
                  latency=10)

    # Scalar destination row index for source row t.
    dest = ct.load(dest_loc, index=(t,), shape=(1,))

    col_offs = ct.arange(TILE, dtype=np.int32) + j * TILE
    row_idx = ct.broadcast_to(dest[:, None], (1, TILE))
    col_idx = col_offs[None, :]

    ct.scatter(out, (row_idx, col_idx), src,
               check_bounds=False,
               latency=10)


@ct.kernel(occupancy=16)
def _scatter_kernel_btN(kv, dest_loc, out,
                        BLOCK_T: ConstInt, TILE: ConstInt):
    """Fallback when T not divisible: multi-row per CTA."""
    bt = ct.bid(0)
    j  = ct.bid(1)

    src = ct.load(kv, index=(bt, j), shape=(BLOCK_T, TILE),
                  padding_mode=ct.PaddingMode.ZERO,
                  latency=10)
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

    # Iter 7: continue trend (BLOCK_T 8→4→2 each improved). Push to 1:
    # each CTA = single source row × single column tile. Eliminates
    # per-CTA broadcast of dest, and the scatter becomes one
    # coalesced contiguous strip per CTA. Maximises CTA-level
    # parallelism for the int8 case (currently worst at 42%).
    BLOCK_T_NOPE = 1
    TILE_NOPE    = 512    # divides 1536 evenly (3 column tiles)
    BLOCK_T_ROPE = 1
    TILE_ROPE    = 64     # full HD_rope row in one tile

    # NOPE
    if BLOCK_T_NOPE == 1:
        grid_nope = (T, ct.cdiv(HD_nope, TILE_NOPE), 1)
        ct.launch(stream, grid_nope, _scatter_kernel_bt1,
                  (kv_nope_2d, dest_loc, out_nope_2d, TILE_NOPE))
    else:
        BLOCK_T_NOPE = _pick_block_t(BLOCK_T_NOPE, T)
        grid_nope = (T // BLOCK_T_NOPE, ct.cdiv(HD_nope, TILE_NOPE), 1)
        ct.launch(stream, grid_nope, _scatter_kernel_btN,
                  (kv_nope_2d, dest_loc, out_nope_2d, BLOCK_T_NOPE, TILE_NOPE))

    # ROPE
    if BLOCK_T_ROPE == 1:
        grid_rope = (T, ct.cdiv(HD_rope, TILE_ROPE), 1)
        ct.launch(stream, grid_rope, _scatter_kernel_bt1,
                  (kv_rope_2d, dest_loc, out_rope_2d, TILE_ROPE))
    else:
        BLOCK_T_ROPE = _pick_block_t(BLOCK_T_ROPE, T)
        grid_rope = (T // BLOCK_T_ROPE, ct.cdiv(HD_rope, TILE_ROPE), 1)
        ct.launch(stream, grid_rope, _scatter_kernel_btN,
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

Pushing `BLOCK_T` to 1 (following the monotone improvement 8→4→2) gives each CTA one source row × one column tile — the scatter is now a single coalesced contiguous strip per CTA with no row-broadcast overhead, which should especially help the int8 case where per-CTA byte volume is small.
