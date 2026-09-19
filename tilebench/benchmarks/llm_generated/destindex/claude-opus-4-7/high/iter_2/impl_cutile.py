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

    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    # Collapse (T, H, D) -> (T, H*D); both are contiguous.
    kv_nope_2d  = kv_nope.view(T, HD_nope)
    kv_rope_2d  = kv_rope.view(T, HD_rope)
    out_nope_2d = out_nope.view(T, HD_nope)
    out_rope_2d = out_rope.view(T, HD_rope)

    stream = torch.cuda.current_stream()

    # Larger BLOCK_T to amortize dest_loc load and improve DRAM efficiency.
    BLOCK_T_NOPE = 16
    TILE_NOPE    = 512    # 1536 / 512 = 3 tiles per row
    BLOCK_T_ROPE = 64
    TILE_ROPE    = 64

    # Handle tail when T is not divisible by BLOCK_T by rounding up;
    # OOB rows produce OOB scatter dests (since dest_loc load is ZERO-padded,
    # but ZERO is a valid dest!). To avoid corruption, fall back to a smaller
    # BLOCK_T that divides T if necessary.
    def pick_block_t(default, T):
        if T % default == 0:
            return default
        for cand in [default // 2, default // 4, default // 8, 8, 4, 2, 1]:
            if cand > 0 and T % cand == 0:
                return cand
        return 1

    BLOCK_T_NOPE = pick_block_t(BLOCK_T_NOPE, T)
    BLOCK_T_ROPE = pick_block_t(BLOCK_T_ROPE, T)

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
