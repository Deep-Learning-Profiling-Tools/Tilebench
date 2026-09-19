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
