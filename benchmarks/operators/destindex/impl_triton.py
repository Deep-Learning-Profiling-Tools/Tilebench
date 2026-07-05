import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_DMODEL": 64, "num_warps": 4, "num_stages": 2}


@triton.jit
def copy_by_dest_kernel(
    kv_ptr,
    dest_ptr,
    out_ptr,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dest_index = tl.load(dest_ptr + token_id).to(tl.int32)

    base_kv  = kv_ptr  + token_id * stride_kv_bs + head_id * stride_kv_h
    base_out = out_ptr + dest_index * stride_o_bs + head_id * stride_o_h

    offs_d = tl.arange(0, BLOCK_DMODEL)
    for d_start in tl.range(0, head_dim, BLOCK_DMODEL):
        cur_offs = d_start + offs_d
        mask_d   = cur_offs < head_dim
        v = tl.load(base_kv  + cur_offs * stride_kv_d, mask=mask_d, other=0.0)
        tl.store(base_out + cur_offs * stride_o_d, v, mask=mask_d)


_copy_by_dest_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_DMODEL": bd}, num_warps=nw, num_stages=ns)
        for bd in [32, 64, 128]
        for nw in [1, 2, 4]
        for ns in [1, 2]
    ],
    key=["head_dim"],
)(copy_by_dest_kernel)


def _launch_copy(kv: torch.Tensor, dest_loc: torch.Tensor, out: torch.Tensor, autotune: bool):
    seq_len, head_num, head_dim = kv.shape
    grid = (seq_len, head_num)
    if autotune:
        _copy_by_dest_kernel_autotuned[grid](
            kv, dest_loc, out,
            kv.stride(0), kv.stride(1), kv.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            head_dim,
        )
    else:
        cfg = _DEFAULT_CONFIG
        copy_by_dest_kernel[grid](
            kv, dest_loc, out,
            kv.stride(0), kv.stride(1), kv.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            head_dim,
            BLOCK_DMODEL=cfg["BLOCK_DMODEL"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    autotune: bool = False,
):
    out_nope = o_nope.clone()
    out_rope = o_rope.clone()
    _launch_copy(kv_nope, dest_loc, out_nope, autotune)
    _launch_copy(kv_rope, dest_loc, out_rope, autotune)
    return out_nope, out_rope


def get_last_config() -> dict | None:
    cfg = _copy_by_dest_kernel_autotuned.best_config
    if cfg is None:
        return None
    return {
        "BLOCK_DMODEL": cfg.kwargs["BLOCK_DMODEL"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
