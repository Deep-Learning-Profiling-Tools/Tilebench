import torch
import triton
import triton.language as tl


@triton.jit
def _copy_by_dest_kernel(
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
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_dim
    dest_index = tl.load(dest_ptr + token_id).to(tl.int32)

    kv_ptrs = kv_ptr + token_id * stride_kv_bs + head_id * stride_kv_h + offs_d * stride_kv_d
    out_ptrs = out_ptr + dest_index * stride_o_bs + head_id * stride_o_h + offs_d * stride_o_d

    v = tl.load(kv_ptrs, mask=mask_d, other=0.0)
    tl.store(out_ptrs, v, mask=mask_d)


def _launch_copy(kv: torch.Tensor, dest_loc: torch.Tensor, out: torch.Tensor):
    seq_len, head_num, head_dim = kv.shape
    block_dmodel = triton.next_power_of_2(head_dim)
    grid = (seq_len, head_num)
    _copy_by_dest_kernel[grid](
        kv,
        dest_loc,
        out,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        head_dim,
        BLOCK_DMODEL=block_dmodel,
        num_warps=2,
        num_stages=1,
    )


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    block_size: int = None,
):
    del block_size
    out_nope = o_nope.clone()
    out_rope = o_rope.clone()
    _launch_copy(kv_nope, dest_loc, out_nope)
    _launch_copy(kv_rope, dest_loc, out_rope)
    return out_nope, out_rope
