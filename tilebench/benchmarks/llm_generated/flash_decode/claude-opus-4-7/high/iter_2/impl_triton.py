import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _split_kernel(
    mid_o_ptr, mid_o_lse_ptr, b_seqlen_ptr,
    partial_o_ptr, partial_lse_ptr,
    stride_mb, stride_mh, stride_mk, stride_md,
    stride_lb, stride_lh, stride_lk,
    stride_pob, stride_poh, stride_pos, stride_pod,
    stride_plb, stride_plh, stride_pls,
    block_seq,
    HEAD_DIM: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    seqlen = tl.load(b_seqlen_ptr + b)
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    k_start = s * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < valid_blocks

    offs_d = tl.arange(0, HEAD_DIM)

    lse_ptrs = mid_o_lse_ptr + b * stride_lb + h * stride_lh + offs_k * stride_lk
    lse = tl.load(lse_ptrs, mask=mask_k, other=-float('inf')).to(tl.float32)

    m = tl.max(lse, axis=0)
    # Safe max: avoid -inf - -inf = NaN when split is fully invalid
    safe_m = tl.where(m > -float('inf'), m, 0.0)
    w = tl.exp(lse - safe_m)
    w = tl.where(mask_k, w, 0.0)
    l = tl.sum(w, axis=0)

    val_ptrs = (mid_o_ptr + b * stride_mb + h * stride_mh
                + offs_k[:, None] * stride_mk + offs_d[None, :] * stride_md)
    vals = tl.load(val_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)
    acc = tl.sum(vals * w[:, None], axis=0)

    is_valid = l > 0.0
    inv_l = tl.where(is_valid, 1.0 / l, 0.0)
    partial_out = acc * inv_l
    local_lse = tl.where(is_valid, tl.log(l) + m, -float('inf'))

    out_ptrs = (partial_o_ptr + b * stride_pob + h * stride_poh
                + s * stride_pos + offs_d * stride_pod)
    tl.store(out_ptrs, partial_out)

    lse_out_ptr = partial_lse_ptr + b * stride_plb + h * stride_plh + s * stride_pls
    tl.store(lse_out_ptr, local_lse)


@triton.jit
def _merge_kernel(
    partial_o_ptr, partial_lse_ptr, output_ptr,
    stride_pob, stride_poh, stride_pos, stride_pod,
    stride_plb, stride_plh, stride_pls,
    stride_ob, stride_oh, stride_od,
    HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, HEAD_DIM)

    lse_ptrs = partial_lse_ptr + b * stride_plb + h * stride_plh + offs_s * stride_pls
    lses = tl.load(lse_ptrs).to(tl.float32)

    global_m = tl.max(lses, axis=0)
    safe_m = tl.where(global_m > -float('inf'), global_m, 0.0)
    w = tl.exp(lses - safe_m)
    l = tl.sum(w, axis=0)

    acc_ptrs = (partial_o_ptr + b * stride_pob + h * stride_poh
                + offs_s[:, None] * stride_pos + offs_d[None, :] * stride_pod)
    accs = tl.load(acc_ptrs).to(tl.float32)

    numer = tl.sum(accs * w[:, None], axis=0)
    out = numer / l

    out_ptrs = output_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out.to(output_ptr.dtype.element_ty))


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_K = 32
    HEAD_DIM_C = _next_pow2(head_dim)
    K_SPLITS = triton.cdiv(num_blocks, BLOCK_K)
    BLOCK_S = _next_pow2(K_SPLITS) if K_SPLITS > 1 else 1
    # ensure tl.arange usable (pow2)
    if BLOCK_S < 2:
        BLOCK_S = 2

    partial_o = torch.zeros((batch, heads, BLOCK_S, head_dim),
                            device=mid_o.device, dtype=torch.float32)
    partial_lse = torch.full((batch, heads, BLOCK_S), -float('inf'),
                             device=mid_o.device, dtype=torch.float32)

    grid1 = (batch, heads, K_SPLITS)
    _split_kernel[grid1](
        mid_o, mid_o_lse, b_seqlen,
        partial_o, partial_lse,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_lse.stride(0), mid_o_lse.stride(1), mid_o_lse.stride(2),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        block_seq,
        HEAD_DIM=HEAD_DIM_C, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    grid2 = (batch, heads)
    _merge_kernel[grid2](
        partial_o, partial_lse, output,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        HEAD_DIM=HEAD_DIM_C, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_S": BLOCK_S, "K_SPLITS": K_SPLITS,
        "HEAD_DIM": HEAD_DIM_C, "num_warps": 4, "num_stages": 2,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
