import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_decode_kernel(
    mid_o_ptr, mid_o_lse_ptr, b_seqlen_ptr, output_ptr,
    stride_mb, stride_mh, stride_mk, stride_md,
    stride_lb, stride_lh, stride_lk,
    stride_ob, stride_oh, stride_od,
    block_seq, NUM_BLOCKS, HEAD_DIM,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d_block = tl.program_id(2)

    seqlen = tl.load(b_seqlen_ptr + b)
    valid_blocks = (seqlen + block_seq - 1) // block_seq

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    lse_row_off = b * stride_lb + h * stride_lh
    mid_o_row_off = b * stride_mb + h * stride_mh

    # Single-pass online merge: running max + scaled numer/denom
    m_running = -float('inf')
    denom = 0.0
    numer = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for kk in range(0, NUM_BLOCKS, BLOCK_K):
        k_offs = kk + offs_k
        mask_k = k_offs < valid_blocks

        lse = tl.load(mid_o_lse_ptr + lse_row_off + k_offs * stride_lk,
                      mask=mask_k, other=-float('inf')).to(tl.float32)

        ptrs = (mid_o_ptr + mid_o_row_off
                + k_offs[:, None] * stride_mk
                + offs_d[None, :] * stride_md)
        full_mask = mask_k[:, None] & d_mask[None, :]
        vals = tl.load(ptrs, mask=full_mask, other=0.0).to(tl.float32)

        m_block = tl.max(lse, axis=0)
        m_new = tl.maximum(m_running, m_block)
        scale = tl.exp(m_running - m_new)
        w = tl.exp(lse - m_new)
        w = tl.where(mask_k, w, 0.0)

        denom = denom * scale + tl.sum(w, axis=0)
        numer = numer * scale + tl.sum(vals * w[:, None], axis=0)
        m_running = m_new

    result = numer / denom
    out_ptrs = output_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, result.to(output_ptr.dtype.element_ty), mask=d_mask)


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    if isinstance(block_seq, torch.Tensor):
        block_seq = int(block_seq.item())
    else:
        block_seq = int(block_seq)

    batch, heads, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, heads, head_dim),
                         device=mid_o.device, dtype=mid_o.dtype)

    BLOCK_D = 16
    BLOCK_K = 128
    num_warps = 4
    num_stages = 3

    D_SPLITS = triton.cdiv(head_dim, BLOCK_D)
    grid = (batch, heads, D_SPLITS)
    _flash_decode_kernel[grid](
        mid_o, mid_o_lse, b_seqlen, output,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_lse.stride(0), mid_o_lse.stride(1), mid_o_lse.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_seq, num_blocks, head_dim,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K, "BLOCK_D": BLOCK_D, "D_SPLITS": D_SPLITS,
        "num_warps": num_warps, "num_stages": num_stages,
        "single_pass": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
