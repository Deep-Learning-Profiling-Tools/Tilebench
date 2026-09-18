import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _moe_topk_kernel(logits_ptr, weights_ptr, indices_ptr,
                     M, E, stride_lm, stride_le,
                     K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_E)
    row_mask = row_offs < M
    col_mask = col_offs < E

    ptrs = logits_ptr + row_offs[:, None] * stride_lm + col_offs[None, :] * stride_le
    load_mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(ptrs, mask=load_mask, other=-float('inf')).to(tl.float32)

    top_vals = tl.full((BLOCK_M, K), -float('inf'), tl.float32)
    top_idxs = tl.zeros((BLOCK_M, K), dtype=tl.int32)
    k_pos = tl.arange(0, K)

    for i in tl.static_range(K):
        v, idx = tl.max(x, axis=1, return_indices=True,
                        return_indices_tie_break_left=True)
        idx_i32 = idx.to(tl.int32)
        pos = K - 1 - i
        sel = k_pos[None, :] == pos
        top_vals = tl.where(sel, v[:, None], top_vals)
        top_idxs = tl.where(sel, idx_i32[:, None], top_idxs)
        x = tl.where(col_offs[None, :] == idx_i32[:, None], -float('inf'), x)

    m_val = tl.max(top_vals, axis=1, keep_dims=True)
    ex = tl.exp(top_vals - m_val)
    sm = tl.sum(ex, axis=1, keep_dims=True)
    w = ex / sm

    out_col_offs = tl.arange(0, K)
    out_ptrs_w = weights_ptr + row_offs[:, None] * K + out_col_offs[None, :]
    out_ptrs_i = indices_ptr + row_offs[:, None] * K + out_col_offs[None, :]
    out_mask = row_mask[:, None]
    tl.store(out_ptrs_w, w.to(weights_ptr.dtype.element_ty), mask=out_mask)
    tl.store(out_ptrs_i, top_idxs, mask=out_mask)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_M = 64
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M),)
    _moe_topk_kernel[grid](
        logits, weights, indices,
        M, E, logits.stride(0), logits.stride(1),
        K=k, BLOCK_M=BLOCK_M, BLOCK_E=BLOCK_E,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
