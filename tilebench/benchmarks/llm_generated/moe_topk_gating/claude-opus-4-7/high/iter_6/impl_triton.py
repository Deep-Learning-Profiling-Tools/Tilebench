import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _top2_combine(va1, ia1, va2, ia2, vb1, ib1, vb2, ib2):
    a_wins = (va1 > vb1) | ((va1 == vb1) & (ia1 < ib1))
    v_top1 = tl.where(a_wins, va1, vb1)
    i_top1 = tl.where(a_wins, ia1, ib1)
    v_a_run = tl.where(a_wins, va2, va1)
    i_a_run = tl.where(a_wins, ia2, ia1)
    v_b_run = tl.where(a_wins, vb1, vb2)
    i_b_run = tl.where(a_wins, ib1, ib2)
    a2_wins = (v_a_run > v_b_run) | ((v_a_run == v_b_run) & (i_a_run < i_b_run))
    v_top2 = tl.where(a2_wins, v_a_run, v_b_run)
    i_top2 = tl.where(a2_wins, i_a_run, i_b_run)
    return v_top1, i_top1, v_top2, i_top2


@triton.jit
def _moe_topk2_kernel(logits_ptr, weights_ptr, indices_ptr,
                      M, E, stride_lm, stride_le,
                      BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_E)
    row_mask = row_offs < M
    col_mask = col_offs < E

    ptrs = logits_ptr + row_offs[:, None] * stride_lm + col_offs[None, :] * stride_le
    load_mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(ptrs, mask=load_mask, other=-float('inf')).to(tl.float32)

    col_i32 = col_offs.to(tl.int32)
    col_bcast = tl.broadcast_to(col_i32[None, :], (BLOCK_M, BLOCK_E))
    neg_inf_t = tl.full((BLOCK_M, BLOCK_E), -float('inf'), tl.float32)
    sentinel = tl.full((BLOCK_M, BLOCK_E), BLOCK_E, tl.int32)

    v1, i1, v2, i2 = tl.reduce(
        (x, col_bcast, neg_inf_t, sentinel),
        axis=1, combine_fn=_top2_combine,
    )

    m = tl.maximum(v1, v2)
    e1 = tl.exp(v1 - m)
    e2 = tl.exp(v2 - m)
    s = e1 + e2
    w1 = e1 / s  # weight for v1 (largest of top-2)
    w2 = e2 / s  # weight for v2 (smaller of top-2)

    out_dtype = weights_ptr.dtype.element_ty
    base_w = weights_ptr + row_offs * 2
    base_i = indices_ptr + row_offs * 2
    # layout: position 0 = smaller (v2); position 1 = larger (v1)
    tl.store(base_w + 0, w2.to(out_dtype), mask=row_mask)
    tl.store(base_w + 1, w1.to(out_dtype), mask=row_mask)
    tl.store(base_i + 0, i2, mask=row_mask)
    tl.store(base_i + 1, i1, mask=row_mask)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_M = 32
    num_warps = 4
    num_stages = 2

    assert k == 2, "this kernel specialized for k=2 (matches config.yaml)"

    grid = (triton.cdiv(M, BLOCK_M),)
    _moe_topk2_kernel[grid](
        logits, weights, indices,
        M, E, logits.stride(0), logits.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_E=BLOCK_E,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "algo": "single_pass_top2",
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
