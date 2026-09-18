import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _top2_combine(a_v1, a_i1, a_v2, a_i2, b_v1, b_i1, b_v2, b_i2):
    take_a1 = (a_v1 > b_v1) | ((a_v1 == b_v1) & (a_i1 <= b_i1))

    out_v1 = tl.where(take_a1, a_v1, b_v1)
    out_i1 = tl.where(take_a1, a_i1, b_i1)

    cand_a_v = tl.where(take_a1, a_v2, a_v1)
    cand_a_i = tl.where(take_a1, a_i2, a_i1)
    cand_b_v = tl.where(take_a1, b_v1, b_v2)
    cand_b_i = tl.where(take_a1, b_i1, b_i2)

    take_a2 = (cand_a_v > cand_b_v) | ((cand_a_v == cand_b_v) & (cand_a_i <= cand_b_i))
    out_v2 = tl.where(take_a2, cand_a_v, cand_b_v)
    out_i2 = tl.where(take_a2, cand_a_i, cand_b_i)

    return out_v1, out_i1, out_v2, out_i2


@triton.jit
def _moe_topk2_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rb = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)
    rows = rb * BLOCK_M + offs_m
    row_mask = rows < M

    neg_vals = tl.full((BLOCK_M, BLOCK_H), -float("inf"), tl.float32)
    big_idxs = tl.full((BLOCK_M, BLOCK_H), 2147483647, tl.int32)

    vals0 = tl.load(
        logits_ptr + rows[:, None] * E + offs_h[None, :],
        mask=row_mask[:, None] & (offs_h[None, :] < E),
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    idxs0 = tl.broadcast_to(offs_h[None, :], (BLOCK_M, BLOCK_H))

    h0_v1, h0_i1, h0_v0, h0_i0 = tl.reduce(
        (vals0, idxs0, neg_vals, big_idxs),
        axis=1,
        combine_fn=_top2_combine,
    )

    offs_e1 = offs_h + BLOCK_H
    vals1 = tl.load(
        logits_ptr + rows[:, None] * E + offs_e1[None, :],
        mask=row_mask[:, None] & (offs_e1[None, :] < E),
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    idxs1 = tl.broadcast_to(offs_e1[None, :], (BLOCK_M, BLOCK_H))

    h1_v1, h1_i1, h1_v0, h1_i0 = tl.reduce(
        (vals1, idxs1, neg_vals, big_idxs),
        axis=1,
        combine_fn=_top2_combine,
    )

    val1, idx1, val0, idx0 = _top2_combine(
        h0_v1, h0_i1, h0_v0, h0_i0,
        h1_v1, h1_i1, h1_v0, h1_i0,
    )

    v0 = val0.to(tl.float32)
    v1 = val1.to(tl.float32)
    e = tl.exp(v0 - v1)
    inv_den = 1.0 / (e + 1.0)
    w0 = e * inv_den
    w1 = inv_den

    offs_k = tl.arange(0, 2)
    w_out = tl.where(offs_k[None, :] == 0, w0[:, None], w1[:, None])
    i_out = tl.where(offs_k[None, :] == 0, idx0[:, None], idx1[:, None])
    out_mask = row_mask[:, None] & (offs_k[None, :] < K)

    tl.store(weights_ptr + rows[:, None] * K + offs_k[None, :], w_out, mask=out_mask)
    tl.store(indices_ptr + rows[:, None] * K + offs_k[None, :], i_out, mask=out_mask)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    BLOCK_M = 32
    BLOCK_H = 64
    BLOCK_E = 128
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M),)

    _moe_topk2_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M,
        E=E,
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_E": BLOCK_E,
            "SPLIT_E": BLOCK_H,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "cache_modifier": ".cg",
            "one_pass_top2": True,
            "split_reduce": True,
            "persistent": False,
            "combined_store": True,
            "softmax_reciprocal": True,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
