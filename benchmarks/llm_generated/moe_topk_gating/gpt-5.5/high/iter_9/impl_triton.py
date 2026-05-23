import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _moe_topk2_argmax2_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    rb = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    rows = rb * BLOCK_M + offs_m

    vals = tl.load(
        logits_ptr + rows[:, None] * E + offs_e[None, :],
        eviction_policy="evict_first",
    )

    # Match the reference's iterative max + mask-out algorithm directly.
    # tl.max(..., return_indices=True) carries the first-occurrence index
    # through the reduction, avoiding the value-only top2 rescan reductions.
    val_best, idx_best = tl.max(
        vals,
        axis=1,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    vals_second = tl.where(offs_e[None, :] != idx_best[:, None], vals, -float("inf"))
    val_second, idx_second = tl.max(
        vals_second,
        axis=1,
        return_indices=True,
        return_indices_tie_break_left=True,
    )

    v0 = val_second.to(tl.float32)
    v1 = val_best.to(tl.float32)
    e = tl.exp(v0 - v1)
    inv_den = 1.0 / (e + 1.0)
    w0 = e * inv_den
    w1 = inv_den

    offs_k = tl.arange(0, 2)
    w_out = tl.where(offs_k[None, :] == 0, w0[:, None], w1[:, None])
    i_out = tl.where(
        offs_k[None, :] == 0,
        idx_second[:, None].to(tl.int32),
        idx_best[:, None].to(tl.int32),
    )

    tl.store(weights_ptr + rows[:, None] * 2 + offs_k[None, :], w_out)
    tl.store(indices_ptr + rows[:, None] * 2 + offs_k[None, :], i_out)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    BLOCK_M = 32
    BLOCK_E = 128
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M),)

    _moe_topk2_argmax2_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        E=E,
        BLOCK_M=BLOCK_M,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_E": BLOCK_E,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "builtin_argmax_two_pass": True,
            "iterative_max_mask": True,
            "index_rescan": False,
            "one_pass_top2": False,
            "split_reduce": False,
            "persistent": False,
            "combined_store": True,
            "softmax_reciprocal": True,
            "static_E128_K2_fastpath": True,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
