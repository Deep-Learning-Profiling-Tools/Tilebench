import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 4, "num_stages": 2}


@triton.jit
def _moe_topk_gating_kernel(
    logits_ptr,
    topk_w_ptr,
    topk_idx_ptr,
    E,
    K,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Row-wise iterative top-K + softmax.
    Grid: (M,) — one program per row.
    Algorithm (unchanged from the original LeetGPU kernel):
      1. Load full row of logits with -inf padding for OOB lanes.
      2. K iterations: find (max, argmax) of current logits tile, write them
         into the top-K buffer at position i via tl.where(offsets_k == i, ...),
         then clear the chosen position in logits with -inf.
      3. Numerically stable softmax on top-K values.
    """
    pid = tl.program_id(0)
    offsets_le = tl.arange(0, BLOCK_SIZE_E)
    mask_le = offsets_le < E

    logits = tl.load(logits_ptr + pid * E + offsets_le, mask=mask_le, other=float("-inf"))

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    mask_k = offsets_k < K

    topk_vals = tl.full((BLOCK_SIZE_K,), value=float("-inf"), dtype=tl.float32)
    topk_idxs = tl.full((BLOCK_SIZE_K,), value=0, dtype=tl.int32)

    for i in range(K):
        curr_max_val = tl.max(logits, axis=-1)
        curr_max_idx = tl.argmax(logits, axis=-1)

        topk_vals = tl.where(offsets_k == i, curr_max_val, topk_vals)
        topk_idxs = tl.where(offsets_k == i, curr_max_idx, topk_idxs)

        logits = tl.where(offsets_le == curr_max_idx, float("-inf"), logits)

    mx = tl.max(topk_vals, axis=-1)
    topk_vals = tl.exp(topk_vals - mx)
    topk_vals = topk_vals / tl.sum(topk_vals, axis=-1)

    tl.store(topk_w_ptr + pid * K + offsets_k, topk_vals, mask=mask_k)
    tl.store(topk_idx_ptr + pid * K + offsets_k, topk_idxs, mask=mask_k)


# Only num_warps / num_stages are tunable; BLOCK_SIZE_E and BLOCK_SIZE_K are fixed by E, K.
_moe_topk_gating_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4, 8]
        for ns in [1, 2, 3]
    ],
    key=["E", "K"],
)(_moe_topk_gating_kernel)


def run(logits: torch.Tensor, M: int, E: int, k: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    topk_weights = torch.empty(M, k, dtype=logits.dtype, device=logits.device)
    topk_indices = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    block_size_e = triton.next_power_of_2(E)
    block_size_k = triton.next_power_of_2(k)
    grid = (M,)

    if autotune:
        _moe_topk_gating_kernel_autotuned[grid](
            logits, topk_weights, topk_indices,
            E, k,
            BLOCK_SIZE_E=block_size_e,
            BLOCK_SIZE_K=block_size_k,
        )
    else:
        cfg = _DEFAULT_CONFIG
        _moe_topk_gating_kernel[grid](
            logits, topk_weights, topk_indices,
            E, k,
            BLOCK_SIZE_E=block_size_e,
            BLOCK_SIZE_K=block_size_k,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    cfg = getattr(_moe_topk_gating_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
