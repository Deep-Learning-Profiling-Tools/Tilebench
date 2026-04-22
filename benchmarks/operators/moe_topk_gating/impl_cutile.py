from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(occupancy=2)
_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]
_last_autotune_config = None


@ct.kernel
def _moe_topk_gating_kernel(
    logits_ptr,
    topk_w_ptr,
    topk_idx_ptr,
    E,
    K: ConstInt,
    BLOCK_SIZE_E: ConstInt,
    BLOCK_SIZE_K: ConstInt,
):
    """
    Row-wise iterative top-K + softmax — mirrors Triton's method exactly:
      1. Load full row of logits as a (1, BLOCK_SIZE_E) tile with padding_mode=NEG_INF
         so OOB lanes do not affect max/argmax.
      2. K iterations: reduce to (max, argmax), write to the top-K buffer at
         position i via ct.where(offsets_k == i, ...), then clear the chosen
         position in logits with -inf.
      3. Numerically stable softmax on top-K values.

    K must be ConstInt because cuTile's Python for-loop is compile-time unrolled;
    this is a DSL constraint on the loop construct, not a change to the algorithm.
    """
    row = ct.bid(0)

    offsets_le = ct.arange(BLOCK_SIZE_E, dtype=np.int32)[None, :]   # (1, BLOCK_SIZE_E)
    offsets_k = ct.arange(BLOCK_SIZE_K, dtype=np.int32)[None, :]    # (1, BLOCK_SIZE_K)

    # Load full row; OOB lanes become -inf so they can't win max/argmax.
    logits = ct.load(
        logits_ptr,
        index=(row, 0),
        shape=(1, BLOCK_SIZE_E),
        padding_mode=ct.PaddingMode.NEG_INF,
    )
    logits = ct.astype(logits, np.float32)

    topk_vals = ct.full((1, BLOCK_SIZE_K), -float("inf"), dtype=np.float32)
    topk_idxs = ct.full((1, BLOCK_SIZE_K), 0, dtype=np.int32)

    for i in range(K):  # compile-time unrolled
        curr_max_val = ct.max(logits, axis=-1, keepdims=True)    # (1, 1)
        curr_max_idx = ct.argmax(logits, axis=-1, keepdims=True) # (1, 1) int32

        # Place curr_max at position i in the top-K buffer.
        topk_vals = ct.where(offsets_k == (K - 1 - i), curr_max_val, topk_vals)
        topk_idxs = ct.where(offsets_k == (K - 1 - i), curr_max_idx, topk_idxs)

        # Mask out the chosen position in logits so it won't win next iteration.
        logits = ct.where(offsets_le == curr_max_idx, -float("inf"), logits)

    # Numerically stable softmax on top-K values.
    mx = ct.max(topk_vals, axis=-1, keepdims=True)   # (1, 1)
    topk_vals = ct.exp(topk_vals - mx)
    s = ct.sum(topk_vals, axis=-1, keepdims=True)    # (1, 1)
    topk_vals = topk_vals / s

    topk_vals_out = ct.astype(topk_vals, topk_w_ptr.dtype)

    # ct.store silently drops OOB col positions (BLOCK_SIZE_K > K case).
    ct.store(topk_w_ptr, index=(row, 0), tile=topk_vals_out)
    ct.store(topk_idx_ptr, index=(row, 0), tile=topk_idxs)


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


def run(logits: torch.Tensor, M: int, E: int, k: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """cuTile MoE Top-K gating — direct mirror of the Triton method."""
    global _last_autotune_config

    topk_weights = torch.empty(M, k, dtype=logits.dtype, device=logits.device)
    topk_indices = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    block_size_e = _next_pow2(E)
    block_size_k = _next_pow2(k)
    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=_moe_topk_gating_kernel,
            args_fn=lambda cfg: (
                logits, topk_weights, topk_indices,
                E, k, block_size_e, block_size_k,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {"occupancy": result.tuned_config.occupancy}
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(
            stream, grid, _moe_topk_gating_kernel,
            (logits, topk_weights, topk_indices,
             E, k, block_size_e, block_size_k),
        )

    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return _last_autotune_config
