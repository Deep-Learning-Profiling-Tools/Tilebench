from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(occupancy=16)
_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [8, 16, 32]]
_last_autotune_config: dict = {}


@ct.kernel
def moe_topk_gating_kernel(
    logits_ptr,
    topk_w_ptr,
    topk_idx_ptr,
    E,
    K: ConstInt,
    BLOCK_SIZE_E: ConstInt,
    BLOCK_SIZE_K: ConstInt,
):
    row = ct.bid(0)

    offsets_le = ct.arange(BLOCK_SIZE_E, dtype=ct.int32)[None, :]
    offsets_k = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)[None, :]


    logits = ct.load(
        logits_ptr,
        index=(row, 0),
        shape=(1, BLOCK_SIZE_E),
        padding_mode=ct.PaddingMode.NEG_INF,
    )
    logits = ct.astype(logits, ct.float32)

    topk_vals = ct.full((1, BLOCK_SIZE_K), -float("inf"), dtype=ct.float32)
    topk_idxs = ct.full((1, BLOCK_SIZE_K), 0, dtype=ct.int32)

    for i in range(K):
        curr_max_val = ct.max(logits, axis=-1, keepdims=True)
        curr_max_idx = ct.argmax(logits, axis=-1, keepdims=True)


        topk_vals = ct.where(offsets_k == (K - 1 - i), curr_max_val, topk_vals)
        topk_idxs = ct.where(offsets_k == (K - 1 - i), curr_max_idx, topk_idxs)


        logits = ct.where(offsets_le == curr_max_idx, -float("inf"), logits)


    mx = ct.max(topk_vals, axis=-1, keepdims=True)
    topk_vals = ct.exp(topk_vals - mx)
    s = ct.sum(topk_vals, axis=-1, keepdims=True)
    topk_vals = topk_vals / s

    topk_vals_out = ct.astype(topk_vals, topk_w_ptr.dtype)


    ct.store(topk_w_ptr, index=(row, 0), tile=topk_vals_out)
    ct.store(topk_idx_ptr, index=(row, 0), tile=topk_idxs)


_tuner = CutileAutotuner(moe_topk_gating_kernel)


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


def run(logits: torch.Tensor, M: int, E: int, k: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    topk_weights = torch.empty(M, k, dtype=logits.dtype, device=logits.device)
    topk_indices = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    block_size_e = _next_pow2(E)
    block_size_k = _next_pow2(k)
    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(M, E, k),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (
                logits, topk_weights, topk_indices,
                E, k, block_size_e, block_size_k,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (logits, topk_weights, topk_indices,
         E, k, block_size_e, block_size_k),
    )

    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
