```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _top2_combine(a_v1, a_i1, a_v2, a_i2, b_v1, b_i1, b_v2, b_i2):
    take_a1 = (a_v1 > b_v1) | ((a_v1 == b_v1) & (a_i1 <= b_i1))

    out_v1 = ct.where(take_a1, a_v1, b_v1)
    out_i1 = ct.where(take_a1, a_i1, b_i1)

    cand_a_v = ct.where(take_a1, a_v2, a_v1)
    cand_a_i = ct.where(take_a1, a_i2, a_i1)
    cand_b_v = ct.where(take_a1, b_v1, b_v2)
    cand_b_i = ct.where(take_a1, b_i1, b_i2)

    take_a2 = (cand_a_v > cand_b_v) | ((cand_a_v == cand_b_v) & (cand_a_i <= cand_b_i))
    out_v2 = ct.where(take_a2, cand_a_v, cand_b_v)
    out_i2 = ct.where(take_a2, cand_a_i, cand_b_i)

    return out_v1, out_i1, out_v2, out_i2


@ct.kernel(occupancy=4)
def _moe_topk2_pair_kernel(logits, weights, indices, ROWS: ConstInt, HALF_E: ConstInt):
    bid = ct.bid(0)

    vals0 = ct.load(
        logits,
        index=(bid, 0),
        shape=(ROWS, HALF_E),
        allow_tma=False,
        latency=1,
    )
    vals1 = ct.load(
        logits,
        index=(bid, 1),
        shape=(ROWS, HALF_E),
        allow_tma=False,
        latency=1,
    )

    offs0 = ct.arange(HALF_E, dtype=np.int32)[None, :]
    offs1 = offs0 + HALF_E

    # In each pair (h, h+64), h is the first occurrence on ties.
    take0 = vals0 >= vals1

    pair_v1 = ct.where(take0, vals0, vals1)
    pair_i1 = ct.where(take0, offs0, offs1)
    pair_v2 = ct.where(take0, vals1, vals0)
    pair_i2 = ct.where(take0, offs1, offs0)

    val1, idx1, val0, idx0 = ct.reduce(
        (pair_v1, pair_i1, pair_v2, pair_i2),
        axis=1,
        func=_top2_combine,
        identity=(-np.inf, 2147483647, -np.inf, 2147483647),
        keepdims=True,
    )

    v0 = ct.astype(val0, np.float32)
    v1 = ct.astype(val1, np.float32)
    e = ct.exp(v0 - v1)
    inv_den = 1.0 / (e + 1.0)
    w0 = e * inv_den
    w1 = inv_den

    cols = ct.arange(2, dtype=np.int32)[None, :]
    w_out = ct.where(cols == 0, w0, w1)
    i_out = ct.where(cols == 0, idx0, idx1)

    ct.store(weights, index=(bid, 0), tile=ct.astype(w_out, logits.dtype), allow_tma=False)
    ct.store(indices, index=(bid, 0), tile=ct.astype(i_out, np.int32), allow_tma=False)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    ROWS = 16
    HALF_E = 64
    TILE_E = 128
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, ROWS), 1, 1)
    ct.launch(stream, grid, _moe_topk2_pair_kernel, (logits, topk_weights, topk_indices, ROWS, HALF_E))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ROWS": ROWS,
            "TILE_E": TILE_E,
            "HALF_E": HALF_E,
            "occupancy": occupancy,
            "allow_tma": False,
            "padding_mode": "none_static_divisible",
            "pairwise_preselect": True,
            "pair_tie_simplified": True,
            "implicit_index_broadcast": True,
            "one_pass_top2": True,
            "combined_store": True,
            "softmax_reciprocal": True,
            "static_E128_K2_fastpath": True,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
