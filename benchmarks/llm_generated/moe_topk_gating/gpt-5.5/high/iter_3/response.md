```python title="impl_triton.py"
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
    den = e + 1.0
    w0 = e / den
    w1 = 1.0 / den

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
    num_warps = 8
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
            "one_pass_top2": True,
            "split_reduce": True,
            "persistent": False,
            "combined_store": True,
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
def _moe_topk2_kernel(logits, weights, indices, ROWS: ConstInt, TILE_E: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        logits,
        index=(bid, 0),
        shape=(ROWS, TILE_E),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    offs = ct.arange(TILE_E, dtype=np.int32)[None, :]
    idxs = ct.broadcast_to(offs, (ROWS, TILE_E))
    neg_vals = ct.full((ROWS, TILE_E), -np.inf, dtype=logits.dtype)
    big_idxs = ct.full((ROWS, TILE_E), 2147483647, dtype=np.int32)

    val1, idx1, val0, idx0 = ct.reduce(
        (vals, idxs, neg_vals, big_idxs),
        axis=1,
        func=_top2_combine,
        identity=(-np.inf, 2147483647, -np.inf, 2147483647),
        keepdims=True,
    )

    v0 = ct.astype(val0, np.float32)
    v1 = ct.astype(val1, np.float32)
    e = ct.exp(v0 - v1)
    den = e + 1.0
    w0 = e / den
    w1 = 1.0 / den

    cols = ct.arange(2, dtype=np.int32)[None, :]
    w_out = ct.where(cols == 0, w0, w1)
    i_out = ct.where(cols == 0, idx0, idx1)

    ct.store(weights, index=(bid, 0), tile=ct.astype(w_out, logits.dtype), allow_tma=False)
    ct.store(indices, index=(bid, 0), tile=ct.astype(i_out, np.int32), allow_tma=False)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    ROWS = 8
    TILE_E = 128
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, ROWS), 1, 1)
    ct.launch(stream, grid, _moe_topk2_kernel, (logits, topk_weights, topk_indices, ROWS, TILE_E))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ROWS": ROWS,
            "TILE_E": TILE_E,
            "occupancy": occupancy,
            "allow_tma": False,
            "one_pass_top2": True,
            "combined_store": True,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
