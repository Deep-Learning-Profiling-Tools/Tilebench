Looking at the trajectory:
- Best `triton` is iter 5 at 16.8% — `BLOCK_M=32, num_warps=4, num_stages=2, evict_first`. Iter 7's `cache_modifier=".cg"` broke ptxas, so I revert and try a smaller change: bump `num_warps` to 8 to get more in-flight loads (the kernel is bandwidth-bound at the largest case).
- Best `cutile` is iter 6 at 14.5% with `BLOCK_M=16, single_pass_top2`. Try bumping `BLOCK_M=32` to get more work per CTA while keeping occupancy=4.

```python title="impl_triton.py"
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
    x = tl.load(ptrs, mask=load_mask, other=-float('inf'),
                eviction_policy="evict_first").to(tl.float32)

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
    BLOCK_M = 32
    num_warps = 8
    num_stages = 2

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
        "evict": "first",
    })
    return (weights, indices)


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
def _top2_combine(va1, ia1, va2, ia2, vb1, ib1, vb2, ib2):
    a_wins = (va1 > vb1) | ((va1 == vb1) & (ia1 < ib1))
    v_top1 = ct.where(a_wins, va1, vb1)
    i_top1 = ct.where(a_wins, ia1, ib1)
    v_a_run = ct.where(a_wins, va2, va1)
    i_a_run = ct.where(a_wins, ia2, ia1)
    v_b_run = ct.where(a_wins, vb1, vb2)
    i_b_run = ct.where(a_wins, ib1, ib2)
    a2_wins = (v_a_run > v_b_run) | ((v_a_run == v_b_run) & (i_a_run < i_b_run))
    v_top2 = ct.where(a2_wins, v_a_run, v_b_run)
    i_top2 = ct.where(a2_wins, i_a_run, i_b_run)
    return (v_top1, i_top1, v_top2, i_top2)


@ct.kernel(occupancy=4)
def _moe_topk2_kernel(logits, weights, indices,
                      BLOCK_M: ConstInt, BLOCK_E: ConstInt):
    row_block = ct.bid(0)
    x2d = ct.load(logits, index=(row_block, 0), shape=(BLOCK_M, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF, latency=1)
    x = ct.astype(x2d, np.float32)

    col = ct.arange(BLOCK_E, dtype=np.int32)
    col_bcast = ct.broadcast_to(col[None, :], (BLOCK_M, BLOCK_E))
    neg_inf = ct.full((BLOCK_M, BLOCK_E), -float('inf'), dtype=np.float32)
    sentinel = ct.full((BLOCK_M, BLOCK_E), BLOCK_E, dtype=np.int32)

    v1, i1, v2, i2 = ct.reduce(
        (x, col_bcast, neg_inf, sentinel),
        axis=1, func=_top2_combine,
        identity=(-float('inf'), BLOCK_E, -float('inf'), BLOCK_E),
    )

    m_v = ct.maximum(v1, v2)
    e1 = ct.exp(v1 - m_v)
    e2 = ct.exp(v2 - m_v)
    s = e1 + e2
    w1 = e1 / s
    w2 = e2 / s

    # layout: position 0 = smaller (v2); position 1 = larger (v1)
    w_tile = ct.cat((w2[:, None], w1[:, None]), axis=1)
    i_tile = ct.cat((i2[:, None], i1[:, None]), axis=1)

    ct.store(weights, index=(row_block, 0), tile=ct.astype(w_tile, logits.dtype))
    ct.store(indices, index=(row_block, 0), tile=i_tile)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = 1
    while BLOCK_E < E:
        BLOCK_E *= 2
    BLOCK_M = 32

    assert k == 2, "this kernel specialized for k=2 (matches config.yaml)"

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, BLOCK_M), 1, 1)
    ct.launch(stream, grid, _moe_topk2_kernel,
              (logits, weights, indices, BLOCK_M, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": 4,
        "algo": "single_pass_top2",
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
