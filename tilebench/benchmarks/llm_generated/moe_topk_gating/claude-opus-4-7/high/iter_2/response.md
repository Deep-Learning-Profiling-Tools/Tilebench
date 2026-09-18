Looking at the feedback: both backends are bandwidth-bound but underutilizing. The previous cuTile used 1 row/CTA (20480 CTAs!) which has huge launch overhead. Triton was at 16% with BLOCK_M=32.

My plan:
- **Triton**: Increase BLOCK_M to 128 with num_warps=8 to amortize launch overhead and process more rows per CTA.
- **cuTile**: Process many rows per CTA (BLOCK_M=16) instead of 1, reducing CTAs by 16x.

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
    BLOCK_M = 128
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
def _argmax_combine(v1, i1, v2, i2):
    win1 = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (ct.where(win1, v1, v2), ct.where(win1, i1, i2))


@ct.kernel(occupancy=4)
def _moe_topk_kernel(logits, weights, indices,
                     E: ConstInt, K: ConstInt,
                     BLOCK_M: ConstInt, BLOCK_E: ConstInt):
    pid = ct.bid(0)
    x2d = ct.load(logits, index=(pid, 0), shape=(BLOCK_M, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF)
    x = ct.astype(x2d, np.float32)

    offs_1d = ct.arange(BLOCK_E, dtype=np.int32)
    offs_row = ct.reshape(offs_1d, (1, BLOCK_E))
    offs_2d = ct.broadcast_to(offs_row, (BLOCK_M, BLOCK_E))

    k_arange = ct.arange(K, dtype=np.int32)
    k_row = ct.reshape(k_arange, (1, K))

    top_vals = ct.full((BLOCK_M, K), -float('inf'), dtype=np.float32)
    top_idxs = ct.full((BLOCK_M, K), 0, dtype=np.int32)

    for i in range(K):
        v, idx = ct.reduce(
            (x, offs_2d), axis=1,
            func=_argmax_combine,
            identity=(-float('inf'), BLOCK_E + 1),
        )
        # v, idx: (BLOCK_M,) -> (BLOCK_M, 1)
        v_col = ct.reshape(v, (BLOCK_M, 1))
        idx_col = ct.reshape(idx, (BLOCK_M, 1))

        pos = K - 1 - i
        sel_k = (k_row == pos)  # (1, K)
        sel_k_full = ct.broadcast_to(sel_k, (BLOCK_M, K))
        v_full = ct.broadcast_to(v_col, (BLOCK_M, K))
        idx_full_k = ct.broadcast_to(idx_col, (BLOCK_M, K))
        top_vals = ct.where(sel_k_full, v_full, top_vals)
        top_idxs = ct.where(sel_k_full, idx_full_k, top_idxs)

        # Mask out chosen index in x
        idx_full_e = ct.broadcast_to(idx_col, (BLOCK_M, BLOCK_E))
        chosen = (offs_2d == idx_full_e)
        x = ct.where(chosen, -float('inf'), x)

    m_val = ct.max(top_vals, axis=1, keepdims=True)
    ex = ct.exp(top_vals - m_val)
    sm = ct.sum(ex, axis=1, keepdims=True)
    w = ex / sm

    w_out = ct.astype(w, logits.dtype)
    ct.store(weights, index=(pid, 0), tile=w_out)
    ct.store(indices, index=(pid, 0), tile=top_idxs)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = 1
    while BLOCK_E < E:
        BLOCK_E *= 2

    BLOCK_M = 16
    # Ensure grid covers M (OOB rows handled via padding/silent-drop)
    num_blocks = (M + BLOCK_M - 1) // BLOCK_M

    stream = torch.cuda.current_stream()
    grid = (num_blocks, 1, 1)
    ct.launch(stream, grid, _moe_topk_kernel,
              (logits, weights, indices, E, k, BLOCK_M, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": 4,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
