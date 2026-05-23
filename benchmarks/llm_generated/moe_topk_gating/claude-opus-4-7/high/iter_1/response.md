Looking at iter 0's results:
- **Triton**: 6-12% roofline — bottleneck is launch overhead with M=20480 tiny CTAs (one row each, only 128 elements per row). Need to batch many rows per CTA.
- **cuTile**: Verify failed because `kernel.with_hints()` doesn't exist on the `@ct.kernel` object in this version. Fix by moving `occupancy` into the decorator.

Strategy for iter 1: Triton processes `BLOCK_M=32` rows per CTA (huge launch-overhead reduction); cuTile first becomes verify-clean by fixing the hint API.

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
    BLOCK_M = 32
    num_warps = 4
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
                     E: ConstInt, K: ConstInt, BLOCK_E: ConstInt):
    row = ct.bid(0)
    x2d = ct.load(logits, index=(row, 0), shape=(1, BLOCK_E),
                  padding_mode=ct.PaddingMode.NEG_INF)
    x = ct.astype(x2d, np.float32)
    x = ct.reshape(x, (BLOCK_E,))

    k_arange = ct.arange(K, dtype=np.int32)
    offs = ct.arange(BLOCK_E, dtype=np.int32)
    top_vals = ct.full((K,), -float('inf'), dtype=np.float32)
    top_idxs = ct.full((K,), 0, dtype=np.int32)

    for i in range(K):
        v, idx = ct.reduce(
            (x, offs), axis=0,
            func=_argmax_combine,
            identity=(-float('inf'), BLOCK_E + 1),
        )
        pos = K - 1 - i
        top_vals = ct.where(k_arange == pos, v, top_vals)
        top_idxs = ct.where(k_arange == pos, idx, top_idxs)
        x = ct.where(offs == idx, -float('inf'), x)

    m_val = ct.max(top_vals)
    ex = ct.exp(top_vals - m_val)
    sm = ct.sum(ex)
    w = ex / sm

    w_out = ct.astype(w, logits.dtype)
    ct.store(weights, index=(row, 0), tile=ct.reshape(w_out, (1, K)))
    ct.store(indices, index=(row, 0), tile=ct.reshape(top_idxs, (1, K)))


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = 1
    while BLOCK_E < E:
        BLOCK_E *= 2

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _moe_topk_kernel,
              (logits, weights, indices, E, k, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": 4,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
