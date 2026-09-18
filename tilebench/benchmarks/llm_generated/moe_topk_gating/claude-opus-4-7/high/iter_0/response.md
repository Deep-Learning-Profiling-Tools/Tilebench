Looking at this op: per-row iterative top-K (max + argmax + mask-out) followed by softmax over the K values. E=128, k=2 are small constants so each row fits in one tile (BLOCK_E=128). One CTA per row.

Key correctness detail: the reference's iterative `torch.max` returns first occurrence on ties, so my argmax must also be first-occurrence-on-tie. Triton's `tl.argmax(tie_break_left=True)` handles that; for cuTile I implement argmax via `ct.reduce` with an explicit tie-breaker on index.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _moe_topk_kernel(logits_ptr, weights_ptr, indices_ptr,
                     M, E, stride_lm, stride_le,
                     K: tl.constexpr, BLOCK_E: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    x = tl.load(logits_ptr + row * stride_lm + offs * stride_le,
                mask=mask, other=-float('inf')).to(tl.float32)

    k_arange = tl.arange(0, K)
    top_vals = tl.full((K,), -float('inf'), tl.float32)
    top_idxs = tl.zeros((K,), dtype=tl.int32)

    for i in tl.static_range(K):
        v = tl.max(x, axis=0)
        idx = tl.argmax(x, axis=0).to(tl.int32)
        pos = K - 1 - i
        top_vals = tl.where(k_arange == pos, v, top_vals)
        top_idxs = tl.where(k_arange == pos, idx, top_idxs)
        # Mask the picked position so next iter picks the next-largest.
        x = tl.where(offs == idx, -float('inf'), x)

    m_val = tl.max(top_vals, axis=0)
    ex = tl.exp(top_vals - m_val)
    sm = tl.sum(ex, axis=0)
    w = ex / sm

    out_offs = row * K + k_arange
    tl.store(weights_ptr + out_offs, w.to(weights_ptr.dtype.element_ty))
    tl.store(indices_ptr + out_offs, top_idxs)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    weights = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)

    BLOCK_E = triton.next_power_of_2(E)
    num_warps = 2
    num_stages = 2

    grid = (M,)
    _moe_topk_kernel[grid](
        logits, weights, indices,
        M, E, logits.stride(0), logits.stride(1),
        K=k, BLOCK_E=BLOCK_E,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
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
    # First-occurrence tie-break on equal values.
    win1 = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (ct.where(win1, v1, v2), ct.where(win1, i1, i2))


@ct.kernel
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

    occupancy = 4
    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    kernel = _moe_topk_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, weights, indices, E, k, BLOCK_E))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_E": BLOCK_E,
        "K": k,
        "occupancy": occupancy,
    })
    return (weights, indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: one CTA per row loads the whole logits row (BLOCK_E=128, -inf padded), runs K=2 iterations of max+argmax+mask-out using register-resident top_vals/top_idxs tiles updated via `tl.where`/`ct.where`, then softmaxes the K winners and writes both outputs. Tie-breaking matches the reference (Triton via `tie_break_left`, cuTile via an explicit index-tiebreak `ct.reduce`).
