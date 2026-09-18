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
