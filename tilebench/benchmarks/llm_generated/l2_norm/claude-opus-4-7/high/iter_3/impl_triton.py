import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, K, eps, stride_row,
                    BLOCK_K: tl.constexpr,
                    LOOP_STAGES: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    o_row = out_ptr + row * stride_row

    # Pass 1: accumulate sum of squares (fp32). Use evict_last so the data
    # stays in L2 for the second pass.
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=LOOP_STAGES):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
        acc += x * x
    total = tl.sum(acc, axis=0)
    norm = tl.sqrt(total)
    denom = tl.maximum(norm, eps)
    inv = 1.0 / denom

    # Pass 2: normalize. evict_first since we're done with this row's data.
    for k0 in tl.range(0, K, BLOCK_K, num_stages=LOOP_STAGES):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        y = x * inv
        tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    BLOCK_K = 2048
    num_warps = 8
    num_stages = 2
    LOOP_STAGES = 4  # software-pipeline depth for the per-row K loops

    grid = (M,)
    _l2_norm_kernel[grid](
        x_flat, out_flat, K, float(eps), x_flat.stride(0),
        BLOCK_K=BLOCK_K, LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "LOOP_STAGES": LOOP_STAGES,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
