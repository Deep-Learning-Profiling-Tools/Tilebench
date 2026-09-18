import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * K
    offs = tl.arange(0, BLOCK_K)

    o0 = offs
    m0 = o0 < K
    x0 = tl.load(x_ptr + base + o0, mask=m0, other=0.0, eviction_policy="evict_first")
    x0f = x0.to(tl.float32)
    sum_sq = tl.sum(x0f * x0f, axis=0)

    if K > BLOCK_K:
        o1 = BLOCK_K + offs
        m1 = o1 < K
        x1 = tl.load(x_ptr + base + o1, mask=m1, other=0.0, eviction_policy="evict_first")
        x1f = x1.to(tl.float32)
        sum_sq += tl.sum(x1f * x1f, axis=0)

    if K > 2 * BLOCK_K:
        o2 = 2 * BLOCK_K + offs
        m2 = o2 < K
        x2 = tl.load(x_ptr + base + o2, mask=m2, other=0.0, eviction_policy="evict_first")
        x2f = x2.to(tl.float32)
        sum_sq += tl.sum(x2f * x2f, axis=0)

    if K > 3 * BLOCK_K:
        o3 = 3 * BLOCK_K + offs
        m3 = o3 < K
        x3 = tl.load(x_ptr + base + o3, mask=m3, other=0.0, eviction_policy="evict_first")
        x3f = x3.to(tl.float32)
        sum_sq += tl.sum(x3f * x3f, axis=0)

    if K > 4 * BLOCK_K:
        o4 = 4 * BLOCK_K + offs
        m4 = o4 < K
        x4 = tl.load(x_ptr + base + o4, mask=m4, other=0.0, eviction_policy="evict_first")
        x4f = x4.to(tl.float32)
        sum_sq += tl.sum(x4f * x4f, axis=0)

    eps_f = sum_sq * 0.0 + eps
    norm = tl.sqrt_rn(sum_sq)
    denom = tl.maximum(norm, eps_f)
    rstd = tl.div_rn(1.0, denom)

    tl.store(
        out_ptr + base + o0,
        x0.to(tl.float32) * rstd,
        mask=m0,
        eviction_policy="evict_first",
    )

    if K > BLOCK_K:
        tl.store(
            out_ptr + base + o1,
            x1.to(tl.float32) * rstd,
            mask=m1,
            eviction_policy="evict_first",
        )

    if K > 2 * BLOCK_K:
        tl.store(
            out_ptr + base + o2,
            x2.to(tl.float32) * rstd,
            mask=m2,
            eviction_policy="evict_first",
        )

    if K > 3 * BLOCK_K:
        tl.store(
            out_ptr + base + o3,
            x3.to(tl.float32) * rstd,
            mask=m3,
            eviction_policy="evict_first",
        )

    if K > 4 * BLOCK_K:
        tl.store(
            out_ptr + base + o4,
            x4.to(tl.float32) * rstd,
            mask=m4,
            eviction_policy="evict_first",
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_K = 2048
    MAX_BLOCKS = 5
    num_warps = 8
    num_stages = 3

    grid = (rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "MAX_BLOCKS": MAX_BLOCKS,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
