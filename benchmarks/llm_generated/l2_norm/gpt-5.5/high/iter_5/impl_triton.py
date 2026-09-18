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
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * K

    offs0 = tl.arange(0, BLOCK_MAIN)
    mask0 = offs0 < K
    x0 = tl.load(
        x_ptr + base + offs0,
        mask=mask0,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    sum_sq = tl.sum(x0 * x0, axis=0)

    if K > BLOCK_MAIN:
        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        mask1 = k1 < K
        x1 = tl.load(
            x_ptr + base + k1,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        sum_sq += tl.sum(x1 * x1, axis=0)

    eps_f = sum_sq * 0.0 + eps
    norm = tl.sqrt_rn(sum_sq)
    denom = tl.maximum(norm, eps_f)
    rstd = tl.div_rn(1.0, denom)

    tl.store(
        out_ptr + base + offs0,
        x0 * rstd,
        mask=mask0,
        cache_modifier=".cs",
    )

    if K > BLOCK_MAIN:
        tl.store(
            out_ptr + base + k1,
            x1 * rstd,
            mask=mask1,
            cache_modifier=".cs",
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_MAIN = 8192
    BLOCK_TAIL = 2048
    num_warps = 16
    num_stages = 3

    grid = (rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_MAIN=BLOCK_MAIN,
        BLOCK_TAIL=BLOCK_TAIL,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_MAIN": BLOCK_MAIN,
            "BLOCK_TAIL": BLOCK_TAIL,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_CACHE_CG": 1,
            "STORE_CACHE_CS": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
