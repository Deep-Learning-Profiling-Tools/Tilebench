import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, eps,
                      K: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    sum_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in range(0, K, BLOCK_N):
        k = start + offs
        mask = k < K
        x = tl.load(x_ptr + row * K + k, mask=mask, other=0.0).to(tl.float32)
        sum_vec += x
        sumsq_vec += x * x

    inv_k = 1.0 / K
    mean = tl.sum(sum_vec, axis=0) * inv_k
    var = tl.sum(sumsq_vec, axis=0) * inv_k - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    for start in range(0, K, BLOCK_N):
        k = start + offs
        mask = k < K
        x = tl.load(x_ptr + row * K + k, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + k, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + k, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * K + k, y, mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)

    K = x.shape[-1]
    n_rows = x.numel() // K

    BLOCK_N = 1024
    num_warps = 4
    num_stages = 2

    grid = (n_rows,)
    _layernorm_kernel[grid](
        x, weight, bias, output, eps,
        K=K,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
