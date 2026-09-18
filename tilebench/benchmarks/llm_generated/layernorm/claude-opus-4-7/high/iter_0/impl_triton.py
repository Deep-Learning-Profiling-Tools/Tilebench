import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
               stride_row, N, eps,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row

    # Pass 1: accumulate sum(x) and sum(x^2) in fp32
    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_xx = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += x
        sum_xx += x * x

    inv_N = 1.0 / N
    mean = tl.sum(sum_x, axis=0) * inv_N
    mean_xx = tl.sum(sum_xx, axis=0) * inv_N
    var = mean_xx - mean * mean
    rstd = tl.math.rsqrt(var + eps)

    # Pass 2: normalize and apply weight/bias
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_row + offs, y, mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    out = torch.empty_like(x)
    K = x.shape[-1]
    rows = x.numel() // K
    x2d = x.contiguous().view(rows, K)
    out2d = out.view(rows, K)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 2

    _ln_kernel[(rows,)](
        x2d, weight, bias, out2d,
        x2d.stride(0), K, eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
