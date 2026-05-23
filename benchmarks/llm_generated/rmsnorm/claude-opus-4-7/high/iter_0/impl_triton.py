import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr,
                    stride_row,
                    N, eps,
                    BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    o_row = out_ptr + row * stride_row
    cols = tl.arange(0, BLOCK_N)

    # Pass 1: sum of squares
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + cols
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)

    # Pass 2: normalize and scale
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + cols
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(o_row + offs, y, mask=mask)


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)
    stride_row = x2.stride(0)

    BLOCK_N = 2048 if K >= 2048 else triton.next_power_of_2(K)
    if BLOCK_N < 256:
        BLOCK_N = 256
    num_warps = 8
    num_stages = 2

    grid = (n_rows,)
    _rmsnorm_kernel[grid](
        x2, rms_w, o2,
        stride_row,
        K, eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
