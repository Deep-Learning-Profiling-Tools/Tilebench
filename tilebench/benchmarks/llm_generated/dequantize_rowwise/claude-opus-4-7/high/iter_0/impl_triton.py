import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dequant_kernel(x_ptr, scale_ptr, out_ptr,
                    rows, cols,
                    stride_xr, stride_xc,
                    stride_or, stride_oc,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < rows
    mask_n = offs_n < cols
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xr + offs_n[None, :] * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)

    scale = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0)

    y = x * scale[:, None] * (1.0 / 127.0)
    y = y.to(tl.float16)

    out_ptrs = out_ptr + offs_m[:, None] * stride_or + offs_n[None, :] * stride_oc
    tl.store(out_ptrs, y, mask=mask)


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows, cols = x.shape
    output = torch.empty((rows, cols), dtype=torch.float16, device=x.device)

    BLOCK_M = 8
    BLOCK_N = 512
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
    _dequant_kernel[grid](
        x, state_x, output,
        rows, cols,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
