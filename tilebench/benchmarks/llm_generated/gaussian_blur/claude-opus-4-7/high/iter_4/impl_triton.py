import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_kernel(x_ptr, k_ptr, out_ptr,
                   H, W,
                   KH: tl.constexpr, KW: tl.constexpr,
                   PAD_H: tl.constexpr, PAD_W: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            row = offs_m[:, None] + (kh - PAD_H)
            col = offs_n[None, :] + (kw - PAD_W)
            mask = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            ptrs = x_ptr + row * W + col
            x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            w_val = tl.load(k_ptr + kh * KW + kw).to(tl.float32)
            acc += x_val * w_val

    out_mask = (offs_m[:, None] < H) & (offs_n[None, :] < W)
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))
    _conv2d_kernel[grid](
        input, kernel, output, H, W,
        KH=KH, KW=KW, PAD_H=PAD_H, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
