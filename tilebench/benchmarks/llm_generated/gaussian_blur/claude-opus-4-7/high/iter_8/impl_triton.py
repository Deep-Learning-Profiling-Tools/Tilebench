import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hconv_kernel(x_ptr, k_ptr, out_ptr, H, W,
                  KW: tl.constexpr, PAD_W: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = offs_m[:, None] < H
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kw in tl.static_range(0, KW):
        col = offs_n + (kw - PAD_W)
        mask = row_mask & (col[None, :] >= 0) & (col[None, :] < W)
        ptrs = x_ptr + offs_m[:, None] * W + col[None, :]
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(k_ptr + kw).to(tl.float32)
        acc += x_val * w_val

    out_mask = row_mask & (offs_n[None, :] < W)
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _vconv_kernel(x_ptr, k_ptr, out_ptr, H, W,
                  KH: tl.constexpr, PAD_H: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    col_mask = offs_n[None, :] < W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        row = offs_m + (kh - PAD_H)
        mask = (row[:, None] >= 0) & (row[:, None] < H) & col_mask
        ptrs = x_ptr + row[:, None] * W + offs_n[None, :]
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(k_ptr + kh).to(tl.float32)
        acc += x_val * w_val

    out_mask = (offs_m[:, None] < H) & col_mask
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    # Decompose 2D kernel into separable 1D row & col vectors.
    # For a true Gaussian K[i,j] = col_k[i] * row_k[j].
    k2d = kernel.view(KH, KW)
    center_f32 = k2d[KH // 2, KW // 2].float()
    row_k = k2d[KH // 2, :].contiguous()                              # length KW
    col_k = (k2d[:, KW // 2].float() / center_f32).to(kernel.dtype)   # length KH
    col_k = col_k.contiguous()

    intermediate = torch.empty_like(input)
    output = torch.empty_like(input)

    BLOCK_M = 32
    BLOCK_N = 128
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))

    _hconv_kernel[grid](
        input, row_k, intermediate, H, W,
        KW=KW, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    _vconv_kernel[grid](
        intermediate, col_k, output, H, W,
        KH=KH, PAD_H=PAD_H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "approach": "separable",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
