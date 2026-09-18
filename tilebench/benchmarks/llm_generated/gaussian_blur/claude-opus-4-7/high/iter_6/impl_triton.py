import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hconv_kernel(x_ptr, kw_ptr, out_ptr,
                  H, W,
                  KW: tl.constexpr, PAD_W: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row = offs_m[:, None]
    row_mask = row < H
    for kw in tl.static_range(0, KW):
        col = offs_n[None, :] + (kw - PAD_W)
        mask = row_mask & (col >= 0) & (col < W)
        ptrs = x_ptr + row * W + col
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(kw_ptr + kw).to(tl.float32)
        acc += x_val * w_val
    out_mask = (offs_m[:, None] < H) & (offs_n[None, :] < W)
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _vconv_kernel(x_ptr, kh_ptr, out_ptr,
                  H, W,
                  KH: tl.constexpr, PAD_H: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    col = offs_n[None, :]
    col_mask = col < W
    for kh in tl.static_range(0, KH):
        row = offs_m[:, None] + (kh - PAD_H)
        mask = (row >= 0) & (row < H) & col_mask
        ptrs = x_ptr + row * W + col
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(kh_ptr + kh).to(tl.float32)
        acc += x_val * w_val
    out_mask = (offs_m[:, None] < H) & col_mask
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _decompose_rank1(kernel, KH, KW, device):
    """Extract 1-D factors g_v, g_h such that K = g_v ⊗ g_h.

    For a true Gaussian (rank-1), this is exact and far cheaper than SVD.
    """
    k2d = kernel.float().view(KH, KW)
    ch = KH // 2
    cw = KW // 2
    center_row = k2d[ch, :]                     # [KW]
    center_col = k2d[:, cw]                     # [KH]
    g_v_center = center_row.sum()                # = g_v[ch]  (since g_h sums to 1)
    g_h = center_row / g_v_center                # [KW]
    g_h_center = g_h[cw]
    g_v = center_col / g_h_center                # [KH]
    return g_v.contiguous(), g_h.contiguous()


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    g_v, g_h = _decompose_rank1(kernel, KH, KW, input.device)
    kh_1d = g_v   # vertical 1-D kernel (length KH)
    kw_1d = g_h   # horizontal 1-D kernel (length KW)

    BLOCK_M = 16
    BLOCK_N = 128
    num_warps = 4
    num_stages = 2

    temp = torch.empty(H * W, dtype=torch.float32, device=input.device)
    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))

    _hconv_kernel[grid](
        input, kw_1d, temp, H, W,
        KW=KW, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    _vconv_kernel[grid](
        temp, kh_1d, output, H, W,
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
        "approach": "separable_rank1",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
