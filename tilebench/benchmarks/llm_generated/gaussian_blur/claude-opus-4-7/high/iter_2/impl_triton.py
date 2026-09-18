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

    base_m = pid_m * BLOCK_M - PAD_H
    base_n = pid_n * BLOCK_N - PAD_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            x_block = tl.make_block_ptr(
                base=x_ptr,
                shape=(H, W),
                strides=(W, 1),
                offsets=(base_m + kh, base_n + kw),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            x_val = tl.load(x_block, boundary_check=(0, 1),
                            padding_option="zero").to(tl.float32)
            w_val = tl.load(k_ptr + kh * KW + kw).to(tl.float32)
            acc += x_val * w_val

    out_block = tl.make_block_ptr(
        base=out_ptr,
        shape=(H, W),
        strides=(W, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_block, acc.to(out_ptr.dtype.element_ty),
             boundary_check=(0, 1))


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    BLOCK_M = 32
    BLOCK_N = 128
    num_warps = 4
    num_stages = 3

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
