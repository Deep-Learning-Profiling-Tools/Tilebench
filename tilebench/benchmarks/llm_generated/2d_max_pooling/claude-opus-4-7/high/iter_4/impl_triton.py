import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool_kernel(
    x_ptr, out_ptr,
    H, W, H_out, W_out,
    stride, padding,
    KS: tl.constexpr, BLOCK_W: tl.constexpr,
):
    nc = tl.program_id(0)
    h_out = tl.program_id(1)
    pid_w = tl.program_id(2)

    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_out < W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    NEG_INF = float('-inf')
    acc = tl.full((BLOCK_W,), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h = h_start + kh
        valid_h = (h >= 0) & (h < H)
        row_base = nc_base + h * W
        for kw in tl.static_range(KS):
            w = w_start + kw
            valid = mask_w & valid_h & (w >= 0) & (w < W)
            v = tl.load(x_ptr + row_base + w, mask=valid, other=NEG_INF)
            acc = tl.maximum(acc, v.to(tl.float32))

    out_off = nc * H_out * W_out + h_out * W_out + w_out
    tl.store(out_ptr + out_off, acc, mask=mask_w)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK_W = 256
    num_warps = 4
    num_stages = 2

    NC = N * C
    grid = (NC, H_out, triton.cdiv(W_out, BLOCK_W))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size, BLOCK_W=BLOCK_W,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_W": BLOCK_W,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
