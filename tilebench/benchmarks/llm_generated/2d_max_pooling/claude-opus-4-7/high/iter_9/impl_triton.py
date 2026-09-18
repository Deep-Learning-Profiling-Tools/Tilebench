import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool_kernel(
    x_ptr, out_ptr,
    H, W, H_out, W_out,
    stride, padding,
    KS: tl.constexpr, BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_out = H_out * W_out
    mask_out = offs < n_out

    h_out = offs // W_out
    w_out = offs % W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    NEG_INF = float('-inf')
    acc = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        for kw in tl.static_range(KS):
            h = h_start + kh
            w = w_start + kw
            valid = (h >= 0) & (h < H) & (w >= 0) & (w < W) & mask_out
            idx = nc_base + h * W + w
            v = tl.load(x_ptr + idx, mask=valid, other=NEG_INF)
            acc = tl.maximum(acc, v.to(tl.float32))

    out_off = nc * n_out + offs
    tl.store(out_ptr + out_off, acc, mask=mask_out)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK = 1024
    num_warps = 4
    num_stages = 3

    NC = N * C
    grid = (NC, triton.cdiv(H_out * W_out, BLOCK))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size, BLOCK=BLOCK,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
