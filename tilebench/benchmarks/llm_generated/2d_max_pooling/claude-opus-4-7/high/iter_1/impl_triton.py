import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool_kernel(
    x_ptr, out_ptr,
    H, W, H_out, W_out,
    stride, padding,
    KS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_h = offs_h < H_out
    mask_w = offs_w < W_out

    h_in_base = offs_h * stride - padding  # [BLOCK_H]
    w_in_base = offs_w * stride - padding  # [BLOCK_W]

    nc_base = nc * H * W
    NEG_INF = float('-inf')
    acc = tl.full((BLOCK_H, BLOCK_W), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        for kw in tl.static_range(KS):
            h = h_in_base[:, None] + kh  # [BLOCK_H, 1]
            w = w_in_base[None, :] + kw  # [1, BLOCK_W]
            valid = (h >= 0) & (h < H) & (w >= 0) & (w < W) & mask_h[:, None] & mask_w[None, :]
            idx = nc_base + h * W + w
            v = tl.load(x_ptr + idx, mask=valid, other=NEG_INF)
            acc = tl.maximum(acc, v.to(tl.float32))

    out_base = nc * H_out * W_out
    out_idx = out_base + offs_h[:, None] * W_out + offs_w[None, :]
    out_mask = mask_h[:, None] & mask_w[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK_H = 8
    BLOCK_W = 64
    num_warps = 4
    num_stages = 2

    NC = N * C
    grid = (NC, triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_W": BLOCK_W,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
