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
    ph = tl.program_id(1)

    tl.assume(stride > 0)
    tl.assume(padding >= 0)
    tl.assume(H > 0)
    tl.assume(W > 0)
    tl.assume(H_out > 0)
    tl.assume(W_out > 0)

    h_out = ph * BLOCK_H + tl.arange(0, BLOCK_H)        # [BLOCK_H]
    w_out = tl.arange(0, BLOCK_W)                       # [BLOCK_W]
    mask_h = h_out < H_out
    mask_w = w_out < W_out

    h_in = h_out * stride - padding                     # [BLOCK_H]
    w_in = w_out * stride - padding                     # [BLOCK_W]

    nc_base = nc * H * W
    NEG_INF = float('-inf')
    acc = tl.full((BLOCK_H, BLOCK_W), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h = h_in[:, None] + kh                          # [BLOCK_H, 1]
        valid_h = (h >= 0) & (h < H)
        for kw in tl.static_range(KS):
            w = w_in[None, :] + kw                      # [1, BLOCK_W]
            valid_w = (w >= 0) & (w < W)
            valid = valid_h & valid_w & mask_h[:, None] & mask_w[None, :]
            idx = nc_base + h * W + w
            v = tl.load(x_ptr + idx, mask=valid, other=NEG_INF)
            acc = tl.maximum(acc, v.to(tl.float32))

    out_idx = nc * H_out * W_out + h_out[:, None] * W_out + w_out[None, :]
    mask_out = mask_h[:, None] & mask_w[None, :]
    tl.store(
        out_ptr + out_idx,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_out,
    )


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    # Choose BLOCK_W as smallest pow-2 >= W_out so each program holds full output rows
    # (avoids row-crossing within a tile -> coalesced loads along W).
    BLOCK_W = max(64, triton.next_power_of_2(W_out))
    BLOCK_H = 4
    num_warps = 8
    num_stages = 2

    NC = N * C
    grid = (NC, triton.cdiv(H_out, BLOCK_H))

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
