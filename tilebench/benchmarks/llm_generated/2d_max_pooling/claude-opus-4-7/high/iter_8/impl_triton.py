import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool_kernel(
    x_ptr, out_ptr,
    H, W, H_out, W_out,
    stride, padding,
    KS: tl.constexpr, BW: tl.constexpr,
):
    nc = tl.program_id(0)
    h_out = tl.program_id(1)
    w_blk = tl.program_id(2)

    tl.assume(stride > 0)
    tl.assume(padding >= 0)
    tl.assume(H > 0)
    tl.assume(W > 0)

    w_out_base = w_blk * BW
    w_out_idx = w_out_base + tl.arange(0, BW)
    mask_w = w_out_idx < W_out

    h_in_base = h_out * stride - padding
    nc_base = nc * H * W

    NEG_INF = float('-inf')
    acc = tl.full((BW,), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h_in = h_in_base + kh
        valid_h = (h_in >= 0) & (h_in < H)
        row_base = nc_base + h_in * W
        for kw in tl.static_range(KS):
            w_in = w_out_idx * stride + kw - padding
            valid = valid_h & (w_in >= 0) & (w_in < W) & mask_w
            idx = row_base + w_in
            v = tl.load(x_ptr + idx, mask=valid, other=NEG_INF,
                        eviction_policy="evict_last")
            acc = tl.maximum(acc, v.to(tl.float32))

    out_off = nc * H_out * W_out + h_out * W_out + w_out_idx
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_w)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BW = 256
    num_warps = 4
    num_stages = 2

    NC = N * C
    grid = (NC, H_out, triton.cdiv(W_out, BW))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size, BW=BW,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BW": BW,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
