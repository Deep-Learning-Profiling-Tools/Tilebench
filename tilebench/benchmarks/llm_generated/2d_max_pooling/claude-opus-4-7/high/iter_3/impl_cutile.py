import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _max_pool_kernel(
    x_flat, out_flat,
    H, W, H_out, W_out,
    stride, padding,
    KS: ConstInt, BLOCK: ConstInt,
):
    nc = ct.bid(0)
    pid = ct.bid(1)

    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    n_out = H_out * W_out
    valid_out = offs < n_out

    h_out = offs // W_out
    w_out = offs % W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    acc = ct.full((BLOCK,), -np.inf, dtype=np.float32)
    neg_inf_tile = ct.full((BLOCK,), -np.inf, dtype=np.float32)

    for kh in range(KS):
        for kw in range(KS):
            h = h_start + kh
            w = w_start + kw
            valid = (h >= 0) & (h < H) & (w >= 0) & (w < W) & valid_out
            idx = nc_base + h * W + w
            safe_idx = ct.where(valid, idx, -1)
            v = ct.gather(x_flat, safe_idx, padding_value=0)
            vf = ct.astype(v, np.float32)
            vf = ct.where(valid, vf, neg_inf_tile)
            acc = ct.maximum(acc, vf)

    out_idx = nc * n_out + offs
    safe_out = ct.where(valid_out, out_idx, -1)
    ct.scatter(out_flat, safe_out, ct.astype(acc, x_flat.dtype))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK = 1024
    occupancy = 2

    NC = N * C
    grid = (NC, ct.cdiv(H_out * W_out, BLOCK), 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, grid, _max_pool_kernel,
        (input, output, H, W, H_out, W_out, stride, padding, kernel_size, BLOCK),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "occupancy": occupancy,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
