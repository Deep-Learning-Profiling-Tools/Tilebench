import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _max_pool_kernel(
    x_flat, out_flat,
    H, W, H_out, W_out,
    stride, padding,
    KS: ConstInt, BLOCK_W: ConstInt,
):
    nc = ct.bid(0)
    h_out = ct.bid(1)
    pid_w = ct.bid(2)

    w_out = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=np.int32)
    mask_w = w_out < W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    acc = ct.full((BLOCK_W,), -np.inf, dtype=np.float32)
    neg_inf_tile = ct.full((BLOCK_W,), -np.inf, dtype=np.float32)

    for kh in range(KS):
        h = h_start + kh
        valid_h = (h >= 0) & (h < H)
        row_base = nc_base + h * W
        for kw in range(KS):
            w = w_start + kw
            valid = mask_w & valid_h & (w >= 0) & (w < W)
            idx = row_base + w
            safe_idx = ct.where(valid, idx, 0)
            v = ct.gather(x_flat, safe_idx, padding_value=0)
            vf = ct.astype(v, np.float32)
            vf = ct.where(valid, vf, neg_inf_tile)
            acc = ct.maximum(acc, vf)

    out_idx = nc * H_out * W_out + h_out * W_out + w_out
    safe_out = ct.where(mask_w, out_idx, -1)
    ct.scatter(out_flat, safe_out, ct.astype(acc, x_flat.dtype))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK_W = 512
    occupancy = 4

    NC = N * C
    grid = (NC, H_out, ct.cdiv(W_out, BLOCK_W))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, grid, _max_pool_kernel,
        (input, output, H, W, H_out, W_out, stride, padding, kernel_size, BLOCK_W),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_W": BLOCK_W,
        "occupancy": occupancy,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
