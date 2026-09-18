import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _max_pool2d_kernel(input, output,
                       H: ConstInt, W: ConstInt,
                       H_OUT: ConstInt, W_OUT: ConstInt,
                       KERNEL_SIZE: ConstInt,
                       STRIDE: ConstInt,
                       PADDING: ConstInt,
                       TILE: ConstInt):
    block_w = ct.bid(0)
    row = ct.bid(1)

    cols = block_w * TILE + ct.arange(TILE, dtype=np.int32)
    oh = row % H_OUT
    nc = row // H_OUT

    ih_base = oh * STRIDE - PADDING
    iw_base = cols * STRIDE - PADDING

    acc = ct.full((TILE,), -np.inf, dtype=np.float32)

    for kh in range(0, KERNEL_SIZE):
        ih = ih_base + kh
        valid_h = (ih >= 0) & (ih < H)

        for kw in range(0, KERNEL_SIZE):
            iw = iw_base + kw
            valid = (cols < W_OUT) & valid_h & (iw >= 0) & (iw < W)
            idx = nc * H * W + ih * W + iw
            safe_idx = ct.where(valid, idx, -1)
            vals = ct.astype(ct.gather(input, safe_idx, padding_value=-np.inf), np.float32)
            acc = ct.maximum(acc, vals)

    out_tile = ct.reshape(ct.astype(acc, input.dtype), (1, TILE))
    ct.store(output, index=(row, block_w), tile=out_tile, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), rows, 1)
    kernel = _max_pool2d_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (input, output, H, W, H_out, W_out, kernel_size, stride, padding, TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
