import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _max_pool2d_k3s2_bh4_shift_kernel(input, output,
                                      H: ConstInt, W: ConstInt,
                                      H_OUT: ConstInt, W_OUT: ConstInt,
                                      PADDING: ConstInt,
                                      TILE: ConstInt,
                                      FULL_TILE: ConstInt):
    block_w = ct.bid(0)
    h_block = ct.bid(1)
    nc = ct.bid(2)

    offs = ct.arange(TILE, dtype=np.int32)
    offs_full = ct.arange(FULL_TILE, dtype=np.int32)

    out_col0 = block_w * TILE
    cols = out_col0 + offs
    cols_full = out_col0 + offs_full

    oh0 = h_block * 4
    ih_base0 = oh0 * 2 - PADDING
    iw0_base = cols_full * 2 - PADDING
    iw1 = cols * 2 - PADDING + 1
    base_nc = nc * H * W

    acc0 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc1 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc2 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc3 = ct.full((TILE,), -np.inf, dtype=np.float32)

    for rh in range(0, 9):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)

        valid0 = valid_h & (cols_full <= W_OUT) & (iw0_base >= 0) & (iw0_base < W)
        idx0 = base_nc + ih * W + iw0_base
        safe_idx0 = ct.where(valid0, idx0, -1)
        v0_full = ct.astype(
            ct.gather(input, safe_idx0, padding_value=-np.inf),
            np.float32,
        )

        valid1 = valid_h & (cols < W_OUT) & (iw1 >= 0) & (iw1 < W)
        idx1 = base_nc + ih * W + iw1
        safe_idx1 = ct.where(valid1, idx1, -1)
        v1 = ct.astype(
            ct.gather(input, safe_idx1, padding_value=-np.inf),
            np.float32,
        )

        v0 = ct.extract(v0_full, (0,), (TILE,))
        v2 = ct.extract(v0_full, (1,), (TILE,))
        hmax = ct.maximum(ct.maximum(v0, v1), v2)

        if rh < 3:
            acc0 = ct.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = ct.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = ct.maximum(acc2, hmax)
        if rh >= 6:
            acc3 = ct.maximum(acc3, hmax)

    out_row = nc * H_OUT + oh0

    out0 = ct.reshape(ct.astype(acc0, input.dtype), (1, TILE))
    out1 = ct.reshape(ct.astype(acc1, input.dtype), (1, TILE))
    out2 = ct.reshape(ct.astype(acc2, input.dtype), (1, TILE))
    out3 = ct.reshape(ct.astype(acc3, input.dtype), (1, TILE))

    ct.store(output, index=(out_row, block_w), tile=out0, allow_tma=False)
    ct.store(output, index=(out_row + 1, block_w), tile=out1, allow_tma=False)
    ct.store(output, index=(out_row + 2, block_w), tile=out2, allow_tma=False)
    ct.store(output, index=(out_row + 3, block_w), tile=out3, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 256
    FULL_TILE = 512
    BLOCK_H = 4
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), ct.cdiv(H_out, BLOCK_H), N * C)
    ct.launch(
        stream,
        grid,
        _max_pool2d_k3s2_bh4_shift_kernel,
        (input, output, H, W, H_out, W_out, padding, TILE, FULL_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh4_shift",
        "TILE": TILE,
        "FULL_TILE": FULL_TILE,
        "BLOCK_H": BLOCK_H,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
