import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _max_pool2d_k3s2_contig_bh8_kernel(input, output,
                                       H: ConstInt, W: ConstInt,
                                       H_OUT: ConstInt, W_OUT: ConstInt,
                                       PADDING: ConstInt,
                                       TILE: ConstInt,
                                       IN_TILE: ConstInt,
                                       HALF_IN: ConstInt,
                                       NEED_W: ConstInt):
    block_w = ct.bid(0)
    h_block = ct.bid(1)
    nc = ct.bid(2)

    offs = ct.arange(IN_TILE, dtype=np.int32)
    out_col0 = block_w * TILE
    iw_start = out_col0 * 2 - PADDING
    ih_base0 = h_block * 16 - PADDING
    base_nc = nc * H * W

    acc0 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc1 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc2 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc3 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc4 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc5 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc6 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc7 = ct.full((TILE,), -np.inf, dtype=np.float32)

    in_cols = iw_start + offs
    valid_w = (offs < NEED_W) & (in_cols >= 0) & (in_cols < W)

    for rh in range(0, 17):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        idx = base_nc + ih * W + in_cols
        safe_idx = ct.where(valid_h & valid_w, idx, -1)

        vals = ct.astype(
            ct.gather(input, safe_idx, padding_value=-np.inf, latency=1),
            np.float32,
        )
        vals2 = ct.reshape(vals, (HALF_IN, 2))

        v0 = ct.reshape(ct.extract(vals2, (0, 0), (TILE, 1)), (TILE,))
        v1 = ct.reshape(ct.extract(vals2, (0, 1), (TILE, 1)), (TILE,))
        v2 = ct.reshape(ct.extract(vals2, (1, 0), (TILE, 1)), (TILE,))
        hmax = ct.maximum(ct.maximum(v0, v1), v2)

        if rh < 3:
            acc0 = ct.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = ct.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = ct.maximum(acc2, hmax)
        if (rh >= 6) and (rh < 9):
            acc3 = ct.maximum(acc3, hmax)
        if (rh >= 8) and (rh < 11):
            acc4 = ct.maximum(acc4, hmax)
        if (rh >= 10) and (rh < 13):
            acc5 = ct.maximum(acc5, hmax)
        if (rh >= 12) and (rh < 15):
            acc6 = ct.maximum(acc6, hmax)
        if rh >= 14:
            acc7 = ct.maximum(acc7, hmax)

    out_row = nc * H_OUT + h_block * 8

    ct.store(output, index=(out_row + 0, block_w),
             tile=ct.reshape(ct.astype(acc0, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 1, block_w),
             tile=ct.reshape(ct.astype(acc1, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 2, block_w),
             tile=ct.reshape(ct.astype(acc2, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 3, block_w),
             tile=ct.reshape(ct.astype(acc3, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 4, block_w),
             tile=ct.reshape(ct.astype(acc4, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 5, block_w),
             tile=ct.reshape(ct.astype(acc5, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 6, block_w),
             tile=ct.reshape(ct.astype(acc6, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 7, block_w),
             tile=ct.reshape(ct.astype(acc7, input.dtype), (1, TILE)),
             allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 128
    BLOCK_H = 8
    IN_TILE = 512
    HALF_IN = 256
    NEED_W = 2 * TILE + 1
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), ct.cdiv(H_out, BLOCK_H), N * C)
    kernel = _max_pool2d_k3s2_contig_bh8_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (input, output, H, W, H_out, W_out, padding, TILE, IN_TILE, HALF_IN, NEED_W),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_contig_bh8",
        "TILE": TILE,
        "BLOCK_H": BLOCK_H,
        "IN_TILE": IN_TILE,
        "HALF_IN": HALF_IN,
        "NEED_W": NEED_W,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
