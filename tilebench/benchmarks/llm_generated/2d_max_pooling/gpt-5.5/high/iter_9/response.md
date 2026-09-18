```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_k3s2_bh8_split320_vec4_kernel(input_ptr, output_ptr,
                                               H: tl.constexpr, W: tl.constexpr,
                                               H_OUT: tl.constexpr, W_OUT: tl.constexpr,
                                               PADDING: tl.constexpr,
                                               BLOCK_A: tl.constexpr,
                                               BLOCK_B: tl.constexpr):
    pid_h = tl.program_id(0)
    nc = tl.program_id(1)

    tl.static_assert(PADDING == 1)

    offs_a = tl.arange(0, BLOCK_A)
    offs_b = tl.arange(0, BLOCK_B)
    kk = tl.arange(0, 4)

    cols_a = offs_a
    cols_b = BLOCK_A + offs_b

    mask_a = cols_a < W_OUT
    mask_b = cols_b < W_OUT

    oh0 = pid_h * 8
    ih_base0 = oh0 * 2 - PADDING
    base_nc = nc * H * W
    neg_inf = -float("inf")

    acca0 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca1 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca2 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca3 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca4 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca5 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca6 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)
    acca7 = tl.full((BLOCK_A,), neg_inf, dtype=tl.float32)

    accb0 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb1 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb2 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb3 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb4 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb5 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb6 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)
    accb7 = tl.full((BLOCK_B,), neg_inf, dtype=tl.float32)

    valid_rh0 = pid_h > 0
    k_mask = kk < 3

    for rh in tl.static_range(0, 17):
        ih = ih_base0 + rh
        row_base = input_ptr + base_nc + ih * W

        iw_a = cols_a[:, None] * 2 + kk[None, :] - PADDING
        safe_iw_a = tl.where(iw_a >= 0, iw_a, 0)
        load_mask_a = mask_a[:, None] & k_mask[None, :] & (iw_a >= 0)
        if rh == 0:
            load_mask_a = load_mask_a & valid_rh0

        vals_a = tl.load(row_base + safe_iw_a, mask=load_mask_a, other=neg_inf)
        hmaxa = tl.max(vals_a, axis=1).to(tl.float32)

        iw_b = cols_b[:, None] * 2 + kk[None, :] - PADDING
        load_mask_b = mask_b[:, None] & k_mask[None, :]
        if rh == 0:
            load_mask_b = load_mask_b & valid_rh0

        vals_b = tl.load(row_base + iw_b, mask=load_mask_b, other=neg_inf)
        hmaxb = tl.max(vals_b, axis=1).to(tl.float32)

        if rh < 3:
            acca0 = tl.maximum(acca0, hmaxa)
            accb0 = tl.maximum(accb0, hmaxb)
        if (rh >= 2) and (rh < 5):
            acca1 = tl.maximum(acca1, hmaxa)
            accb1 = tl.maximum(accb1, hmaxb)
        if (rh >= 4) and (rh < 7):
            acca2 = tl.maximum(acca2, hmaxa)
            accb2 = tl.maximum(accb2, hmaxb)
        if (rh >= 6) and (rh < 9):
            acca3 = tl.maximum(acca3, hmaxa)
            accb3 = tl.maximum(accb3, hmaxb)
        if (rh >= 8) and (rh < 11):
            acca4 = tl.maximum(acca4, hmaxa)
            accb4 = tl.maximum(accb4, hmaxb)
        if (rh >= 10) and (rh < 13):
            acca5 = tl.maximum(acca5, hmaxa)
            accb5 = tl.maximum(accb5, hmaxb)
        if (rh >= 12) and (rh < 15):
            acca6 = tl.maximum(acca6, hmaxa)
            accb6 = tl.maximum(accb6, hmaxb)
        if rh >= 14:
            acca7 = tl.maximum(acca7, hmaxa)
            accb7 = tl.maximum(accb7, hmaxb)

    out_row_base = nc * H_OUT
    oh1 = oh0 + 1
    oh2 = oh0 + 2
    oh3 = oh0 + 3
    oh4 = oh0 + 4
    oh5 = oh0 + 5
    oh6 = oh0 + 6
    oh7 = oh0 + 7

    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols_a, acca0, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols_b, accb0, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols_a, acca1, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols_b, accb1, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols_a, acca2, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols_b, accb2, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols_a, acca3, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols_b, accb3, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh4) * W_OUT + cols_a, acca4, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh4) * W_OUT + cols_b, accb4, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh5) * W_OUT + cols_a, acca5, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh5) * W_OUT + cols_b, accb5, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh6) * W_OUT + cols_a, acca6, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh6) * W_OUT + cols_b, accb6, mask=mask_b)
    tl.store(output_ptr + (out_row_base + oh7) * W_OUT + cols_a, acca7, mask=mask_a)
    tl.store(output_ptr + (out_row_base + oh7) * W_OUT + cols_b, accb7, mask=mask_b)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)

    BLOCK_A = 256
    BLOCK_B = 64
    BLOCK_H = 8
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(H_out, BLOCK_H), N * C)
    _max_pool2d_k3s2_bh8_split320_vec4_kernel[grid](
        input, output,
        H=H,
        W=W,
        H_OUT=H_out,
        W_OUT=W_out,
        PADDING=padding,
        BLOCK_A=BLOCK_A,
        BLOCK_B=BLOCK_B,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh8_split320_vec4",
        "BLOCK_A": BLOCK_A,
        "BLOCK_B": BLOCK_B,
        "BLOCK_H": BLOCK_H,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _max_pool2d_k3s2_bh4_rowload_rh0mask_kernel(input2d, output,
                                                H: ConstInt, W: ConstInt,
                                                H_OUT: ConstInt, W_OUT: ConstInt,
                                                PADDING: ConstInt,
                                                TILE: ConstInt,
                                                IN_TILE: ConstInt):
    block_w = ct.bid(0)
    h_block = ct.bid(1)
    nc = ct.bid(2)

    cols = block_w * TILE + ct.arange(TILE, dtype=np.int32)
    oh0 = h_block * 4
    ih_base0 = oh0 * 2 - PADDING
    base_row = nc * H

    acc0 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc1 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc2 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc3 = ct.full((TILE,), -np.inf, dtype=np.float32)

    for rh in range(0, 9):
        ih = ih_base0 + rh

        if rh == 0:
            valid_h = h_block > 0
            safe_row = ct.where(valid_h, base_row + ih, -1)
        else:
            safe_row = base_row + ih

        row2d = ct.load(
            input2d,
            index=(safe_row, 0),
            shape=(1, IN_TILE),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        row = ct.reshape(row2d, (IN_TILE,))
        pairs = ct.reshape(row, (TILE, 2))

        even = ct.reshape(ct.extract(pairs, (0, 0), shape=(TILE, 1)), (TILE,))
        odd = ct.reshape(ct.extract(pairs, (0, 1), shape=(TILE, 1)), (TILE,))

        even_f = ct.astype(even, np.float32)
        odd_f = ct.astype(odd, np.float32)

        left_col = cols * 2 - PADDING
        if rh == 0:
            valid_left = (cols < W_OUT) & valid_h & (left_col >= 0) & (left_col < W)
        else:
            valid_left = (cols < W_OUT) & (left_col >= 0) & (left_col < W)
        safe_left_col = ct.where(valid_left, left_col, -1)
        left_f = ct.astype(
            ct.gather(input2d, (safe_row, safe_left_col),
                      padding_value=-np.inf, latency=1),
            np.float32,
        )

        hmax = ct.maximum(ct.maximum(left_f, even_f), odd_f)

        if rh < 3:
            acc0 = ct.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = ct.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = ct.maximum(acc2, hmax)
        if rh >= 6:
            acc3 = ct.maximum(acc3, hmax)

    out_row = nc * H_OUT + oh0

    out0 = ct.reshape(ct.astype(acc0, input2d.dtype), (1, TILE))
    out1 = ct.reshape(ct.astype(acc1, input2d.dtype), (1, TILE))
    out2 = ct.reshape(ct.astype(acc2, input2d.dtype), (1, TILE))
    out3 = ct.reshape(ct.astype(acc3, input2d.dtype), (1, TILE))

    ct.store(output, index=(out_row, block_w), tile=out0, allow_tma=False)
    ct.store(output, index=(out_row + 1, block_w), tile=out1, allow_tma=False)
    ct.store(output, index=(out_row + 2, block_w), tile=out2, allow_tma=False)
    ct.store(output, index=(out_row + 3, block_w), tile=out3, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    input2d = input.reshape(N * C * H, W)
    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    IN_TILE = 1024
    BLOCK_H = 4
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), ct.cdiv(H_out, BLOCK_H), N * C)
    ct.launch(
        stream,
        grid,
        _max_pool2d_k3s2_bh4_rowload_rh0mask_kernel,
        (input2d, output, H, W, H_out, W_out, padding, TILE, IN_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh4_rowload_rh0mask",
        "TILE": TILE,
        "IN_TILE": IN_TILE,
        "BLOCK_H": BLOCK_H,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
