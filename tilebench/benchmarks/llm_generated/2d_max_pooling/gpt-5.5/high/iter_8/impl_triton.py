import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_k3s2_bh8_split320_rh0mask_kernel(input_ptr, output_ptr,
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

    for rh in tl.static_range(0, 17):
        ih = ih_base0 + rh
        row_base = input_ptr + base_nc + ih * W

        if rh == 0:
            load_mask_a = mask_a & valid_rh0
            load_mask_b = mask_b & valid_rh0
        else:
            load_mask_a = mask_a
            load_mask_b = mask_b

        iw0a = cols_a * 2 - PADDING
        iw1a = cols_a * 2
        iw2a = iw1a + 1
        iw0a_safe = tl.where(iw0a >= 0, iw0a, 0)

        va0 = tl.load(row_base + iw0a_safe, mask=load_mask_a, other=neg_inf).to(tl.float32)
        va1 = tl.load(row_base + iw1a, mask=load_mask_a, other=neg_inf).to(tl.float32)
        va2 = tl.load(row_base + iw2a, mask=load_mask_a, other=neg_inf).to(tl.float32)
        va0 = tl.where(iw0a >= 0, va0, neg_inf)
        hmaxa = tl.maximum(tl.maximum(va0, va1), va2)

        iw0b = cols_b * 2 - PADDING
        iw1b = cols_b * 2
        iw2b = iw1b + 1

        vb0 = tl.load(row_base + iw0b, mask=load_mask_b, other=neg_inf).to(tl.float32)
        vb1 = tl.load(row_base + iw1b, mask=load_mask_b, other=neg_inf).to(tl.float32)
        vb2 = tl.load(row_base + iw2b, mask=load_mask_b, other=neg_inf).to(tl.float32)
        hmaxb = tl.maximum(tl.maximum(vb0, vb1), vb2)

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
    _max_pool2d_k3s2_bh8_split320_rh0mask_kernel[grid](
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
        "mode": "k3s2_bh8_split320_rh0mask",
        "BLOCK_A": BLOCK_A,
        "BLOCK_B": BLOCK_B,
        "BLOCK_H": BLOCK_H,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
