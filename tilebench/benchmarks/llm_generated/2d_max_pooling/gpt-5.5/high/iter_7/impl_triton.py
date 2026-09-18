import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_k3s2_bh8_row640_kernel(input_ptr, output_ptr,
                                        H: tl.constexpr, W: tl.constexpr,
                                        H_OUT: tl.constexpr, W_OUT: tl.constexpr,
                                        PADDING: tl.constexpr,
                                        BLOCK_G0: tl.constexpr,
                                        BLOCK_G1: tl.constexpr,
                                        BLOCK_G2: tl.constexpr,
                                        IN0: tl.constexpr,
                                        IN1: tl.constexpr,
                                        IN2: tl.constexpr,
                                        IS_TOP: tl.constexpr):
    nc = tl.program_id(1)

    tl.static_assert(PADDING == 1)
    tl.static_assert(BLOCK_G0 == 128)
    tl.static_assert(BLOCK_G1 == 128)
    tl.static_assert(BLOCK_G2 == 64)
    tl.static_assert(IN0 == 256)
    tl.static_assert(IN1 == 256)
    tl.static_assert(IN2 == 128)

    if IS_TOP:
        h_block = 0
    else:
        h_block = tl.program_id(0) + 1

    oh0 = h_block * 8
    ih_base0 = oh0 * 2 - PADDING
    base_nc = nc * H * W
    neg_inf = -float("inf")

    offs0 = tl.arange(0, BLOCK_G0)
    offs1 = tl.arange(0, BLOCK_G1)
    offs2 = tl.arange(0, BLOCK_G2)

    cols0 = offs0
    cols1 = BLOCK_G0 + offs1
    cols2 = BLOCK_G0 + BLOCK_G1 + offs2

    mask0 = cols0 < W_OUT
    mask1 = cols1 < W_OUT
    mask2 = cols2 < W_OUT

    in0 = tl.arange(0, IN0)
    in1 = tl.arange(0, IN1)
    in2 = tl.arange(0, IN2)

    acc00 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc01 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc02 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc03 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc04 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc05 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc06 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)
    acc07 = tl.full((BLOCK_G0,), neg_inf, dtype=tl.float32)

    acc10 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc11 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc12 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc13 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc14 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc15 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc16 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)
    acc17 = tl.full((BLOCK_G1,), neg_inf, dtype=tl.float32)

    acc20 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc21 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc22 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc23 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc24 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc25 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc26 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)
    acc27 = tl.full((BLOCK_G2,), neg_inf, dtype=tl.float32)

    idx0_left = offs0 * 2 - 1
    idx0_left_safe = tl.maximum(idx0_left, 0)
    idx1_left = offs1 * 2 - 1
    idx1_left_safe = tl.maximum(idx1_left, 0)
    idx2_left = offs2 * 2 - 1
    idx2_left_safe = tl.maximum(idx2_left, 0)

    for rh in tl.static_range(0, 17):
        if (not IS_TOP) or (rh > 0):
            ih = ih_base0 + rh
            row_base = input_ptr + base_nc + ih * W

            row0 = tl.load(
                row_base + in0,
                mask=in0 < W,
                other=neg_inf,
                eviction_policy="evict_first",
            ).to(tl.float32)
            pair0 = tl.reshape(row0, (BLOCK_G0, 2))
            even0, odd0 = tl.split(pair0)
            left0 = tl.gather(row0, idx0_left_safe, 0)
            left0 = tl.where(offs0 > 0, left0, neg_inf)
            hmax0 = tl.maximum(tl.maximum(left0, even0), odd0)

            row1 = tl.load(
                row_base + IN0 + in1,
                mask=(IN0 + in1) < W,
                other=neg_inf,
                eviction_policy="evict_first",
            ).to(tl.float32)
            pair1 = tl.reshape(row1, (BLOCK_G1, 2))
            even1, odd1 = tl.split(pair1)
            prev0 = tl.load(row_base + (IN0 - 1), mask=(IN0 - 1) < W, other=neg_inf).to(tl.float32)
            left1_local = tl.gather(row1, idx1_left_safe, 0)
            left1 = tl.where(offs1 == 0, prev0, left1_local)
            hmax1 = tl.maximum(tl.maximum(left1, even1), odd1)

            row2 = tl.load(
                row_base + IN0 + IN1 + in2,
                mask=(IN0 + IN1 + in2) < W,
                other=neg_inf,
                eviction_policy="evict_first",
            ).to(tl.float32)
            pair2 = tl.reshape(row2, (BLOCK_G2, 2))
            even2, odd2 = tl.split(pair2)
            prev1 = tl.load(row_base + (IN0 + IN1 - 1), mask=(IN0 + IN1 - 1) < W, other=neg_inf).to(tl.float32)
            left2_local = tl.gather(row2, idx2_left_safe, 0)
            left2 = tl.where(offs2 == 0, prev1, left2_local)
            hmax2 = tl.maximum(tl.maximum(left2, even2), odd2)

            if rh < 3:
                acc00 = tl.maximum(acc00, hmax0)
                acc10 = tl.maximum(acc10, hmax1)
                acc20 = tl.maximum(acc20, hmax2)
            if (rh >= 2) and (rh < 5):
                acc01 = tl.maximum(acc01, hmax0)
                acc11 = tl.maximum(acc11, hmax1)
                acc21 = tl.maximum(acc21, hmax2)
            if (rh >= 4) and (rh < 7):
                acc02 = tl.maximum(acc02, hmax0)
                acc12 = tl.maximum(acc12, hmax1)
                acc22 = tl.maximum(acc22, hmax2)
            if (rh >= 6) and (rh < 9):
                acc03 = tl.maximum(acc03, hmax0)
                acc13 = tl.maximum(acc13, hmax1)
                acc23 = tl.maximum(acc23, hmax2)
            if (rh >= 8) and (rh < 11):
                acc04 = tl.maximum(acc04, hmax0)
                acc14 = tl.maximum(acc14, hmax1)
                acc24 = tl.maximum(acc24, hmax2)
            if (rh >= 10) and (rh < 13):
                acc05 = tl.maximum(acc05, hmax0)
                acc15 = tl.maximum(acc15, hmax1)
                acc25 = tl.maximum(acc25, hmax2)
            if (rh >= 12) and (rh < 15):
                acc06 = tl.maximum(acc06, hmax0)
                acc16 = tl.maximum(acc16, hmax1)
                acc26 = tl.maximum(acc26, hmax2)
            if rh >= 14:
                acc07 = tl.maximum(acc07, hmax0)
                acc17 = tl.maximum(acc17, hmax1)
                acc27 = tl.maximum(acc27, hmax2)

    out_row_base = nc * H_OUT
    oh1 = oh0 + 1
    oh2 = oh0 + 2
    oh3 = oh0 + 3
    oh4 = oh0 + 4
    oh5 = oh0 + 5
    oh6 = oh0 + 6
    oh7 = oh0 + 7

    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols0, acc00, mask=mask0 & (oh0 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols1, acc10, mask=mask1 & (oh0 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols2, acc20, mask=mask2 & (oh0 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols0, acc01, mask=mask0 & (oh1 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols1, acc11, mask=mask1 & (oh1 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols2, acc21, mask=mask2 & (oh1 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols0, acc02, mask=mask0 & (oh2 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols1, acc12, mask=mask1 & (oh2 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols2, acc22, mask=mask2 & (oh2 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols0, acc03, mask=mask0 & (oh3 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols1, acc13, mask=mask1 & (oh3 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols2, acc23, mask=mask2 & (oh3 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh4) * W_OUT + cols0, acc04, mask=mask0 & (oh4 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh4) * W_OUT + cols1, acc14, mask=mask1 & (oh4 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh4) * W_OUT + cols2, acc24, mask=mask2 & (oh4 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh5) * W_OUT + cols0, acc05, mask=mask0 & (oh5 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh5) * W_OUT + cols1, acc15, mask=mask1 & (oh5 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh5) * W_OUT + cols2, acc25, mask=mask2 & (oh5 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh6) * W_OUT + cols0, acc06, mask=mask0 & (oh6 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh6) * W_OUT + cols1, acc16, mask=mask1 & (oh6 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh6) * W_OUT + cols2, acc26, mask=mask2 & (oh6 < H_OUT))

    tl.store(output_ptr + (out_row_base + oh7) * W_OUT + cols0, acc07, mask=mask0 & (oh7 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh7) * W_OUT + cols1, acc17, mask=mask1 & (oh7 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh7) * W_OUT + cols2, acc27, mask=mask2 & (oh7 < H_OUT))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)

    BLOCK_G0 = 128
    BLOCK_G1 = 128
    BLOCK_G2 = 64
    IN0 = 256
    IN1 = 256
    IN2 = 128
    BLOCK_H = 8
    num_warps = 4
    num_stages = 2

    h_blocks = triton.cdiv(H_out, BLOCK_H)

    _max_pool2d_k3s2_bh8_row640_kernel[(1, N * C)](
        input, output,
        H=H,
        W=W,
        H_OUT=H_out,
        W_OUT=W_out,
        PADDING=padding,
        BLOCK_G0=BLOCK_G0,
        BLOCK_G1=BLOCK_G1,
        BLOCK_G2=BLOCK_G2,
        IN0=IN0,
        IN1=IN1,
        IN2=IN2,
        IS_TOP=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if h_blocks > 1:
        _max_pool2d_k3s2_bh8_row640_kernel[(h_blocks - 1, N * C)](
            input, output,
            H=H,
            W=W,
            H_OUT=H_out,
            W_OUT=W_out,
            PADDING=padding,
            BLOCK_G0=BLOCK_G0,
            BLOCK_G1=BLOCK_G1,
            BLOCK_G2=BLOCK_G2,
            IN0=IN0,
            IN1=IN1,
            IN2=IN2,
            IS_TOP=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh8_row640_top_inner",
        "BLOCK_G0": BLOCK_G0,
        "BLOCK_G1": BLOCK_G1,
        "BLOCK_G2": BLOCK_G2,
        "IN0": IN0,
        "IN1": IN1,
        "IN2": IN2,
        "BLOCK_H": BLOCK_H,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
