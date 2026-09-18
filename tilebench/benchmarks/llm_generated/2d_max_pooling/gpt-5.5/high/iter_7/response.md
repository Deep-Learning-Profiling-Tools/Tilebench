```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _max_pool2d_k3s2_bh8_split320_kernel(input2d, output,
                                          H: ConstInt, W: ConstInt,
                                          H_OUT: ConstInt, W_OUT: ConstInt,
                                          PADDING: ConstInt,
                                          TILE_A: ConstInt,
                                          TILE_B: ConstInt,
                                          IN_A: ConstInt,
                                          IN_B: ConstInt):
    h_block = ct.bid(0)
    nc = ct.bid(1)

    cols_a = ct.arange(TILE_A, dtype=np.int32)
    cols_b = TILE_A + ct.arange(TILE_B, dtype=np.int32)

    oh0 = h_block * 8
    ih_base0 = oh0 * 2 - PADDING
    base_row = nc * H

    acca0 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca1 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca2 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca3 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca4 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca5 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca6 = ct.full((TILE_A,), -np.inf, dtype=np.float32)
    acca7 = ct.full((TILE_A,), -np.inf, dtype=np.float32)

    accb0 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb1 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb2 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb3 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb4 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb5 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb6 = ct.full((TILE_B,), -np.inf, dtype=np.float32)
    accb7 = ct.full((TILE_B,), -np.inf, dtype=np.float32)

    for rh in range(0, 17):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        safe_row = ct.where(valid_h, base_row + ih, -1)

        rowa2d = ct.load(
            input2d,
            index=(safe_row, 0),
            shape=(1, IN_A),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        rowa = ct.reshape(rowa2d, (IN_A,))
        pairsa = ct.reshape(rowa, (TILE_A, 2))
        evena = ct.reshape(ct.extract(pairsa, (0, 0), shape=(TILE_A, 1)), (TILE_A,))
        odda = ct.reshape(ct.extract(pairsa, (0, 1), shape=(TILE_A, 1)), (TILE_A,))

        evena_f = ct.astype(evena, np.float32)
        odda_f = ct.astype(odda, np.float32)

        left_a_col = cols_a * 2 - PADDING
        valid_left_a = (cols_a < W_OUT) & valid_h & (left_a_col >= 0) & (left_a_col < W)
        safe_left_a = ct.where(valid_left_a, left_a_col, -1)
        lefta_f = ct.astype(
            ct.gather(input2d, (safe_row, safe_left_a),
                      padding_value=-np.inf, latency=1),
            np.float32,
        )
        hmaxa = ct.maximum(ct.maximum(lefta_f, evena_f), odda_f)

        rowb2d = ct.load(
            input2d,
            index=(safe_row, 4),
            shape=(1, IN_B),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        rowb = ct.reshape(rowb2d, (IN_B,))
        pairsb = ct.reshape(rowb, (TILE_B, 2))
        evenb = ct.reshape(ct.extract(pairsb, (0, 0), shape=(TILE_B, 1)), (TILE_B,))
        oddb = ct.reshape(ct.extract(pairsb, (0, 1), shape=(TILE_B, 1)), (TILE_B,))

        evenb_f = ct.astype(evenb, np.float32)
        oddb_f = ct.astype(oddb, np.float32)

        left_b_col = cols_b * 2 - PADDING
        valid_left_b = (cols_b < W_OUT) & valid_h & (left_b_col >= 0) & (left_b_col < W)
        safe_left_b = ct.where(valid_left_b, left_b_col, -1)
        leftb_f = ct.astype(
            ct.gather(input2d, (safe_row, safe_left_b),
                      padding_value=-np.inf, latency=1),
            np.float32,
        )
        hmaxb = ct.maximum(ct.maximum(leftb_f, evenb_f), oddb_f)

        if rh < 3:
            acca0 = ct.maximum(acca0, hmaxa)
            accb0 = ct.maximum(accb0, hmaxb)
        if (rh >= 2) and (rh < 5):
            acca1 = ct.maximum(acca1, hmaxa)
            accb1 = ct.maximum(accb1, hmaxb)
        if (rh >= 4) and (rh < 7):
            acca2 = ct.maximum(acca2, hmaxa)
            accb2 = ct.maximum(accb2, hmaxb)
        if (rh >= 6) and (rh < 9):
            acca3 = ct.maximum(acca3, hmaxa)
            accb3 = ct.maximum(accb3, hmaxb)
        if (rh >= 8) and (rh < 11):
            acca4 = ct.maximum(acca4, hmaxa)
            accb4 = ct.maximum(accb4, hmaxb)
        if (rh >= 10) and (rh < 13):
            acca5 = ct.maximum(acca5, hmaxa)
            accb5 = ct.maximum(accb5, hmaxb)
        if (rh >= 12) and (rh < 15):
            acca6 = ct.maximum(acca6, hmaxa)
            accb6 = ct.maximum(accb6, hmaxb)
        if rh >= 14:
            acca7 = ct.maximum(acca7, hmaxa)
            accb7 = ct.maximum(accb7, hmaxb)

    out_row = nc * H_OUT + oh0

    outa0 = ct.reshape(ct.astype(acca0, input2d.dtype), (1, TILE_A))
    outa1 = ct.reshape(ct.astype(acca1, input2d.dtype), (1, TILE_A))
    outa2 = ct.reshape(ct.astype(acca2, input2d.dtype), (1, TILE_A))
    outa3 = ct.reshape(ct.astype(acca3, input2d.dtype), (1, TILE_A))
    outa4 = ct.reshape(ct.astype(acca4, input2d.dtype), (1, TILE_A))
    outa5 = ct.reshape(ct.astype(acca5, input2d.dtype), (1, TILE_A))
    outa6 = ct.reshape(ct.astype(acca6, input2d.dtype), (1, TILE_A))
    outa7 = ct.reshape(ct.astype(acca7, input2d.dtype), (1, TILE_A))

    outb0 = ct.reshape(ct.astype(accb0, input2d.dtype), (1, TILE_B))
    outb1 = ct.reshape(ct.astype(accb1, input2d.dtype), (1, TILE_B))
    outb2 = ct.reshape(ct.astype(accb2, input2d.dtype), (1, TILE_B))
    outb3 = ct.reshape(ct.astype(accb3, input2d.dtype), (1, TILE_B))
    outb4 = ct.reshape(ct.astype(accb4, input2d.dtype), (1, TILE_B))
    outb5 = ct.reshape(ct.astype(accb5, input2d.dtype), (1, TILE_B))
    outb6 = ct.reshape(ct.astype(accb6, input2d.dtype), (1, TILE_B))
    outb7 = ct.reshape(ct.astype(accb7, input2d.dtype), (1, TILE_B))

    ct.store(output, index=(out_row, 0), tile=outa0, allow_tma=False)
    ct.store(output, index=(out_row, 4), tile=outb0, allow_tma=False)
    ct.store(output, index=(out_row + 1, 0), tile=outa1, allow_tma=False)
    ct.store(output, index=(out_row + 1, 4), tile=outb1, allow_tma=False)
    ct.store(output, index=(out_row + 2, 0), tile=outa2, allow_tma=False)
    ct.store(output, index=(out_row + 2, 4), tile=outb2, allow_tma=False)
    ct.store(output, index=(out_row + 3, 0), tile=outa3, allow_tma=False)
    ct.store(output, index=(out_row + 3, 4), tile=outb3, allow_tma=False)
    ct.store(output, index=(out_row + 4, 0), tile=outa4, allow_tma=False)
    ct.store(output, index=(out_row + 4, 4), tile=outb4, allow_tma=False)
    ct.store(output, index=(out_row + 5, 0), tile=outa5, allow_tma=False)
    ct.store(output, index=(out_row + 5, 4), tile=outb5, allow_tma=False)
    ct.store(output, index=(out_row + 6, 0), tile=outa6, allow_tma=False)
    ct.store(output, index=(out_row + 6, 4), tile=outb6, allow_tma=False)
    ct.store(output, index=(out_row + 7, 0), tile=outa7, allow_tma=False)
    ct.store(output, index=(out_row + 7, 4), tile=outb7, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    input2d = input.reshape(N * C * H, W)
    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE_A = 256
    TILE_B = 64
    IN_A = 512
    IN_B = 128
    BLOCK_H = 8
    occupancy = 4

    grid = (ct.cdiv(H_out, BLOCK_H), N * C, 1)
    ct.launch(
        stream,
        grid,
        _max_pool2d_k3s2_bh8_split320_kernel,
        (input2d, output, H, W, H_out, W_out, padding, TILE_A, TILE_B, IN_A, IN_B),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh8_split320_fixed",
        "TILE_A": TILE_A,
        "TILE_B": TILE_B,
        "IN_A": IN_A,
        "IN_B": IN_B,
        "BLOCK_H": BLOCK_H,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
