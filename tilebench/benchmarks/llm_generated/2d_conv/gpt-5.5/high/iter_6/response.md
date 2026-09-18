```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_implicit_gemm_kernel(
    input_ptr,
    weight_flat_ptr,
    output_ptr,
    K_TOTAL,
    spatial,
    H,
    W,
    OUT_W,
    COUT,
    CIN_G,
    COUT_G,
    in_s_b,
    in_s_c,
    in_s_h,
    in_s_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_POS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    GROUPS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pos_tiles = tl.cdiv(spatial, BLOCK_POS)
    num_oc_tiles = tl.cdiv(COUT_G, BLOCK_OC)
    tiles_per_bg = num_pos_tiles * num_oc_tiles

    bg = pid // tiles_per_bg
    rem = pid - bg * tiles_per_bg

    batch = bg // GROUPS
    group = bg - batch * GROUPS

    pid_oc = rem // num_pos_tiles
    pid_pos = rem - pid_oc * num_pos_tiles

    offs_oc_g = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_oc = group * COUT_G + offs_oc_g

    offs_pos = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)
    offs_pos = tl.max_contiguous(tl.multiple_of(offs_pos, BLOCK_POS), BLOCK_POS)

    ow = offs_pos % OUT_W
    oh = offs_pos // OUT_W

    offs_k_base = tl.arange(0, BLOCK_K)
    k_hw: tl.constexpr = KH * KW

    acc = tl.zeros((BLOCK_OC, BLOCK_POS), dtype=tl.float32)

    for k_start in tl.range(0, K_TOTAL, BLOCK_K, num_stages=LOOP_STAGES):
        offs_k = k_start + offs_k_base

        ic_g = offs_k // k_hw
        rem_k = offs_k - ic_g * k_hw
        kh = rem_k // KW
        kw = rem_k - kh * KW

        ih = oh[None, :] * STRIDE + kh[:, None] - PADDING
        iw = ow[None, :] * STRIDE + kw[:, None] - PADDING
        offs_ic = group * CIN_G + ic_g

        w_ptrs = weight_flat_ptr + offs_oc[:, None] * K_TOTAL + offs_k[None, :]
        w_mask = (offs_oc_g[:, None] < COUT_G) & (offs_k[None, :] < K_TOTAL)
        w_tile = tl.load(
            w_ptrs,
            mask=w_mask,
            other=0.0,
            eviction_policy="evict_last",
        )

        x_ptrs = (
            input_ptr
            + batch * in_s_b
            + offs_ic[:, None] * in_s_c
            + ih * in_s_h
            + iw * in_s_w
        )
        x_mask = (
            (offs_k[:, None] < K_TOTAL)
            & (offs_pos[None, :] < spatial)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W)
        )
        x_tile = tl.load(
            x_ptrs,
            mask=x_mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        acc = tl.dot(w_tile, x_tile, acc, input_precision=INPUT_PRECISION)

    out_ptrs = (
        output_ptr
        + batch * COUT * spatial
        + offs_oc[:, None] * spatial
        + offs_pos[None, :]
    )
    out_mask = (offs_oc_g[:, None] < COUT_G) & (offs_pos[None, :] < spatial)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _conv2d_direct_fp32_kernel(
    input_ptr,
    weight_flat_ptr,
    output_ptr,
    K_TOTAL,
    spatial,
    H,
    W,
    OUT_W,
    COUT,
    CIN_G,
    COUT_G,
    in_s_b,
    in_s_c,
    in_s_h,
    in_s_w,
    BLOCK_POS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_pos = tl.program_id(0)
    oc = tl.program_id(1)
    batch = tl.program_id(2)

    group = oc // COUT_G
    k_hw: tl.constexpr = KH * KW

    offs_pos = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)
    offs_pos = tl.max_contiguous(tl.multiple_of(offs_pos, BLOCK_POS), BLOCK_POS)

    ow = offs_pos % OUT_W
    oh = offs_pos // OUT_W

    acc = tl.zeros((BLOCK_POS,), dtype=tl.float32)

    for ic_g in tl.range(
        0,
        CIN_G,
        1,
        num_stages=LOOP_STAGES,
        disallow_acc_multi_buffer=True,
    ):
        ic = group * CIN_G + ic_g

        for kh_i in tl.static_range(0, KH):
            ih = oh * STRIDE + kh_i - PADDING
            h_ok = (ih >= 0) & (ih < H)

            for kw_i in tl.static_range(0, KW):
                iw = ow * STRIDE + kw_i - PADDING
                k_idx = ic_g * k_hw + kh_i * KW + kw_i

                x_ptrs = (
                    input_ptr
                    + batch * in_s_b
                    + ic * in_s_c
                    + ih * in_s_h
                    + iw * in_s_w
                )
                x_mask = (offs_pos < spatial) & h_ok & (iw >= 0) & (iw < W)
                x_val = tl.load(
                    x_ptrs,
                    mask=x_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

                w_val = tl.load(
                    weight_flat_ptr + oc * K_TOTAL + k_idx,
                    mask=k_idx < K_TOTAL,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)

                acc = tl.fma(x_val, w_val, acc)

    out_ptrs = output_ptr + batch * COUT * spatial + oc * spatial + offs_pos
    tl.store(out_ptrs, acc, mask=offs_pos < spatial)


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    batch = input.shape[0]
    in_channels = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    out_channels = weight.shape[0]
    KH = weight.shape[2]
    KW = weight.shape[3]

    out_H = (H + 2 * padding - KH) // stride + 1
    out_W = (W + 2 * padding - KW) // stride + 1
    spatial = out_H * out_W

    output = torch.empty(
        (batch, out_channels, out_H, out_W),
        device=input.device,
        dtype=input.dtype,
    )

    cin_g = in_channels // groups
    cout_g = out_channels // groups
    k_total = cin_g * KH * KW

    weight_flat = weight.reshape(out_channels, k_total)
    output_flat = output.reshape(batch, out_channels, spatial)

    BLOCK_OC = 128
    BLOCK_POS = 128
    BLOCK_K = 64
    DIRECT_BLOCK_POS = 256
    num_warps = 8
    num_stages = 3
    direct_num_stages = 1
    input_precision = "tf32"

    if input.dtype == torch.float32:
        grid = (
            triton.cdiv(spatial, DIRECT_BLOCK_POS),
            out_channels,
            batch,
        )
        _conv2d_direct_fp32_kernel[grid](
            input,
            weight_flat,
            output_flat,
            k_total,
            spatial,
            H,
            W,
            out_W,
            out_channels,
            cin_g,
            cout_g,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            BLOCK_POS=DIRECT_BLOCK_POS,
            KH=KH,
            KW=KW,
            STRIDE=stride,
            PADDING=padding,
            LOOP_STAGES=direct_num_stages,
            num_warps=num_warps,
            num_stages=direct_num_stages,
        )
    else:
        grid = (
            batch
            * groups
            * triton.cdiv(cout_g, BLOCK_OC)
            * triton.cdiv(spatial, BLOCK_POS),
        )

        _conv2d_implicit_gemm_kernel[grid](
            input,
            weight_flat,
            output_flat,
            k_total,
            spatial,
            H,
            W,
            out_W,
            out_channels,
            cin_g,
            cout_g,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            BLOCK_OC=BLOCK_OC,
            BLOCK_POS=BLOCK_POS,
            BLOCK_K=BLOCK_K,
            KH=KH,
            KW=KW,
            STRIDE=stride,
            PADDING=padding,
            GROUPS=groups,
            LOOP_STAGES=num_stages,
            INPUT_PRECISION=input_precision,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "mode": "fp16_implicit_gemm_mma_fp32_ordered_direct_fma",
            "BLOCK_OC": BLOCK_OC,
            "BLOCK_POS": BLOCK_POS,
            "BLOCK_K": BLOCK_K,
            "DIRECT_BLOCK_POS": DIRECT_BLOCK_POS,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "direct_num_stages": direct_num_stages,
            "input_precision": input_precision,
            "tf32_reference_disabled": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _conv2d_implicit_gemm_kernel(
    inp,
    w_flat,
    out3,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE: ConstInt,
    PADDING: ConstInt,
    GROUPS: ConstInt,
    TO: ConstInt,
    TP: ConstInt,
    TK: ConstInt,
):
    pid = ct.bid(0)

    CIN = inp.shape[1]
    H = inp.shape[2]
    W = inp.shape[3]
    COUT = out3.shape[1]
    SPATIAL = out3.shape[2]

    OW = (W + 2 * PADDING - KW) // STRIDE + 1
    CIN_G = CIN // GROUPS
    COUT_G = COUT // GROUPS
    K_TOTAL = CIN_G * KH * KW
    KHW = KH * KW

    num_pos_tiles = ct.cdiv(SPATIAL, TP)
    num_oc_tiles = ct.cdiv(COUT_G, TO)
    tiles_per_bg = num_pos_tiles * num_oc_tiles

    bg = pid // tiles_per_bg
    rem = pid - bg * tiles_per_bg

    batch = bg // GROUPS
    group = bg - batch * GROUPS

    pid_oc = rem // num_pos_tiles
    pid_pos = rem - pid_oc * num_pos_tiles

    acc = ct.full((TO, TP), 0.0, dtype=np.float32)

    offs_o_col = ct.arange(TO, dtype=np.int32)[:, None]
    offs_p_row = ct.arange(TP, dtype=np.int32)[None, :]
    offs_k_col = ct.arange(TK, dtype=np.int32)[:, None]
    offs_k_row = ct.arange(TK, dtype=np.int32)[None, :]

    pos = pid_pos * TP + offs_p_row
    ow = pos - (pos // OW) * OW
    oh = pos // OW

    for kt in range(0, ct.cdiv(K_TOTAL, TK)):
        if (COUT_G % TO) == 0:
            row_tile = group * (COUT_G // TO) + pid_oc
            w_tile = ct.load(
                w_flat,
                index=(row_tile, kt),
                shape=(TO, TK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=2,
            )
        else:
            oc_w = group * COUT_G + pid_oc * TO + offs_o_col
            kk_w = kt * TK + offs_k_row
            w_tile = ct.gather(
                w_flat,
                (oc_w, kk_w),
                padding_value=0.0,
                check_bounds=True,
                latency=2,
            )
            w_valid = (oc_w < (group + 1) * COUT_G) & (kk_w < K_TOTAL)
            w_tile = ct.where(w_valid, w_tile, 0.0)

        kk = kt * TK + offs_k_col
        ic_g = kk // KHW
        rem_k = kk - ic_g * KHW
        kh = rem_k // KW
        kw = rem_k - kh * KW

        ih = oh * STRIDE + kh - PADDING
        iw = ow * STRIDE + kw - PADDING
        ic = group * CIN_G + ic_g

        x_tile = ct.gather(
            inp,
            (batch, ic, ih, iw),
            padding_value=0.0,
            check_bounds=True,
            latency=1,
        )
        x_valid = (kk < K_TOTAL) & (pos < SPATIAL)
        x_tile = ct.where(x_valid, x_tile, 0.0)

        acc = ct.mma(w_tile, x_tile, acc)

    out_tile = ct.astype(acc, inp.dtype)

    if (COUT_G % TO) == 0:
        row_tile = group * (COUT_G // TO) + pid_oc
        ct.store(
            out3,
            index=(batch, row_tile, pid_pos),
            tile=out_tile.reshape((1, TO, TP)),
            latency=1,
        )
    else:
        oc = group * COUT_G + pid_oc * TO + offs_o_col
        ct.scatter(
            out3,
            (batch, oc, pos),
            out_tile,
            check_bounds=True,
            latency=1,
        )


@ct.kernel
def _conv2d_direct_fp32_kernel(
    inp,
    w_flat,
    out3,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE: ConstInt,
    PADDING: ConstInt,
    GROUPS: ConstInt,
    TP: ConstInt,
):
    pid_pos = ct.bid(0)
    oc = ct.bid(1)
    batch = ct.bid(2)

    CIN = inp.shape[1]
    H = inp.shape[2]
    W = inp.shape[3]
    COUT = out3.shape[1]
    SPATIAL = out3.shape[2]

    OW = (W + 2 * PADDING - KW) // STRIDE + 1
    CIN_G = CIN // GROUPS
    COUT_G = COUT // GROUPS
    KHW = KH * KW

    group = oc // COUT_G

    offs = ct.arange(TP, dtype=np.int32)
    pos = pid_pos * TP + offs
    ow = pos - (pos // OW) * OW
    oh = pos // OW

    acc = ct.full((TP,), 0.0, dtype=np.float32)

    for ic_g in range(0, CIN_G):
        ic = group * CIN_G + ic_g

        for kh in range(0, KH):
            ih = oh * STRIDE + kh - PADDING

            for kw in range(0, KW):
                iw = ow * STRIDE + kw - PADDING
                k_idx = ic_g * KHW + kh * KW + kw

                x_val = ct.gather(
                    inp,
                    (batch, ic, ih, iw),
                    padding_value=0.0,
                    check_bounds=True,
                    latency=1,
                )
                w_val = ct.load(
                    w_flat,
                    index=(oc, k_idx),
                    shape=(),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=2,
                )

                acc = acc + ct.astype(x_val, np.float32) * ct.astype(w_val, np.float32)

    out_tile = ct.astype(acc, inp.dtype)
    ct.scatter(
        out3,
        (batch, oc, pos),
        out_tile,
        check_bounds=True,
        latency=1,
    )


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    batch = input.shape[0]
    in_channels = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    out_channels = weight.shape[0]
    KH = weight.shape[2]
    KW = weight.shape[3]

    out_H = (H + 2 * padding - KH) // stride + 1
    out_W = (W + 2 * padding - KW) // stride + 1
    spatial = out_H * out_W

    output = torch.empty(
        (batch, out_channels, out_H, out_W),
        device=input.device,
        dtype=input.dtype,
    )

    cin_g = in_channels // groups
    k_total = cin_g * KH * KW

    weight_flat = weight.reshape(out_channels, k_total)
    output_flat = output.reshape(batch, out_channels, spatial)

    TO = 128
    TP = 128
    TK = 64
    DIRECT_TP = 256
    occupancy = 1

    stream = torch.cuda.current_stream()

    if input.dtype == torch.float32:
        grid = (
            ct.cdiv(spatial, DIRECT_TP),
            out_channels,
            batch,
        )
        kernel = _conv2d_direct_fp32_kernel.with_hints(occupancy=occupancy)
        ct.launch(
            stream,
            grid,
            kernel,
            (
                input,
                weight_flat,
                output_flat,
                KH,
                KW,
                stride,
                padding,
                groups,
                DIRECT_TP,
            ),
        )
    else:
        grid = (
            batch
            * groups
            * ct.cdiv(out_channels // groups, TO)
            * ct.cdiv(spatial, TP),
            1,
            1,
        )
        kernel = _conv2d_implicit_gemm_kernel.with_hints(occupancy=occupancy)
        ct.launch(
            stream,
            grid,
            kernel,
            (
                input,
                weight_flat,
                output_flat,
                KH,
                KW,
                stride,
                padding,
                groups,
                TO,
                TP,
                TK,
            ),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "mode": "fp16_implicit_gemm_mma_fp32_ordered_direct_fma",
            "TO": TO,
            "TP": TP,
            "TK": TK,
            "DIRECT_TP": DIRECT_TP,
            "occupancy": occupancy,
            "fp32_precision": "ordered_direct_fma",
            "tf32_reference_disabled": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
