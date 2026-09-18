```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_implicit_gemm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    w_s_o,
    w_s_i,
    w_s_h,
    w_s_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_POS: tl.constexpr,
    BLOCK_C: tl.constexpr,
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
    ow = offs_pos % OUT_W
    oh = offs_pos // OUT_W

    offs_c_base = tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_OC, BLOCK_POS), dtype=tl.float32)

    for c_start in tl.range(0, CIN_G, BLOCK_C, num_stages=LOOP_STAGES):
        offs_c_g = c_start + offs_c_base
        offs_ic = group * CIN_G + offs_c_g

        for kh in tl.static_range(0, KH):
            ih = oh[None, :] * STRIDE + kh - PADDING
            ih_valid = (ih >= 0) & (ih < H)

            for kw in tl.static_range(0, KW):
                iw = ow[None, :] * STRIDE + kw - PADDING
                iw_valid = (iw >= 0) & (iw < W)

                w_ptrs = (
                    weight_ptr
                    + offs_oc[:, None] * w_s_o
                    + offs_c_g[None, :] * w_s_i
                    + kh * w_s_h
                    + kw * w_s_w
                )
                w_mask = (offs_oc_g[:, None] < COUT_G) & (offs_c_g[None, :] < CIN_G)
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
                    (offs_c_g[:, None] < CIN_G)
                    & (offs_pos[None, :] < spatial)
                    & ih_valid
                    & iw_valid
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
    output_flat = output.reshape(batch, out_channels, spatial)

    BLOCK_OC = 128
    BLOCK_POS = 128
    BLOCK_C = 64
    num_warps = 8
    num_stages = 2
    input_precision = "tf32x3"

    cin_g = in_channels // groups
    cout_g = out_channels // groups

    grid = (
        batch
        * groups
        * triton.cdiv(cout_g, BLOCK_OC)
        * triton.cdiv(spatial, BLOCK_POS),
    )

    _conv2d_implicit_gemm_kernel[grid](
        input,
        weight,
        output_flat,
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
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        BLOCK_OC=BLOCK_OC,
        BLOCK_POS=BLOCK_POS,
        BLOCK_C=BLOCK_C,
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
            "BLOCK_OC": BLOCK_OC,
            "BLOCK_POS": BLOCK_POS,
            "BLOCK_C": BLOCK_C,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "input_precision": input_precision,
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

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=1)
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

    weight_flat = weight.reshape(out_channels, -1)
    output_flat = output.reshape(batch, out_channels, spatial)

    TO = 128
    TP = 128
    TK = 64
    occupancy = 1

    cout_g = out_channels // groups
    grid = (
        batch
        * groups
        * ct.cdiv(cout_g, TO)
        * ct.cdiv(spatial, TP),
        1,
        1,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _conv2d_implicit_gemm_kernel,
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
            "TO": TO,
            "TP": TP,
            "TK": TK,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
