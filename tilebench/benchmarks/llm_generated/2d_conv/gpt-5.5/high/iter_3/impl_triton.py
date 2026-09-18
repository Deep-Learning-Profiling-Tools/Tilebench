import torch
import triton
import triton.language as tl

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
    BLOCK_OC: tl.constexpr,
    BLOCK_POS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    GROUPS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
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

    k_hw: tl.constexpr = KH * KW
    acc = tl.zeros((BLOCK_OC, BLOCK_POS), dtype=tl.float32)

    for k in tl.range(0, K_TOTAL, 1, num_stages=LOOP_STAGES):
        ic_g = k // k_hw
        rem_k = k - ic_g * k_hw
        kh = rem_k // KW
        kw = rem_k - kh * KW

        ih = oh * STRIDE + kh - PADDING
        iw = ow * STRIDE + kw - PADDING
        ic = group * CIN_G + ic_g

        w = tl.load(
            weight_flat_ptr + offs_oc * K_TOTAL + k,
            mask=offs_oc_g < COUT_G,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        x = tl.load(
            input_ptr
            + batch * in_s_b
            + ic * in_s_c
            + ih * in_s_h
            + iw * in_s_w,
            mask=(offs_pos < spatial) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W),
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        acc += w[:, None] * x[None, :]

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

    cin_g = in_channels // groups
    cout_g = out_channels // groups
    k_total = cin_g * KH * KW

    weight_flat = weight.reshape(out_channels, k_total)
    output_flat = output.reshape(batch, out_channels, spatial)

    if input.dtype == torch.float32:
        BLOCK_OC = 32
        BLOCK_POS = 128
        num_warps = 4
        num_stages = 2

        grid = (
            batch
            * groups
            * triton.cdiv(cout_g, BLOCK_OC)
            * triton.cdiv(spatial, BLOCK_POS),
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
            BLOCK_OC=BLOCK_OC,
            BLOCK_POS=BLOCK_POS,
            KH=KH,
            KW=KW,
            STRIDE=stride,
            PADDING=padding,
            GROUPS=groups,
            LOOP_STAGES=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "mode": "direct_fp32",
                "BLOCK_OC": BLOCK_OC,
                "BLOCK_POS": BLOCK_POS,
                "BLOCK_K": 1,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )
    else:
        BLOCK_OC = 64
        BLOCK_POS = 256
        BLOCK_K = 64
        num_warps = 8
        num_stages = 3
        input_precision = "tf32x3"

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
                "mode": "implicit_gemm_mma",
                "BLOCK_OC": BLOCK_OC,
                "BLOCK_POS": BLOCK_POS,
                "BLOCK_K": BLOCK_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "input_precision": input_precision,
            }
        )

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
