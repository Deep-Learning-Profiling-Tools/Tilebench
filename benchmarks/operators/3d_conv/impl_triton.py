import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_DHW": 64,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 64,
    "num_warps": 4,
    "num_stages": 3,
}


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_D, in_H, in_W,
    out_D, out_H, out_W,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    groups,
    out_channels_per_group,
    in_channels_per_group,
    # strides
    stride_input_b, stride_input_c, stride_input_d, stride_input_h, stride_input_w,
    stride_weight_oc, stride_weight_ic, stride_weight_kd, stride_weight_kh, stride_weight_kw,
    stride_output_b, stride_output_c, stride_output_d, stride_output_h, stride_output_w,
    # tile sizes
    BLOCK_SIZE_BATCH_DHW: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
):
    """
    Implicit GEMM Conv3d kernel — the 3D analog of 2d_conv's implicit GEMM.
    Grid: (cdiv(batch*out_D*out_H*out_W, BLOCK_SIZE_BATCH_DHW),
           cdiv(out_channels_per_group, BLOCK_SIZE_OUT_FEAT),
           groups)
    Each block computes a (BLOCK_SIZE_BATCH_DHW x BLOCK_SIZE_OUT_FEAT) output
    tile, contracting over (in_channels_per_group * kD * kH * kW) via tl.dot.
    """
    pid_bdhw = tl.program_id(0)
    pid_oc   = tl.program_id(1)
    group_id = tl.program_id(2)

    # Tile offsets into (batch*out_D*out_H*out_W) and output-channels-per-group
    bdhw_offsets = pid_bdhw * BLOCK_SIZE_BATCH_DHW + tl.arange(0, BLOCK_SIZE_BATCH_DHW)
    oc_offsets   = pid_oc   * BLOCK_SIZE_OUT_FEAT  + tl.arange(0, BLOCK_SIZE_OUT_FEAT)

    # Decode bdhw → (b, od, oh, ow)
    out_HW  = out_H * out_W
    out_DHW = out_D * out_HW
    b_idx   = bdhw_offsets // out_DHW
    dhw_idx = bdhw_offsets %  out_DHW
    od_idx  = dhw_idx // out_HW
    hw_idx  = dhw_idx %  out_HW
    oh_idx  = hw_idx // out_W
    ow_idx  = hw_idx %  out_W

    # Absolute output channel
    oc_abs = group_id * out_channels_per_group + oc_offsets

    # Input channel base for this group
    ic_base = group_id * in_channels_per_group

    acc = tl.zeros((BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)

    # Iterate over (in_channels_per_group * kD * kH * kW) in BLOCK_SIZE_IN_FEAT chunks
    kHW  = kH * kW
    kDHW = kD * kHW
    total_in_feat = in_channels_per_group * kDHW
    for in_feat_start in range(0, tl.cdiv(total_in_feat, BLOCK_SIZE_IN_FEAT)):
        in_feat_offsets = in_feat_start * BLOCK_SIZE_IN_FEAT + tl.arange(0, BLOCK_SIZE_IN_FEAT)

        # Decode in_feat → (ic, kd, kh, kw)
        ic_local = in_feat_offsets // kDHW
        kdhw     = in_feat_offsets %  kDHW
        kd_idx   = kdhw // kHW
        khw      = kdhw %  kHW
        kh_idx   = khw // kW
        kw_idx   = khw %  kW

        # Absolute input channel
        ic_abs = ic_base + ic_local

        # Input spatial positions
        id_idx = od_idx[:, None] * stride_d + kd_idx[None, :] - pad_d
        ih_idx = oh_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw_idx = ow_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        # Masks
        in_feat_mask = in_feat_offsets < total_in_feat
        bdhw_mask    = bdhw_offsets < batch * out_DHW
        valid_d      = (id_idx >= 0) & (id_idx < in_D)
        valid_h      = (ih_idx >= 0) & (ih_idx < in_H)
        valid_w      = (iw_idx >= 0) & (iw_idx < in_W)
        valid_b      = b_idx[:, None] < batch
        in_mask      = bdhw_mask[:, None] & valid_d & valid_h & valid_w & valid_b & in_feat_mask[None, :]

        # Load input tile: (BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_IN_FEAT)
        in_ptrs = (
            input_ptr
            + b_idx[:, None] * stride_input_b
            + ic_abs[None, :] * stride_input_c
            + tl.where(valid_d, id_idx, 0) * stride_input_d
            + tl.where(valid_h, ih_idx, 0) * stride_input_h
            + tl.where(valid_w, iw_idx, 0) * stride_input_w
        )
        in_tile = tl.load(in_ptrs, mask=in_mask, other=0.0)

        # Load weight tile: (BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT)
        # weight layout: (out_channels, in_channels_per_group, kD, kH, kW)
        oc_mask     = oc_abs < out_channels
        weight_mask = in_feat_mask[:, None] & oc_mask[None, :]
        w_ptrs = (
            weight_ptr
            + oc_abs[None, :] * stride_weight_oc
            + ic_local[:, None] * stride_weight_ic
            + kd_idx[:, None] * stride_weight_kd
            + kh_idx[:, None] * stride_weight_kh
            + kw_idx[:, None] * stride_weight_kw
        )
        w_tile = tl.load(w_ptrs, mask=weight_mask, other=0.0)

        # input_precision="tf32" enables TF32 Tensor Cores for fp32 inputs;
        # no-op for fp16, which already uses HMMA with fp32 accumulation.
        # Matches torch (cuDNN TF32) and impl_cutile's tfloat32 cast.
        acc += tl.dot(in_tile, w_tile, input_precision="tf32")

    # Store output tile
    out_mask = (bdhw_offsets < batch * out_DHW)[:, None] & (oc_abs < out_channels)[None, :]
    out_ptrs = (
        output_ptr
        + b_idx[:, None] * stride_output_b
        + oc_abs[None, :] * stride_output_c
        + od_idx[:, None] * stride_output_d
        + oh_idx[:, None] * stride_output_h
        + ow_idx[:, None] * stride_output_w
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


_conv3d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BATCH_DHW": bs_bdhw,
                "BLOCK_SIZE_IN_FEAT": bs_in,
                "BLOCK_SIZE_OUT_FEAT": bs_out,
            },
            num_warps=nw,
            num_stages=ns,
        )
        for bs_bdhw in [32, 64, 128]
        for bs_in   in [16, 32, 64]
        for bs_out  in [64, 128]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
        if bs_bdhw * bs_out >= nw * 256
        and bs_bdhw * bs_out <= 128 * 128
    ],
    key=["batch", "in_channels", "out_channels", "in_D", "in_H", "in_W",
         "kD", "kH", "kW", "groups", "stride_d", "pad_d"],
    warmup=1,
    rep=3,
)(conv3d_kernel)


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs,
):
    """
    Triton Conv3d forward — implicit GEMM.
    input:  (batch, in_channels, D, H, W)
    weight: (out_channels, in_channels // groups, kD, kH, kW)
    """
    assert input.is_contiguous() and weight.is_contiguous()
    batch, in_channels, in_D, in_H, in_W = input.shape
    out_channels, in_channels_per_group, kD, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_D = (in_D + 2 * padding - kD) // stride + 1
    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1
    out_channels_per_group = out_channels // groups

    output = torch.empty((batch, out_channels, out_D, out_H, out_W),
                         device=input.device, dtype=input.dtype)

    total_bdhw = batch * out_D * out_H * out_W

    common_args = dict(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_D=in_D,
        in_H=in_H,
        in_W=in_W,
        out_D=out_D,
        out_H=out_H,
        out_W=out_W,
        kD=kD,
        kH=kH,
        kW=kW,
        stride_d=stride,
        stride_h=stride,
        stride_w=stride,
        pad_d=padding,
        pad_h=padding,
        pad_w=padding,
        groups=groups,
        out_channels_per_group=out_channels_per_group,
        in_channels_per_group=in_channels_per_group,
        stride_input_b=input.stride(0),
        stride_input_c=input.stride(1),
        stride_input_d=input.stride(2),
        stride_input_h=input.stride(3),
        stride_input_w=input.stride(4),
        stride_weight_oc=weight.stride(0),
        stride_weight_ic=weight.stride(1),
        stride_weight_kd=weight.stride(2),
        stride_weight_kh=weight.stride(3),
        stride_weight_kw=weight.stride(4),
        stride_output_b=output.stride(0),
        stride_output_c=output.stride(1),
        stride_output_d=output.stride(2),
        stride_output_h=output.stride(3),
        stride_output_w=output.stride(4),
    )

    if autotune:
        grid = lambda meta: (
            triton.cdiv(total_bdhw, meta["BLOCK_SIZE_BATCH_DHW"]),
            triton.cdiv(out_channels_per_group, meta["BLOCK_SIZE_OUT_FEAT"]),
            groups,
        )
        _conv3d_kernel_autotuned[grid](input, weight, output, **common_args)
    else:
        cfg = _DEFAULT_CONFIG
        bs_bdhw = cfg["BLOCK_SIZE_BATCH_DHW"]
        bs_out  = cfg["BLOCK_SIZE_OUT_FEAT"]
        grid = (
            triton.cdiv(total_bdhw, bs_bdhw),
            triton.cdiv(out_channels_per_group, bs_out),
            groups,
        )
        conv3d_kernel[grid](
            input, weight, output,
            **common_args,
            BLOCK_SIZE_BATCH_DHW=bs_bdhw,
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=bs_out,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_conv3d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_BATCH_DHW": cfg.kwargs["BLOCK_SIZE_BATCH_DHW"],
        "BLOCK_SIZE_IN_FEAT":   cfg.kwargs["BLOCK_SIZE_IN_FEAT"],
        "BLOCK_SIZE_OUT_FEAT":  cfg.kwargs["BLOCK_SIZE_OUT_FEAT"],
        "num_warps":            cfg.num_warps,
        "num_stages":           cfg.num_stages,
    }
