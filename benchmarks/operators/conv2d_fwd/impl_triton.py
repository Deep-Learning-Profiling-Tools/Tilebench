import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_HEIGHT_WIDTH": 32,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 32,
    "num_warps": 4,
    "num_stages": 2,
}


@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_H, in_W,
    out_H, out_W,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    groups,
    out_channels_per_group,
    in_channels_per_group,
    # strides
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_oc, stride_weight_ic, stride_weight_kh, stride_weight_kw,
    stride_output_b, stride_output_c, stride_output_h, stride_output_w,
    # tile sizes
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    fp16_flag: tl.constexpr,
    tf32_flag: tl.constexpr,
):
    """
    Implicit GEMM Conv2d kernel.
    Grid: (cdiv(batch*out_H*out_W, BLOCK_SIZE_BATCH_HEIGHT_WIDTH),
           cdiv(out_channels_per_group, BLOCK_SIZE_OUT_FEAT),
           groups)
    Each block computes a (BLOCK_SIZE_BATCH_HEIGHT_WIDTH x BLOCK_SIZE_OUT_FEAT) output tile.
    """
    pid_bhw   = tl.program_id(0)
    pid_oc    = tl.program_id(1)
    group_id  = tl.program_id(2)

    # Tile offsets into (batch*out_H*out_W) and output-channels-per-group
    bhw_offsets = pid_bhw * BLOCK_SIZE_BATCH_HEIGHT_WIDTH + tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH)
    oc_offsets  = pid_oc  * BLOCK_SIZE_OUT_FEAT            + tl.arange(0, BLOCK_SIZE_OUT_FEAT)

    # Decode bhw → (b, oh, ow)
    b_idx  = bhw_offsets // (out_H * out_W)
    hw_idx = bhw_offsets %  (out_H * out_W)
    oh_idx = hw_idx // out_W
    ow_idx = hw_idx %  out_W

    # Absolute output channel
    oc_abs = group_id * out_channels_per_group + oc_offsets

    # Input channel base for this group
    ic_base = group_id * in_channels_per_group

    acc = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)

    # Iterate over (in_channels_per_group * kH * kW) in blocks of BLOCK_SIZE_IN_FEAT
    total_in_feat = in_channels_per_group * kH * kW
    for in_feat_start in range(0, tl.cdiv(total_in_feat, BLOCK_SIZE_IN_FEAT)):
        in_feat_offsets = in_feat_start * BLOCK_SIZE_IN_FEAT + tl.arange(0, BLOCK_SIZE_IN_FEAT)

        # Decode in_feat → (ic, kh, kw)
        ic_local = in_feat_offsets // (kH * kW)
        kh_idx   = (in_feat_offsets % (kH * kW)) // kW
        kw_idx   = in_feat_offsets % kW

        # Absolute input channel
        ic_abs = ic_base + ic_local

        # Input spatial positions
        ih_idx = oh_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw_idx = ow_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        # Masks
        in_feat_mask = in_feat_offsets < total_in_feat
        bhw_mask     = bhw_offsets < batch * out_H * out_W
        valid_h      = (ih_idx >= 0) & (ih_idx < in_H)
        valid_w      = (iw_idx >= 0) & (iw_idx < in_W)
        valid_b      = b_idx[:, None] < batch
        in_mask      = bhw_mask[:, None] & valid_h & valid_w & valid_b & in_feat_mask[None, :]

        # Load input tile: (BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT)
        in_ptrs = (
            input_ptr
            + b_idx[:, None] * stride_input_b
            + ic_abs[None, :] * stride_input_c
            + tl.where(valid_h, ih_idx, 0) * stride_input_h
            + tl.where(valid_w, iw_idx, 0) * stride_input_w
        )
        in_tile = tl.load(in_ptrs, mask=in_mask, other=0.0)

        # Load weight tile: (BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT)
        # weight layout: (out_channels, in_channels_per_group, kH, kW)
        oc_mask    = oc_abs < out_channels
        weight_mask = in_feat_mask[:, None] & oc_mask[None, :]
        w_ptrs = (
            weight_ptr
            + oc_abs[None, :] * stride_weight_oc
            + ic_local[:, None] * stride_weight_ic
            + kh_idx[:, None] * stride_weight_kh
            + kw_idx[:, None] * stride_weight_kw
        )
        w_tile = tl.load(w_ptrs, mask=weight_mask, other=0.0)

        if fp16_flag:
            in_tile = in_tile.to(tl.float16)
            w_tile  = w_tile.to(tl.float16)
            acc += tl.dot(in_tile, w_tile, allow_tf32=False)
        elif tf32_flag:
            in_tile = in_tile.to(tl.float32)
            w_tile  = w_tile.to(tl.float32)
            acc += tl.dot(in_tile, w_tile, allow_tf32=True)
        else:
            acc += tl.dot(in_tile.to(tl.float32), w_tile.to(tl.float32), allow_tf32=False)

    # Store output tile
    out_mask = (bhw_offsets < batch * out_H * out_W)[:, None] & (oc_abs < out_channels)[None, :]
    out_ptrs = (
        output_ptr
        + b_idx[:, None] * stride_output_b
        + oc_abs[None, :] * stride_output_c
        + oh_idx[:, None] * stride_output_h
        + ow_idx[:, None] * stride_output_w
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


_conv2d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BATCH_HEIGHT_WIDTH": bs_bhw,
                "BLOCK_SIZE_IN_FEAT": bs_in,
                "BLOCK_SIZE_OUT_FEAT": bs_out,
            },
            num_warps=nw,
            num_stages=ns,
        )
        for bs_bhw in [16, 32, 64]
        for bs_in  in [16, 32]
        for bs_out in [32, 64]
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["batch", "in_channels", "out_channels", "in_H", "in_W", "kH", "kW"],
)(_conv2d_kernel)


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
    Triton Conv2d forward — implicit GEMM.
    input:  (batch, in_channels, H, W)
    weight: (out_channels, in_channels // groups, kH, kW)
    """
    assert input.is_contiguous() and weight.is_contiguous()
    batch, in_channels, in_H, in_W = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1
    out_channels_per_group = out_channels // groups

    output = torch.empty((batch, out_channels, out_H, out_W), device=input.device, dtype=input.dtype)

    fp16_flag = input.dtype == torch.float16
    tf32_flag = input.dtype == torch.float32

    total_bhw = batch * out_H * out_W

    common_args = dict(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_H=in_H,
        in_W=in_W,
        out_H=out_H,
        out_W=out_W,
        kH=kH,
        kW=kW,
        stride_h=stride,
        stride_w=stride,
        pad_h=padding,
        pad_w=padding,
        groups=groups,
        out_channels_per_group=out_channels_per_group,
        in_channels_per_group=in_channels_per_group,
        stride_input_b=input.stride(0),
        stride_input_c=input.stride(1),
        stride_input_h=input.stride(2),
        stride_input_w=input.stride(3),
        stride_weight_oc=weight.stride(0),
        stride_weight_ic=weight.stride(1),
        stride_weight_kh=weight.stride(2),
        stride_weight_kw=weight.stride(3),
        stride_output_b=output.stride(0),
        stride_output_c=output.stride(1),
        stride_output_h=output.stride(2),
        stride_output_w=output.stride(3),
        fp16_flag=fp16_flag,
        tf32_flag=tf32_flag,
    )

    if autotune:
        grid = lambda meta: (
            triton.cdiv(total_bhw, meta["BLOCK_SIZE_BATCH_HEIGHT_WIDTH"]),
            triton.cdiv(out_channels_per_group, meta["BLOCK_SIZE_OUT_FEAT"]),
            groups,
        )
        _conv2d_kernel_autotuned[grid](input, weight, output, **common_args)
    else:
        cfg = _DEFAULT_CONFIG
        bs_bhw = cfg["BLOCK_SIZE_BATCH_HEIGHT_WIDTH"]
        bs_out = cfg["BLOCK_SIZE_OUT_FEAT"]
        grid = (
            triton.cdiv(total_bhw, bs_bhw),
            triton.cdiv(out_channels_per_group, bs_out),
            groups,
        )
        _conv2d_kernel[grid](
            input, weight, output,
            **common_args,
            BLOCK_SIZE_BATCH_HEIGHT_WIDTH=bs_bhw,
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=bs_out,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_conv2d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_BATCH_HEIGHT_WIDTH": cfg.kwargs["BLOCK_SIZE_BATCH_HEIGHT_WIDTH"],
        "BLOCK_SIZE_IN_FEAT":            cfg.kwargs["BLOCK_SIZE_IN_FEAT"],
        "BLOCK_SIZE_OUT_FEAT":           cfg.kwargs["BLOCK_SIZE_OUT_FEAT"],
        "num_warps":                     cfg.num_warps,
        "num_stages":                    cfg.num_stages,
    }
