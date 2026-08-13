from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(block_bdhw=64, block_in=32, block_out=64, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(block_bdhw=bbdhw, block_in=bin_, block_out=bout, occupancy=occ)
    for bbdhw in [32, 64, 128]
    for bin_ in [16, 32, 64]
    for bout in [64, 128]
    for occ in [4, 8, 16, 32]
    if bbdhw * bout >= 256 and bbdhw * bout <= 128 * 128
]


@ct.kernel
def conv3d_kernel(
    input_flat,
    weight_flat,
    output_flat,
    batch, in_channels, out_channels,
    in_D, in_H, in_W,
    out_D, out_H, out_W,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_channels_per_group, in_channels_per_group,

    stride_input_b, stride_input_c, stride_input_d, stride_input_h, stride_input_w,
    stride_weight_oc, stride_weight_ic, stride_weight_kd, stride_weight_kh, stride_weight_kw,
    stride_output_b, stride_output_c, stride_output_d, stride_output_h, stride_output_w,
    BLOCK_BDHW: ConstInt,
    BLOCK_IN: ConstInt,
    BLOCK_OUT: ConstInt,
):
    pid_bdhw = ct.bid(0)
    pid_oc = ct.bid(1)
    group_id = ct.bid(2)

    out_HW = out_H * out_W
    out_DHW = out_D * out_HW
    total_bdhw = batch * out_DHW
    kHW = kH * kW
    kDHW = kD * kHW
    total_in_feat = in_channels_per_group * kDHW


    bdhw_offsets = pid_bdhw * BLOCK_BDHW + ct.arange(BLOCK_BDHW, dtype=ct.int32)
    oc_offsets = pid_oc * BLOCK_OUT + ct.arange(BLOCK_OUT, dtype=ct.int32)


    b_idx = bdhw_offsets // out_DHW
    dhw_idx = bdhw_offsets % out_DHW
    od_idx = dhw_idx // out_HW
    hw_idx = dhw_idx % out_HW
    oh_idx = hw_idx // out_W
    ow_idx = hw_idx % out_W

    oc_abs = group_id * out_channels_per_group + oc_offsets
    ic_base = group_id * in_channels_per_group

    bdhw_mask = bdhw_offsets < total_bdhw
    oc_mask = oc_abs < out_channels

    acc = ct.full((BLOCK_BDHW, BLOCK_OUT), 0.0, dtype=ct.float32)


    mma_dtype = ct.tfloat32 if input_flat.dtype == ct.float32 else input_flat.dtype

    for in_feat_start in range(0, ct.cdiv(total_in_feat, BLOCK_IN)):
        in_feat_offsets = in_feat_start * BLOCK_IN + ct.arange(BLOCK_IN, dtype=ct.int32)


        ic_local = in_feat_offsets // kDHW
        kdhw = in_feat_offsets % kDHW
        kd_idx = kdhw // kHW
        khw = kdhw % kHW
        kh_idx = khw // kW
        kw_idx = khw % kW
        ic_abs = ic_base + ic_local

        in_feat_mask = in_feat_offsets < total_in_feat


        id_idx = ct.expand_dims(od_idx, 1) * stride_d + ct.expand_dims(kd_idx, 0) - pad_d
        ih_idx = ct.expand_dims(oh_idx, 1) * stride_h + ct.expand_dims(kh_idx, 0) - pad_h
        iw_idx = ct.expand_dims(ow_idx, 1) * stride_w + ct.expand_dims(kw_idx, 0) - pad_w

        valid_d = ct.bitwise_and(id_idx >= 0, id_idx < in_D)
        valid_h = ct.bitwise_and(ih_idx >= 0, ih_idx < in_H)
        valid_w = ct.bitwise_and(iw_idx >= 0, iw_idx < in_W)
        valid_b = ct.expand_dims(b_idx, 1) < batch
        in_mask = ct.bitwise_and(
            ct.bitwise_and(ct.expand_dims(bdhw_mask, 1), valid_b),
            ct.bitwise_and(
                ct.bitwise_and(valid_d, ct.bitwise_and(valid_h, valid_w)),
                ct.expand_dims(in_feat_mask, 0),
            ),
        )


        id_safe = ct.where(valid_d, id_idx, 0)
        ih_safe = ct.where(valid_h, ih_idx, 0)
        iw_safe = ct.where(valid_w, iw_idx, 0)
        b_safe = ct.where(valid_b, ct.expand_dims(b_idx, 1), 0)
        ic_abs_2d = ct.expand_dims(ic_abs, 0)


        in_lin = (
            b_safe * stride_input_b
            + ic_abs_2d * stride_input_c
            + id_safe * stride_input_d
            + ih_safe * stride_input_h
            + iw_safe * stride_input_w
        )
        in_tile = ct.gather(input_flat, in_lin, padding_value=0.0)
        in_tile = ct.where(in_mask, in_tile, 0.0)


        oc_abs_2d = ct.expand_dims(oc_abs, 0)
        ic_local_2d = ct.expand_dims(ic_local, 1)
        kd_idx_2d = ct.expand_dims(kd_idx, 1)
        kh_idx_2d = ct.expand_dims(kh_idx, 1)
        kw_idx_2d = ct.expand_dims(kw_idx, 1)
        in_feat_mask_2d = ct.expand_dims(in_feat_mask, 1)
        oc_mask_2d = ct.expand_dims(oc_mask, 0)
        weight_mask = ct.bitwise_and(in_feat_mask_2d, oc_mask_2d)

        w_lin = (
            oc_abs_2d * stride_weight_oc
            + ic_local_2d * stride_weight_ic
            + kd_idx_2d * stride_weight_kd
            + kh_idx_2d * stride_weight_kh
            + kw_idx_2d * stride_weight_kw
        )
        w_tile = ct.gather(weight_flat, w_lin, padding_value=0.0)
        w_tile = ct.where(weight_mask, w_tile, 0.0)

        acc = ct.mma(ct.astype(in_tile, mma_dtype),
                     ct.astype(w_tile, mma_dtype), acc)


    out_mask = ct.bitwise_and(
        ct.expand_dims(bdhw_mask, 1),
        ct.expand_dims(oc_mask, 0),
    )
    b_idx_2d = ct.expand_dims(b_idx, 1)
    od_idx_2d = ct.expand_dims(od_idx, 1)
    oh_idx_2d = ct.expand_dims(oh_idx, 1)
    ow_idx_2d = ct.expand_dims(ow_idx, 1)
    oc_abs_out_2d = ct.expand_dims(oc_abs, 0)

    out_lin = (
        b_idx_2d * stride_output_b
        + oc_abs_out_2d * stride_output_c
        + od_idx_2d * stride_output_d
        + oh_idx_2d * stride_output_h
        + ow_idx_2d * stride_output_w
    )

    out_lin = ct.where(out_mask, out_lin, batch * out_channels * out_DHW)

    acc_cast = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_lin, acc_cast)


_tuner = CutileAutotuner(conv3d_kernel)


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

    input_flat = input.view(-1)
    weight_flat = weight.view(-1)
    output_flat = output.view(-1)

    stream = torch.cuda.current_stream()

    runtime_args = (
        input_flat, weight_flat, output_flat,
        batch, in_channels, out_channels,
        in_D, in_H, in_W,
        out_D, out_H, out_W,
        kD, kH, kW,
        stride, stride, stride,
        padding, padding, padding,
        out_channels_per_group, in_channels_per_group,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3), input.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
    )
    total_bdhw = batch * out_D * out_H * out_W

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(batch, in_channels, out_channels, in_D, in_H, in_W,
                       kD, kH, kW, stride, padding, groups, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                (total_bdhw + cfg.block_bdhw - 1) // cfg.block_bdhw,
                (out_channels_per_group + cfg.block_out - 1) // cfg.block_out,
                groups,
            ),
            args_fn=lambda cfg: runtime_args + (cfg.block_bdhw, cfg.block_in, cfg.block_out),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "block_bdhw": cfg.block_bdhw,
            "block_in":   cfg.block_in,
            "block_out":  cfg.block_out,
            "occupancy":  cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (
        (total_bdhw + cfg.block_bdhw - 1) // cfg.block_bdhw,
        (out_channels_per_group + cfg.block_out - 1) // cfg.block_out,
        groups,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        runtime_args + (cfg.block_bdhw, cfg.block_in, cfg.block_out),
    )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
