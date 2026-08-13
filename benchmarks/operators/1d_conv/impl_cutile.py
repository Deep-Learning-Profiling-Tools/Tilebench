from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(block_bl=64, block_in=32, block_out=64, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(block_bl=bbl, block_in=bin_, block_out=bout, occupancy=occ)
    for bbl in [32, 64, 128]
    for bin_ in [16, 32, 64]
    for bout in [64, 128]
    for occ in [4, 8, 16, 32]
    if bbl * bout >= 256 and bbl * bout <= 128 * 128
]


@ct.kernel
def conv1d_kernel(
    input_flat,
    weight_flat,
    output_flat,
    batch, in_channels, out_channels,
    in_L, out_L,
    kL,
    stride_l, pad_l,
    out_channels_per_group, in_channels_per_group,

    stride_input_b, stride_input_c, stride_input_l,
    stride_weight_oc, stride_weight_ic, stride_weight_kl,
    stride_output_b, stride_output_c, stride_output_l,
    BLOCK_BL: ConstInt,
    BLOCK_IN: ConstInt,
    BLOCK_OUT: ConstInt,
):
    pid_bl = ct.bid(0)
    pid_oc = ct.bid(1)
    group_id = ct.bid(2)

    total_bl = batch * out_L
    total_in_feat = in_channels_per_group * kL


    bl_offsets = pid_bl * BLOCK_BL + ct.arange(BLOCK_BL, dtype=ct.int32)
    oc_offsets = pid_oc * BLOCK_OUT + ct.arange(BLOCK_OUT, dtype=ct.int32)


    b_idx = bl_offsets // out_L
    ol_idx = bl_offsets % out_L

    oc_abs = group_id * out_channels_per_group + oc_offsets
    ic_base = group_id * in_channels_per_group

    bl_mask = bl_offsets < total_bl
    oc_mask = oc_abs < out_channels

    acc = ct.full((BLOCK_BL, BLOCK_OUT), 0.0, dtype=ct.float32)


    mma_dtype = ct.tfloat32 if input_flat.dtype == ct.float32 else input_flat.dtype

    for in_feat_start in range(0, ct.cdiv(total_in_feat, BLOCK_IN)):
        in_feat_offsets = in_feat_start * BLOCK_IN + ct.arange(BLOCK_IN, dtype=ct.int32)


        ic_local = in_feat_offsets // kL
        kl_idx = in_feat_offsets % kL
        ic_abs = ic_base + ic_local

        in_feat_mask = in_feat_offsets < total_in_feat


        il_idx = ct.expand_dims(ol_idx, 1) * stride_l + ct.expand_dims(kl_idx, 0) - pad_l

        valid_l = ct.bitwise_and(il_idx >= 0, il_idx < in_L)
        valid_b = ct.expand_dims(b_idx, 1) < batch
        in_mask = ct.bitwise_and(
            ct.bitwise_and(ct.expand_dims(bl_mask, 1), valid_b),
            ct.bitwise_and(valid_l, ct.expand_dims(in_feat_mask, 0)),
        )


        il_safe = ct.where(valid_l, il_idx, 0)
        b_safe = ct.where(valid_b, ct.expand_dims(b_idx, 1), 0)
        ic_abs_2d = ct.expand_dims(ic_abs, 0)


        in_lin = (
            b_safe * stride_input_b
            + ic_abs_2d * stride_input_c
            + il_safe * stride_input_l
        )
        in_tile = ct.gather(input_flat, in_lin, padding_value=0.0)
        in_tile = ct.where(in_mask, in_tile, 0.0)


        oc_abs_2d = ct.expand_dims(oc_abs, 0)
        ic_local_2d = ct.expand_dims(ic_local, 1)
        kl_idx_2d = ct.expand_dims(kl_idx, 1)
        in_feat_mask_2d = ct.expand_dims(in_feat_mask, 1)
        oc_mask_2d = ct.expand_dims(oc_mask, 0)
        weight_mask = ct.bitwise_and(in_feat_mask_2d, oc_mask_2d)

        w_lin = (
            oc_abs_2d * stride_weight_oc
            + ic_local_2d * stride_weight_ic
            + kl_idx_2d * stride_weight_kl
        )
        w_tile = ct.gather(weight_flat, w_lin, padding_value=0.0)
        w_tile = ct.where(weight_mask, w_tile, 0.0)

        acc = ct.mma(ct.astype(in_tile, mma_dtype),
                     ct.astype(w_tile, mma_dtype), acc)


    out_mask = ct.bitwise_and(
        ct.expand_dims(bl_mask, 1),
        ct.expand_dims(oc_mask, 0),
    )
    b_idx_2d = ct.expand_dims(b_idx, 1)
    ol_idx_2d = ct.expand_dims(ol_idx, 1)
    oc_abs_out_2d = ct.expand_dims(oc_abs, 0)

    out_lin = (
        b_idx_2d * stride_output_b
        + oc_abs_out_2d * stride_output_c
        + ol_idx_2d * stride_output_l
    )

    out_lin = ct.where(out_mask, out_lin, batch * out_channels * out_L)

    acc_cast = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_lin, acc_cast)


_tuner = CutileAutotuner(conv1d_kernel)


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
    batch, in_channels, in_L = input.shape
    out_channels, in_channels_per_group, kL = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_L = (in_L + 2 * padding - kL) // stride + 1
    out_channels_per_group = out_channels // groups

    output = torch.empty((batch, out_channels, out_L),
                         device=input.device, dtype=input.dtype)

    input_flat = input.view(-1)
    weight_flat = weight.view(-1)
    output_flat = output.view(-1)

    stream = torch.cuda.current_stream()

    runtime_args = (
        input_flat, weight_flat, output_flat,
        batch, in_channels, out_channels,
        in_L, out_L,
        kL,
        stride, padding,
        out_channels_per_group, in_channels_per_group,
        input.stride(0), input.stride(1), input.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
    )
    total_bl = batch * out_L

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(batch, in_channels, out_channels, in_L,
                       kL, stride, padding, groups, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                (total_bl + cfg.block_bl - 1) // cfg.block_bl,
                (out_channels_per_group + cfg.block_out - 1) // cfg.block_out,
                groups,
            ),
            args_fn=lambda cfg: runtime_args + (cfg.block_bl, cfg.block_in, cfg.block_out),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "block_bl":  cfg.block_bl,
            "block_in":  cfg.block_in,
            "block_out": cfg.block_out,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (
        (total_bl + cfg.block_bl - 1) // cfg.block_bl,
        (out_channels_per_group + cfg.block_out - 1) // cfg.block_out,
        groups,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        runtime_args + (cfg.block_bl, cfg.block_in, cfg.block_out),
    )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
