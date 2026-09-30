import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_LENGTH": 64,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 64,
    "num_warps": 4,
    "num_stages": 3,
}


@triton.jit
def conv1d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_L, out_L,
    kL,
    stride_l, pad_l,
    groups,
    out_channels_per_group,
    in_channels_per_group,

    stride_input_b, stride_input_c, stride_input_l,
    stride_weight_oc, stride_weight_ic, stride_weight_kl,
    stride_output_b, stride_output_c, stride_output_l,

    BLOCK_SIZE_BATCH_LENGTH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
):
    pid_bl   = tl.program_id(0)
    pid_oc   = tl.program_id(1)
    group_id = tl.program_id(2)


    bl_offsets = pid_bl * BLOCK_SIZE_BATCH_LENGTH + tl.arange(0, BLOCK_SIZE_BATCH_LENGTH)
    oc_offsets = pid_oc * BLOCK_SIZE_OUT_FEAT     + tl.arange(0, BLOCK_SIZE_OUT_FEAT)


    b_idx  = bl_offsets // out_L
    ol_idx = bl_offsets %  out_L


    oc_abs = group_id * out_channels_per_group + oc_offsets


    ic_base = group_id * in_channels_per_group

    acc = tl.zeros((BLOCK_SIZE_BATCH_LENGTH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)


    total_in_feat = in_channels_per_group * kL
    for in_feat_start in range(0, tl.cdiv(total_in_feat, BLOCK_SIZE_IN_FEAT)):
        in_feat_offsets = in_feat_start * BLOCK_SIZE_IN_FEAT + tl.arange(0, BLOCK_SIZE_IN_FEAT)


        ic_local = in_feat_offsets // kL
        kl_idx   = in_feat_offsets %  kL


        ic_abs = ic_base + ic_local


        il_idx = ol_idx[:, None] * stride_l + kl_idx[None, :] - pad_l


        in_feat_mask = in_feat_offsets < total_in_feat
        bl_mask      = bl_offsets < batch * out_L
        valid_l      = (il_idx >= 0) & (il_idx < in_L)
        valid_b      = b_idx[:, None] < batch
        in_mask      = bl_mask[:, None] & valid_l & valid_b & in_feat_mask[None, :]


        in_ptrs = (
            input_ptr
            + b_idx[:, None] * stride_input_b
            + ic_abs[None, :] * stride_input_c
            + tl.where(valid_l, il_idx, 0) * stride_input_l
        )
        in_tile = tl.load(in_ptrs, mask=in_mask, other=0.0)


        oc_mask     = oc_abs < out_channels
        weight_mask = in_feat_mask[:, None] & oc_mask[None, :]
        w_ptrs = (
            weight_ptr
            + oc_abs[None, :] * stride_weight_oc
            + ic_local[:, None] * stride_weight_ic
            + kl_idx[:, None] * stride_weight_kl
        )
        w_tile = tl.load(w_ptrs, mask=weight_mask, other=0.0)


        acc += tl.dot(in_tile, w_tile, input_precision="tf32")


    out_mask = (bl_offsets < batch * out_L)[:, None] & (oc_abs < out_channels)[None, :]
    out_ptrs = (
        output_ptr
        + b_idx[:, None] * stride_output_b
        + oc_abs[None, :] * stride_output_c
        + ol_idx[:, None] * stride_output_l
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


_conv1d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BATCH_LENGTH": bs_bl,
                "BLOCK_SIZE_IN_FEAT": bs_in,
                "BLOCK_SIZE_OUT_FEAT": bs_out,
            },
            num_warps=nw,
            num_stages=ns,
        )
        for bs_bl  in [32, 64, 128]
        for bs_in  in [16, 32, 64]
        for bs_out in [64, 128]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
        if bs_bl * bs_out >= nw * 256
        and bs_bl * bs_out <= 128 * 128
    ],
    key=["batch", "in_channels", "out_channels", "in_L",
         "kL", "groups", "stride_l", "pad_l"],
    warmup=1,
    rep=3,
)(conv1d_kernel)


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

    output = torch.empty((batch, out_channels, out_L), device=input.device, dtype=input.dtype)

    total_bl = batch * out_L

    common_args = dict(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_L=in_L,
        out_L=out_L,
        kL=kL,
        stride_l=stride,
        pad_l=padding,
        groups=groups,
        out_channels_per_group=out_channels_per_group,
        in_channels_per_group=in_channels_per_group,
        stride_input_b=input.stride(0),
        stride_input_c=input.stride(1),
        stride_input_l=input.stride(2),
        stride_weight_oc=weight.stride(0),
        stride_weight_ic=weight.stride(1),
        stride_weight_kl=weight.stride(2),
        stride_output_b=output.stride(0),
        stride_output_c=output.stride(1),
        stride_output_l=output.stride(2),
    )

    if autotune:
        grid = lambda meta: (
            triton.cdiv(total_bl, meta["BLOCK_SIZE_BATCH_LENGTH"]),
            triton.cdiv(out_channels_per_group, meta["BLOCK_SIZE_OUT_FEAT"]),
            groups,
        )
        _conv1d_kernel_autotuned[grid](input, weight, output, **common_args)
    else:
        cfg = _DEFAULT_CONFIG
        bs_bl  = cfg["BLOCK_SIZE_BATCH_LENGTH"]
        bs_out = cfg["BLOCK_SIZE_OUT_FEAT"]
        grid = (
            triton.cdiv(total_bl, bs_bl),
            triton.cdiv(out_channels_per_group, bs_out),
            groups,
        )
        conv1d_kernel[grid](
            input, weight, output,
            **common_args,
            BLOCK_SIZE_BATCH_LENGTH=bs_bl,
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=bs_out,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_conv1d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_BATCH_LENGTH": cfg.kwargs["BLOCK_SIZE_BATCH_LENGTH"],
        "BLOCK_SIZE_IN_FEAT":      cfg.kwargs["BLOCK_SIZE_IN_FEAT"],
        "BLOCK_SIZE_OUT_FEAT":     cfg.kwargs["BLOCK_SIZE_OUT_FEAT"],
        "num_warps":               cfg.num_warps,
        "num_stages":              cfg.num_stages,
    }
