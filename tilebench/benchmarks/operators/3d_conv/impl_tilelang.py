import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_DHW": 64,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 64,
    "threads": 128,
    "num_stages": 3,
}
_last_autotune_config: dict = {}


def conv3d_configs():
    return [
        dict(
            BLOCK_SIZE_BATCH_DHW=bs_bdhw,
            BLOCK_SIZE_IN_FEAT=bs_in,
            BLOCK_SIZE_OUT_FEAT=bs_out,
            threads=nt,
            num_stages=ns,
        )
        for bs_bdhw in [32, 64, 128]
        for bs_in in [16, 32, 64]
        for bs_out in [64, 128]
        for nt in [64, 128, 256]
        for ns in [2, 3, 4]
    ]


@tilelang.autotune(configs=conv3d_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def conv3d_kernel(
    input,
    weight,
    output,
    stride,
    padding,
    groups,
    dtype,
    BLOCK_SIZE_BATCH_DHW: int = 64,
    BLOCK_SIZE_IN_FEAT: int = 32,
    BLOCK_SIZE_OUT_FEAT: int = 64,
    threads: int = 128,
    num_stages: int = 3,
):
    BATCH, IN_CHANNELS, IN_D, IN_H, IN_W = T.const(
        "BATCH, IN_CHANNELS, IN_D, IN_H, IN_W"
    )
    OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KD, KH, KW = T.const(
        "OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KD, KH, KW"
    )
    OUT_D, OUT_H, OUT_W = T.const("OUT_D, OUT_H, OUT_W")

    input: T.Tensor((BATCH, IN_CHANNELS, IN_D, IN_H, IN_W), dtype)
    weight: T.Tensor((OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KD, KH, KW), dtype)
    output: T.Tensor((BATCH, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), dtype)
    output_flat = T.Tensor(
        (BATCH * OUT_CHANNELS * OUT_D * OUT_H * OUT_W,), dtype, output.data
    )

    out_hw = OUT_H * OUT_W
    out_dhw = OUT_D * out_hw
    total_bdhw = BATCH * out_dhw
    out_channels_per_group = OUT_CHANNELS // groups
    kernel_hw = KH * KW
    kernel_dhw = KD * kernel_hw
    total_in_feat = IN_CHANNELS_PER_GROUP * kernel_dhw

    with T.Kernel(
        T.ceildiv(total_bdhw, BLOCK_SIZE_BATCH_DHW),
        T.ceildiv(out_channels_per_group, BLOCK_SIZE_OUT_FEAT),
        groups,
        threads=threads,
    ) as (pid_bdhw, pid_oc, group_id):
        input_tile = T.alloc_shared((BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_IN_FEAT), dtype)
        weight_tile = T.alloc_shared((BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT), dtype)
        acc = T.alloc_fragment(
            (BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_OUT_FEAT), "float32"
        )
        bdhw_offsets = T.alloc_fragment((BLOCK_SIZE_BATCH_DHW,), "int32")
        batch_ids = T.alloc_fragment((BLOCK_SIZE_BATCH_DHW,), "int32")
        out_ds = T.alloc_fragment((BLOCK_SIZE_BATCH_DHW,), "int32")
        out_rows = T.alloc_fragment((BLOCK_SIZE_BATCH_DHW,), "int32")
        out_cols = T.alloc_fragment((BLOCK_SIZE_BATCH_DHW,), "int32")
        feat_offsets = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        in_channel_locals = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        kernel_ds = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        kernel_rows = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        kernel_cols = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        use_tmem = dtype != "float32"
        if use_tmem:
            acc_tmem = T.alloc_tmem(
                (BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_OUT_FEAT), "float32"
            )
            mbar = T.alloc_barrier(1)
        else:
            T.clear(acc)

        for i in T.Parallel(BLOCK_SIZE_BATCH_DHW):
            bdhw_offsets[i] = pid_bdhw * BLOCK_SIZE_BATCH_DHW + i
            batch_ids[i] = bdhw_offsets[i] // out_dhw
            dhw_id = bdhw_offsets[i] % out_dhw
            out_ds[i] = dhw_id // out_hw
            hw_id = dhw_id % out_hw
            out_rows[i] = hw_id // OUT_W
            out_cols[i] = hw_id % OUT_W

        for feat_block in T.Pipelined(
            T.ceildiv(total_in_feat, BLOCK_SIZE_IN_FEAT), num_stages=num_stages
        ):
            for j in T.Parallel(BLOCK_SIZE_IN_FEAT):
                feat_offsets[j] = feat_block * BLOCK_SIZE_IN_FEAT + j
                in_channel_locals[j] = feat_offsets[j] // kernel_dhw
                kernel_rem = feat_offsets[j] % kernel_dhw
                kernel_ds[j] = kernel_rem // kernel_hw
                kernel_hw_rem = kernel_rem % kernel_hw
                kernel_rows[j] = kernel_hw_rem // KW
                kernel_cols[j] = kernel_hw_rem % KW

            for i, j in T.Parallel(BLOCK_SIZE_BATCH_DHW, BLOCK_SIZE_IN_FEAT):
                in_d = out_ds[i] * stride + kernel_ds[j] - padding
                in_row = out_rows[i] * stride + kernel_rows[j] - padding
                in_col = out_cols[i] * stride + kernel_cols[j] - padding
                in_channel = group_id * IN_CHANNELS_PER_GROUP + in_channel_locals[j]

                input_tile[i, j] = T.Select(
                    bdhw_offsets[i] < total_bdhw
                    and feat_offsets[j] < total_in_feat
                    and in_d >= 0
                    and in_d < IN_D
                    and in_row >= 0
                    and in_row < IN_H
                    and in_col >= 0
                    and in_col < IN_W,
                    input[batch_ids[i], in_channel, in_d, in_row, in_col],
                    T.cast(0.0, dtype),
                )

            for i, j in T.Parallel(BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT):
                out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
                out_channel = group_id * out_channels_per_group + out_channel_local

                weight_tile[i, j] = T.Select(
                    feat_offsets[i] < total_in_feat
                    and out_channel_local < out_channels_per_group,
                    weight[
                        out_channel,
                        in_channel_locals[i],
                        kernel_ds[i],
                        kernel_rows[i],
                        kernel_cols[i],
                    ],
                    T.cast(0.0, dtype),
                )

            T.sync_threads()
            if use_tmem:
                T.gemm(
                    input_tile,
                    weight_tile,
                    acc_tmem,
                    mbar=mbar,
                    clear_accum=feat_block == 0,
                )
            else:
                T.gemm(input_tile, weight_tile, acc)

        if use_tmem:
            T.copy(acc_tmem, acc)

        for tile_idx in T.Parallel(BLOCK_SIZE_BATCH_DHW * BLOCK_SIZE_OUT_FEAT):
            i = tile_idx // BLOCK_SIZE_OUT_FEAT
            j = tile_idx % BLOCK_SIZE_OUT_FEAT
            out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
            out_channel = group_id * out_channels_per_group + out_channel_local
            if out_channel_local < out_channels_per_group:
                output_idx = (
                    ((batch_ids[i] * OUT_CHANNELS + out_channel) * OUT_D + out_ds[i])
                    * OUT_H
                    + out_rows[i]
                ) * OUT_W + out_cols[i]
                output_flat[output_idx] = T.Cast(dtype, acc[i, j])


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
    output = torch.empty(
        (batch, out_channels, out_D, out_H, out_W),
        device=input.device,
        dtype=input.dtype,
    )
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input, weight, output):
            tuned_kernel = conv3d_kernel.compile(
                input,
                weight,
                output,
                stride=stride,
                padding=padding,
                groups=groups,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input, weight, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        conv3d_kernel(
            input,
            weight,
            output,
            stride=stride,
            padding=padding,
            groups=groups,
            dtype=dtype,
            BLOCK_SIZE_BATCH_DHW=cfg["BLOCK_SIZE_BATCH_DHW"],
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=cfg["BLOCK_SIZE_OUT_FEAT"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
