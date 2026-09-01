import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_LENGTH": 64,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 64,
    "threads": 128,
    "num_stages": 3,
}
_last_autotune_config: dict = {}


def conv1d_configs():
    return [
        dict(
            BLOCK_SIZE_BATCH_LENGTH=bs_bl,
            BLOCK_SIZE_IN_FEAT=bs_in,
            BLOCK_SIZE_OUT_FEAT=bs_out,
            threads=nt,
            num_stages=ns,
        )
        for bs_bl in [32, 64, 128]
        for bs_in in [16, 32, 64]
        for bs_out in [64, 128]
        for nt in [64, 128, 256]
        for ns in [2, 3, 4]
    ]


@tilelang.autotune(configs=conv1d_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def conv1d_kernel(
    input,
    weight,
    output,
    stride,
    padding,
    groups,
    dtype,
    BLOCK_SIZE_BATCH_LENGTH: int = 64,
    BLOCK_SIZE_IN_FEAT: int = 32,
    BLOCK_SIZE_OUT_FEAT: int = 64,
    threads: int = 128,
    num_stages: int = 3,
):
    BATCH, IN_CHANNELS, IN_L = T.const("BATCH, IN_CHANNELS, IN_L")
    OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KL = T.const(
        "OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KL"
    )
    OUT_L = T.const("OUT_L")

    input: T.Tensor((BATCH, IN_CHANNELS, IN_L), dtype)
    weight: T.Tensor((OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KL), dtype)
    output: T.Tensor((BATCH, OUT_CHANNELS, OUT_L), dtype)
    output_flat = T.Tensor((BATCH * OUT_CHANNELS * OUT_L,), dtype, output.data)

    total_bl = BATCH * OUT_L
    out_channels_per_group = OUT_CHANNELS // groups
    total_in_feat = IN_CHANNELS_PER_GROUP * KL

    with T.Kernel(
        T.ceildiv(total_bl, BLOCK_SIZE_BATCH_LENGTH),
        T.ceildiv(out_channels_per_group, BLOCK_SIZE_OUT_FEAT),
        groups,
        threads=threads,
    ) as (pid_bl, pid_oc, group_id):
        input_tile = T.alloc_shared(
            (BLOCK_SIZE_BATCH_LENGTH, BLOCK_SIZE_IN_FEAT), dtype
        )
        weight_tile = T.alloc_shared((BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT), dtype)
        acc = T.alloc_fragment(
            (BLOCK_SIZE_BATCH_LENGTH, BLOCK_SIZE_OUT_FEAT), "float32"
        )
        bl_offsets = T.alloc_fragment((BLOCK_SIZE_BATCH_LENGTH,), "int32")
        batch_ids = T.alloc_fragment((BLOCK_SIZE_BATCH_LENGTH,), "int32")
        out_ls = T.alloc_fragment((BLOCK_SIZE_BATCH_LENGTH,), "int32")
        feat_offsets = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        in_channel_locals = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        kernel_ls = T.alloc_fragment((BLOCK_SIZE_IN_FEAT,), "int32")
        use_tmem = dtype != "float32"
        if use_tmem:
            acc_tmem = T.alloc_tmem(
                (BLOCK_SIZE_BATCH_LENGTH, BLOCK_SIZE_OUT_FEAT), "float32"
            )
            mbar = T.alloc_barrier(1)
        else:
            T.clear(acc)

        for i in T.Parallel(BLOCK_SIZE_BATCH_LENGTH):
            bl_offsets[i] = pid_bl * BLOCK_SIZE_BATCH_LENGTH + i
            batch_ids[i] = bl_offsets[i] // OUT_L
            out_ls[i] = bl_offsets[i] % OUT_L

        for feat_block in T.Pipelined(
            T.ceildiv(total_in_feat, BLOCK_SIZE_IN_FEAT), num_stages=num_stages
        ):
            for j in T.Parallel(BLOCK_SIZE_IN_FEAT):
                feat_offsets[j] = feat_block * BLOCK_SIZE_IN_FEAT + j
                in_channel_locals[j] = feat_offsets[j] // KL
                kernel_ls[j] = feat_offsets[j] % KL

            for i, j in T.Parallel(BLOCK_SIZE_BATCH_LENGTH, BLOCK_SIZE_IN_FEAT):
                in_l = out_ls[i] * stride + kernel_ls[j] - padding
                in_channel = group_id * IN_CHANNELS_PER_GROUP + in_channel_locals[j]

                input_tile[i, j] = T.Select(
                    bl_offsets[i] < total_bl
                    and feat_offsets[j] < total_in_feat
                    and in_l >= 0
                    and in_l < IN_L,
                    input[batch_ids[i], in_channel, in_l],
                    0.0,
                )

            for i, j in T.Parallel(BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT):
                out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
                out_channel = group_id * out_channels_per_group + out_channel_local

                weight_tile[i, j] = T.Select(
                    feat_offsets[i] < total_in_feat
                    and out_channel_local < out_channels_per_group,
                    weight[out_channel, in_channel_locals[i], kernel_ls[i]],
                    0.0,
                )

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

        for tile_idx in T.Parallel(BLOCK_SIZE_BATCH_LENGTH * BLOCK_SIZE_OUT_FEAT):
            i = tile_idx // BLOCK_SIZE_OUT_FEAT
            j = tile_idx % BLOCK_SIZE_OUT_FEAT
            out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
            out_channel = group_id * out_channels_per_group + out_channel_local
            if out_channel_local < out_channels_per_group:
                output_idx = (
                    batch_ids[i] * OUT_CHANNELS + out_channel
                ) * OUT_L + out_ls[i]
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
    batch, in_channels, in_L = input.shape
    out_channels, in_channels_per_group, kL = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_L = (in_L + 2 * padding - kL) // stride + 1
    output = torch.empty(
        (batch, out_channels, out_L), device=input.device, dtype=input.dtype
    )
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input, weight, output):
            tuned_kernel = conv1d_kernel.compile(
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
        conv1d_kernel(
            input,
            weight,
            output,
            stride=stride,
            padding=padding,
            groups=groups,
            dtype=dtype,
            BLOCK_SIZE_BATCH_LENGTH=cfg["BLOCK_SIZE_BATCH_LENGTH"],
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=cfg["BLOCK_SIZE_OUT_FEAT"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
