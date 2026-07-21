import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_BATCH_HEIGHT_WIDTH": 64,
    "BLOCK_SIZE_IN_FEAT": 32,
    "BLOCK_SIZE_OUT_FEAT": 64,
    "threads": 128,
    "num_stages": 1,
}
_last_autotune_config: dict = {}


def conv2d_configs():
    return [
        dict(
            BLOCK_SIZE_BATCH_HEIGHT_WIDTH=bs_bhw,
            BLOCK_SIZE_IN_FEAT=bs_in,
            BLOCK_SIZE_OUT_FEAT=bs_out,
            threads=nt,
            num_stages=ns,
        )
        for bs_bhw in [32, 64, 128]
        for bs_in in [16, 32, 64]
        for bs_out in [64, 128]
        for nt in [64, 128, 256]
        for ns in [1]
    ]


# what is the algo?
# input is shape [batch, in_channels, h_in, w_in]
# weight is shape [out_channels, in_channels, h_k, w_k]
# output is shape [batch, out_channels, h_o, w_o]

# we want to utilize tensor cores -> need to do GEMM
# batch = # of samples
# per sample we have in_channels layers of 2d plane of h_in by w_in
# then we do the operation on per channel then sum up
# so all the in_channels contribute to one element
# and then we do that for # of out_channels
# resulting in output shape of [batch, out_channels, h_o, w_o]

# now how do we use GEMM here?

# well lets start from dot product
# flatten last 3 dim on input and weight for sliding window basically
# dot producting those vectors give the scalar we need
# but we need to do this out_channel times
# so imagine row is the values of input (A)
# then in B (given A @ B) each col needs to be the kernel weights
# so # of cols needs to be out_channels
# so shape of B should be [in_channels x h_k x w_k, out_channels]
# so shape of A should be [h_o x w_o, in_channels x h_k x w_k]
# so then A @ B gives us shape [h_o x w_o, out_channels]
# so pretty easy to covnert each row represents an element in therid,
# and each col in the row js repesents for that out_channels so easy reshape
# then just do this process independenlty per batch i believe


@tilelang.autotune(configs=conv2d_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def conv2d_kernel(
    input,
    weight,
    output,
    stride,
    padding,
    groups,
    dtype,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: int = 64,
    BLOCK_SIZE_IN_FEAT: int = 32,
    BLOCK_SIZE_OUT_FEAT: int = 64,
    threads: int = 128,
    num_stages: int = 1,
):
    BATCH, IN_CHANNELS, IN_H, IN_W = T.const("BATCH, IN_CHANNELS, IN_H, IN_W")
    OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KH, KW = T.const(
        "OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KH, KW"
    )
    OUT_H, OUT_W = T.const("OUT_H, OUT_W")

    input: T.Tensor((BATCH, IN_CHANNELS, IN_H, IN_W), dtype)
    weight: T.Tensor((OUT_CHANNELS, IN_CHANNELS_PER_GROUP, KH, KW), dtype)
    output: T.Tensor((BATCH, OUT_CHANNELS, OUT_H, OUT_W), dtype)
    output_flat = T.Tensor(
        (BATCH * OUT_CHANNELS * OUT_H * OUT_W,), dtype, output.data
    )

    out_hw = OUT_H * OUT_W
    total_bhw = BATCH * out_hw
    out_channels_per_group = OUT_CHANNELS // groups
    total_in_feat = IN_CHANNELS_PER_GROUP * KH * KW
    kernel_hw = KH * KW

    with T.Kernel(
        T.ceildiv(total_bhw, BLOCK_SIZE_BATCH_HEIGHT_WIDTH),
        T.ceildiv(out_channels_per_group, BLOCK_SIZE_OUT_FEAT),
        groups,
        threads=threads,
    ) as (pid_bhw, pid_oc, group_id):
        input_tile = T.alloc_shared(
            (BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT), dtype
        )
        weight_tile = T.alloc_shared((BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT), dtype)
        output_tile = T.alloc_shared(
            (BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), dtype
        )
        acc = T.alloc_fragment(
            (BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), "float32"
        )
        use_tmem = dtype != "float32"
        if use_tmem:
            acc_tmem = T.alloc_tmem(
                (BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), "float32"
            )
            mbar = T.alloc_barrier(1)
        else:
            T.clear(acc)

        for feat_block in T.Pipelined(
            T.ceildiv(total_in_feat, BLOCK_SIZE_IN_FEAT), num_stages=num_stages
        ):
            for i, j in T.Parallel(BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT):
                bhw = pid_bhw * BLOCK_SIZE_BATCH_HEIGHT_WIDTH + i
                feat = feat_block * BLOCK_SIZE_IN_FEAT + j
                batch_id = bhw // out_hw
                hw_id = bhw % out_hw
                out_row = hw_id // OUT_W
                out_col = hw_id % OUT_W
                in_channel_local = feat // kernel_hw
                kernel_rem = feat % kernel_hw
                kernel_row = kernel_rem // KW
                kernel_col = kernel_rem % KW
                in_row = out_row * stride + kernel_row - padding
                in_col = out_col * stride + kernel_col - padding
                in_channel = group_id * IN_CHANNELS_PER_GROUP + in_channel_local

                if (
                    bhw < total_bhw
                    and feat < total_in_feat
                    and in_row >= 0
                    and in_row < IN_H
                    and in_col >= 0
                    and in_col < IN_W
                ):
                    input_tile[i, j] = input[batch_id, in_channel, in_row, in_col]
                else:
                    input_tile[i, j] = T.Cast(dtype, 0.0)

            for i, j in T.Parallel(BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT):
                feat = feat_block * BLOCK_SIZE_IN_FEAT + i
                out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
                out_channel = group_id * out_channels_per_group + out_channel_local
                in_channel_local = feat // kernel_hw
                kernel_rem = feat % kernel_hw
                kernel_row = kernel_rem // KW
                kernel_col = kernel_rem % KW

                if feat < total_in_feat and out_channel_local < out_channels_per_group:
                    weight_tile[i, j] = weight[
                        out_channel, in_channel_local, kernel_row, kernel_col
                    ]
                else:
                    weight_tile[i, j] = T.Cast(dtype, 0.0)

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
        T.copy(acc, output_tile)
        T.sync_threads()

        for tile_idx in T.Parallel(
            BLOCK_SIZE_BATCH_HEIGHT_WIDTH * BLOCK_SIZE_OUT_FEAT
        ):
            i = tile_idx // BLOCK_SIZE_OUT_FEAT
            j = tile_idx % BLOCK_SIZE_OUT_FEAT
            bhw = pid_bhw * BLOCK_SIZE_BATCH_HEIGHT_WIDTH + i
            out_channel_local = pid_oc * BLOCK_SIZE_OUT_FEAT + j
            out_channel = group_id * out_channels_per_group + out_channel_local
            batch_id = bhw // out_hw
            hw_id = bhw % out_hw
            out_row = hw_id // OUT_W
            out_col = hw_id % OUT_W
            if bhw < total_bhw and out_channel_local < out_channels_per_group:
                output_idx = (
                    (batch_id * OUT_CHANNELS + out_channel) * OUT_H + out_row
                ) * OUT_W + out_col
                output_flat[output_idx] = output_tile[i, j]


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
    batch, in_channels, in_H, in_W = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1
    output = torch.empty(
        (batch, out_channels, out_H, out_W),
        device=input.device,
        dtype=input.dtype,
    )
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input, weight, output):
            tuned_kernel = conv2d_kernel.compile(
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
        conv2d_kernel(
            input,
            weight,
            output,
            stride=stride,
            padding=padding,
            groups=groups,
            dtype=dtype,
            BLOCK_SIZE_BATCH_HEIGHT_WIDTH=cfg["BLOCK_SIZE_BATCH_HEIGHT_WIDTH"],
            BLOCK_SIZE_IN_FEAT=cfg["BLOCK_SIZE_IN_FEAT"],
            BLOCK_SIZE_OUT_FEAT=cfg["BLOCK_SIZE_OUT_FEAT"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
