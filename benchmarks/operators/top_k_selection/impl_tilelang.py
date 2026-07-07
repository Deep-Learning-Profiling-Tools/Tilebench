import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def bitonic_step_configs():
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in [512, 1024, 2048]
        for nt in [64, 128, 256]
    ]


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@tilelang.autotune(configs=bitonic_step_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def bitonic_step_kernel(padding_len, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        input_padding: T.Tensor((padding_len,), dtype),
        N: T.int32,
        stage: T.int32,
        stride: T.int32,
    ):
        neg_inf = -T.infinity(dtype)
        with T.Kernel(T.ceildiv(padding_len, BLOCK_SIZE * 2), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK_SIZE):
                offset = pid * BLOCK_SIZE + local_idx

                slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
                slice_2_offset = slice_1_offset + stride

                valid_1 = slice_1_offset < N
                valid_2 = slice_2_offset < N
                slice_1_t = T.if_then_else(valid_1, input_padding[slice_1_offset], neg_inf)
                slice_2_t = T.if_then_else(valid_2, input_padding[slice_2_offset], neg_inf)

                descend = ((slice_1_offset // stage) % 2) == 1
                greater = slice_1_t > slice_2_t
                swap = descend == greater

                new_slice_1_t = T.if_then_else(swap, slice_2_t, slice_1_t)
                new_slice_2_t = T.if_then_else(swap, slice_1_t, slice_2_t)

                if valid_1:
                    input_padding[slice_1_offset] = new_slice_1_t
                if valid_2:
                    input_padding[slice_2_offset] = new_slice_2_t

    return main


def run(input: torch.Tensor, N: int, k: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()
    dtype = str(input.dtype).removeprefix("torch.")
    padding_len = _next_pow2(N)
    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    if autotune:
        with set_autotune_inputs(input_padding, padding_len, 2, 1):
            kernel = bitonic_step_kernel(padding_len, dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        BLOCK_SIZE = int(block_size) if block_size is not None else cfg["BLOCK_SIZE"]
        kernel = bitonic_step_kernel(
            padding_len, dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            threads=cfg["threads"],
        )

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            kernel(input_padding, padding_len, stage, stride)
            stride >>= 1
        stage <<= 1

    return input_padding[:k].clone()


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
