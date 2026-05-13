import math
import torch
import cuda.tile as ct

_LAST_CONFIG = None


@ct.kernel
def bitonic_sort_desc_kernel(
    input_ptr,
    N,
    stage,
    stride,
):
    pid = ct.bid(0)

    slice_1_offset = (pid // stride) * (2 * stride) + (pid % stride)
    slice_2_offset = slice_1_offset + stride

    valid_1 = slice_1_offset < N
    valid_2 = slice_2_offset < N

    safe_1 = ct.where(valid_1, slice_1_offset, 0)
    safe_2 = ct.where(valid_2, slice_2_offset, 0)

    slice_1_t = ct.load(input_ptr, index=(safe_1,), shape=())
    slice_2_t = ct.load(input_ptr, index=(safe_2,), shape=())

    slice_1_t = ct.where(valid_1, slice_1_t, -float("inf"))
    slice_2_t = ct.where(valid_2, slice_2_t, -float("inf"))

    descend = ((slice_1_offset // stage) % 2) == 1
    greater = slice_1_t > slice_2_t
    swap = descend == greater

    new_slice_1_t = ct.where(swap, slice_2_t, slice_1_t)
    new_slice_2_t = ct.where(swap, slice_1_t, slice_2_t)

    ct.store(input_ptr, index=(slice_1_offset,), tile=new_slice_1_t)
    ct.store(input_ptr, index=(slice_2_offset,), tile=new_slice_2_t)


def run(
    input,
    N: int,
    k: int,
    BLOCK_SIZE: int = 1024,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()

    padding_len = 1
    while padding_len < N:
        padding_len <<= 1

    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            pair_count = padding_len // 2
            ct.launch(
                torch.cuda.current_stream(),
                (pair_count, 1, 1),
                bitonic_sort_desc_kernel,
                (
                    input_padding,
                    padding_len,
                    stage,
                    stride,
                ),
            )
            stride >>= 1
        stage <<= 1

    output = input_padding[:k].clone()

    _LAST_CONFIG = {
        "BLOCK_SIZE": int(BLOCK_SIZE),
        "kernel_style": "scalar_pairwise_bitonic",
    }
    return output


def get_last_config() -> dict | None:
    return _LAST_CONFIG
