import torch
import triton
import triton.language as tl

_LAST_CONFIG = None

_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 1024,
    "num_warps": 4,
    "num_stages": 1,
}


@triton.jit
def bitonic_sort_desc_kernel(
    input_ptr,
    N,
    stage,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
    slice_2_offset = slice_1_offset + stride

    slice_1_t = tl.load(
        input_ptr + slice_1_offset,
        mask=slice_1_offset < N,
        other=-float("inf"),
    )
    slice_2_t = tl.load(
        input_ptr + slice_2_offset,
        mask=slice_2_offset < N,
        other=-float("inf"),
    )

    descend = ((slice_1_offset // stage) % 2) == 1
    greater = slice_1_t > slice_2_t
    swap = descend == greater

    new_slice_1_t = tl.where(swap, slice_2_t, slice_1_t)
    new_slice_2_t = tl.where(swap, slice_1_t, slice_2_t)

    tl.store(input_ptr + slice_1_offset, new_slice_1_t, mask=slice_1_offset < N)
    tl.store(input_ptr + slice_2_offset, new_slice_2_t, mask=slice_2_offset < N)


bitonic_sort_desc_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=1)
        for bs in [256, 512, 1024]
        for nw in [1, 2, 4]
    ],
    key=["N"],
)(bitonic_sort_desc_kernel)


def _next_power_of_2_py(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


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

    BLOCK_SIZE = int(BLOCK_SIZE)

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()

    padding_len = _next_power_of_2_py(N)
    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            if autotune:
                grid = lambda meta: (
                    triton.cdiv(padding_len, meta["BLOCK_SIZE"] * 2),
                )
                bitonic_sort_desc_kernel_autotuned[grid](
                    input_padding,
                    padding_len,
                    stage,
                    stride,
                )
            else:
                grid = (triton.cdiv(padding_len, BLOCK_SIZE * 2),)
                bitonic_sort_desc_kernel[grid](
                    input_padding,
                    padding_len,
                    stage,
                    stride,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=_DEFAULT_CONFIG["num_warps"],
                    num_stages=_DEFAULT_CONFIG["num_stages"],
                )
            stride >>= 1
        stage <<= 1

    output = input_padding[:k].clone()

    _LAST_CONFIG = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": _DEFAULT_CONFIG["num_warps"],
        "num_stages": _DEFAULT_CONFIG["num_stages"],
        "autotune": bool(autotune),
    }
    return output


def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    padding_len = triton.next_power_of_2(N)
    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    BLOCK_SIZE = 1024
    grid = lambda metadata: (triton.cdiv(padding_len, metadata["BLOCK_SIZE"] * 2),)

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            bitonic_sort_desc_kernel[grid](
                input_padding,
                padding_len,
                stage,
                stride,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            stride >>= 1
        stage <<= 1

    output.copy_(input_padding[:k])


def get_last_config() -> dict | None:
    cfg = getattr(bitonic_sort_desc_kernel_autotuned, "best_config", None)
    if cfg is None:
        return _LAST_CONFIG
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }