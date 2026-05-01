"""Top-k via multi-launch descending bitonic sort.

Algorithm:
  1. Pad the input to next_pow2(N) with -inf.
  2. Outer loop over `stage in [2, 4, ..., padding_len]`; inner loop over
     `stride in [stage/2, stage/4, ..., 1]`. Each (stage, stride) launches
     a single compare-exchange kernel that, for each of the
     padding_len/2 pairs, swaps the two endpoints based on whether the
     pair lies in an ascending or descending sub-sequence.
  3. The first k elements of the sorted buffer are the top-k (descending).

The compare-exchange kernel is autotuned via `triton.autotune(key=["N"])`,
so the first launch in step 2 runs the sweep and all subsequent
log²(padding_len)/2 launches are cache hits — equivalent to running the
sweep once per padding_len.
"""
import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 1024,
    "num_warps": 4,
    "num_stages": 1,
}


@triton.jit
def _bitonic_step_kernel(
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

    slice_1_t = tl.load(input_ptr + slice_1_offset,
                        mask=slice_1_offset < N, other=-float("inf"))
    slice_2_t = tl.load(input_ptr + slice_2_offset,
                        mask=slice_2_offset < N, other=-float("inf"))

    descend = ((slice_1_offset // stage) % 2) == 1
    greater = slice_1_t > slice_2_t
    swap = descend == greater

    new_slice_1_t = tl.where(swap, slice_2_t, slice_1_t)
    new_slice_2_t = tl.where(swap, slice_1_t, slice_2_t)

    tl.store(input_ptr + slice_1_offset, new_slice_1_t, mask=slice_1_offset < N)
    tl.store(input_ptr + slice_2_offset, new_slice_2_t, mask=slice_2_offset < N)


# Sweep mirrors impl_cutile.py 1-to-1:
#   BLOCK_SIZE  ↔ tile             same values
#   num_warps   ↔ occupancy        nw * occ ≈ 64 on B200, so Triton's
#                                   nw ∈ [2, 4, 8] pairs with cuTile's
#                                   occ ∈ [32, 16, 8].
# num_stages is fixed to 1 — this kernel has no runtime K-loop, just
# load → compare → store, so software pipelining has nothing to overlap.
_bitonic_step_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=1)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["N"],
)(_bitonic_step_kernel)


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def run(input: torch.Tensor, N: int, k: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()
    padding_len = _next_pow2(N)
    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    BLOCK_SIZE = (
        int(block_size) if block_size is not None else _DEFAULT_CONFIG["BLOCK_SIZE"]
    )

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            if autotune:
                grid = lambda meta: (triton.cdiv(padding_len, meta["BLOCK_SIZE"] * 2),)
                _bitonic_step_kernel_autotuned[grid](
                    input_padding, padding_len, stage, stride,
                )
            else:
                grid = (triton.cdiv(padding_len, BLOCK_SIZE * 2),)
                _bitonic_step_kernel[grid](
                    input_padding, padding_len, stage, stride,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=_DEFAULT_CONFIG["num_warps"],
                    num_stages=_DEFAULT_CONFIG["num_stages"],
                )
            stride >>= 1
        stage <<= 1

    return input_padding[:k].clone()


def get_last_config() -> dict | None:
    cfg = getattr(_bitonic_step_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
