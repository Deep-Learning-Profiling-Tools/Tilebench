import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 2}
_BLOCK_SIZE = 1024
_BLOCK_BB = 128  # prefix-sum kernel for the second-layer buffer (hardcoded in LeetGPU)


@triton.jit
def count_ones_in_block(input, block_sum, N, bit, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    block = tl.load(input + offset, mask=mask, other=0)

    bit_mask = ((block >> bit) & 1).to(tl.int32)
    tl.store(block_sum + program_id, tl.sum(bit_mask))


@triton.jit
def count_ones_per_block_blocks(first_layer_sum, block_block_sum, K, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < K

    block_of_blocks = tl.load(first_layer_sum + offset, mask=mask, other=0)
    tl.store(block_block_sum + program_id, tl.sum(block_of_blocks))


@triton.jit
def compute_prefix_sums_per_block_of_blocks(block_block_sum, global_ones, L, BLOCK_SIZE: tl.constexpr):
    # Grid of 1 process
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < L

    block_block_sums = tl.load(block_block_sum + offset, mask=mask, other=0)

    tl.store(block_block_sum + offset, tl.cumsum(block_block_sums) - block_block_sums, mask=mask)
    tl.store(global_ones, tl.sum(block_block_sums))


@triton.jit
def compute_prefix_sums_per_block(first_layer_sum, block_block_sum, K, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < K

    sum_in_blocks = tl.load(first_layer_sum + offset, mask=mask, other=0)
    prefix_sum_for_blocks = tl.load(block_block_sum + program_id)

    tl.store(
        first_layer_sum + offset,
        tl.cumsum(sum_in_blocks) - sum_in_blocks + prefix_sum_for_blocks,
        mask=mask,
    )


@triton.jit
def radix_sort_kernel(input, output, first_layer_sum, global_ones, bit, N, BLOCK_SIZE: tl.constexpr):
    input = input.to(tl.pointer_type(tl.uint32))
    output = output.to(tl.pointer_type(tl.uint32))

    program_id = tl.program_id(axis=0)
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    ones_before = tl.load(first_layer_sum + program_id)
    zeros_before = program_id * BLOCK_SIZE - ones_before

    block = tl.load(input + offset, mask=mask, other=0)
    mask_bits = ((block >> bit) & 1).to(tl.int32)

    ones_in_block = tl.cumsum(mask_bits)
    ones_rank = ones_in_block - mask_bits

    zeros_in_block = tl.cumsum(1 - mask_bits)
    zeros_rank = zeros_in_block - (1 - mask_bits)

    global_zeros = N - tl.load(global_ones)

    offset_values = tl.where(
        mask_bits == 0,
        zeros_before.to(tl.int32) + zeros_rank.to(tl.int32),
        global_zeros.to(tl.int32) + ones_before.to(tl.int32) + ones_rank.to(tl.int32),
    )

    tl.store(output + offset_values, block, mask=mask)


# Autotune the scatter kernel (dominant cost). BLOCK_SIZE is fixed at _BLOCK_SIZE
# because other kernels share the same block layout; only num_warps / num_stages vary.
_radix_sort_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw)
        for nw in [2, 4, 8]
    ],
    key=["N"],
    warmup=1,
    rep=3,
)(radix_sort_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    32-pass 1-bit radix sort matching the LeetGPU algorithm:
      for each bit in 0..31:
        1. per-block count of bit=1
        2. block-of-blocks count + hierarchical prefix sum
        3. per-block prefix sum
        4. scatter to final (zeros first, ones after)
    """
    if N <= 1:
        return input.clone()

    # Clone input so we don't mutate the caller's tensor; output is the other buffer.
    work = input.clone()
    output = torch.empty_like(input)

    grid = (triton.cdiv(N, _BLOCK_SIZE),)
    grid_second = (triton.cdiv(grid[0], _BLOCK_SIZE),)
    grid_third = (1,)

    first_layer = torch.empty((grid[0],), dtype=torch.int32, device=input.device)
    second_layer = torch.empty((grid_second[0],), dtype=torch.int32, device=input.device)
    global_ones = torch.empty((), dtype=torch.int32, device=input.device)

    cfg = _DEFAULT_CONFIG

    for bit in range(32):
        count_ones_in_block[grid](work, first_layer, N, bit, _BLOCK_SIZE)
        count_ones_per_block_blocks[grid_second](first_layer, second_layer, grid[0], _BLOCK_SIZE)
        compute_prefix_sums_per_block_of_blocks[grid_third](
            second_layer, global_ones, grid_second[0], _BLOCK_BB,
        )
        compute_prefix_sums_per_block[grid_second](first_layer, second_layer, grid[0], _BLOCK_SIZE)

        if autotune:
            _radix_sort_kernel_autotuned[grid](
                work, output, first_layer, global_ones, bit, N, _BLOCK_SIZE,
            )
        else:
            radix_sort_kernel[grid](
                work, output, first_layer, global_ones, bit, N, _BLOCK_SIZE,
                num_warps=cfg["num_warps"],
            )

        # Ping-pong: the pass's result becomes the next pass's input. A
        # pointer swap instead of `work.copy_(output)` saves a full
        # read+write of the array per pass (32 device copies ≈ 5 GB of
        # traffic at n=20M). Mirrored in impl_cutile.py.
        work, output = output, work

    return work


def get_last_config() -> dict | None:
    cfg = getattr(_radix_sort_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps}
