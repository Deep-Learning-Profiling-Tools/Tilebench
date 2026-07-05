import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK": 256, "num_warps": 4}


@triton.jit
def compute_block_sums_kernel(
    input_ptr,
    block_sum_ptr,
    block_sq_sum_ptr,
    N,
    C,
    BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    channel_id = tl.program_id(1)

    row_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = row_offsets < N

    x = tl.load(input_ptr + row_offsets * C + channel_id, mask=mask, other=0.0).to(tl.float32)

    local_sum = tl.sum(x, axis=0)
    local_sq_sum = tl.sum(x * x, axis=0)

    out_idx = block_id * C + channel_id
    tl.store(block_sum_ptr + out_idx, local_sum)
    tl.store(block_sq_sum_ptr + out_idx, local_sq_sum)


@triton.jit
def compute_mean_invstd_kernel(
    block_sum_ptr,
    block_sq_sum_ptr,
    mean_ptr,
    inv_std_ptr,
    N,
    C,
    NUM_BLOCKS,
    BLOCK_B: tl.constexpr,
    eps,
):
    channel_id = tl.program_id(0)

    block_offsets = tl.arange(0, BLOCK_B)
    mask = block_offsets < NUM_BLOCKS

    sums = tl.load(block_sum_ptr + block_offsets * C + channel_id, mask=mask, other=0.0)
    sq_sums = tl.load(block_sq_sum_ptr + block_offsets * C + channel_id, mask=mask, other=0.0)

    total_sum = tl.sum(sums, axis=0)
    total_sq_sum = tl.sum(sq_sums, axis=0)

    mean = total_sum / N
    var = total_sq_sum / N - mean * mean
    var = tl.maximum(var, 0.0)

    tl.store(mean_ptr + channel_id, mean)
    tl.store(inv_std_ptr + channel_id, tl.rsqrt(var + eps))


@triton.jit
def apply_batch_norm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    mean_ptr,
    inv_std_ptr,
    total_elements,
    C,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    channel_id = offsets % C

    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + channel_id, mask=mask)
    inv_std = tl.load(inv_std_ptr + channel_id, mask=mask)
    gamma = tl.load(gamma_ptr + channel_id, mask=mask).to(tl.float32)
    beta = tl.load(beta_ptr + channel_id, mask=mask).to(tl.float32)

    y = (x - mean) * inv_std * gamma + beta
    tl.store(output_ptr + offsets, y, mask=mask)


# Only the apply kernel is autotuned (dominates total work for large N*C).
# Block sums and mean/invstd kernels use fixed defaults.
_apply_batch_norm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048]
        for nw in [4, 8]
    ],
    key=["total_elements"],
)(apply_batch_norm_kernel)


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    BLOCK_N = 1024
    NUM_BLOCKS = triton.cdiv(N, BLOCK_N)

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)

    # Kernel 1: per-(block, channel) partial sums.
    compute_block_sums_kernel[(NUM_BLOCKS, C)](
        input, block_sum, block_sq_sum, N, C, BLOCK_N=BLOCK_N,
    )

    # Kernel 2: finish reduction per channel, compute mean/inv_std.
    BLOCK_B = triton.next_power_of_2(NUM_BLOCKS)
    compute_mean_invstd_kernel[(C,)](
        block_sum, block_sq_sum, mean, inv_std,
        N, C, NUM_BLOCKS, BLOCK_B=BLOCK_B, eps=eps,
    )

    # Kernel 3: apply normalization element-wise.
    total_elements = N * C
    if autotune:
        grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK"]),)
        _apply_batch_norm_kernel_autotuned[grid](
            input, gamma, beta, output, mean, inv_std,
            total_elements, C,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total_elements, cfg["BLOCK"]),)
        apply_batch_norm_kernel[grid](
            input, gamma, beta, output, mean, inv_std,
            total_elements, C,
            BLOCK=cfg["BLOCK"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_apply_batch_norm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK": cfg.kwargs["BLOCK"],
        "num_warps": cfg.num_warps,
    }
