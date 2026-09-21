import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"ROWS": 16, "num_warps": 8}


_STEP = 8
_K1_BLOCK_N = 64


@triton.jit
def compute_block_sums_kernel(
    input_ptr,
    block_sum_ptr,
    block_sq_sum_ptr,
    N,
    C: tl.constexpr,
    C_P2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STEP: tl.constexpr,
):
    block_id = tl.program_id(0)
    cols = tl.arange(0, C_P2)
    col_mask = cols < C

    acc_sum = tl.zeros([C_P2], dtype=tl.float32)
    acc_sq = tl.zeros([C_P2], dtype=tl.float32)

    row0 = block_id * BLOCK_N
    for r in tl.static_range(0, BLOCK_N, STEP):
        rows = row0 + r + tl.arange(0, STEP)
        mask = rows < N
        x = tl.load(input_ptr + rows[:, None] * C + cols[None, :],
                    mask=mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sq += tl.sum(x * x, axis=0)

    tl.store(block_sum_ptr + block_id * C + cols, acc_sum, mask=col_mask)
    tl.store(block_sq_sum_ptr + block_id * C + cols, acc_sq, mask=col_mask)


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
    N,
    C: tl.constexpr,
    C_P2: tl.constexpr,
    ROWS: tl.constexpr,
    STEP: tl.constexpr,
):
    block_id = tl.program_id(0)
    cols = tl.arange(0, C_P2)
    col_mask = cols < C

    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0)
    inv_std = tl.load(inv_std_ptr + cols, mask=col_mask, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scale = inv_std * gamma
    shift = beta - mean * scale

    row0 = block_id * ROWS
    for r in tl.static_range(0, ROWS, STEP):
        rows = row0 + r + tl.arange(0, STEP)
        mask = rows < N
        ptrs = rows[:, None] * C + cols[None, :]
        x = tl.load(input_ptr + ptrs, mask=mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        y = x * scale[None, :] + shift[None, :]
        tl.store(output_ptr + ptrs, y, mask=mask[:, None] & col_mask[None, :])


_apply_batch_norm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"ROWS": rows}, num_warps=nw)
        for rows in [8, 16, 32, 64]
        for nw in [4, 8]
    ],
    key=["N"],
)(apply_batch_norm_kernel)


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    NUM_BLOCKS = triton.cdiv(N, _K1_BLOCK_N)

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)


    compute_block_sums_kernel[(NUM_BLOCKS,)](
        input, block_sum, block_sq_sum, N, C=C, C_P2=triton.next_power_of_2(C),
        BLOCK_N=_K1_BLOCK_N, STEP=_STEP,
        num_warps=4,
    )


    BLOCK_B = triton.next_power_of_2(NUM_BLOCKS)
    compute_mean_invstd_kernel[(C,)](
        block_sum, block_sq_sum, mean, inv_std,
        N, C, NUM_BLOCKS, BLOCK_B=BLOCK_B, eps=eps,
    )


    if autotune:
        grid = lambda meta: (triton.cdiv(N, meta["ROWS"]),)
        _apply_batch_norm_kernel_autotuned[grid](
            input, gamma, beta, output, mean, inv_std,
            N, C=C, C_P2=triton.next_power_of_2(C), STEP=_STEP,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(N, cfg["ROWS"]),)
        apply_batch_norm_kernel[grid](
            input, gamma, beta, output, mean, inv_std,
            N, C=C, C_P2=triton.next_power_of_2(C), ROWS=cfg["ROWS"], STEP=_STEP,
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_apply_batch_norm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "ROWS": cfg.kwargs["ROWS"],
        "num_warps": cfg.num_warps,
    }
