from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=256, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


@ct.kernel
def compute_block_sums_kernel(
    input_ptr,
    block_sum_ptr,
    block_sq_sum_ptr,
    N,
    C,
    BLOCK_N: ConstInt,
):
    """Per-(block, channel) partial sum / sum-of-squares — mirror of Triton kernel 1."""
    block_id = ct.bid(0)
    channel_id = ct.bid(1)

    row_offsets = block_id * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
    mask = row_offsets < N

    # x[row, channel] = input[row * C + channel] in flat layout.
    input_idx = row_offsets * C + channel_id
    idx_safe = ct.where(mask, input_idx, -1)
    x = ct.gather(input_ptr, idx_safe, padding_value=0.0)
    x = ct.astype(x, ct.float32)

    local_sum = ct.sum(x, axis=0, keepdims=True)        # (1,)
    local_sq_sum = ct.sum(x * x, axis=0, keepdims=True) # (1,)

    # Store at flat index block_id * C + channel_id (block-uniform → ct.store ok).
    out_idx = block_id * C + channel_id
    ct.store(block_sum_ptr, index=(out_idx,), tile=local_sum)
    ct.store(block_sq_sum_ptr, index=(out_idx,), tile=local_sq_sum)


@ct.kernel
def compute_mean_invstd_kernel(
    block_sum_ptr,
    block_sq_sum_ptr,
    mean_ptr,
    inv_std_ptr,
    N,
    C,
    NUM_BLOCKS,
    BLOCK_B: ConstInt,
    eps,
):
    """Finish per-channel reduction and emit mean / inv_std — mirror of Triton kernel 2."""
    channel_id = ct.bid(0)

    block_offsets = ct.arange(BLOCK_B, dtype=ct.int32)
    mask = block_offsets < NUM_BLOCKS

    idx = block_offsets * C + channel_id
    idx_safe = ct.where(mask, idx, -1)
    sums = ct.gather(block_sum_ptr, idx_safe, padding_value=0.0)
    sq_sums = ct.gather(block_sq_sum_ptr, idx_safe, padding_value=0.0)

    total_sum = ct.sum(sums, axis=0, keepdims=True)        # (1,)
    total_sq_sum = ct.sum(sq_sums, axis=0, keepdims=True)  # (1,)

    mean = total_sum / N
    var = total_sq_sum / N - mean * mean
    var = ct.maximum(var, 0.0)
    inv_std = ct.rsqrt(var + eps)

    ct.store(mean_ptr, index=(channel_id,), tile=mean)
    ct.store(inv_std_ptr, index=(channel_id,), tile=inv_std)


@ct.kernel
def apply_batch_norm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    mean_ptr,
    inv_std_ptr,
    total_elements,
    C,
    TILE: ConstInt,
):
    """Element-wise apply y = (x - mean) * inv_std * gamma + beta — mirror of Triton kernel 3."""
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # Tile-aligned x load (OOB tail → 0.0; corresponding output write silently dropped).
    x = ct.load(input_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x, ct.float32)

    # Per-element gather of the per-channel parameters.
    # channel_id = offsets % C is always in [0, C), so no OOB clamping needed.
    channel_id = offsets % C
    mean = ct.gather(mean_ptr, channel_id)
    inv_std = ct.gather(inv_std_ptr, channel_id)
    gamma = ct.gather(gamma_ptr, channel_id)
    beta = ct.gather(beta_ptr, channel_id)
    gamma = ct.astype(gamma, ct.float32)
    beta = ct.astype(beta, ct.float32)

    y = (x - mean) * inv_std * gamma + beta
    y_out = ct.astype(y, output_ptr.dtype)
    ct.store(output_ptr, index=(bid,), tile=y_out)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
# Mirrors Triton's @triton.autotune(key=["total_elements"]) — kernel 3 is the
# dominant cost; kernels 1 and 2 use fixed defaults on both sides.
_tuner = CutileAutotuner(apply_batch_norm_kernel)


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile batch normalization — direct mirror of the 3-kernel Triton pipeline:
      kernel 1: per-(block, channel) partial sums  (grid: NUM_BLOCKS x C)
      kernel 2: finish reduction → mean / inv_std  (grid: C)
      kernel 3: element-wise apply                  (grid: cdiv(N*C, TILE))
    """

    output = torch.empty_like(input)
    BLOCK_N = 1024
    NUM_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)

    # ct.gather/store work on flat 1D arrays — match Triton's flat indexing layout.
    input_flat = input.contiguous().view(-1)
    output_flat = output.view(-1)
    block_sum_flat = block_sum.view(-1)
    block_sq_sum_flat = block_sq_sum.view(-1)

    stream = torch.cuda.current_stream()

    # Kernel 1 — fixed config (small, not autotuned).
    ct.launch(stream, (NUM_BLOCKS, C, 1), compute_block_sums_kernel,
              (input_flat, block_sum_flat, block_sq_sum_flat, N, C, BLOCK_N))

    # Kernel 2 — fixed config.
    BLOCK_B = _next_pow2(NUM_BLOCKS)
    ct.launch(stream, (C, 1, 1), compute_mean_invstd_kernel,
              (block_sum_flat, block_sq_sum_flat, mean, inv_std,
               N, C, NUM_BLOCKS, BLOCK_B, eps))

    # Kernel 3 — autotuned (dominant cost for large N*C).
    total_elements = N * C
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(total_elements,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(total_elements, cfg.tile), 1, 1),
            args_fn=lambda cfg: (
                input_flat, gamma, beta, output_flat, mean, inv_std,
                total_elements, C, cfg.tile,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(total_elements, cfg.tile), 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (input_flat, gamma, beta, output_flat, mean, inv_std,
               total_elements, C, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
