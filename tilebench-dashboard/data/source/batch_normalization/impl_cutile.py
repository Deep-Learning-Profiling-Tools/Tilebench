from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]


_DEFAULT_CONFIG = SimpleNamespace(rows=16, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(rows=r, occupancy=occ)
    for r in [8, 16, 32, 64]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


_STEP = 8
_K1_BLOCK_N = 64


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


@ct.kernel
def compute_block_sums_kernel(
    input_2d,
    block_sum_2d,
    block_sq_sum_2d,
    N,
    C: ConstInt,
    C_P2: ConstInt,
    BLOCK_N: ConstInt,
    STEP: ConstInt,
):
    block_id = ct.bid(0)

    acc_sum = ct.zeros((1, C_P2), dtype=ct.float32)
    acc_sq = ct.zeros((1, C_P2), dtype=ct.float32)

    tiles_per_block = BLOCK_N // STEP
    for i in range(tiles_per_block):
        x = ct.load(input_2d,
                    index=(block_id * tiles_per_block + i, 0),
                    shape=(STEP, C_P2),
                    padding_mode=ct.PaddingMode.ZERO)
        x = ct.astype(x, ct.float32)
        acc_sum = acc_sum + ct.sum(x, axis=0, keepdims=True)
        acc_sq = acc_sq + ct.sum(x * x, axis=0, keepdims=True)

    ct.store(block_sum_2d, index=(block_id, 0), tile=acc_sum)
    ct.store(block_sq_sum_2d, index=(block_id, 0), tile=acc_sq)


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
    channel_id = ct.bid(0)

    block_offsets = ct.arange(BLOCK_B, dtype=ct.int32)
    mask = block_offsets < NUM_BLOCKS

    idx = block_offsets * C + channel_id
    idx_safe = ct.where(mask, idx, -1)
    sums = ct.gather(block_sum_ptr, idx_safe, padding_value=0.0)
    sq_sums = ct.gather(block_sq_sum_ptr, idx_safe, padding_value=0.0)

    total_sum = ct.sum(sums, axis=0, keepdims=True)
    total_sq_sum = ct.sum(sq_sums, axis=0, keepdims=True)

    mean = total_sum / N
    var = total_sq_sum / N - mean * mean
    var = ct.maximum(var, 0.0)
    inv_std = ct.rsqrt(var + eps)

    ct.store(mean_ptr, index=(channel_id,), tile=mean)
    ct.store(inv_std_ptr, index=(channel_id,), tile=inv_std)


@ct.kernel
def apply_batch_norm_kernel(
    input_2d,
    gamma_ptr,
    beta_ptr,
    output_2d,
    mean_ptr,
    inv_std_ptr,
    N,
    C: ConstInt,
    C_P2: ConstInt,
    ROWS: ConstInt,
    STEP: ConstInt,
):
    block_id = ct.bid(0)

    mean = ct.reshape(ct.load(mean_ptr, index=(0,), shape=(C_P2,),
                              padding_mode=ct.PaddingMode.ZERO), (1, C_P2))
    inv_std = ct.reshape(ct.load(inv_std_ptr, index=(0,), shape=(C_P2,),
                                 padding_mode=ct.PaddingMode.ZERO), (1, C_P2))
    gamma = ct.reshape(ct.astype(ct.load(gamma_ptr, index=(0,), shape=(C_P2,),
                                         padding_mode=ct.PaddingMode.ZERO), ct.float32), (1, C_P2))
    beta = ct.reshape(ct.astype(ct.load(beta_ptr, index=(0,), shape=(C_P2,),
                                        padding_mode=ct.PaddingMode.ZERO), ct.float32), (1, C_P2))
    scale = inv_std * gamma
    shift = beta - mean * scale

    tiles_per_block = ROWS // STEP
    for i in range(tiles_per_block):
        tile_row = block_id * tiles_per_block + i
        x = ct.load(input_2d, index=(tile_row, 0), shape=(STEP, C_P2),
                    padding_mode=ct.PaddingMode.ZERO)
        x = ct.astype(x, ct.float32)
        y = x * scale + shift
        ct.store(output_2d, index=(tile_row, 0), tile=ct.astype(y, output_2d.dtype))


_tuner = CutileAutotuner(apply_batch_norm_kernel)


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    output = torch.empty_like(input)
    NUM_BLOCKS = (N + _K1_BLOCK_N - 1) // _K1_BLOCK_N

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)

    input_2d = input.contiguous().view(N, C)
    output_2d = output.view(N, C)
    block_sum_flat = block_sum.view(-1)
    block_sq_sum_flat = block_sq_sum.view(-1)

    stream = torch.cuda.current_stream()


    C_P2 = _next_pow2(C)
    ct.launch(stream, (NUM_BLOCKS, 1, 1), compute_block_sums_kernel,
              (input_2d, block_sum, block_sq_sum, N, C, C_P2, _K1_BLOCK_N, _STEP))


    BLOCK_B = _next_pow2(NUM_BLOCKS)
    ct.launch(stream, (C, 1, 1), compute_mean_invstd_kernel,
              (block_sum_flat, block_sq_sum_flat, mean, inv_std,
               N, C, NUM_BLOCKS, BLOCK_B, eps))


    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N, C, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(N, cfg.rows), 1, 1),
            args_fn=lambda cfg: (
                input_2d, gamma, beta, output_2d, mean, inv_std,
                N, C, C_P2, cfg.rows, _STEP,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "rows":      cfg.rows,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(N, cfg.rows), 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (input_2d, gamma, beta, output_2d, mean, inv_std,
               N, C, C_P2, cfg.rows, _STEP))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
