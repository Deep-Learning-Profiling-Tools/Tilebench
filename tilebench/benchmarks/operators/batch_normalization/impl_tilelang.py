import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tilelang.math import next_power_of_2
_DEFAULT_CONFIG = {"ROWS": 16, "threads": 256}
_last_autotune_config: dict = {}
def apply_batch_norm_configs():
    ROWS = [8, 16, 32, 64]
    threads = [128, 256]
    return [
        dict(ROWS=rows, threads=nt)
        for rows in ROWS
        for nt in threads
    ]

_STEP = 8
_K1_BLOCK_N = 64

@tilelang.jit
def compute_block_sums_kernel(
    input, block_sum, block_sq_sum, N, C, NUM_BLOCKS, dtype,
    C_P2: int, BLOCK_N: int, STEP: int, threads: int = 128
):
    input: T.Tensor((N, C), dtype)
    block_sum: T.Tensor((NUM_BLOCKS, C), "float32")
    block_sq_sum: T.Tensor((NUM_BLOCKS, C), "float32")
    with T.Kernel(NUM_BLOCKS, threads=threads) as block_id:
        acc_sum = T.alloc_fragment((C_P2,), "float32")
        acc_sq = T.alloc_fragment((C_P2,), "float32")
        input_local = T.alloc_fragment((STEP, C_P2), "float32")
        input_sq_local = T.alloc_fragment((STEP, C_P2), "float32")
        tile_sum = T.alloc_fragment((C_P2,), "float32")
        tile_sq_sum = T.alloc_fragment((C_P2,), "float32")
        T.fill(acc_sum, 0.0)
        T.fill(acc_sq, 0.0)

        for r in T.unroll(0, BLOCK_N, STEP):
            for i, col in T.Parallel(STEP, C_P2):
                row = block_id * BLOCK_N + r + i
                input_local[i, col] = T.cast(input[row, col], "float32")
                input_sq_local[i, col] = input_local[i, col] * input_local[i, col]
            T.reduce_sum(input_local, tile_sum, dim=0, clear=True)
            T.reduce_sum(input_sq_local, tile_sq_sum, dim=0, clear=True)
            for col in T.Parallel(C_P2):
                acc_sum[col] += tile_sum[col]
                acc_sq[col] += tile_sq_sum[col]

        T.copy(acc_sum, block_sum[block_id, 0:C_P2])
        T.copy(acc_sq, block_sq_sum[block_id, 0:C_P2])

@tilelang.jit
def compute_mean_invstd_kernel(
    block_sum, block_sq_sum, mean, inv_std, N, C, NUM_BLOCKS,
    dtype, eps,
    BLOCK_B: int, threads: int = 128
):
    block_sum: T.Tensor((NUM_BLOCKS, C), dtype)
    block_sq_sum: T.Tensor((NUM_BLOCKS, C), dtype)
    mean: T.Tensor((C,), "float32")
    inv_std: T.Tensor((C,), "float32")
    with T.Kernel(C, threads=threads) as channel_id:
        sums = T.alloc_fragment((BLOCK_B, 1), dtype)
        sq_sums = T.alloc_fragment((BLOCK_B, 1), dtype)
        T.copy(block_sum[0:BLOCK_B, channel_id:channel_id + 1], sums)
        T.copy(block_sq_sum[0:BLOCK_B, channel_id:channel_id + 1], sq_sums)
        total_sum = T.alloc_fragment((1, ), dtype)
        total_sq_sum = T.alloc_fragment((1, ), dtype)
        mean_local = T.alloc_fragment((1, ), dtype)
        inv_std_local = T.alloc_fragment((1, ), dtype)
        
        T.reduce_sum(sums, total_sum, dim = 0, clear=True)
        T.reduce_sum(sq_sums, total_sq_sum, dim = 0, clear=True)

        mean_local[0] = total_sum[0] / N
        var_raw = total_sq_sum[0] / N - mean_local[0] * mean_local[0]
        var_clamped = T.max(var_raw, 0.0)
        inv_std_local[0] = T.rsqrt(var_clamped + eps)

        T.copy(mean_local, mean[channel_id])
        T.copy(inv_std_local, inv_std[channel_id])

@tilelang.autotune(configs=apply_batch_norm_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def apply_batch_norm_kernel(
    input,
    gamma,
    beta,
    output,
    mean,
    inv_std,
    N,
    C,
    dtype,
    C_P2: int,
    ROWS: int = 16,
    STEP: int = 8,
    threads: int = 128
):
    input: T.Tensor((N, C), dtype)
    output: T.Tensor((N, C), dtype)
    gamma: T.Tensor((C, ), dtype)
    beta: T.Tensor((C, ), dtype)
    mean: T.Tensor((C, ), "float32")
    inv_std: T.Tensor((C, ), "float32")

    with T.Kernel(T.ceildiv(N, ROWS), threads=threads) as block_id:
        mean_local = T.alloc_fragment((C_P2,), "float32")
        inv_std_local = T.alloc_fragment((C_P2,), "float32")
        gamma_local = T.alloc_fragment((C_P2,), "float32")
        beta_local = T.alloc_fragment((C_P2,), "float32")
        scale = T.alloc_fragment((C_P2,), "float32")
        shift = T.alloc_fragment((C_P2,), "float32")
        x = T.alloc_fragment((STEP, C_P2), "float32")
        y = T.alloc_fragment((STEP, C_P2), "float32")

        T.copy(mean[0:C_P2], mean_local)
        T.copy(inv_std[0:C_P2], inv_std_local)
        T.copy(gamma[0:C_P2], gamma_local)
        T.copy(beta[0:C_P2], beta_local)

        for col in T.Parallel(C_P2):
            scale[col] = inv_std_local[col] * gamma_local[col]
            shift[col] = beta_local[col] - mean_local[col] * scale[col]

        for r in T.unroll(0, ROWS, STEP):
            row_start = block_id * ROWS + r
            T.copy(input[row_start:row_start + STEP, 0:C_P2], x)
            for i, col in T.Parallel(STEP, C_P2):
                y[i, col] = x[i, col] * scale[col] + shift[col]
            T.copy(y, output[row_start:row_start + STEP, 0:C_P2])
    

def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    dtype = str(input.dtype).removeprefix("torch.")

    BLOCK_N = _K1_BLOCK_N
    NUM_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
    C_P2 = next_power_of_2(C)

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)
    input_2d = input.contiguous().view(N, C)
    output_2d = output.view(N, C)

    compute_block_sums_kernel(
        input_2d, block_sum, block_sq_sum,
        N, C, NUM_BLOCKS, dtype, C_P2,
        BLOCK_N=BLOCK_N,
        STEP=_STEP,
        threads=128,
    )

    BLOCK_B = next_power_of_2(NUM_BLOCKS)
    compute_mean_invstd_kernel(
        block_sum, block_sq_sum, mean, inv_std,
        N, C, NUM_BLOCKS, "float32", eps,
        BLOCK_B=BLOCK_B,
        threads=128,
    )

    if autotune:
        with set_autotune_inputs(input_2d, gamma, beta, output_2d, mean, inv_std):
            tuned_kernel = apply_batch_norm_kernel.compile(
                input_2d, gamma, beta, output_2d, mean, inv_std,
                N, C, dtype, C_P2,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input_2d, gamma, beta, output_2d, mean, inv_std)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        apply_batch_norm_kernel(
            input_2d, gamma, beta, output_2d, mean, inv_std,
            N, C, dtype, C_P2,
            ROWS=cfg["ROWS"],
            STEP=_STEP,
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
        

    
