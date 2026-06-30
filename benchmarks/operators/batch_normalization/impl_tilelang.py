import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tilelang.math import next_power_of_2
_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "threads": 128}
_last_autotune_config: dict = {}
def apply_batch_norm_configs():
    BLOCK_SIZE = [256, 512, 1024, 2048]
    threads = [128, 256]
    return [
        dict(BLOCK=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

@tilelang.jit
def compute_block_sums_kernel(
    input, block_sum, block_sq_sum, N, C, NUM_BLOCKS, dtype, 
    BLOCK_N: int = 1024, threads: int = 128
):
    total_elements = T.const("total_elements")
    block_elements = T.const("block_elements")
    input: T.Tensor((total_elements,), dtype)
    block_sum: T.Tensor((block_elements,), "float32")
    block_sq_sum: T.Tensor((block_elements,), "float32")
    with T.Kernel(T.ceildiv(N, BLOCK_N), C, threads=threads) as (block_id, channel_id):
        
        local_sum = T.alloc_fragment((1,), "float32")
        local_sq_sum = T.alloc_fragment((1, ), "float32")
        input_local = T.alloc_fragment((BLOCK_N,), "float32")
        input_sq_local = T.alloc_fragment((BLOCK_N,), "float32")
        # have to call reduce sum over the whole column
        # do NOT want to use T.serial at all
        for i in T.Parallel(BLOCK_N):
            row = block_id * BLOCK_N + i
            if row < N:
                input_local[i] = T.Cast("float32", input[row * C + channel_id])
            else:
                input_local[i] = 0.0
            input_sq_local[i] = input_local[i] * input_local[i]
        T.reduce_sum(input_local, local_sum, dim=0, clear = True)
        T.reduce_sum(input_sq_local, local_sq_sum, dim=0, clear = True)

        block_idx = block_id * C + channel_id
        T.copy(local_sum, block_sum[block_idx])
        T.copy(local_sq_sum, block_sq_sum[block_idx])

@tilelang.jit
def compute_mean_invstd_kernel(
    block_sum, block_sq_sum, mean, inv_std, N, C, NUM_BLOCKS,
    dtype, eps,
    BLOCK_B, threads: int = 128
):
    block_elements = T.const("block_elements")
    block_sum: T.Tensor((block_elements,), dtype)
    block_sq_sum: T.Tensor((block_elements,), dtype)
    mean: T.Tensor((C,), "float32")
    inv_std: T.Tensor((C,), "float32")
    with T.Kernel(C, threads=threads) as channel_id:
        sums = T.alloc_fragment((BLOCK_B, ), dtype)
        sq_sums = T.alloc_fragment((BLOCK_B, ), dtype)
        for i in T.Parallel(BLOCK_B):
            if i < NUM_BLOCKS:
                block_idx = i * C + channel_id
                sums[i] = block_sum[block_idx]
                sq_sums[i] = block_sq_sum[block_idx]
            else:
                sums[i] = 0.0
                sq_sums[i] = 0.0
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
    C,
    dtype,
    BLOCK: int = 256,
    threads: int = 128
):
    total_elements = T.const("total_elements")
    input: T.Tensor((total_elements, ), dtype)
    output: T.Tensor((total_elements, ), dtype)
    gamma: T.Tensor((C, ), dtype)
    beta: T.Tensor((C, ), dtype)
    mean: T.Tensor((C, ), "float32")
    inv_std: T.Tensor((C, ), "float32")

    with T.Kernel(T.ceildiv(total_elements, BLOCK), threads=threads) as pid:
        start = pid * BLOCK
        channel_id = T.alloc_fragment((BLOCK, ), "int32")
        x = T.alloc_fragment((BLOCK, ), dtype)
        mean_local = T.alloc_fragment((BLOCK, ), "float32")
        inv_std_local = T.alloc_fragment((BLOCK, ), "float32")
        gamma_local = T.alloc_fragment((BLOCK, ), dtype)
        beta_local = T.alloc_fragment((BLOCK, ), dtype)
        y = T.alloc_fragment((BLOCK, ), dtype)
        T.copy(input[start], x)
        for i in T.Parallel(BLOCK):
            channel_id[i] = (start + i) % C
            mean_local[i] = mean[channel_id[i]]
            inv_std_local[i] = inv_std[channel_id[i]] 
            gamma_local[i] = gamma[channel_id[i]]
            beta_local[i] = beta[channel_id[i]]
            y[i] = (x[i] - mean_local[i]) * inv_std_local[i] * gamma_local[i] + beta_local[i]
        T.copy(y, output[start])
    

def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    dtype = str(input.dtype).removeprefix("torch.")

    BLOCK_N = 1024
    NUM_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N

    block_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    block_sq_sum = torch.empty((NUM_BLOCKS, C), device=input.device, dtype=torch.float32)
    mean = torch.empty((C,), device=input.device, dtype=torch.float32)
    inv_std = torch.empty((C,), device=input.device, dtype=torch.float32)
    input_flat = input.contiguous().view(-1)
    output_flat = output.view(-1)
    block_sum_flat = block_sum.view(-1)
    block_sq_sum_flat = block_sq_sum.view(-1)

    # Kernel 1: per-(block, channel) partial sums.
    compute_block_sums_kernel(
        input_flat, block_sum_flat, block_sq_sum_flat,
        N, C, NUM_BLOCKS, dtype,
        BLOCK_N=BLOCK_N,
        threads=128,
    )

    # Kernel 2: finish reduction per channel, compute mean/inv_std.
    BLOCK_B = next_power_of_2(NUM_BLOCKS)
    compute_mean_invstd_kernel(
        block_sum_flat, block_sq_sum_flat, mean, inv_std,
        N, C, NUM_BLOCKS, "float32", eps,
        BLOCK_B=BLOCK_B,
        threads=128,
    )

    # Kernel 3: apply normalization element-wise.
    if autotune:
        with set_autotune_inputs(input_flat, gamma, beta, output_flat, mean, inv_std):
            tuned_kernel = apply_batch_norm_kernel.compile(
                input_flat, gamma, beta, output_flat, mean, inv_std,
                C, dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input_flat, gamma, beta, output_flat, mean, inv_std)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        apply_batch_norm_kernel(
            input_flat, gamma, beta, output_flat, mean, inv_std,
            C, dtype,
            BLOCK=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
        

    
