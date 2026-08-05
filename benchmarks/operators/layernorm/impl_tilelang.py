import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "threads": 256, "num_stages": 2}
_last_autotune_config: dict = {}


def layernorm_configs():
    BLOCK_N = [512, 1024, 2048]
    threads = [64, 128, 256]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_N=bn, threads=nt, num_stages=ns)
        for bn in BLOCK_N
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=layernorm_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def layernorm_kernel(X, weight, bias, Y, dtype, eps,
                     BLOCK_N: int = 1024, threads: int = 256,
                     num_stages: int = 2):
    M = T.dynamic("M")
    N = T.const("N")
    X: T.Tensor((M, N), dtype)
    weight: T.Tensor((N,), dtype)
    bias: T.Tensor((N,), dtype)
    Y: T.Tensor((M, N), dtype)

    accum_dtype = "float"
    with T.Kernel(M, threads=threads) as row:
        sum_local = T.alloc_fragment((1, BLOCK_N), accum_dtype)
        sumsq_local = T.alloc_fragment((1, BLOCK_N), accum_dtype)

        sum_row = T.alloc_fragment((1,), accum_dtype)
        sumsq_row = T.alloc_fragment((1,), accum_dtype)
        mean_row = T.alloc_fragment((1,), accum_dtype)
        rstd_row = T.alloc_fragment((1,), accum_dtype)

        T.fill(sum_local, 0.0)
        T.fill(sumsq_local, 0.0)

        for off in T.serial(0, N, BLOCK_N):
            for i, j in T.Parallel(1, BLOCK_N):
                col = off + j
                x_val = T.Cast(accum_dtype, X[row, col])
                sum_local[i, j] += x_val
                sumsq_local[i, j] += x_val * x_val

        T.reduce_sum(sum_local, sum_row, dim=1)
        T.reduce_sum(sumsq_local, sumsq_row, dim=1)

        mean_row[0] = sum_row[0] / N
        variance = sumsq_row[0] / N - mean_row[0] * mean_row[0]
        rstd_row[0] = T.rsqrt(variance + T.Cast(accum_dtype, eps))

        for off in T.serial(0, N, BLOCK_N):
            for i, j in T.Parallel(1, BLOCK_N):
                col = off + j
                x_val = T.Cast(accum_dtype, X[row, col])
                norm = (x_val - mean_row[i]) * rstd_row[i]
                Y[row, col] = T.Cast(
                    dtype,
                    norm
                    * T.Cast(accum_dtype, weight[col])
                    + T.Cast(accum_dtype, bias[col]),
                )


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        eps: float = 1e-5, autotune: bool = False, **kwargs) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    orig_shape = x.shape
    K = orig_shape[-1]
    batch_M = x.numel() // K
    x_2d = x.contiguous().reshape(batch_M, K)
    out = torch.empty_like(x)
    out_2d = out.reshape(batch_M, K)

    if autotune:
        with set_autotune_inputs(x_2d, weight, bias, out_2d):
            kernel = layernorm_kernel.compile(
                x_2d, weight, bias, out_2d,
                dtype=dtype,
                eps=eps,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x_2d, weight, bias, out_2d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        layernorm_kernel(
            x_2d, weight, bias, out_2d, dtype, eps,
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
