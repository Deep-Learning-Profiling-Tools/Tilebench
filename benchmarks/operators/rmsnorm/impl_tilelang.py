import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "threads": 256, "num_stages": 2}
_last_autotune_config: dict = {}


def rmsnorm_configs():
    BLOCK_N = [512, 1024, 2048]
    threads = [64, 128, 256]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_N=bn, threads=nt, num_stages=ns)
        for bn in BLOCK_N
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=rmsnorm_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def rmsnorm_kernel(
    X, rms_w, Y, dtype, eps,
    BLOCK_N: int = 1024, threads: int = 256, num_stages: int = 2,
):
    M = T.dynamic("M")
    N = T.const("N")
    X: T.Tensor((M, N), dtype)
    rms_w: T.Tensor((N,), dtype)
    Y: T.Tensor((M, N), dtype)

    accum_dtype = "float"

    with T.Kernel(M, threads=threads) as row:
        sumsq_local = T.alloc_fragment((1, BLOCK_N), accum_dtype)
        row_sum = T.alloc_fragment((1,), accum_dtype)
        inv_rms = T.alloc_fragment((1,), accum_dtype)

        T.fill(sumsq_local, 0.0)

        for off in T.serial(0, N, BLOCK_N):
            for i, j in T.Parallel(1, BLOCK_N):
                col = off + j
                x_val = T.Cast(accum_dtype, X[row, col])
                sumsq_local[i, j] += x_val * x_val

        T.reduce_sum(sumsq_local, row_sum, dim=1)

        for i in T.Parallel(1):
            inv_rms[i] = T.rsqrt(row_sum[i] / N + T.Cast(accum_dtype, eps))

        for off in T.serial(0, N, BLOCK_N):
            for i, j in T.Parallel(1, BLOCK_N):
                col = off + j
                x_val = T.Cast(accum_dtype, X[row, col])
                weight = T.Cast(accum_dtype, rms_w[col])
                Y[row, col] = T.Cast(dtype, x_val * inv_rms[i] * weight)


def run(
    x: torch.Tensor,
    rms_w: torch.Tensor,
    eps: float = 1e-6,
    autotune: bool = False,
    **kwargs,
) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    orig_shape = x.shape
    K = orig_shape[-1]
    batch_M = x.numel() // K

    x_2d = x.reshape(batch_M, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    out = torch.empty_like(x)
    out_2d = out.reshape(batch_M, K)

    if autotune:
        with set_autotune_inputs(x_2d, rms_w, out_2d):
            kernel = rmsnorm_kernel.compile(
                x_2d, rms_w, out_2d,
                dtype=dtype,
                eps=eps,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x_2d, rms_w, out_2d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        rmsnorm_kernel(
            x_2d, rms_w, out_2d,
            dtype, eps,
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
