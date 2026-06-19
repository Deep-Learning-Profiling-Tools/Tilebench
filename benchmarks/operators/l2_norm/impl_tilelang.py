import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "threads": 256, "num_stages": 2}
_last_autotune_config: dict = {}


def l2_norm_fwd_configs():
    BLOCK_N = [512, 1024, 2048]
    threads = [64, 128, 256]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_N=bn, threads=nt, num_stages=ns)
        for bn in BLOCK_N
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=l2_norm_fwd_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def l2_norm_fwd_kernel(X, Y, dtype, eps,
                       BLOCK_N: int = 1024, threads: int = 256,
                       num_stages: int = 2):
    M = T.dynamic("M")
    N = T.const("N")
    X: T.Tensor((M, N), dtype)
    Y: T.Tensor((M, N), dtype)

    accum_dtype = "float"

    # need to use BLOCK_SIZE
    with T.Kernel(M, threads=threads) as row:
        X_local = T.alloc_fragment((1, N), accum_dtype)
        X_sq_local = T.alloc_fragment((1, N), accum_dtype)
        row_sum = T.alloc_fragment((1,), accum_dtype)
        inv_norm = T.alloc_fragment((1,), accum_dtype)
        #could also use T.copy here instead of parallel, but then 
        #we will need to use T.Parallel anyway for the squaured calc
        for i, j in T.Parallel(1, N):
            X_local[i, j] = T.Cast(accum_dtype, X[row, j])
            X_sq_local[i, j] = X_local[i, j] * X_local[i, j]
            
        T.reduce_sum(X_sq_local, row_sum, dim=1)
        inv_norm[0] = T.rsqrt(row_sum[0] + T.Cast(accum_dtype, eps))
        for i, j in T.Parallel(1, N):
            Y[row, j] = T.Cast(dtype, X_local[i, j] * inv_norm[i])




def run(x: torch.Tensor, eps: float = 1e-6, autotune: bool = False, **kwargs) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    y_2d = torch.empty_like(x_2d)
    N = x_2d.shape[-1]
    M = x_2d.shape[0]

    if autotune:
        with set_autotune_inputs(x_2d, y_2d):
            kernel = l2_norm_fwd_kernel.compile(
                x_2d, y_2d,
                dtype=dtype,
                eps=eps,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x_2d, y_2d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        l2_norm_fwd_kernel(
            x_2d, y_2d, dtype, eps,
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )
    return y_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
