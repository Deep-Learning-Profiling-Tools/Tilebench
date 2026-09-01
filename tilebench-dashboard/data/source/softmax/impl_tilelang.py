import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def softmax_config():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

@tilelang.autotune(configs=softmax_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def softmax_online_kernel(x, y, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    M = T.dynamic("M")
    N = T.const("N")
    x: T.Tensor((M, N), dtype)
    y: T.Tensor((M, N), dtype)
    with T.Kernel(M, threads=threads) as row:
        X_local = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        Y_local = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        x_exp_local = T.alloc_fragment((BLOCK_SIZE,) , "float32")
        tile_sum_local = T.alloc_fragment((1,) , "float32")
        l = T.alloc_fragment((1,), "float32")
        running_max = T.alloc_fragment((1,), "float32")
        tile_max = T.alloc_fragment((1, ), "float32")
        past_max = T.alloc_fragment((1,), "float32")
        l[0] = 0.0
        running_max[0] = -T.infinity("float32")
        past_max[0] = -T.infinity("float32")
        for start in T.serial(0, N, BLOCK_SIZE):
            end = T.min(start + BLOCK_SIZE, N)
            
            T.fill(X_local, -T.infinity("float32"))
            T.fill(x_exp_local, 0.0)
            T.copy(x[row : row + 1, start : end], X_local)
            T.reduce_max(X_local, tile_max, dim = 0)
            running_max[0] = T.max(running_max[0], tile_max[0])

            for i in T.Parallel(BLOCK_SIZE):
                col = start + i
                if col < N:
                    x_exp_local[i] = T.exp(X_local[i] - running_max[0])
            T.reduce_sum(x_exp_local, tile_sum_local, dim = 0 )
            
            l[0] = l[0] * T.exp(past_max[0] - running_max[0]) + tile_sum_local[0]
            past_max[0] = running_max[0]
        
        for start in T.serial(0, N, BLOCK_SIZE):
            end = T.min(start + BLOCK_SIZE, N)
            T.copy(x[row : row + 1, start : end], X_local)
            coef = 1.0 / l[0] 
            for i in T.Parallel(BLOCK_SIZE):
                col = start + i
                if col < N:
                    Y_local[i] = T.Cast(dtype, T.exp(X_local[i] - running_max[0]) * coef)
            T.copy(Y_local, y[row, start : end])


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty_like(x)

    if autotune:
        with set_autotune_inputs(x, output):
            kernel = softmax_online_kernel.compile(x, output, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, output)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        if block_size is not None:
            cfg["BLOCK_SIZE"] = block_size
        softmax_online_kernel(
            x, output, dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
