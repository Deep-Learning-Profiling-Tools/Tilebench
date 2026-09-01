import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def sigmoid_configs():
    BLOCK_SIZE = [1024, 2048, 4096, 8192]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs = sigmoid_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def sigmoid_kernel(x, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements, ), dtype)
    output: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            output[idx] = T.sigmoid(T.cast(x[idx], "float32"))


def run(X: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    dtype = str(X.dtype).removeprefix("torch.")
    output = torch.empty_like(X)
    if autotune:
        with set_autotune_inputs(X, output):
            kernel = sigmoid_kernel.compile(X, output, dtype = dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(X, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        sigmoid_kernel(
            X, output, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
