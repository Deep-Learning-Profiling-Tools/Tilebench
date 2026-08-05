import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128, "num_stages": 2}
_last_autotune_config: dict = {}
def relu_configs():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs = relu_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit


def relu_kernel(x, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements, ), dtype)
    output: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        start = pid * BLOCK_SIZE 
        x_reg = T.alloc_fragment((BLOCK_SIZE,), dtype)
        output_reg = T.alloc_fragment((BLOCK_SIZE,), dtype)
        T.copy(x[start : start + BLOCK_SIZE], x_reg)
        for local_idx in T.Parallel(BLOCK_SIZE):
            value = x_reg[local_idx]
            output_reg[local_idx] = T.Select(value >= 0, value, 0)
        T.copy(output_reg, output[start : start + BLOCK_SIZE])

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty_like(x)
    if autotune:
        with set_autotune_inputs(x, output):
            kernel = relu_kernel.compile(x, output, dtype = dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        relu_kernel(
            x, output, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output
def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
