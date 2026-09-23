import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}
def vector_add_configs():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

@tilelang.autotune(configs=vector_add_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit

def add_kernel(x, y, output, dtype, BLOCK_SIZE: int =  1024, threads: int = 128 ):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements, ), dtype)
    y: T.Tensor((n_elements, ), dtype)
    output: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        start = pid * BLOCK_SIZE
        x_reg = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        y_reg = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        out_reg = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        T.copy(x[start], x_reg)
        T.copy(y[start], y_reg)
        for local_idx in T.Parallel(BLOCK_SIZE):
            out_reg[local_idx] = x_reg[local_idx] + y_reg[local_idx]
        T.copy(out_reg, output[start])

def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty_like(x)
    if autotune:
        with set_autotune_inputs(x, y, output):
            kernel = add_kernel.compile(x, y, output, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, y, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        add_kernel(
            x, y, output, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
