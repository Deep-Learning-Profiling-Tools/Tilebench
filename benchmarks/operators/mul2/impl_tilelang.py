import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config = None
def mul2_configs():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs = mul2_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def mul2_kernel(x, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements, ), dtype)
    output: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            if idx < n_elements:
                output[idx] = x[idx] * 2

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config
    dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty_like(x)
    if autotune:
        with set_autotune_inputs(x, output):
            kernel = mul2_kernel.compile(x, output, dtype = dtype)
        _last_autotune_config = dict(kernel.config)
        kernel(x, output)
    else:
        _last_autotune_config = None
        cfg = _DEFAULT_CONFIG
        mul2_kernel(
            x, output, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output

def get_last_config() -> dict | None:
    return _last_autotune_config
