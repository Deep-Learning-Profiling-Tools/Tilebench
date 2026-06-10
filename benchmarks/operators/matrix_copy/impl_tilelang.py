import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config = None


def matrix_copy_configs():
    BLOCK_SIZE = [1024, 2048, 4096]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

#remove _ at start

@tilelang.autotune(configs = matrix_copy_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def matrix_copy_kernel(A, B, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    A: T.Tensor((n_elements, ), dtype)
    B: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            if idx < n_elements:
                B[idx] = A[idx]
                
# flatten to 1d as torch + cuTile implentation does the same
def run(A: torch.Tensor, N : int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config
    dtype = str(A.dtype).removeprefix("torch.")
    B = torch.empty_like(A)
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    if autotune:
        with set_autotune_inputs(A_flat, B_flat):
            kernel = matrix_copy_kernel.compile(A_flat, B_flat, dtype = dtype)
        _last_autotune_config = dict(kernel.config)
        kernel(A_flat, B_flat)
    else:
        _last_autotune_config = None
        cfg = _DEFAULT_CONFIG
        matrix_copy_kernel(
            A_flat, B_flat, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return B


def get_last_config() -> dict | None:
    return _last_autotune_config
