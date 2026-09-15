import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_TILE": 64, "threads": 128}
_last_autotune_config: dict = {}

def matrix_transpose_config():
    BLOCK_TILE = [32, 64, 128]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_TILE = bt, threads = nt)
        for bt in BLOCK_TILE
        for nt in threads
    ]

@tilelang.autotune(configs=matrix_transpose_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def matrix_transpose_kernel(
    x, output, dtype, 
    BLOCK_TILE: int = 64, threads: int = 128
):
    M, N = T.const("M, N")
    x: T.Tensor((M, N), dtype)
    output: T.Tensor((N, M), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_TILE), T.ceildiv(N, BLOCK_TILE), threads=threads) as (pid_m, pid_n):
        tile = T.alloc_shared((BLOCK_TILE, BLOCK_TILE), dtype)
        tile_T = T.alloc_shared((BLOCK_TILE, BLOCK_TILE), dtype)
        T.copy(
            x[pid_m * BLOCK_TILE : (pid_m + 1) * BLOCK_TILE,
              pid_n * BLOCK_TILE : (pid_n + 1) * BLOCK_TILE],
            tile,
        )
        T.transpose(tile, tile_T)
        T.copy(
            tile_T,
            output[pid_n * BLOCK_TILE : (pid_n + 1) * BLOCK_TILE,
                   pid_m * BLOCK_TILE : (pid_m + 1) * BLOCK_TILE],
        )
def run(x: torch.Tensor, block_size: int = 1024, 
        autotune: bool = False, **kwargs) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    if autotune:
        with set_autotune_inputs(x, output):
            kernel = matrix_transpose_kernel.compile(
                x, output, dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        matrix_transpose_kernel(
            x, output,
            dtype=dtype,
            BLOCK_TILE = cfg["BLOCK_TILE"],
            threads = cfg["threads"]
        )
    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None


    
