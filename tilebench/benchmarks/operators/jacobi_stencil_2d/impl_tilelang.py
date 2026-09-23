import torch  
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_R": 1,
    "BLOCK_SIZE_C": 1024,
    "threads": 128,
}
_last_autotune_config: dict = {}
def jacobi_stencil_configs():
    BLOCK_SIZE_R = [1, 2, 4]
    BLOCK_SIZE_C = [256, 512, 1024, 2048]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE_R=br, BLOCK_SIZE_C=bc, threads=nt)
        for br in BLOCK_SIZE_R
        for bc in BLOCK_SIZE_C
        for nt in threads
    ]
@tilelang.autotune(configs=jacobi_stencil_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def jacobi_stencil_kernel(input, output, dtype,
                          BLOCK_SIZE_R: int = 1, 
                          BLOCK_SIZE_C: int = 1024,
                          threads: int = 128):

    M, N = T.const("M, N")
    input: T.Tensor((M, N), dtype)
    output: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_SIZE_R), T.ceildiv(N, BLOCK_SIZE_C), threads=threads) as (pid_r, pid_c):
        local_tile_out = T.alloc_fragment((BLOCK_SIZE_R, BLOCK_SIZE_C), dtype)
        start_r = pid_r * BLOCK_SIZE_R 
        start_c = pid_c * BLOCK_SIZE_C
        T.copy(input[start_r, start_c], local_tile_out)
        quarter = T.cast(0.25, dtype)
        for i, j in T.Parallel(BLOCK_SIZE_R, BLOCK_SIZE_C):
            gb_r = i + start_r 
            gb_c = j + start_c
            is_inner = gb_r >= 1 and gb_c >= 1 and gb_r < M - 1 and gb_c < N - 1
            # top + bottom + left + right, matching the torch reference and the
            # Triton/cuTile kernels: fp16 adds round at each step, so summing the
            # neighbours in a different order drifts by 1 ULP on ~31% of points.
            avg = quarter * (
                input[gb_r - 1, gb_c] +
                input[gb_r + 1, gb_c] +
                input[gb_r, gb_c - 1] +
                input[gb_r, gb_c + 1]
            )
            local_tile_out[i, j] = T.Select(is_inner, avg, local_tile_out[i, j])
        T.copy(local_tile_out, output[start_r, start_c])
        
def run(input: torch.Tensor, rows: int, cols: int,
    block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    dtype = str(input.dtype).removeprefix("torch.")
    if autotune:
        with set_autotune_inputs(input, output):
            tuned_kernel = jacobi_stencil_kernel.compile(
                input, output, dtype=dtype
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        jacobi_stencil_kernel(
            input, output, dtype=dtype,
            BLOCK_SIZE_R=cfg["BLOCK_SIZE_R"],
            BLOCK_SIZE_C=cfg["BLOCK_SIZE_C"],
            threads=cfg["threads"]
        )
    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
        
