import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def interleave_configs():
    BLOCK_SIZE = [1024, 2048, 4096, 8192]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=interleave_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def interleave_kernel(A, B, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    A: T.Tensor((n_elements,), dtype)
    B: T.Tensor((n_elements,), dtype)
    output: T.Tensor((2 * n_elements,), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        start = pid * BLOCK_SIZE
        A_reg = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        B_reg = T.alloc_fragment((BLOCK_SIZE, ), dtype)
        output_reg = T.alloc_fragment((2 * BLOCK_SIZE, ), dtype)
        T.copy(A[start], A_reg)
        T.copy(B[start], B_reg)
        for out_local_idx in T.Parallel(2 * BLOCK_SIZE):
            in_local_idx = out_local_idx // 2
            output_reg[out_local_idx] = T.Select(
                out_local_idx % 2 == 0,
                A_reg[in_local_idx],
                B_reg[in_local_idx],
            )
        T.copy(output_reg, output[2 * start])


def run(A: torch.Tensor, B: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    dtype = str(A.dtype).removeprefix("torch.")
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    if autotune:
        with set_autotune_inputs(A, B, output):
            kernel = interleave_kernel.compile(A, B, output, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(A, B, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        interleave_kernel(
            A, B, output, dtype=dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
