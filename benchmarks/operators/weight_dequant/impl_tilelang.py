import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def weight_dequant_configs():
    BLOCK_SIZE = [256, 512, 1024, 2048, 4096]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=weight_dequant_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def weight_dequant_kernel(
    X, S, output, dtype,
    N, TILE_SIZE,
    BLOCK_SIZE: int = 1024, threads: int = 128,
):
    n_elements = T.dynamic("n_elements")
    S_rows, S_cols = T.dynamic("S_rows, S_cols")
    X: T.Tensor((n_elements,), dtype)
    S: T.Tensor((S_rows, S_cols), dtype)
    output: T.Tensor((n_elements,), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = pid * BLOCK_SIZE + local_idx
            if idx < n_elements:
                row = idx // N
                col = idx % N
                s_row = row // TILE_SIZE
                s_col = col // TILE_SIZE
                value = T.cast(X[idx], "float32") * T.cast(S[s_row, s_col], "float32")
                output[idx] = T.cast(value, dtype)


def run(
    X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
    block_size: int = 1024, autotune: bool = False, **kwargs,
) -> torch.Tensor:
    dtype = str(X.dtype).removeprefix("torch.")
    output = torch.empty((M, N), dtype=X.dtype, device=X.device)
    X_flat = X.contiguous().view(-1)
    output_flat = output.view(-1)

    if autotune:
        with set_autotune_inputs(X_flat, S, output_flat):
            kernel = weight_dequant_kernel.compile(
                X_flat, S, output_flat,
                dtype=dtype,
                N=N,
                TILE_SIZE=TILE_SIZE,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(X_flat, S, output_flat)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        weight_dequant_kernel(
            X_flat, S, output_flat,
            dtype=dtype,
            N=N,
            TILE_SIZE=TILE_SIZE,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
