import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "threads": 128, "num_stages": 2}
_last_autotune_config: dict = {}


def mean_reduction_configs():
    BLOCK_N = [512, 1024, 2048]
    threads = [64, 128, 256]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_N=bn, threads=nt, num_stages=ns)
        for bn in BLOCK_N
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=mean_reduction_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def mean_reduction_kernel(
    x, output, in_dtype, out_dtype,
    BLOCK_N: int = 1024, threads: int = 128, num_stages: int = 2,
):
    M = T.dynamic("M")
    N = T.const("N")
    x: T.Tensor((M, N), in_dtype)
    output: T.Tensor((M,), out_dtype)

    with T.Kernel(M, threads=threads) as row:
        acc = T.alloc_fragment((BLOCK_N,) , "float32")
        row_sum = T.alloc_fragment((1,), "float32")
        T.fill(acc, 0.0)

        for tile_idx in T.Pipelined(T.ceildiv(N, BLOCK_N), num_stages=num_stages):
            for local_col in T.Parallel(BLOCK_N):
                col = tile_idx * BLOCK_N + local_col
                acc[local_col] += T.cast(x[row, col], "float32")

        T.reduce_sum(acc, row_sum, dim=0)
        output[row] = row_sum[0] / N


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    in_dtype = str(x.dtype).removeprefix("torch.")
    out_dtype = "float32"

    if x.ndim == 2 and dim == 1:
        x2d = x.contiguous()
    else:
        dims = list(range(x.ndim))
        dims.remove(dim % x.ndim)
        dims.append(dim % x.ndim)
        x2d = x.permute(dims).contiguous().reshape(-1, x.shape[dim])

    M, N = x2d.shape
    output = torch.empty(M, dtype=torch.float32, device=x.device)

    if autotune:
        with set_autotune_inputs(x2d, output):
            kernel = mean_reduction_kernel.compile(
                x2d, output,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x2d, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        mean_reduction_kernel(
            x2d, output,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
