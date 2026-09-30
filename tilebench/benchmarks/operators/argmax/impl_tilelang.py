import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_N": 256, "threads": 128, "num_stages": 2}
_last_autotune_config: dict = {}


def argmax_rowwise_config():
    BLOCK_N = [256, 512, 1024, 2048]
    threads = [128, 256, 512]
    num_stages = [2, 3, 4]
    return [
        dict(BLOCK_N=bn, threads=nt, num_stages=ns)
        for bn in BLOCK_N
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=argmax_rowwise_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def argmax_rowwise_kernel(X, Out, dtype, BLOCK_N: int = 256, threads: int = 128, num_stages: int = 2):
    M = T.dynamic("M")
    N = T.const("N")
    X: T.Tensor((M, N), dtype)
    Out: T.Tensor((M,), "int64")

    with T.Kernel(M, threads=threads) as row:
        x_tile = T.alloc_fragment((BLOCK_N,), "float32")
        idx_tile = T.alloc_fragment((BLOCK_N,), "int32")
        tile_max = T.alloc_fragment((1,), "float32")
        tile_idx = T.alloc_fragment((1,), "int32")
        best_val = T.alloc_fragment((1,), "float32")
        best_idx = T.alloc_fragment((1,), "int32")

        best_val[0] = -T.infinity("float32")
        best_idx[0] = 0

        T.annotate_safe_value({X: -T.infinity(dtype)})
        for tile in T.Pipelined(T.ceildiv(N, BLOCK_N), num_stages=num_stages):
            start = tile * BLOCK_N

            T.fill(x_tile, -T.infinity("float32"))
            for i in T.Parallel(BLOCK_N):
                col = start + i
                x_tile[i] = T.Cast("float32", X[row, col])
            T.reduce_max(x_tile, tile_max, dim=0, clear=True)

            for i in T.Parallel(BLOCK_N):
                col = start + i
                idx_tile[i] = T.Select(x_tile[i] == tile_max[0], col, N)

            T.reduce_min(idx_tile, tile_idx, dim=0, clear=True)

            better = tile_max[0] > best_val[0]
            best_val[0] = T.Select(better, tile_max[0], best_val[0])
            best_idx[0] = T.Select(better, tile_idx[0], best_idx[0])

        Out[row] = T.cast(best_idx[0], "int64")


def run(
    x: torch.Tensor,
    dim: int = 1,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    dtype = str(x.dtype).removeprefix("torch.")

    if dim == 1:
        x2d = x.contiguous()
    else:
        x2d = x.transpose(0, 1).contiguous()

    M, _ = x2d.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    if autotune:
        with set_autotune_inputs(x2d, output):
            kernel = argmax_rowwise_kernel.compile(
                x2d, output,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x2d, output)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        argmax_rowwise_kernel(
            x2d, output, dtype,
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
