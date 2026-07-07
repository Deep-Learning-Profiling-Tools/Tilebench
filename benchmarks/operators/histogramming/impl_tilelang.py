import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 1024,
    "NUM_PARTIAL": 256,
    "BLOCK_ROWS": 64,
    "BLOCK_BINS": 256,
    "threads_partial": 128,
    "threads_reduce": 128,
}
_last_autotune_config: dict = {}


def histogram_partial_configs():
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in [1024, 2048]
        for nt in [128, 256]
    ]


def histogram_reduce_configs():
    return [
        dict(BLOCK_ROWS=br, BLOCK_BINS=bb, threads=nt)
        for br in [64, 128]
        for bb in [64, 128]
        for nt in [128, 256]
    ]


@tilelang.autotune(configs=histogram_partial_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def histogram_partial_kernel(x, partial, BLOCK_SIZE: int = 1024, threads: int = 128):
    N, num_partials, num_bins = T.const("N, num_partials, num_bins")
    x: T.Tensor((N,), "int32")
    partial: T.Tensor((num_partials, num_bins), "int32")

    with T.Kernel(num_partials, threads=threads) as pid:
        for chunk_idx in T.serial(pid, T.ceildiv(N, BLOCK_SIZE), num_partials):
            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = chunk_idx * BLOCK_SIZE + local_idx
                in_bounds = idx < N
                val = T.if_then_else(in_bounds, x[idx], 0)
                valid = in_bounds and (val >= 0) and (val < num_bins)
                if valid:
                    T.atomic_add(partial[pid, val], 1)


@tilelang.autotune(configs=histogram_reduce_configs(), warmup=3, rep=10, timeout=60)
@tilelang.jit
def histogram_reduce_kernel(partial, histogram, BLOCK_ROWS: int = 64, BLOCK_BINS: int = 256, threads: int = 128):
    num_partials, num_bins = T.const("num_partials, num_bins")
    partial: T.Tensor((num_partials, num_bins), "int32")
    histogram: T.Tensor((num_bins,), "int32")

    with T.Kernel(T.ceildiv(num_bins, BLOCK_BINS), threads=threads) as pid_b:
        acc = T.alloc_fragment((BLOCK_BINS,), "int32")
        tile = T.alloc_fragment((BLOCK_ROWS, BLOCK_BINS), "int32")
        tile_sum = T.alloc_fragment((BLOCK_BINS,), "int32")
        T.fill(acc, 0)

        for row_start in T.serial(0, num_partials, BLOCK_ROWS):
            T.fill(tile, 0)
            T.copy(
                partial[
                    row_start : row_start + BLOCK_ROWS,
                    pid_b * BLOCK_BINS : (pid_b + 1) * BLOCK_BINS,
                ],
                tile,
            )
            T.reduce_sum(tile, tile_sum, dim=0, clear=True)
            for local_bin in T.Parallel(BLOCK_BINS):
                acc[local_bin] += tile_sum[local_bin]

        for local_bin in T.Parallel(BLOCK_BINS):
            bin_idx = pid_b * BLOCK_BINS + local_bin
            if bin_idx < num_bins:
                histogram[bin_idx] = acc[local_bin]


def run(input: torch.Tensor, N: int, num_bins: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    cfg = _DEFAULT_CONFIG
    BLOCK_SIZE = int(block_size) if block_size is not None else cfg["BLOCK_SIZE"]

    input = input.contiguous()
    histogram = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    num_partials = min(cfg["NUM_PARTIAL"], (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)

    if autotune:
        with set_autotune_inputs(input, partial):
            partial_kernel = histogram_partial_kernel.compile(input, partial)
        partial_cfg = dict(partial_kernel.config or {})
        partial.zero_()
        partial_kernel(input, partial)

        with set_autotune_inputs(partial, histogram):
            reduce_kernel = histogram_reduce_kernel.compile(partial, histogram)
        reduce_cfg = dict(reduce_kernel.config or {})
        reduce_kernel(partial, histogram)

        _last_autotune_config.clear()
        _last_autotune_config.update({
            "partial_BLOCK_SIZE": partial_cfg.get("BLOCK_SIZE"),
            "partial_threads": partial_cfg.get("threads"),
            "reduce_BLOCK_ROWS": reduce_cfg.get("BLOCK_ROWS"),
            "reduce_BLOCK_BINS": reduce_cfg.get("BLOCK_BINS"),
            "reduce_threads": reduce_cfg.get("threads"),
        })
    else:
        _last_autotune_config.clear()
        histogram_partial_kernel(
            input, partial,
            BLOCK_SIZE=BLOCK_SIZE,
            threads=cfg["threads_partial"],
        )
        histogram_reduce_kernel(
            partial, histogram,
            BLOCK_ROWS=cfg["BLOCK_ROWS"],
            BLOCK_BINS=cfg["BLOCK_BINS"],
            threads=cfg["threads_reduce"],
        )

    return histogram


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
