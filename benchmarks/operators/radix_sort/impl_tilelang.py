import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tilelang.math import next_power_of_2


_DEFAULT_CONFIG = {"threads": 128}
_BLOCK_SIZE = 1024
_RADIX_BITS = 2
_RADIX = 1 << _RADIX_BITS
_FIELD_BITS = 16
_FIELD_MASK = (1 << _FIELD_BITS) - 1
_last_autotune_config: dict = {}


def scatter_configs():
    return [dict(threads=nt) for nt in [64, 128, 256]]


@tilelang.jit
def radix_histogram_kernel(
    N,
    K,
    BLOCK_SIZE: int = 1024,
    RADIX: int = 4,
    FIELD_BITS: int = 16,
    FIELD_MASK: int = 65535,
    threads: int = 128,
):
    @T.prim_func
    def main(
        input: T.Tensor((N,), "int32"),
        hist: T.Tensor((RADIX * K,), "int32"),
        shift: T.int32,
    ):
        with T.Kernel(K, threads=threads) as pid:
            packed = T.alloc_fragment((BLOCK_SIZE,), "int64")
            total = T.alloc_fragment((1,), "int64")

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                valid = idx < N
                block = input[idx]
                digit = T.bitwise_and(T.shift_right(block, shift), RADIX - 1)
                packed[local_idx] = T.shift_left(
                    T.cast(valid, "int64"),
                    digit * FIELD_BITS,
                )

            T.reduce_sum(packed, total, dim=0, clear=True)

            for digit in T.Parallel(RADIX):
                hist[digit * K + pid] = T.cast(
                    T.bitwise_and(
                        T.shift_right(total[0], digit * FIELD_BITS),
                        FIELD_MASK,
                    ),
                    "int32",
                )

    return main


@tilelang.jit
def radix_sum_chunks_kernel(M, G2, BLOCK_SIZE: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        src: T.Tensor((M,), "int32"),
        dst: T.Tensor((G2,), "int32"),
    ):
        with T.Kernel(G2, threads=threads) as pid:
            vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
            total = T.alloc_fragment((1,), "int32")

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                vals[local_idx] = src[idx]

            T.reduce_sum(vals, total, dim=0, clear=True)
            dst[pid] = total[0]

    return main


@tilelang.jit
def radix_scan_chunk_sums_kernel(G2, BLOCK_BB: int = 1024, threads: int = 128):
    @T.prim_func
    def main(sums: T.Tensor((G2,), "int32")):
        with T.Kernel(1, threads=threads) as _:
            vals = T.alloc_fragment((BLOCK_BB,), "int32")
            original = T.alloc_fragment((BLOCK_BB,), "int32")

            for local_idx in T.Parallel(BLOCK_BB):
                valid = local_idx < G2
                vals[local_idx] = sums[local_idx]
                original[local_idx] = vals[local_idx]

            T.cumsum(vals, dim=0)

            for local_idx in T.Parallel(BLOCK_BB):
                if local_idx < G2:
                    sums[local_idx] = vals[local_idx] - original[local_idx]

    return main


@tilelang.jit
def radix_scan_chunks_kernel(M, G2, BLOCK_SIZE: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        src: T.Tensor((M,), "int32"),
        chunk_offsets: T.Tensor((G2,), "int32"),
    ):
        with T.Kernel(G2, threads=threads) as pid:
            vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
            original = T.alloc_fragment((BLOCK_SIZE,), "int32")
            base = chunk_offsets[pid]

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                vals[local_idx] = src[idx]
                original[local_idx] = vals[local_idx]

            T.cumsum(vals, dim=0)

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                if idx < M:
                    src[idx] = vals[local_idx] - original[local_idx] + base

    return main


@tilelang.autotune(configs=scatter_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def radix_scatter_kernel(
    N,
    K,
    BLOCK_SIZE: int = 1024,
    RADIX: int = 4,
    FIELD_BITS: int = 16,
    FIELD_MASK: int = 65535,
    threads: int = 128,
):
    @T.prim_func
    def main(
        input: T.Tensor((N,), "int32"),
        output: T.Tensor((N,), "int32"),
        hist: T.Tensor((RADIX * K,), "int32"),
        shift: T.int32,
    ):
        with T.Kernel(K, threads=threads) as pid:
            block = T.alloc_fragment((BLOCK_SIZE,), "int32")
            digit_vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
            packed = T.alloc_fragment((BLOCK_SIZE,), "int64")

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                valid = idx < N
                val = input[idx]
                digit = T.bitwise_and(T.shift_right(val, shift), RADIX - 1)
                block[local_idx] = val
                digit_vals[local_idx] = digit
                packed[local_idx] = T.shift_left(
                    T.cast(valid, "int64"),
                    digit * FIELD_BITS,
                )

            T.cumsum(packed, dim=0)

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                if idx < N:
                    digit = digit_vals[local_idx]
                    field = digit * FIELD_BITS
                    rank = T.cast(
                        T.bitwise_and(
                            T.shift_right(
                                packed[local_idx]
                                - T.shift_left(T.cast(1, "int64"), field),
                                field,
                            ),
                            FIELD_MASK,
                        ),
                        "int32",
                    )
                    base = hist[digit * K + pid]
                    output[base + rank] = block[local_idx]

    return main


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if N <= 1:
        return input.clone()

    work = input.clone()
    output = torch.empty_like(input)

    BLOCK_SIZE = _BLOCK_SIZE
    K = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    M = _RADIX * K
    G2 = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    BB = max(1024, next_power_of_2(G2))

    hist = torch.empty((M,), dtype=torch.int32, device=input.device)
    chunk_sums = torch.empty((G2,), dtype=torch.int32, device=input.device)

    hist_kernel = radix_histogram_kernel(
        N,
        K,
        BLOCK_SIZE=BLOCK_SIZE,
        RADIX=_RADIX,
        FIELD_BITS=_FIELD_BITS,
        FIELD_MASK=_FIELD_MASK,
        threads=128,
    )
    sum_chunks_kernel = radix_sum_chunks_kernel(
        M,
        G2,
        BLOCK_SIZE=BLOCK_SIZE,
        threads=128,
    )
    scan_chunk_sums_kernel = radix_scan_chunk_sums_kernel(
        G2,
        BLOCK_BB=BB,
        threads=128,
    )
    scan_chunks_kernel = radix_scan_chunks_kernel(
        M,
        G2,
        BLOCK_SIZE=BLOCK_SIZE,
        threads=128,
    )

    if autotune:
        with set_autotune_inputs(work, output, hist, 0):
            scatter_kernel = radix_scatter_kernel(
                N,
                K,
                BLOCK_SIZE=BLOCK_SIZE,
                RADIX=_RADIX,
                FIELD_BITS=_FIELD_BITS,
                FIELD_MASK=_FIELD_MASK,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(scatter_kernel.config or {}))
    else:
        _last_autotune_config.clear()
        scatter_kernel = radix_scatter_kernel(
            N,
            K,
            BLOCK_SIZE=BLOCK_SIZE,
            RADIX=_RADIX,
            FIELD_BITS=_FIELD_BITS,
            FIELD_MASK=_FIELD_MASK,
            threads=_DEFAULT_CONFIG["threads"],
        )

    for shift in range(0, 32, _RADIX_BITS):
        hist_kernel(work, hist, shift)
        sum_chunks_kernel(hist, chunk_sums)
        scan_chunk_sums_kernel(chunk_sums)
        scan_chunks_kernel(hist, chunk_sums)
        scatter_kernel(work, output, hist, shift)
        work, output = output, work

    return work


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
