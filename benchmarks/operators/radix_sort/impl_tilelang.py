import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {"threads": 64}
_BLOCK_SIZE = 1024
_BLOCK_BB = 128
_last_autotune_config: dict = {}


def scatter_configs():
    return [dict(threads=nt) for nt in [64, 128, 256]]


@tilelang.jit
def count_ones_in_block_kernel(N, BLOCK_SIZE: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        input: T.Tensor((N,), "int32"),
        block_sum: T.Tensor((T.ceildiv(N, BLOCK_SIZE),), "int32"),
        bit: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=threads) as pid:
            bits = T.alloc_fragment((BLOCK_SIZE,), "int32")
            total = T.alloc_fragment((1,), "int32")

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                val = T.if_then_else(idx < N, input[idx], 0)
                bits[local_idx] = T.bitwise_and(T.shift_right(val, bit), 1)

            T.reduce_sum(bits, total, dim=0, clear=True)
            block_sum[pid] = total[0]

    return main


@tilelang.jit
def count_ones_per_block_blocks_kernel(first_layer_sum, block_block_sum, BLOCK_SIZE: int = 1024, threads: int = 128):
    K, L = T.const("K, L")
    first_layer_sum: T.Tensor((K,), "int32")
    block_block_sum: T.Tensor((L,), "int32")

    with T.Kernel(T.ceildiv(K, BLOCK_SIZE), threads=threads) as pid:
        vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
        total = T.alloc_fragment((1,), "int32")

        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = pid * BLOCK_SIZE + local_idx
            vals[local_idx] = T.if_then_else(idx < K, first_layer_sum[idx], 0)

        T.reduce_sum(vals, total, dim=0, clear=True)
        block_block_sum[pid] = total[0]


@tilelang.jit
def compute_prefix_sums_bb_kernel(block_block_sum, global_ones, BLOCK_SIZE: int = 128, threads: int = 128):
    L = T.const("L")
    block_block_sum: T.Tensor((L,), "int32")
    global_ones: T.Tensor((1,), "int32")

    with T.Kernel(1, threads=threads) as _:
        vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
        original = T.alloc_fragment((BLOCK_SIZE,), "int32")
        total = T.alloc_fragment((1,), "int32")

        for local_idx in T.Parallel(BLOCK_SIZE):
            valid = local_idx < L
            vals[local_idx] = T.if_then_else(valid, block_block_sum[local_idx], 0)
            original[local_idx] = vals[local_idx]

        T.cumsum(vals, dim=0)

        for local_idx in T.Parallel(BLOCK_SIZE):
            if local_idx < L:
                block_block_sum[local_idx] = vals[local_idx] - original[local_idx]

        T.reduce_sum(original, total, dim=0, clear=True)
        global_ones[0] = total[0]


@tilelang.jit
def compute_prefix_sums_per_block_kernel(first_layer_sum, block_block_sum, BLOCK_SIZE: int = 1024, threads: int = 128):
    K, L = T.const("K, L")
    first_layer_sum: T.Tensor((K,), "int32")
    block_block_sum: T.Tensor((L,), "int32")

    with T.Kernel(T.ceildiv(K, BLOCK_SIZE), threads=threads) as pid:
        vals = T.alloc_fragment((BLOCK_SIZE,), "int32")
        original = T.alloc_fragment((BLOCK_SIZE,), "int32")
        prefix = block_block_sum[pid]

        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = pid * BLOCK_SIZE + local_idx
            vals[local_idx] = T.if_then_else(idx < K, first_layer_sum[idx], 0)
            original[local_idx] = vals[local_idx]

        T.cumsum(vals, dim=0)

        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = pid * BLOCK_SIZE + local_idx
            if idx < K:
                first_layer_sum[idx] = vals[local_idx] - original[local_idx] + prefix


@tilelang.autotune(configs=scatter_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def radix_sort_kernel(N, BLOCK_SIZE: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        input: T.Tensor((N,), "int32"),
        output: T.Tensor((N,), "int32"),
        first_layer_sum: T.Tensor((T.ceildiv(N, BLOCK_SIZE),), "int32"),
        global_ones: T.Tensor((1,), "int32"),
        bit: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=threads) as pid:
            block = T.alloc_fragment((BLOCK_SIZE,), "int32")
            bit_values = T.alloc_fragment((BLOCK_SIZE,), "int32")
            ones_scan = T.alloc_fragment((BLOCK_SIZE,), "int32")
            zeros_scan = T.alloc_fragment((BLOCK_SIZE,), "int32")

            ones_before = first_layer_sum[pid]
            zeros_before = pid * BLOCK_SIZE - ones_before
            global_zeros = N - global_ones[0]

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                valid = idx < N
                val = T.if_then_else(valid, input[idx], 0)
                bit_value = T.bitwise_and(T.shift_right(val, bit), 1)
                block[local_idx] = val
                bit_values[local_idx] = bit_value
                ones_scan[local_idx] = bit_value
                zeros_scan[local_idx] = 1 - bit_value

            T.cumsum(ones_scan, dim=0)
            T.cumsum(zeros_scan, dim=0)

            for local_idx in T.Parallel(BLOCK_SIZE):
                idx = pid * BLOCK_SIZE + local_idx
                if idx < N:
                    bit_value = bit_values[local_idx]
                    ones_rank = ones_scan[local_idx] - bit_value
                    zeros_rank = zeros_scan[local_idx] - (1 - bit_value)
                    dest = T.if_then_else(
                        bit_value == 0,
                        zeros_before + zeros_rank,
                        global_zeros + ones_before + ones_rank,
                    )
                    output[dest] = block[local_idx]

    return main


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if N <= 1:
        return input.clone()

    work = input.clone()
    output = torch.empty_like(input)

    BLOCK_SIZE = _BLOCK_SIZE
    K = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    L = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    first_layer = torch.empty((K,), dtype=torch.int32, device=input.device)
    second_layer = torch.empty((L,), dtype=torch.int32, device=input.device)
    global_ones = torch.empty((1,), dtype=torch.int32, device=input.device)

    count_kernel = count_ones_in_block_kernel(N, BLOCK_SIZE=BLOCK_SIZE, threads=128)

    if autotune:
        with set_autotune_inputs(work, output, first_layer, global_ones, 0):
            scatter_kernel = radix_sort_kernel(N, BLOCK_SIZE=BLOCK_SIZE)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(scatter_kernel.config or {}))
    else:
        _last_autotune_config.clear()
        scatter_kernel = radix_sort_kernel(
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            threads=_DEFAULT_CONFIG["threads"],
        )

    for bit in range(32):
        count_kernel(work, first_layer, bit)
        count_ones_per_block_blocks_kernel(
            first_layer,
            second_layer,
            BLOCK_SIZE=BLOCK_SIZE,
            threads=128,
        )
        compute_prefix_sums_bb_kernel(
            second_layer,
            global_ones,
            BLOCK_SIZE=_BLOCK_BB,
            threads=128,
        )
        compute_prefix_sums_per_block_kernel(
            first_layer,
            second_layer,
            BLOCK_SIZE=BLOCK_SIZE,
            threads=128,
        )
        scatter_kernel(work, output, first_layer, global_ones, bit)
        work.copy_(output)

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
