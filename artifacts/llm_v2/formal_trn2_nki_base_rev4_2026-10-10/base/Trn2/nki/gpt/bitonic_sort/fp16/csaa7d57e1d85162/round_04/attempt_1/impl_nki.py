import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _exchange_results(a, b, descending):
    swap = nl.ndarray(a.shape, dtype=nl.uint8, buffer=nl.sbuf)
    if descending:
        nisa.tensor_tensor(dst=swap, data1=a, data2=b, op=nl.less)
    else:
        nisa.tensor_tensor(dst=swap, data1=a, data2=b, op=nl.greater)

    result_a = nl.where(swap, b, a)
    result_b = nl.where(swap, a, b)
    return result_a, result_b


def _local_exchange(a, b, descending):
    result_a, result_b = _exchange_results(a, b, descending)
    nisa.tensor_copy(dst=a, src=result_a)
    nisa.tensor_copy(dst=b, src=result_b)


@nki.jit
def _initialize(data, work, N, B):
    M = work.shape[0]
    pid = nl.program_id(0)
    P = 128
    width = 2 * B
    tiles = M // (2 * P * width)
    output_view = work.reshape((2, tiles, P, width))

    for tile in range(tiles):
        base = pid * (M // 2) + tile * P * width
        values = nl.ndarray((P, width), dtype=data.dtype, buffer=nl.sbuf)

        valid = min(P * width, max(0, N - base))
        full_rows = valid // width
        remainder = valid - full_rows * width

        with nl.no_reorder():
            if valid < P * width:
                nisa.memset(dst=values, value=float("inf"))

            if full_rows > 0:
                source = data[
                    base:base + full_rows * width
                ].reshape((full_rows, width))
                nisa.dma_copy(dst=values[0:full_rows, :], src=source)

            if remainder > 0:
                start = base + full_rows * width
                source = data[start:start + remainder].reshape((1, remainder))
                nisa.dma_copy(
                    dst=values[full_rows:full_rows + 1, 0:remainder],
                    src=source,
                )

            k = 2
            while k <= B:
                j = k // 2
                while j > 0:
                    pattern = [
                        [width, P],
                        [2 * k, width // (2 * k)],
                        [2 * j, k // (2 * j)],
                        [1, j],
                    ]

                    a = values.ap(pattern=pattern, offset=0)
                    b = values.ap(pattern=pattern, offset=j)
                    _local_exchange(a, b, False)

                    a = values.ap(pattern=pattern, offset=k)
                    b = values.ap(pattern=pattern, offset=k + j)
                    _local_exchange(a, b, True)
                    j //= 2
                k *= 2

            nisa.dma_copy(dst=output_view[pid, tile, :, :], src=values)

    return work


def _global_tile(source_a, source_b, descending):
    a = nl.ndarray(source_a.shape, dtype=source_a.dtype, buffer=nl.sbuf)
    b = nl.ndarray(source_b.shape, dtype=source_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=source_a)
    nisa.dma_copy(dst=b, src=source_b)

    result_a, result_b = _exchange_results(a, b, descending)
    nisa.dma_copy(dst=source_a, src=result_a)
    nisa.dma_copy(dst=source_b, src=result_b)


@nki.jit
def _global_stage(work, k, j, free_cap):
    M = work.shape[0]
    pid = nl.program_id(0)
    F = min(free_cap, j)
    Q = j // F
    descending = False

    if k < M:
        R = M // (2 * k)
        S = k // (2 * j)
        base_core = pid * k
        descending = pid == 1

        if R >= S and R >= Q:
            P = min(128, R)
            nr = R // P
            ns = S
            nq = Q
            r_stride = 2 * k * P
            s_stride = 2 * j
            q_stride = F
            partition_stride = 2 * k
        elif S >= Q:
            P = min(128, S)
            nr = R
            ns = S // P
            nq = Q
            r_stride = 2 * k
            s_stride = 2 * j * P
            q_stride = F
            partition_stride = 2 * j
        else:
            P = min(128, Q)
            nr = R
            ns = S
            nq = Q // P
            r_stride = 2 * k
            s_stride = 2 * j
            q_stride = P * F
            partition_stride = F
    else:
        S = M // (2 * j)
        nq = 1
        q_stride = 1

        if S >= Q:
            P = min(128, S // 2)
            nr = S // (2 * P)
            ns = Q
            r_stride = 2 * j * P
            s_stride = F
            partition_stride = 2 * j
            base_core = pid * (M // 2)
        else:
            P = min(128, Q // 2)
            nr = S
            ns = Q // (2 * P)
            r_stride = 2 * j
            s_stride = P * F
            partition_stride = F
            base_core = pid * (j // 2)

    # Every access has a compile-time element offset. In particular, these
    # accesses do not dynamically index an access-pattern view.
    for r in range(nr):
        for s in range(ns):
            for q in range(nq):
                offset = (
                    base_core
                    + r * r_stride
                    + s * s_stride
                    + q * q_stride
                )
                source_a = work.ap(
                    pattern=[[partition_stride, P], [1, F]],
                    offset=offset,
                )
                source_b = work.ap(
                    pattern=[[partition_stride, P], [1, F]],
                    offset=offset + j,
                )
                _global_tile(source_a, source_b, descending)

    return work


def _tail_tile(source, C, descending):
    P = source.shape[0]
    values = nl.ndarray((P, C), dtype=source.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=source)

    with nl.no_reorder():
        j = C // 2
        while j > 0:
            pattern = [
                [C, P],
                [2 * j, C // (2 * j)],
                [1, j],
            ]
            a = values.ap(pattern=pattern, offset=0)
            b = values.ap(pattern=pattern, offset=j)
            _local_exchange(a, b, descending)
            j //= 2

        nisa.dma_copy(dst=source, src=values)


@nki.jit
def _local_tail(work, k, C):
    M = work.shape[0]
    pid = nl.program_id(0)
    descending = False

    if k < M:
        R = M // (2 * k)
        S = k // C
        base_core = pid * k
        descending = pid == 1

        if S >= R:
            P = min(128, S)
            nr = R
            nt = S // P
            r_stride = 2 * k
            t_stride = P * C
            partition_stride = C
        else:
            P = min(128, R)
            nr = R // P
            nt = S
            r_stride = 2 * k * P
            t_stride = C
            partition_stride = 2 * k
    else:
        P = 128
        nr = M // (2 * P * C)
        nt = 1
        r_stride = P * C
        t_stride = C
        partition_stride = C
        base_core = pid * (M // 2)

    for r in range(nr):
        for t in range(nt):
            offset = base_core + r * r_stride + t * t_stride
            source = work.ap(
                pattern=[[partition_stride, P], [1, C]],
                offset=offset,
            )
            _tail_tile(source, C, descending)

    return work


def run(data: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    if N <= 1:
        _LAST_CONFIG = {"algorithm": "copy", "length": N}
        return data.clone()

    M = 1 << ((N - 1).bit_length())
    B = min(16384, M // 512)
    C = 2 * B
    free_cap = 8192

    L = M.bit_length() - 1
    initial_levels = B.bit_length() - 1
    global_levels = L - (C.bit_length() - 1)
    global_passes = global_levels * (global_levels + 1) // 2
    tail_groups = L - initial_levels

    _LAST_CONFIG = {
        "algorithm": "strict_in_place_bitonic",
        "lnc": 2,
        "length": N,
        "padded_length": M,
        "initial_sort_block": B,
        "local_merge_block": C,
        "initial_partitions": 128,
        "initial_tiles_per_core": M // (256 * C),
        "global_free_cap": free_cap,
        "global_free_width_rule": "min(cap, partner_distance)",
        "partition_limit": 128,
        "global_partition_axis": "largest_of_regions_pairs_chunks",
        "tile_offsets": "compile_time",
        "network_stages": L * (L + 1) // 2,
        "global_passes": global_passes,
        "local_tail_groups": tail_groups,
        "synchronization": "kernel_boundaries",
        "kernel_launches": 1 + global_passes + tail_groups,
    }

    work = torch.empty((M,), dtype=data.dtype, device=data.device)
    _initialize[2](data, work, N, B)

    k = C
    while k <= M:
        j = k // 2
        while j >= C:
            _global_stage[2](work, k, j, free_cap)
            j //= 2

        # The remaining distances C/2, ..., 1 are confined to aligned
        # C-element blocks, each owned entirely by one physical core.
        _local_tail[2](work, k, C)
        k *= 2

    return work[:N]


def get_last_config() -> dict:
    return _LAST_CONFIG
