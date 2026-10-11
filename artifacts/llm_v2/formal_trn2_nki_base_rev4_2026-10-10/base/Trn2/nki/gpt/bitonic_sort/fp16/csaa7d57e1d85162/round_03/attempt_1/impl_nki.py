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
def _global_stage(work, k, j, F):
    M = work.shape[0]
    pid = nl.program_id(0)
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

    # Every dynamic DMA index selects only the leading dimension of an
    # explicit [tile, partition, free] access-pattern view.
    if nr >= ns and nr >= nq:
        for s in range(ns):
            for q in range(nq):
                base = base_core + s * s_stride + q * q_stride

                if nr <= 4:
                    for r in range(nr):
                        offset = base + r * r_stride
                        source_a = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset,
                        )
                        source_b = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset + j,
                        )
                        _global_tile(source_a, source_b, descending)
                else:
                    grid_a = work.ap(
                        pattern=[
                            [r_stride, nr],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base,
                    )
                    grid_b = work.ap(
                        pattern=[
                            [r_stride, nr],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base + j,
                    )

                    def body_r(i):
                        _global_tile(grid_a[i], grid_b[i], descending)

                    nl.fori_loop(0, nr, body_r)

    elif ns >= nq:
        for r in range(nr):
            for q in range(nq):
                base = base_core + r * r_stride + q * q_stride

                if ns <= 4:
                    for s in range(ns):
                        offset = base + s * s_stride
                        source_a = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset,
                        )
                        source_b = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset + j,
                        )
                        _global_tile(source_a, source_b, descending)
                else:
                    grid_a = work.ap(
                        pattern=[
                            [s_stride, ns],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base,
                    )
                    grid_b = work.ap(
                        pattern=[
                            [s_stride, ns],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base + j,
                    )

                    def body_s(i):
                        _global_tile(grid_a[i], grid_b[i], descending)

                    nl.fori_loop(0, ns, body_s)

    else:
        for r in range(nr):
            for s in range(ns):
                base = base_core + r * r_stride + s * s_stride

                if nq <= 4:
                    for q in range(nq):
                        offset = base + q * q_stride
                        source_a = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset,
                        )
                        source_b = work.ap(
                            pattern=[[partition_stride, P], [1, F]],
                            offset=offset + j,
                        )
                        _global_tile(source_a, source_b, descending)
                else:
                    grid_a = work.ap(
                        pattern=[
                            [q_stride, nq],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base,
                    )
                    grid_b = work.ap(
                        pattern=[
                            [q_stride, nq],
                            [partition_stride, P],
                            [1, F],
                        ],
                        offset=base + j,
                    )

                    def body_q(i):
                        _global_tile(grid_a[i], grid_b[i], descending)

                    nl.fori_loop(0, nq, body_q)

    return work


def _tail_tile(source, B, descending):
    P = source.shape[0]
    values = nl.ndarray((P, B), dtype=source.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=source)

    with nl.no_reorder():
        j = B // 2
        while j > 0:
            pattern = [
                [B, P],
                [2 * j, B // (2 * j)],
                [1, j],
            ]
            a = values.ap(pattern=pattern, offset=0)
            b = values.ap(pattern=pattern, offset=j)
            _local_exchange(a, b, descending)
            j //= 2

        nisa.dma_copy(dst=source, src=values)


@nki.jit
def _local_tail(work, k, B):
    M = work.shape[0]
    pid = nl.program_id(0)
    descending = False

    if k < M:
        R = M // (2 * k)
        S = k // B
        base_core = pid * k
        descending = pid == 1

        if S >= R:
            P = min(128, S)
            nr = R
            nt = S // P
            r_stride = 2 * k
            t_stride = P * B
            partition_stride = B
        else:
            P = min(128, R)
            nr = R // P
            nt = S
            r_stride = 2 * k * P
            t_stride = B
            partition_stride = 2 * k
    else:
        P = 128
        nr = M // (2 * P * B)
        nt = 1
        r_stride = P * B
        t_stride = B
        partition_stride = B
        base_core = pid * (M // 2)

    if nr >= nt:
        for t in range(nt):
            base = base_core + t * t_stride

            if nr <= 4:
                for r in range(nr):
                    source = work.ap(
                        pattern=[[partition_stride, P], [1, B]],
                        offset=base + r * r_stride,
                    )
                    _tail_tile(source, B, descending)
            else:
                grid = work.ap(
                    pattern=[
                        [r_stride, nr],
                        [partition_stride, P],
                        [1, B],
                    ],
                    offset=base,
                )

                def body_r(i):
                    _tail_tile(grid[i], B, descending)

                nl.fori_loop(0, nr, body_r)
    else:
        for r in range(nr):
            base = base_core + r * r_stride

            if nt <= 4:
                for t in range(nt):
                    source = work.ap(
                        pattern=[[partition_stride, P], [1, B]],
                        offset=base + t * t_stride,
                    )
                    _tail_tile(source, B, descending)
            else:
                grid = work.ap(
                    pattern=[
                        [t_stride, nt],
                        [partition_stride, P],
                        [1, B],
                    ],
                    offset=base,
                )

                def body_t(i):
                    _tail_tile(grid[i], B, descending)

                nl.fori_loop(0, nt, body_t)

    return work


def run(data: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    if N <= 1:
        _LAST_CONFIG = {"algorithm": "copy", "length": N}
        return data.clone()

    M = 1 << ((N - 1).bit_length())
    B = min(16384, M // 512)
    F = 1024
    L = M.bit_length() - 1
    local_log = B.bit_length() - 1
    remaining_levels = L - local_log
    global_passes = remaining_levels * (remaining_levels + 1) // 2

    _LAST_CONFIG = {
        "algorithm": "strict_in_place_bitonic",
        "lnc": 2,
        "length": N,
        "padded_length": M,
        "local_block": B,
        "initial_partitions": 128,
        "initial_tiles_per_core": M // (512 * B),
        "global_free_width": F,
        "global_partition_limit": 128,
        "global_partition_axis": "largest_of_regions_pairs_chunks",
        "dynamic_indexing": "leading_tile_axis",
        "static_tile_limit": 4,
        "network_stages": L * (L + 1) // 2,
        "global_passes": global_passes,
        "local_tail_groups": remaining_levels,
        "synchronization": "kernel_boundaries",
        "kernel_launches": 1 + global_passes + remaining_levels,
    }

    # All device writes target this one allocation.
    work = torch.empty((M,), dtype=data.dtype, device=data.device)
    _initialize[2](data, work, N, B)

    k = 2 * B
    while k <= M:
        j = k // 2
        while j >= B:
            _global_stage[2](work, k, j, F)
            j //= 2

        _local_tail[2](work, k, B)
        k *= 2

    return work[:N]


def get_last_config() -> dict:
    return _LAST_CONFIG
